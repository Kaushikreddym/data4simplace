"""Tests for the calibration package.

The optimiser itself is torch's and the model is torchcrop's; what is tested
here is the wiring that has been got wrong before, and the three properties a
calibration silently fails without:

* the **season alignment** — an autumn emergence belongs to the *following*
  harvest year, and getting it wrong still produces a plausible season length;
* the **batch invariant** — one region and one year per forward pass, because a
  table ordinate cannot be ``[B]``-shaped and the engine takes one scalar
  ``start_doy``;
* the **gradient path** — that the maturity date moves with ``tsum1``, which it
  does not if the date is read off the clamped DVS trajectory.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchcrop")

from cropmodelling4eu.calibration.config import (  # noqa: E402
    CLMS_HARVEST_OFFSET_DAYS,
    CalibrationConfig,
    DataConfig,
    LossConfig,
)
from cropmodelling4eu.calibration.dataset import (  # noqa: E402
    StratifiedFractionSampler,
)
from cropmodelling4eu.calibration.losses import (  # noqa: E402
    RunningClimatology,
    segment_mean,
    sigmoid_fall_day,
)
from cropmodelling4eu.calibration.observations import (  # noqa: E402
    ObservationSet,
    doy_to_date,
    load_observations,
)
from cropmodelling4eu.calibration.problem import (  # noqa: E402
    CalibrationProblem,
    SimplexGroup,
    load_stage_specs,
)
from cropmodelling4eu.calibration.splits import make_split  # noqa: E402

STAGE_FILES = ("phenology", "yield", "lai")


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


def test_autumn_stage_moves_to_the_previous_calendar_year():
    """An emergence at DOY 301 of harvest year 2020 happened in autumn 2019.

    This is the alignment bug that keeps coming back: pivoting on the raw year
    pairs one season's emergence with the previous season's harvest and still
    produces a ~300 d season length and a near-zero bias.
    """
    autumn = doy_to_date(pd.Series([2020]), pd.Series([301.0]), autumn=True)
    assert autumn.iloc[0].year == 2019

    january = doy_to_date(pd.Series([2020]), pd.Series([20.0]), autumn=True)
    assert january.iloc[0].year == 2020

    harvest = doy_to_date(pd.Series([2020]), pd.Series([200.0]), autumn=False)
    assert harvest.iloc[0].year == 2020


def test_clms_harvest_carries_the_measured_offset():
    """``CPMCH`` is corrected onto the ground scale, not left for tsum2."""
    clms = pd.DataFrame(
        {
            "adm_id": ["DE111"],
            "year": [2020],
            "obs_emergence_doy": [300.0],
            "obs_harvest_doy": [200.0],
            "emergence_doy_p10": [290.0], "emergence_doy_p90": [310.0],
            "harvest_doy_p10": [195.0], "harvest_doy_p90": [205.0],
        }
    )
    table = load_observations(clms=clms).table
    harvest = table.loc[table["variable"] == "harvest_date", "value"].iloc[0]
    naive = pd.Timestamp("2020-01-01") + pd.Timedelta(days=199)
    assert (pd.Timestamp(harvest) - naive).days == pytest.approx(
        CLMS_HARVEST_OFFSET_DAYS, abs=1
    )


def test_a_missing_date_is_dropped_not_turned_into_a_sentinel():
    """`pd.to_numeric` maps NaT to the int64 minimum, which is finite.

    So a station-season with no observed date would enter the loss as
    -9.22e18 ns — a date 292 years before the Big Bang — and neither the
    `notna` filter nor the loss's `isfinite` mask would drop it. PEP725 has
    3 800 seasons with no emergence and 4 566 with no heading, so this is the
    common case, not the edge one.
    """
    pep = pd.DataFrame(
        {
            "SimplaceID": [11, 22],
            "harvest_year": [2020, 2020],
            "emergence_doy": [300.0, np.nan],   # the second station has none
            "harvest_doy": [200.0, 205.0],
            "heading_doy": [150.0, np.nan],
        }
    )
    table = load_observations(pep=pep).table
    assert set(table.loc[table["variable"] == "emergence_date", "unit"]) == {"11"}
    assert set(table.loc[table["variable"] == "harvest_date", "unit"]) == {"11", "22"}
    assert (table["value"] > 0).all(), "no NaT sentinel survived"


def test_a_wider_pixel_spread_is_a_weaker_observation():
    """The within-unit spread is the observation's precision, so it is a weight."""
    clms = pd.DataFrame(
        {
            "adm_id": ["TIGHT", "WIDE"],
            "year": [2020, 2020],
            "obs_emergence_doy": [300.0, 300.0],
            "obs_harvest_doy": [200.0, 200.0],
            "emergence_doy_p10": [298.0, 270.0],
            "emergence_doy_p90": [302.0, 330.0],
            "harvest_doy_p10": [199.0, 199.0],
            "harvest_doy_p90": [201.0, 201.0],
        }
    )
    table = load_observations(clms=clms).table
    emergence = table[table["variable"] == "emergence_date"].set_index("unit")
    assert emergence.loc["TIGHT", "weight"] > emergence.loc["WIDE", "weight"]


def test_yield_is_split_into_a_level_and_a_detrended_anomaly():
    """CyBench carries a technology trend LINTUL-5 cannot produce.

    Fitting raw levels lets the parameters absorb it, so the level and the
    anomaly are separate rows and the anomaly is the residual from a per-unit
    line — not from the unit mean, which would leave the trend inside it.
    """
    years = np.arange(2000, 2020)
    yields = pd.DataFrame(
        {
            "adm_id": "DE111",
            "harvest_year": years,
            # A pure trend: 5 t/ha rising by 0.05 t/ha a year, no interannual
            # variation at all.
            "yield": 5.0 + 0.05 * (years - 2000),
            "harvest_area": 1000.0,
            "country": "DE",
        }
    )
    table = load_observations(yields=yields).table
    anomaly = table.loc[table["variable"] == "yield_anomaly", "value"]
    assert anomaly.abs().max() < 1e-6, "a pure trend must leave no anomaly"

    level = table.loc[table["variable"] == "yield_level", "value"]
    assert level.nunique() == 1
    assert float(level.iloc[0]) == pytest.approx(5.475, abs=1e-3)


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #


def _unit_years(n_regions: int = 3, n_units: int = 10, n_years: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "unit": f"R{r}U{u}", "unit_kind": "adm", "region_id": r,
                "year": 2017 + y, "n_cells": 4,
            }
            for r in range(n_regions)
            for u in range(n_units)
            for y in range(n_years)
        ]
    )


def test_every_batch_is_one_region():
    """The invariant the whole dataset module exists to hold.

    A table ordinate cannot be ``[B]``-shaped — ``rebuild_table`` stacks scalar
    ordinates — so a forward pass carries exactly one region's parameters.
    Years are *not* constrained: every window in a region opens on the same
    day-of-year, so seasons share one relative axis and one scalar
    ``start_doy``, and mixing them is what keeps a batch full.
    """
    sampler = StratifiedFractionSampler(_unit_years(), fraction=0.5, batch_size=16)
    batches = sampler.epoch(0)
    assert batches
    for batch in batches:
        assert batch["region_id"].nunique() == 1
    assert max(b["year"].nunique() for b in batches) > 1, (
        "year-homogeneous batches would waste most of every forward pass"
    )


def test_the_draw_is_stratified_by_region_and_year():
    """An unstratified draw can produce a 'wet years only' epoch."""
    sampler = StratifiedFractionSampler(_unit_years(), fraction=0.2, batch_size=64)
    drawn = pd.concat(sampler.epoch(0))
    assert drawn["region_id"].nunique() == 3
    assert drawn["year"].nunique() == 8


def test_a_stratum_too_small_for_the_fraction_still_contributes():
    """At 20 % a region-year with three units would otherwise draw nothing."""
    small = _unit_years(n_regions=1, n_units=3, n_years=2)
    drawn = pd.concat(StratifiedFractionSampler(small, 0.2, 64).epoch(0))
    assert len(drawn) == 2, "one unit per stratum is the floor"


def test_batches_are_packed_by_cell_years_not_unit_years():
    """The batch limit is what a forward pass costs, which is cells."""
    frame = _unit_years(n_regions=1, n_units=20, n_years=1)
    frame["n_cells"] = 10
    for batch in StratifiedFractionSampler(frame, 1.0, 32).epoch(0):
        assert batch["n_cells"].sum() <= 32 or len(batch) == 1


def test_epochs_are_deterministic_and_differ_between_themselves():
    sampler = StratifiedFractionSampler(_unit_years(), 0.2, 64, seed=7)
    first = pd.concat(sampler.epoch(0))
    assert first.equals(pd.concat(sampler.epoch(0))), "a killed run must resume identically"
    assert not first.equals(pd.concat(sampler.epoch(1)))


# --------------------------------------------------------------------------- #
# Region merges — a deliberate override of the silhouette's split
# --------------------------------------------------------------------------- #


def test_a_merge_folds_the_named_clusters_and_renames_for_its_parts():
    """The merged region must say what it is made of.

    Renumbering after a merge is what would otherwise make ``PAN-1`` point at a
    different region than it did before — a silent change of meaning in the one
    label the parameter tables are keyed by.
    """
    from cropmodelling4eu.calibration.regions import _apply_merges

    labels = np.array([0] * 5 + [1] * 3 + [2] * 4)
    merged, origins, applied = _apply_merges(labels, "PAN", (("PAN-1", "PAN-3"),))

    assert applied == {("PAN-1", "PAN-3")}
    assert len(set(merged.tolist())) == 2
    names = {int(k): "PAN-" + "+".join(str(i + 1) for i in v) for k, v in origins.items()}
    assert sorted(names.values()) == ["PAN-1+3", "PAN-2"]
    # The two merged clusters end in one label, the untouched one on its own.
    assert merged[0] == merged[8] and merged[5] != merged[0]


def test_a_zone_the_merge_does_not_name_is_untouched():
    from cropmodelling4eu.calibration.regions import _apply_merges

    labels = np.array([0] * 5 + [1] * 3 + [2] * 4)
    merged, origins, applied = _apply_merges(labels, "CON", (("PAN-1", "PAN-3"),))

    assert not applied
    assert np.array_equal(merged, labels)
    assert origins == {0: (0,), 1: (1,), 2: (2,)}


def test_a_merge_naming_a_cluster_the_zone_lacks_raises():
    """A stale name must not leave the run fitting regions the config merged."""
    from cropmodelling4eu.calibration.regions import _apply_merges

    with pytest.raises(ValueError, match="did not produce"):
        _apply_merges(np.array([0, 0, 1, 1]), "PAN", (("PAN-1", "PAN-3"),))


def test_a_merge_across_two_zones_is_refused():
    """A region is a subdivision of one EnZ zone; a pair spanning two is not."""
    from cropmodelling4eu.calibration.config import RegionConfig

    with pytest.raises(ValueError, match="same EnZ zone|share an EnZ zone"):
        RegionConfig(merge_regions=(("PAN-1", "CON-2"),))


# --------------------------------------------------------------------------- #
# Sowing convention
# --------------------------------------------------------------------------- #


def _pool_inputs():
    """Minimal regions / observations / bundle stubs for :func:`build_pool`."""
    from cropmodelling4eu.calibration.regions import CalibrationRegions

    table = pd.DataFrame(
        {
            "SimplaceID": [1, 2, 3, 4],
            "lon": 10.0, "lat": 52.0,
            "adm_id": pd.array(["DE111"] * 4, dtype="string"),
            "region_id": 1,
            "region": pd.array(["CON-1"] * 4, dtype="string"),
            "enz_name": "CON", "mean_yield": 7.0,
        }
    )
    summary = pd.DataFrame(
        [{"region_id": 1, "region": "CON-1", "n_cells": 4, "n_units": 1,
          "enz": "CON", "mean_yield": 7.0}]
    )
    observations = ObservationSet(
        pd.DataFrame(
            {
                "unit": pd.array(["DE111", "DE111"], dtype="string"),
                "unit_kind": "adm", "year": [2020, 2021],
                "variable": "harvest_date", "value": [1.0, 2.0],
                "weight": 1.0, "support": "adm", "source": "clms",
            }
        )
    )

    class _Site:
        @staticmethod
        def sowing_doy(ids):
            return np.full(len(ids), 285)          # the export's fixed calendar

    class _Bundle:
        ids = np.array([1, 2, 3, 4], dtype=np.int64)
        site = _Site()

    return CalibrationRegions(table, summary), observations, _Bundle()


def test_the_site_calendar_gives_every_season_the_same_sowing_date():
    """Which is why it cannot be the convention a phenology stage fits on.

    `site.csv` publishes one date per cell, so simulated emergence has no
    interannual variation at all — and the anomaly is what stage 1 is for.
    """
    from cropmodelling4eu.calibration.dataset import build_pool

    regions, observations, bundle = _pool_inputs()
    pool = build_pool(observations, regions, bundle, DataConfig())
    assert pool.groupby("SimplaceID")["sowing_doy"].nunique().eq(1).all()


def test_a_simulated_sowing_table_replaces_the_calendar_per_season():
    """The convention the production run is on, read from a static file."""
    from cropmodelling4eu.calibration.dataset import build_pool

    regions, observations, bundle = _pool_inputs()
    sowing = pd.DataFrame(
        [
            {"SimplaceID": cell, "year": year, "sowing_doy": 260 + offset}
            for cell in (1, 2, 3, 4)
            for year, offset in ((2020, 0), (2021, 5))
        ]
    )
    pool = build_pool(observations, regions, bundle, DataConfig(), sowing)
    assert pool.groupby("SimplaceID")["sowing_doy"].nunique().eq(2).all()
    assert pool["anchor_doy"].eq(260).all(), "the anchor is the region's earliest"


def test_a_cell_season_the_sowing_table_misses_is_dropped_not_fallen_back():
    """One pool must never mix two sowing conventions."""
    from cropmodelling4eu.calibration.dataset import build_pool

    regions, observations, bundle = _pool_inputs()
    partial = pd.DataFrame(
        [{"SimplaceID": cell, "year": 2020, "sowing_doy": 260} for cell in (1, 2, 3, 4)]
    )
    pool = build_pool(observations, regions, bundle, DataConfig(), partial)
    assert set(pool["year"]) == {2020}, "2021 has no row in the table"


def test_a_cell_can_sit_in_two_regions_and_needs_an_anchor_per_region():
    """PEP725 makes a cell its own observation unit; CyBench puts it in an adm.

    Those two units can land in different regions, and each region anchors its
    window on its own earliest sowing day. Keyed on (cell, year) alone the cache
    keeps one window and the cell then disagrees with its batch-mates about
    which day the window opens — which is exactly the failure that killed the
    first full run.
    """
    from cropmodelling4eu.calibration.dataset import WeatherCache

    assert WeatherCache.KEY == ("SimplaceID", "year", "anchor_doy")

    pool = pd.DataFrame(
        {
            "SimplaceID": [5, 5],
            "year": [2020, 2020],
            "anchor_doy": [240, 260],      # same cell, two regions' anchors
        }
    )
    wanted = pool[list(WeatherCache.KEY)].drop_duplicates()
    assert len(wanted) == 2, "both windows must be cached, not one of them"


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #


def test_segment_mean_drops_a_cell_that_never_reached_the_threshold():
    """A NaN is a cell with no date, not a zero — it must not poison the unit."""
    values = torch.tensor([10.0, float("nan"), 20.0, float("nan")])
    index = torch.tensor([0, 0, 1, 1])
    mean, valid = segment_mean(values, index, 2)
    assert float(mean[0]) == pytest.approx(10.0)
    assert bool(valid[0]) and bool(valid[1])

    all_nan = torch.tensor([float("nan"), float("nan")])
    mean, valid = segment_mean(all_nan, torch.tensor([0, 0]), 1)
    assert not bool(valid[0])


def test_segment_mean_is_weighted_and_differentiable():
    values = torch.tensor([2.0, 4.0], requires_grad=True)
    mean, _ = segment_mean(
        values, torch.tensor([0, 0]), 1, weights=torch.tensor([3.0, 1.0])
    )
    assert float(mean[0]) == pytest.approx(2.5)
    mean.sum().backward()
    # The weighted mean is linear, so the gradient is the normalised weight.
    assert values.grad.tolist() == pytest.approx([0.75, 0.25])


def test_sigmoid_fall_day_handles_a_hump_that_crossing_day_cannot():
    """fAPAR rises, plateaus and falls, so the first day below half the peak is day 0."""
    series = torch.tensor([[0.0, 0.5, 1.0, 1.0, 0.5, 0.0]])
    day = sigmoid_fall_day(series, torch.tensor([0.5]), sharpness=50.0)
    assert 2.0 < float(day[0]) < 4.0


def test_the_climatology_is_detached():
    """It is a reference level; a gradient through it is a second level term."""
    climatology = RunningClimatology(momentum=0.5)
    values = torch.tensor([10.0], requires_grad=True)
    reference = climatology.update("x", np.array(["u"]), values, torch.tensor([True]))
    assert not reference.requires_grad
    # First sighting seeds its own reference, so the anomaly is zero rather
    # than undefined.
    assert float(reference[0]) == pytest.approx(10.0)

    # The reference a batch is scored against is the climatology *before* that
    # batch; the EMA moves afterwards.
    second = climatology.update(
        "x", np.array(["u"]), torch.tensor([20.0]), torch.tensor([True])
    )
    assert float(second[0]) == pytest.approx(10.0)
    third = climatology.update(
        "x", np.array(["u"]), torch.tensor([20.0]), torch.tensor([True])
    )
    assert float(third[0]) == pytest.approx(15.0), "momentum 0.5 on 10 -> 20"


def test_the_climatology_does_not_depend_on_position_in_the_batch():
    """A batch spans several years, so one unit appears in it more than once.

    Updating as we go would score each season against a level that had already
    absorbed the earlier seasons of the same batch — two seasons of one unit
    compared against two different references.
    """
    climatology = RunningClimatology(momentum=0.5)
    units = np.array(["u", "u", "u"])
    reference = climatology.update(
        "x", units, torch.tensor([10.0, 20.0, 30.0]), torch.tensor([True] * 3)
    )
    # One level for all three seasons, and on a first sighting it is the mean of
    # them — the natural first estimate of the unit's climatology.
    assert reference.tolist() == pytest.approx([20.0, 20.0, 20.0])

    # Reversing the batch cannot change what any season is scored against.
    other = RunningClimatology(momentum=0.5)
    reversed_reference = other.update(
        "x", units, torch.tensor([30.0, 20.0, 10.0]), torch.tensor([True] * 3)
    )
    assert reversed_reference.tolist() == pytest.approx(reference.tolist())


# --------------------------------------------------------------------------- #
# Parameter specs and the problem
# --------------------------------------------------------------------------- #


#: The harmonised SUSTAg ``WW`` block's table knots and vernalisation scalars.
#:
#: The shipped spec files address table ordinates **by abscissa**
#: (``crop.vernrt@-10.0``), so they are written against a specific crop file —
#: the one the production run uses, written to
#: ``<TC_OUT_DIR>/workspace/crop_wheat.yaml`` by
#: :func:`cropmodelling4eu.torchcrop.params.write_crop_yaml`. torchcrop's
#: *bundled* wheat preset has different knots (``vernrt`` at 0 and 1, not at
#: -10 .. 30), so a spec resolved against it raises — which is the correct
#: behaviour and is what
#: :func:`test_a_spec_against_the_wrong_crop_file_raises` pins.
#:
#: Reproduced here rather than read off the cluster so the tests run anywhere.
SUSTAG_TABLES: dict[str, list[list[float]]] = {
    "dtsmtb": [[-5.0, 0.0], [0.0, 0.0], [30.0, 30.0], [45.0, 30.0]],
    "phottb": [[0.0, 0.0], [17.0, 1.0]],
    "vernrt": [[-10.0, 0.0], [-4.0, 0.0], [3.0, 1.0], [10.0, 1.0],
               [17.0, 0.0], [30.0, 0.0]],
    "slatb": [[0.0, 0.02], [0.4, 0.02], [1.0, 0.015], [2.0, 0.015]],
    "ruetb": [[0.0, 2.6], [1.0, 2.5], [1.3, 2.0], [2.0, 0.9]],
    "fltb": [[0.0, 0.65], [0.25, 0.65], [0.5, 0.5], [0.646, 0.3],
             [0.95, 0.0], [2.0, 0.0]],
    "fstb": [[0.0, 0.35], [0.25, 0.35], [0.5, 0.5], [0.646, 0.7],
             [0.95, 1.0], [1.0, 0.0], [2.0, 0.0]],
    "fotb": [[0.0, 0.0], [0.99, 0.0], [1.0, 1.0], [2.0, 1.0]],
    "rdrltb": [[-10.0, 0.0], [10.0, 0.02], [15.0, 0.03], [30.0, 0.05],
               [50.0, 0.09]],
}

#: SIMPLACE's vernalisation block, which torchcrop's preset has no slot for.
SUSTAG_SCALARS: dict[str, float] = {
    "idsl": 2.0, "versat": 70.0, "vbase": 14.0,
    "tsum1": 1125.0, "tsum2": 1000.0, "tsumem": 60.0,
}


@pytest.fixture()
def crop_params():
    """torchcrop's wheat preset, moved onto the harmonised SUSTAg WW crop."""
    from torchcrop import CropParameters

    crop = CropParameters(crop_name="wheat")
    for name, table in SUSTAG_TABLES.items():
        setattr(crop, name, torch.tensor(table, dtype=torch.float32))
    for name, value in SUSTAG_SCALARS.items():
        setattr(crop, name, torch.tensor(value, dtype=torch.float32))
    return crop


class _Holder:
    def __init__(self, crop):
        from torchcrop.parameters.site_params import SiteParameters
        from torchcrop.parameters.soil_params import SoilParameters

        self.crop_params = crop
        self.soil_params = SoilParameters()
        self.site_params = SiteParameters()


def _stage_path(stage: str):
    from cropmodelling4eu.calibration.config import STAGE_PARAMETERS
    from pathlib import Path

    import cropmodelling4eu.calibration as package

    return Path(package.__file__).parent / STAGE_PARAMETERS[stage]


@pytest.mark.parametrize("stage", STAGE_FILES)
def test_every_shipped_spec_file_loads_and_resolves(stage, crop_params):
    """A spec naming a field torchcrop does not have must fail here, not mid-run."""
    specs = load_stage_specs(_stage_path(stage))
    assert specs.specs
    problem = CalibrationProblem(
        {1: _Holder(crop_params)}, specs.specs, specs.groups, specs.simplex,
        specs.priors, specs.ratio_priors,
    )
    values = problem.materialize(1)
    assert set(values) >= {s.name for s in specs.specs}


def test_a_spec_against_the_wrong_crop_file_raises():
    """A table ordinate is addressed by abscissa, so the crop file is a contract.

    torchcrop's bundled preset puts ``vernrt`` at DVS 0 and 1; the harmonised
    SUSTAg crop puts it at -10 .. 30 C. Resolving the shipped spec against the
    wrong one must fail loudly rather than calibrating some other row.
    """
    from torchcrop import CropParameters

    # Tier 2, because tier 1 is four scalars and addresses no table at all.
    specs = load_stage_specs(_stage_path("phenology"), max_tier=2)
    with pytest.raises(ValueError, match="no table row"):
        CalibrationProblem(
            {1: _Holder(CropParameters(crop_name="wheat"))}, specs.specs, specs.groups
        )


def test_blocked_parameters_are_excluded_with_their_reason(crop_params):
    """A stage that cannot identify something has to say why, in its output."""
    specs = load_stage_specs(_stage_path("yield"))
    names = {s.name for s in specs.specs}
    assert "crop.nlue" not in names, "the N group is unidentifiable at nni 0.985"
    assert "initial mineral N" in specs.blocked["crop.nlue"]

    freed = load_stage_specs(_stage_path("yield"), free_blocked=True)
    assert "crop.nlue" in {s.name for s in freed.specs}


def test_tier_two_needs_a_third_dated_stage(crop_params):
    """The identifiability rule: free parameters <= observed stages, per region.

    Two dated stages (CLMS emergence and harvest) identify the four thermal
    scalars and nothing else. Freeing the vernalisation block and the
    photoperiod ramp alongside them puts the parameter correlations at
    0.997-0.999, which is what the tier split encodes.
    """
    tier1 = {s.name for s in load_stage_specs(_stage_path("phenology")).specs}
    tier2 = {s.name for s in load_stage_specs(_stage_path("phenology"), max_tier=2).specs}
    assert tier1 == {"crop.tsumem", "crop.tsum1", "crop.tsum2", "crop.versat"}
    for name in ("crop.vbase", "crop.phottb@0.0", "crop.vernrt@3.0",
                 "crop.minimal_vernalisation_factor"):
        assert name not in tier1 and name in tier2


def test_no_latent_is_seeded_on_its_bound(crop_params):
    """A latent at a bound has a saturated sigmoid and is frozen, not free.

    Tier 2, because that is where the SUSTAg crop's on-the-bound values are:
    the ``vernrt`` plateau at 0 and 1 and both ``phottb`` knots.
    """
    specs = load_stage_specs(_stage_path("phenology"), max_tier=2)
    problem = CalibrationProblem(
        {1: _Holder(crop_params)}, specs.specs, specs.groups, specs.simplex
    )
    values = problem.materialize(1)
    for spec in specs.specs:
        lo, hi = spec.bounds
        value = float(values[spec.name])
        assert lo < value < hi, f"{spec.name} sits on a bound and cannot move"


def test_the_simplex_group_keeps_the_partitioning_valid(crop_params):
    """``fl + fs + fo = 1`` is a constraint ConstraintGroup cannot express.

    Calibrating the three independently produces an invalid partitioning that
    still runs, which is the failure mode worth a test: nothing raises.
    """
    from torchcrop.calibration import ParameterSpec

    specs = [
        ParameterSpec("crop.fltb@0.646", bounds=(0.0, 1.0), init=0.8),
        ParameterSpec("crop.fstb@0.646", bounds=(0.0, 1.0), init=0.9),
    ]
    problem = CalibrationProblem(
        {1: _Holder(crop_params)},
        specs,
        simplex=[SimplexGroup(members=("crop.fltb@0.646", "crop.fstb@0.646"))],
    )
    values = problem.materialize(1)
    total = float(values["crop.fltb@0.646"]) + float(values["crop.fstb@0.646"])
    assert total <= 1.0 + 1e-6
    # And the rescaled values reach the crop's own tables, not just the dict.
    row = (crop_params.fltb[:, 0] - 0.646).abs().argmin()
    assert float(crop_params.fltb[row, 1]) == pytest.approx(
        float(values["crop.fltb@0.646"]), abs=1e-6
    )


def test_the_penalties_touch_only_the_batch_s_region(crop_params):
    """A batch carries one region, so its penalties must too.

    Summed over every region they are applied once per *batch* instead of once
    per *step*: with 51 batches over 29 regions that pulls each region toward
    its prior ~28 times for every time its data is seen, and the prior wins by
    arithmetic rather than by evidence. It also makes every batch touch every
    region's latents, so no two batches can be run independently — which is
    what forces the whole calibration onto one core.
    """
    import copy

    from torchcrop.calibration import ParameterSpec

    specs = [ParameterSpec("crop.tsum1", bounds=(500.0, 1300.0))]
    problem = CalibrationProblem(
        {r: _Holder(copy.deepcopy(crop_params)) for r in (1, 2, 3)},
        specs,
        priors={"crop.tsum1": (860.0, 200.0)},
    )
    with torch.no_grad():                       # make the regions differ
        problem.manager(2)._latents["crop__tsum1"] += 1.0
        problem.manager(3)._latents["crop__tsum1"] -= 1.0

    problem.zero_grad(set_to_none=True)
    (problem.prior_penalty(1.0, 1) + problem.pooling_penalty(1.0, 1)).backward()

    touched = {
        r for r in (1, 2, 3)
        if problem.manager(r)._latents["crop__tsum1"].grad is not None
    }
    assert touched == {1}, "regions 2 and 3 were not in this batch"


def test_regions_may_free_different_parameters(crop_params):
    """The identifiability rule is *per region*, so the spec set must be too.

    PEP725 reaches 11 of 28 regions on this domain. Requiring one shared spec
    set would freeze the vernalisation block everywhere because 17 regions lack
    a third dated stage — throwing away the information the other 11 have.
    """
    import copy

    from torchcrop.calibration import ParameterSpec

    tier1 = [ParameterSpec("crop.tsum1", bounds=(500.0, 1300.0))]
    tier2 = tier1 + [ParameterSpec("crop.vbase", bounds=(0.0, 25.0))]
    problem = CalibrationProblem(
        {r: _Holder(copy.deepcopy(crop_params)) for r in (1, 2, 3)},
        {1: tier2, 2: tier1, 3: tier1},
    )
    assert set(problem.materialize(1)) == {"crop.tsum1", "crop.vbase"}
    assert set(problem.materialize(2)) == {"crop.tsum1"}
    assert set(problem.spec_names) == {"crop.tsum1", "crop.vbase"}

    # A parameter free in one region pools across the regions that have it, and
    # is simply absent from the others rather than erroring.
    assert float(problem.pooling_penalty(1.0, 2)) >= 0.0
    assert float(problem.pooling_penalty(1.0, 1)) >= 0.0


def test_the_ratio_prior_is_released_only_where_anthesis_is_observed(crop_params):
    """Elsewhere it is the only thing holding a near-singular direction."""
    import copy

    from torchcrop.calibration import ParameterSpec

    specs = [
        ParameterSpec("crop.tsum1", bounds=(500.0, 1300.0)),
        ParameterSpec("crop.tsum2", bounds=(400.0, 1350.0)),
    ]
    key = ("crop.tsum1", "crop.tsum2")
    problem = CalibrationProblem(
        {r: _Holder(copy.deepcopy(crop_params)) for r in (1, 2)},
        specs,
        ratio_priors={1: {key: (0.30, 0.06)}, 2: {key: (0.30, 0.30)}},
    )
    tight = float(problem.prior_penalty(1.0, 1))
    loose = float(problem.prior_penalty(1.0, 2))
    assert tight > loose, "a released prior must pull less on the same values"


def test_two_simplex_groups_on_one_table_keep_both_gradients(crop_params):
    """`rebuild_table` detaches every ordinate not in its own updates.

    So writing a table once per simplex group makes the second write wipe the
    gradient the first established. With `fltb` and `fstb` each carrying a group
    at DVS 0.646 and another at 0.95, three of the four partitioning ordinates
    silently stopped training — while a finite difference showed a clear
    response and the true analytic gradient was ~540.
    """
    from cropmodelling4eu.calibration.problem import SimplexGroup
    from torchcrop.calibration import ParameterSpec

    names = ["crop.fltb@0.646", "crop.fstb@0.646", "crop.fltb@0.95", "crop.fstb@0.95"]
    problem = CalibrationProblem(
        {1: _Holder(crop_params)},
        [ParameterSpec(n, bounds=(0.0, 1.0)) for n in names],
        simplex=[
            SimplexGroup(members=("crop.fltb@0.646", "crop.fstb@0.646")),
            SimplexGroup(members=("crop.fltb@0.95", "crop.fstb@0.95")),
        ],
    )
    problem.materialize(1)

    # Every freed ordinate must still be connected to its latent *in the table
    # the model will read*, not merely in the returned value dict.
    for field in ("fltb", "fstb"):
        table = getattr(crop_params, field)
        assert table.grad_fn is not None, f"{field} is detached from the graph"

    problem.zero_grad(set_to_none=True)
    (crop_params.fltb.sum() + crop_params.fstb.sum()).backward()
    dead = [
        n for n in names
        if problem.manager(1)._latents[n.replace(".", "__").replace("@", "__")
                                       .replace("[", "__").replace("]", "__")].grad is None
    ]
    assert not dead, f"no gradient reaches {dead} through the rebuilt table"


def test_pooling_shrinks_a_region_toward_the_others(crop_params):
    """Hierarchical pooling, so a data-poor region inherits the global fit."""
    import copy

    from torchcrop.calibration import ParameterSpec

    specs = [ParameterSpec("crop.tsum1", bounds=(500.0, 1300.0))]
    problem = CalibrationProblem(
        {r: _Holder(copy.deepcopy(crop_params)) for r in (1, 2, 3)}, specs
    )
    assert float(problem.pooling_penalty(1.0, 1)) == pytest.approx(0.0), "identical starts"

    with torch.no_grad():
        problem.manager(1)._latents["crop__tsum1"] += 2.0
    assert float(problem.pooling_penalty(1.0, 1)) > 0.0
    # Region 2 has not moved, but the mean has, so it feels the pull too.
    assert float(problem.pooling_penalty(1.0, 2)) > 0.0


# --------------------------------------------------------------------------- #
# Region parallelism
# --------------------------------------------------------------------------- #


def test_a_wave_never_repeats_a_region():
    """Two batches of one region must stay sequential; different regions need not.

    The second batch of a region reads the parameters the first wrote, so
    running them together would lose one of the two updates. Different regions
    share no latents at all — that is what makes the parallelism exact.
    """
    from cropmodelling4eu.calibration.parallel import waves

    batches = [
        pd.DataFrame({"region_id": [r]})
        for r in (1, 1, 1, 2, 2, 3)
    ]
    grouped = waves(batches)
    for wave in grouped:
        regions = [int(b["region_id"].iloc[0]) for b in wave]
        assert len(regions) == len(set(regions))
    # Nothing is lost, and the region needing three passes sets the depth.
    assert sum(len(w) for w in grouped) == len(batches)
    assert len(grouped) == 3


def test_waves_preserve_every_batch_for_a_single_region():
    from cropmodelling4eu.calibration.parallel import waves

    batches = [pd.DataFrame({"region_id": [7]}) for _ in range(4)]
    grouped = waves(batches)
    assert len(grouped) == 4 and all(len(w) == 1 for w in grouped)


def test_applying_a_gradient_steps_only_its_own_region(crop_params):
    """Adam skips parameters whose grad is None, which is what scopes the step."""
    import copy

    from cropmodelling4eu.calibration.parallel import BatchOutcome
    from torchcrop.calibration import ParameterSpec

    specs = [ParameterSpec("crop.tsum1", bounds=(500.0, 1300.0))]
    problem = CalibrationProblem(
        {r: _Holder(copy.deepcopy(crop_params)) for r in (1, 2)}, specs
    )

    class _Cal:                       # only `_apply` is under test
        _apply = __import__(
            "cropmodelling4eu.calibration.trainer", fromlist=["Calibrator"]
        ).Calibrator._apply

    optimizer = torch.optim.Adam(problem.parameters(), lr=0.1)
    before = {r: float(problem.manager(r)._latents["crop__tsum1"]) for r in (1, 2)}
    _Cal()._apply(
        problem, optimizer,
        BatchOutcome(region_id=1, grads={"crop__tsum1": 1.0}, loss=0.0, report={}),
        grad_clip=1.0,
    )
    after = {r: float(problem.manager(r)._latents["crop__tsum1"]) for r in (1, 2)}
    assert after[1] != before[1], "region 1 was in the batch"
    assert after[2] == before[2], "region 2 was not, and must not have moved"


def test_a_batch_with_no_evidence_takes_no_step(crop_params):
    """The prior must not move a region on a batch that observes nothing.

    A batch whose unit-years carry no observation for any active stage still
    has a non-zero *penalty*, so `total.backward()` yields a prior-only
    gradient and the region would step toward its prior on no evidence — the
    same failure as applying the penalties once per batch, by another route.
    """
    from cropmodelling4eu.calibration.trainer import Calibrator

    # The guard is the `scored_stages` count, which `_batch_loss` sets and
    # `batch_gradients` reads before it ever calls backward().
    source = inspect.getsource(Calibrator.batch_gradients)
    assert 'report.get("scored_stages")' in source
    assert source.index('report.get("scored_stages")') < source.index("total.backward()")


def test_a_stage_only_draws_unit_years_it_can_score():
    """A yield-only unit contributes no phenology target, so stage 1 must skip it.

    Drawing it costs a full 465-day forward pass and produces a loss with no
    terms in it — and it makes the reported training-set size a number the
    stage cannot actually fit on.
    """
    from cropmodelling4eu.calibration.trainer import STAGE_VARIABLES

    # `scorable_unit_years` is pure set logic over the observation table; the
    # part worth pinning is that the stage's variable sets do not overlap, so
    # the filter cannot silently keep everything.
    phenology = set(STAGE_VARIABLES["phenology"])
    yields = set(STAGE_VARIABLES["yield"])
    canopy = set(STAGE_VARIABLES["lai"])
    assert phenology & yields == set()
    assert phenology & canopy == set()
    assert yields & canopy == set()


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_phenology_uses_leave_one_year_out():
    """CLMS covers eight seasons, so a 15 % year block is one year."""
    split = make_split(_unit_years(), mode="loyo")
    assert len(split.held_out_years) == 2
    assert set(split.train["year"]) & set(split.val["year"]) == set()


def test_the_split_holds_out_units_as_well_as_years():
    """A model can extrapolate in time and fail to transfer in space."""
    split = make_split(_unit_years(n_units=20), mode="block", test_fraction=0.2)
    assert split.held_out_units
    assert set(split.train["unit"]) & set(split.held_out_units) == set()


def test_a_reference_with_a_short_window_survives_into_training():
    """CLMS exists only in 2017-2024 — exactly the years a block split holds out.

    A 70/15/15 split of 25 years removes the last eight, which is the whole CLMS
    record. The split reads as perfectly ordinary while silently leaving the
    primary phenology reference with nothing to train on.
    """
    pool = pd.DataFrame(
        [
            {"unit": f"U{u}", "unit_kind": "adm", "region_id": u % 3,
             "year": y, "n_cells": 4}
            for u in range(30) for y in range(2000, 2025)
        ]
    )
    unguarded = make_split(pool, mode="block")
    assert set(range(2017, 2025)) <= set(unguarded.held_out_years), (
        "this is the failure the guard exists for"
    )

    guarded = make_split(pool, mode="block", protect=(2017, 2024))
    trained = set(range(2017, 2025)) - set(guarded.held_out_years)
    assert len(trained) >= 4, "at least half the protected window must train"


def test_a_pool_too_short_to_split_raises():
    two_years = _unit_years(n_years=2)
    with pytest.raises(ValueError, match="3 harvest years"):
        make_split(two_years)


# --------------------------------------------------------------------------- #
# The gradient path — the property a silent failure hides in
# --------------------------------------------------------------------------- #


def _toy_weather(n_days: int = 400, batch: int = 2):
    """Warm, bright, well-watered days: a season that certainly matures."""
    from torchcrop import WeatherDriver

    doy = (np.arange(n_days) % 365) + 1.0
    channels = np.stack(
        [
            doy,
            np.full(n_days, 14.0),   # davtmp
            np.full(n_days, 9.0),    # tmin
            np.full(n_days, 19.0),   # tmax
            np.full(n_days, 15.0),   # irrad
            np.full(n_days, 3.0),    # rain
            np.full(n_days, 1.2),    # vp
            np.full(n_days, 2.0),    # wind
        ],
        axis=1,
    )
    data = np.broadcast_to(channels, (batch, n_days, 8)).copy()
    return WeatherDriver(torch.as_tensor(data, dtype=torch.float32))


def _toy_model(crop):
    from torchcrop import Lintul5Model
    from torchcrop.parameters.site_params import SiteParameters
    from torchcrop.parameters.soil_params import SoilParameters

    site = SiteParameters()
    site.idpl = torch.tensor(1.0)
    return Lintul5Model(crop, SoilParameters(), site)


def test_the_checkpointed_forward_reproduces_the_stock_one(crop_params):
    """O(sqrt(T)) memory must cost accuracy nothing at all."""
    from cropmodelling4eu.calibration.checkpointed import run_trajectories

    model = _toy_model(crop_params)
    weather = _toy_weather()
    with torch.no_grad():
        plain = run_trajectories(model, weather, 1, keep=("dvs", "lai"), segment=0)
        chunked = run_trajectories(model, weather, 1, keep=("dvs", "lai"), segment=20)
    for name in ("dvs", "lai"):
        assert torch.allclose(plain[name], chunked[name])


def test_the_maturity_date_moves_with_tsum1(crop_params):
    """The gradient the clamped DVS trajectory silently destroys.

    ``euler_update`` clamps ``dvs`` to ``[0, 2]``, so on the maturity day the
    stored value is exactly 2.0 and ``crossing_day``'s interpolation fraction
    lands on its own ``clamp(0, 1)`` boundary — derivative zero, date a whole
    number, and nothing raises. Reading the *uncapped* trajectory is what keeps
    the date differentiable.
    """
    from cropmodelling4eu.calibration.checkpointed import run_trajectories
    from cropmodelling4eu.torchcrop.run import crossing_day

    model = _toy_model(crop_params)
    weather = _toy_weather()

    tsum1 = torch.tensor(float(crop_params.tsum1), requires_grad=True)
    model.crop_params.tsum1 = tsum1
    trajectories = run_trajectories(
        model, weather, 1, keep=("dvs", "dvs_uncapped"), segment=0
    )

    capped = crossing_day(trajectories["dvs"], 2.0)
    uncapped = crossing_day(trajectories["dvs_uncapped"], 2.0)
    assert torch.isfinite(uncapped).all(), "the toy season must mature"

    # The dates agree to within the day the clamp costs.
    assert float((capped - uncapped).abs().max()) < 1.5

    capped_grad, = torch.autograd.grad(capped.sum(), tsum1, retain_graph=True)
    uncapped_grad, = torch.autograd.grad(uncapped.sum(), tsum1)
    assert float(capped_grad) == 0.0, "this is the failure the fix is for"
    assert abs(float(uncapped_grad)) > 1e-3


# --------------------------------------------------------------------------- #
# Container ownership — what a stage may leave behind for the next one
# --------------------------------------------------------------------------- #


def _tensor_fields(holder):
    """Every tensor field of the three containers, as ``(name, tensor)``."""
    from dataclasses import fields as dataclass_fields

    return [
        (f"{label}.{entry.name}", value)
        for label, container in (
            ("crop", holder.crop_params),
            ("soil", holder.soil_params),
            ("site", holder.site_params),
        )
        for entry in dataclass_fields(container)
        if isinstance(value := getattr(container, entry.name), torch.Tensor)
    ]


def _toy_loss(holder, weather):
    """A scalar that depends on the containers, through the full day loop."""
    from cropmodelling4eu.calibration.checkpointed import run_trajectories

    model = _toy_model(holder.crop_params)
    return run_trajectories(model, weather, 1, keep=("wso",), segment=0)["wso"].sum()


def test_a_stage_leaves_no_graph_in_the_containers_for_the_next_one(crop_params):
    """The crash that killed every stage-2 run: SLURM 820638 and its siblings.

    ``Calibrator`` keeps **one** ``CropParameters`` per region alive across every
    stage, and ``materialize`` writes ``bij.forward(latent)`` into it. The stock
    manager launders only the tables *it* targets, so a scalar written by stage 1
    is still there — with its graph — when stage 2 forwards through it. The first
    backward frees that bijection's saved tensors; the second raises.
    """
    from torchcrop.calibration import ParameterSpec

    holder = _Holder(crop_params)
    weather = _toy_weather(n_days=220)

    stage1 = CalibrationProblem(
        {1: holder}, [ParameterSpec("crop.tsum1", bounds=(900.0, 1400.0))]
    )
    stage1.materialize(1)
    _toy_loss(holder, weather).backward()
    # The end-of-stage `parameter_table()`, which is where the graph was left.
    stage1.values()

    # A disjoint spec set, exactly as the yield stage frees only `ruetb`.
    stage2 = CalibrationProblem(
        {1: holder}, [ParameterSpec("crop.ruetb@0.0", bounds=(2.0, 3.5))]
    )
    for _ in range(2):
        stage2.zero_grad(set_to_none=True)
        stage2.materialize(1)
        _toy_loss(holder, weather).backward()

    grads = [p.grad for p in stage2.parameters() if p.grad is not None]
    assert grads, "the second stage must still receive a gradient"


def test_reading_the_values_out_does_not_write_a_graph_back(crop_params):
    """``values()`` wants floats, but ``materialize`` writes the containers."""
    from torchcrop.calibration import ParameterSpec

    holder = _Holder(crop_params)
    problem = CalibrationProblem(
        {1: holder},
        [
            ParameterSpec("crop.tsum1", bounds=(900.0, 1400.0)),
            ParameterSpec("crop.ruetb@0.0", bounds=(2.0, 3.5)),
        ],
    )
    assert problem.values()[1]
    live = [name for name, t in _tensor_fields(holder) if t.grad_fn is not None]
    assert not live, f"graph left in {live}"


def test_laundering_the_containers_spares_a_learnable_parameter(crop_params):
    """``nn.Parameter`` fields are torchcrop's own direct-calibration path.

    ``Lintul5Model.learnable_parameter_groups`` collects them straight off the
    containers, so a blanket ``detach`` would hand the caller's optimiser a leaf
    the model no longer reads — a quieter failure than the one being fixed.
    """
    from torchcrop.calibration import ParameterSpec

    holder = _Holder(crop_params)
    learnable = torch.nn.Parameter(torch.tensor(0.9))
    holder.crop_params.rdrshm = learnable

    problem = CalibrationProblem(
        {1: holder}, [ParameterSpec("crop.tsum1", bounds=(900.0, 1400.0))]
    )
    problem.materialize(1)
    problem.detach_containers()

    assert holder.crop_params.rdrshm is learnable
    assert isinstance(holder.crop_params.rdrshm, torch.nn.Parameter)
