"""Tests for the torchcrop runner.

The model itself is torchcrop's to test; what is tested here is the wiring —
that the export's soil, site, schedule and weather reach LINTUL-5 as the right
parameters, and that per-cell sowing dates are grouped rather than averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchcrop")

from cropmodelling4eu.export import resolve_export  # noqa: E402
from cropmodelling4eu.torchcrop.run import (  # noqa: E402
    build_site_params,
    build_soil_params,
    crossing_day,
    fertilizer_from_dvs,
    group_by_sowing,
    run_cells,
    run_shard,
    sowing_plan,
)

from .conftest import TEST_CELLS  # noqa: E402


# --------------------------------------------------------------------------- #
# Sowing groups
# --------------------------------------------------------------------------- #


def test_group_by_sowing_splits_the_shard():
    ids = np.array([1, 2, 3, 4, 5])
    doys = np.array([280, 295, 280, 295, 280])
    groups = group_by_sowing(ids, doys)

    assert sorted(groups) == [280, 295]
    assert groups[280].tolist() == [1, 3, 5]
    assert groups[295].tolist() == [2, 4]
    # Every cell lands in exactly one group.
    assert sum(g.size for g in groups.values()) == ids.size


def test_group_by_sowing_keeps_a_single_date_in_one_group():
    """The old constant-DOY behaviour is the one-group case, unchanged."""
    ids = np.arange(1, 11)
    groups = group_by_sowing(ids, np.full(10, 270))
    assert list(groups) == [270]
    assert groups[270].tolist() == ids.tolist()


def test_shard_groups_match_the_site_table(run_config):
    """The fixture's two sowing dates must survive into two groups."""
    bundle = resolve_export(run_config)
    ids = bundle.ids
    groups = group_by_sowing(ids, bundle.site.sowing_doy(ids))
    assert sorted(groups) == [280, 295]


def test_sowing_table_groups_by_year_not_by_doy():
    """A simulated-sowing table must not fragment a year by per-cell date --
    that fragmentation (one batch per (year, doy)) is exactly the slowdown
    this grouping is meant to avoid."""
    ids = np.array([1, 2, 3, 4])
    sowing = pd.DataFrame(
        {
            "SimplaceID": [1, 2, 3, 4, 1, 2, 3, 4],
            "year": [2000, 2000, 2000, 2000, 2001, 2001, 2001, 2001],
            # Every cell disagrees with every other -- four groups under the
            # old (year, doy) grouping, one under the new (year) grouping.
            "sowing_doy": [270, 275, 280, 285, 268, 271, 279, 290],
        }
    )
    plan = sowing_plan(ids, [2000, 2001], np.full(4, 270), sowing)

    assert len(plan) == 2
    for years, group, doy in plan:
        assert len(years) == 1
        assert sorted(group.tolist()) == [1, 2, 3, 4]
        assert doy.size == 4


# --------------------------------------------------------------------------- #
# Parameter construction
# --------------------------------------------------------------------------- #


def test_soil_params_integrate_over_the_files_own_layers(run_config):
    bundle = resolve_export(run_config)
    soil = bundle.soil.select(bundle.ids)
    params = build_soil_params(soil, rootzone_m=1.0, profile_bottom_m=2.0)

    n = bundle.ids.size
    assert params.wcfc.shape == (n,)
    # The fixture profile is uniform, so the rooting-zone mean is the layer value.
    assert params.wcfc.numpy() == pytest.approx(0.31)
    assert params.wcwp.numpy() == pytest.approx(0.14)
    assert params.wcst.numpy() == pytest.approx(0.42)
    # Air-dry is clamped below the wilting point.
    assert (params.wcad.numpy() < params.wcwp.numpy()).all()
    # Mineral N: (12 + 28) kg/ha in each of the five rooting-zone layers,
    # kg/ha -> g/m2 is a factor of 10.
    assert params.nminti.numpy() == pytest.approx((12.0 + 28.0) * 5 / 10.0)


def test_soil_params_follow_a_different_layer_geometry(run_config, export_dir):
    """Halving the profile depth must change the depth-integrated stock."""
    bundle = resolve_export(run_config)
    deep = build_soil_params(bundle.soil, rootzone_m=1.0, profile_bottom_m=2.0)
    shallow = build_soil_params(bundle.soil, rootzone_m=0.3, profile_bottom_m=2.0)
    # A 0.3 m rooting zone holds two layers of stock, not five.
    assert shallow.nminti.numpy() == pytest.approx((12.0 + 28.0) * 2 / 10.0)
    assert (shallow.nminti.numpy() < deep.nminti.numpy()).all()


def test_site_params_come_from_the_export(run_config):
    bundle = resolve_export(run_config)
    ids = np.array(TEST_CELLS)
    params = build_site_params(ids, 2000, 280, bundle)

    # Altitude is per cell, from site.csv -- not a constant zero.
    assert params.altitude.numpy().tolist() == [10.0, 20.0, 300.0, 1200.0]
    assert params.altitude.numpy().std() > 0
    # CO2 is the export's series for that year.
    assert params.co2.numpy() == pytest.approx(369.6)
    # idpl is the group's sowing day, which the window was built from.
    assert params.idpl.numpy() == pytest.approx(280.0)
    # Latitude is decoded from the grid.
    assert params.latitude.numpy().max() <= run_config.grid.max_lat


def test_site_params_use_the_configured_grid(run_config):
    """A different grid must move the cells, not silently reuse Europe's."""
    bundle = resolve_export(run_config)
    ids = np.array([TEST_CELLS[0]])
    here = build_site_params(ids, 2000, 280, bundle).latitude.numpy()[0]
    assert run_config.grid.min_lat <= here <= run_config.grid.max_lat


# --------------------------------------------------------------------------- #
# Fertilizer placement
# --------------------------------------------------------------------------- #


def test_fertilizer_is_placed_at_the_first_day_past_each_stage():
    ids = np.array([1])
    # A monotone DVS trajectory over 10 days, with the leading pre-sowing entry.
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 1.0, 10)]])
    plans = {1: np.array([[0.25, 5.0, 0.0, 0.0], [0.9, 3.0, 0.0, 0.0]])}

    applied = fertilizer_from_dvs(ids, dvs, plans)
    assert applied.shape == (1, 10, 3)
    # Both doses land, on exactly one day each.
    assert applied[0, :, 0].sum().item() == pytest.approx(8.0)
    assert int((applied[0, :, 0] > 0).sum()) == 2
    # The DVS 0.25 dose comes first.
    days = torch.nonzero(applied[0, :, 0]).flatten().tolist()
    assert days[0] < days[1]


def test_cells_without_a_plan_run_unfertilised():
    ids = np.array([1, 999])
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 1.0, 5)]] * 2)
    plans = {1: np.array([[0.25, 5.0, 0.0, 0.0]])}

    applied = fertilizer_from_dvs(ids, dvs, plans)
    assert applied[0].sum().item() == pytest.approx(5.0)
    assert applied[1].sum().item() == 0.0


def test_stages_never_reached_are_not_applied():
    """A dose keyed past the trajectory's end must not be dumped on day 0."""
    ids = np.array([1])
    dvs = torch.tensor([[0.0, *np.linspace(0.0, 0.5, 5)]])
    plans = {1: np.array([[0.9, 4.0, 0.0, 0.0]])}
    assert fertilizer_from_dvs(ids, dvs, plans).sum().item() == 0.0


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_run_shard_writes_a_parquet(tmp_path, run_config):
    """One shard over the fixture export, end to end through LINTUL-5."""
    out = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    assert out.is_file()

    frame = pd.read_parquet(out)
    assert set(frame["SimplaceID"]) <= set(TEST_CELLS)
    for column in ("yield_g_m2", "biomass_g_m2", "max_lai", "sowing_doy", "irri"):
        assert column in frame.columns
    # Both sowing groups reach the output, so grouping did not drop a batch.
    assert sorted(frame["sowing_doy"].unique()) == [280, 295]
    # Yields are finite and not absurd for winter wheat [g/m2].
    assert frame["yield_g_m2"].notna().all()
    assert frame["yield_g_m2"].between(0, 3000).all()


@pytest.mark.slow
def test_shard_is_idempotent(tmp_path, run_config):
    first = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    mtime = first.stat().st_mtime_ns
    again = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    assert again == first and again.stat().st_mtime_ns == mtime


@pytest.mark.slow
def test_run_shard_daily_writes_a_matching_trajectory_file(tmp_path, run_config):
    """--daily writes a second Parquet beside the summary, same cells and
    seasons, one row per (cell, day, variable) for exactly the requested set."""
    out = run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path, daily=True)
    daily_path = tmp_path / "torchcrop_daily_shard_000.parquet"
    assert daily_path.is_file()

    summary = pd.read_parquet(out)
    daily = pd.read_parquet(daily_path)
    assert set(daily["variable"].unique()) == {"LAI", "AGB", "NNI", "TRANRF"}
    assert set(daily["SimplaceID"]) <= set(summary["SimplaceID"])
    assert set(daily["year"]) <= set(summary["year"])
    assert daily["value"].notna().all()


@pytest.mark.slow
def test_run_shard_daily_is_idempotent_with_the_summary(tmp_path, run_config):
    """A shard already done (both files present) is not re-run even with
    --daily; a summary-only shard from an earlier run is topped up rather
    than silently treated as complete."""
    run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path)
    daily_path = tmp_path / "torchcrop_daily_shard_000.parquet"
    assert not daily_path.exists()

    run_shard(run_config, shard=0, n_shards=1, out_dir=tmp_path, daily=True)
    assert daily_path.is_file()


@pytest.mark.slow
def test_run_cells_both_matches_separate_summary_and_daily_calls(run_config):
    """mode="both" shares one _simulate call per batch (see run_shard's own
    daily=True path) -- it must still reproduce exactly what mode="summary"
    and mode="daily" give when run separately, the two-pass smoke test used
    before run_cells_torchcrop.py --daily-out existed."""
    ids = np.array(TEST_CELLS)

    summary, daily = run_cells(run_config, ids, [2000], mode="both")
    summary_only = run_cells(run_config, ids, [2000], mode="summary")
    daily_only = run_cells(run_config, ids, [2000], mode="daily")

    pd.testing.assert_frame_equal(
        summary.sort_values("SimplaceID").reset_index(drop=True),
        summary_only.sort_values("SimplaceID").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        daily.sort_values(["SimplaceID", "date", "variable"]).reset_index(drop=True),
        daily_only.sort_values(["SimplaceID", "date", "variable"]).reset_index(drop=True),
    )


@pytest.mark.slow
def test_merged_sowing_batch_matches_singleton_runs(run_config):
    """Cells sowing on different simulated dates in the same year now share a
    batch (see sowing_plan): each must still latch on its *own* idpl inside
    the shared, earliest-anchored window and reproduce exactly what running
    it alone -- one cell, its own window -- would give. This is the
    correctness check for widening batches beyond a single sowing date."""
    ids = np.array(TEST_CELLS)
    sowing = pd.DataFrame(
        {"SimplaceID": ids, "year": 2000, "sowing_doy": [270, 278, 288, 296]}
    )

    merged = run_cells(run_config, ids, [2000], sowing=sowing)
    singles = pd.concat(
        [
            run_cells(run_config, np.array([sid]), [2000], sowing=sowing)
            for sid in ids
        ],
        ignore_index=True,
    )

    merged = merged.sort_values("SimplaceID").reset_index(drop=True)
    singles = singles.sort_values("SimplaceID").reset_index(drop=True)
    assert merged["SimplaceID"].tolist() == singles["SimplaceID"].tolist()
    # sowing_doy/days_to_maturity/max_lai are exact -- integers or a single
    # forward-Euler max, with no cross-cell reduction to reorder. The
    # continuous stress means (tranrf_mean, nni_mean) accumulate over ~200
    # daily steps, so a batch-of-4 vs. batch-of-1 op can round differently in
    # float32; a real windowing/idpl bug would show up as a wrong sowing day
    # or a several-percent yield/stress shift, not float32 noise, so a loose
    # tolerance still catches it.
    for column in ("sowing_doy", "days_to_maturity", "max_lai"):
        np.testing.assert_allclose(
            merged[column].to_numpy(dtype=float),
            singles[column].to_numpy(dtype=float),
            rtol=1e-5, atol=1e-6, err_msg=column,
        )
    for column in ("yield_g_m2", "tranrf_mean", "nni_mean"):
        np.testing.assert_allclose(
            merged[column].to_numpy(dtype=float),
            singles[column].to_numpy(dtype=float),
            rtol=1e-3, atol=1e-4, err_msg=column,
        )


# --------------------------------------------------------------------------- #
# Emergence
# --------------------------------------------------------------------------- #


def test_crossing_day_interpolates_between_whole_days():
    """The date is reported in days, so a whole-day argmax is a whole-day bias."""
    x = torch.tensor([
        [0.0, 10.0, 20.0, 30.0, 40.0],   # 25 falls midway between steps 2 and 3
        [0.0, 30.0, 60.0, 90.0, 120.0],  # and 5/6 of the way through step 0
        [0.0, 1.0, 2.0, 3.0, 4.0],       # never reaches 25
    ])
    day = crossing_day(x, 25.0)
    assert day[0].item() == pytest.approx(2.5)
    assert day[1].item() == pytest.approx(25.0 / 30.0)
    # A row that never crosses has no crossing day. Clamping it to the last
    # step would make "never emerged" indistinguishable from "emerged late".
    assert torch.isnan(day[2])


def test_crossing_day_is_differentiable_in_the_threshold():
    """tsumem's only gradient path in stage 1 runs through here."""
    x = torch.tensor([[0.0, 10.0, 20.0, 30.0, 40.0]])
    thr = torch.tensor(25.0, requires_grad=True)
    crossing_day(x, thr).sum().backward()
    # d(day)/d(threshold) = 1 / (x1 - x0) = 1/10.
    assert thr.grad.item() == pytest.approx(0.1)


def test_crossing_day_handles_a_threshold_already_met_at_the_first_step():
    x = torch.tensor([[50.0, 60.0, 70.0]])
    assert crossing_day(x, 25.0).item() == pytest.approx(0.0)


@pytest.mark.slow
def test_run_writes_an_emergence_day_between_sowing_and_maturity(run_config):
    """Emergence is dated on the thermal-time clock, not on DVS.

    DVS is pinned at 0 until the crop emerges, so it carries no emergence
    signal at all -- the check that matters is that the column exists, is
    fractional (an argmax would make it integral), and lands inside the
    season it belongs to.
    """
    frame = run_cells(run_config, np.array(TEST_CELLS), [2000])

    assert "days_to_emergence" in frame.columns
    emergence = frame["days_to_emergence"]
    assert emergence.notna().all()
    # Emergence follows sowing and precedes maturity, for every cell.
    assert (emergence > 0).all()
    assert (emergence < frame["days_to_maturity"]).all()
    # Linear interpolation, not argmax: at least one cell lands off a whole day.
    assert not np.allclose(emergence, emergence.round())


# --------------------------------------------------------------------------- #
# Which SIMPLACE crop the run is harmonised against
# --------------------------------------------------------------------------- #


def _multi_crop_xml(tmp_path):
    """A LINTUL5_crop.xml-shaped file: several blocks, keyed `Crop`, tables
    stored as one interleaved <value> list rather than as an x/y pair."""
    path = tmp_path / "LINTUL5_crop.xml"
    path.write_text(
        "<crops>\n"
        "  <crop>\n"
        "    <parameter id='Crop'>MAIZ</parameter>\n"
        "    <parameter id='TSUM1'>800.0</parameter>\n"
        "    <parameter id='IDSL'>0</parameter>\n"
        "  </crop>\n"
        "  <crop>\n"
        "    <parameter id='Crop'>WW</parameter>\n"
        "    <parameter id='TSUM1'>1125.0</parameter>\n"
        "    <parameter id='IDSL'>2</parameter>\n"
        "    <parameter id='VERSAT'>70</parameter>\n"
        "    <parameter id='VBASE'>14</parameter>\n"
        "    <parameter id='PHOTTB'>\n"
        "      <value>0</value><value>0.0</value>\n"
        "      <value>17</value><value>1.0</value>\n"
        "    </parameter>\n"
        "  </crop>\n"
        "</crops>\n"
    )
    return path


def test_a_multi_crop_file_without_a_named_block_is_refused(tmp_path):
    """Taking the first block reads LINTUL5_crop.xml as maize, and the wheat
    run that follows looks entirely normal."""
    from cropmodelling4eu.torchcrop.params import load_simplace_crop

    with pytest.raises(ValueError, match="no crop was named"):
        load_simplace_crop(_multi_crop_xml(tmp_path))


def test_crop_block_is_selected_by_either_key_spelling(tmp_path):
    """Brandenburg keys on CropName, EU SUSTAg on Crop; neither file says so."""
    from cropmodelling4eu.torchcrop.params import load_simplace_crop

    block = load_simplace_crop(_multi_crop_xml(tmp_path), "WW")
    assert block["TSUM1"] == 1125.0
    assert block["IDSL"] == 2.0


def test_an_unknown_block_name_lists_what_the_file_holds(tmp_path):
    from cropmodelling4eu.torchcrop.params import load_simplace_crop

    with pytest.raises(ValueError, match=r"MAIZ.*WW"):
        load_simplace_crop(_multi_crop_xml(tmp_path), "winter_wheat")


def test_interleaved_tables_are_read_as_x_y_pairs(tmp_path):
    """SUSTAg stores a table as one <value> list, Brandenburg as two params.
    Reading only the latter leaves every SUSTAg table on torchcrop's preset."""
    from cropmodelling4eu.torchcrop.params import load_simplace_crop, simplace_table

    block = load_simplace_crop(_multi_crop_xml(tmp_path), "WW")
    assert simplace_table(block, "phottb") == [[0.0, 0.0], [17.0, 1.0]]
    # A table the file does not hold stays absent rather than becoming empty.
    assert simplace_table(block, "slatb") is None


def test_vernalisation_survives_a_preset_with_no_slot_for_it(tmp_path):
    """idsl = 2 with versat == vbase is a no-op: torchcrop returns vernfac = 1.

    The bundled wheat preset has no versat/vbase/vernrt entries, so filling
    only its own slots would drop them and leave the crop running at idsl = 2
    with a TSUM1 that was calibrated *with* vernalisation.
    """
    from torchcrop import CropParameters

    from cropmodelling4eu.torchcrop.params import write_crop_yaml

    out = write_crop_yaml(
        tmp_path / "crop_wheat.yaml",
        simplace_crop_xml=_multi_crop_xml(tmp_path),
        crop_name="wheat",
        simplace_crop="WW",
    )
    params = CropParameters(config_file=str(out))
    assert float(params.idsl) == 2.0
    assert float(params.versat) == 70.0
    assert float(params.vbase) == 14.0
    assert float(params.versat) != float(params.vbase)
