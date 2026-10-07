"""One long observation table over four references.

``(unit, unit_kind, year, variable, value, weight, support)`` — CLMS phenology
(primary), CyBench yield, CyBench fAPAR and PEP725, in one shape, so the
sampler draws from one pool and the loss dispatches on ``variable`` rather than
on which file a number came from.

Three things are done **here** rather than in a loss, because they are
properties of the reference and not of the model:

* **The CLMS level offsets are applied to the observation.** ``CPMCH`` sits
  18.7 d before PEP725 and ``CPMCE`` 6.0 d after it, identically on the 10 km
  and NUTS supports and constant year to year. Correcting the reference is
  better than letting ``tsum2`` absorb a product bias, and it keeps the loss
  in days on the ground scale.
* **A phenology date becomes an absolute date, not a day-of-year.** The loss
  works in days since the simulation window's start, which removes circular
  arithmetic from it entirely — but that conversion needs to know that a winter
  crop's emergence at DOY 301 belongs to the autumn *before* its harvest year.
  That is the single alignment mistake that still produces a plausible season
  length and a plausible-looking near-zero bias while destroying the
  interannual correlation, so it is done once, here.
* **CyBench yield is detrended per unit**, and the trend is kept. LINTUL-5
  cannot produce a technology trend — no cultivar change, no management
  change — so fitting raw levels lets the parameters absorb it. The level and
  the anomaly are carried as two rows so the objective can weight them apart.

PEP725 is **wrapped, never reimplemented**: ``data4simplace.phenology.pep725``
already does the calendar-year to harvest-year alignment, the containment match
and the circular statistics, and re-deriving them is how the alignment bug
comes back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.calibration.config import (
    AUTUMN_CUT_DOY,
    CLMS_EMERGENCE_OFFSET_DAYS,
    CLMS_HARVEST_OFFSET_DAYS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "COLUMNS",
    "ObservationSet",
    "PHENOLOGY_VARIABLES",
    "doy_to_date",
    "load_observations",
]

#: The long table's schema.
COLUMNS: tuple[str, ...] = (
    "unit", "unit_kind", "year", "variable", "value", "weight", "support", "source",
)

#: Variables the phenology loss reads, and the accumulator each is dated on.
#: This is not a free choice: DVS is pinned at 0 until the crop emerges, so it
#: carries no emergence signal at all, and its first non-zero step is one day
#: late because the rate on day *t* produces the state on day *t + 1*.
PHENOLOGY_VARIABLES: dict[str, str] = {
    "emergence_date": "tsump",
    "anthesis_date": "dvs",
    "harvest_date": "dvs",
}


def doy_to_date(
    harvest_year: pd.Series | np.ndarray, doy: pd.Series | np.ndarray, autumn: bool
) -> pd.Series:
    """A day-of-year on a **harvest year** to the absolute date it names.

    An autumn stage (sowing, emergence) observed at or past
    :data:`~cropmodelling4eu.calibration.config.AUTUMN_CUT_DOY` happened in the
    calendar year *before* the harvest it belongs to; a January emergence did
    not. Harvest never moves.
    """
    year = pd.Series(np.asarray(harvest_year, dtype="int64"))
    day = pd.Series(np.asarray(doy, dtype="float64"))
    if autumn:
        year = year - (day >= AUTUMN_CUT_DOY).astype(int)
    base = pd.to_datetime(year.astype(str), format="%Y")
    return base + pd.to_timedelta(day - 1.0, unit="D")


@dataclass(frozen=True, slots=True)
class ObservationSet:
    """The pooled observations and the sampling units they define."""

    table: pd.DataFrame

    def variables(self) -> list[str]:
        return sorted(self.table["variable"].unique())

    def for_variables(self, variables: tuple[str, ...]) -> "ObservationSet":
        """The subset a stage reads."""
        return ObservationSet(
            self.table[self.table["variable"].isin(variables)].reset_index(drop=True)
        )

    def unit_years(self) -> pd.DataFrame:
        """``(unit, unit_kind, year)`` — the sampling units, deduplicated.

        The sampling unit is an `adm_id`-year or a station-year, never a bare
        cell: CyBench and CLMS both report on `adm_id`, so the loss support and
        the sampling support are the same object.
        """
        return (
            self.table[["unit", "unit_kind", "year"]]
            .drop_duplicates()
            .sort_values(["unit_kind", "unit", "year"])
            .reset_index(drop=True)
        )

    def summarise(self) -> str:
        counts = self.table.groupby(["variable", "source"], observed=True).agg(
            rows=("value", "size"), units=("unit", "nunique"), years=("year", "nunique")
        )
        lines = [f"{len(self.table)} observations over {self.table['unit'].nunique()} units"]
        for (variable, source), row in counts.iterrows():
            lines.append(
                f"  {variable:<18} {source:<8} {row.rows:>7} rows  "
                f"{row.units:>5} units  {row.years:>3} years"
            )
        return "\n".join(lines)


def _long(
    frame: pd.DataFrame,
    variable: str,
    value_col: str,
    *,
    unit_kind: str,
    support: str,
    source: str,
    weight_col: str | None = None,
) -> pd.DataFrame:
    raw = frame[value_col]
    if pd.api.types.is_datetime64_any_dtype(raw):
        # A date column must be tested for missingness **before** it is turned
        # into a number. `pd.to_numeric` on datetime64 yields nanoseconds, and
        # it maps NaT to the int64 minimum (-9.22e18) rather than to NaN — a
        # perfectly finite value that neither `notna` here nor the loss's own
        # `isfinite` mask would ever drop. On PEP725, where 3 800 station-seasons
        # have no emergence date and 4 566 no heading date, that silently feeds
        # the loss a date 292 years before the Big Bang.
        present = raw.notna()
        value = raw.astype("int64")
    else:
        value = pd.to_numeric(raw, errors="coerce")
        present = value.notna()

    out = pd.DataFrame(
        {
            "unit": frame["unit"].astype("string"),
            "unit_kind": unit_kind,
            "year": frame["year"].astype("int64"),
            "variable": variable,
            "value": value,
            "weight": 1.0 if weight_col is None else frame[weight_col].astype(float),
            "support": support,
            "source": source,
        }
    )
    return out[present.to_numpy()].reset_index(drop=True)


def _clms_rows(clms: pd.DataFrame) -> pd.DataFrame:
    """CLMS emergence and harvest as **offset-corrected absolute dates**."""
    frame = clms.rename(columns={"adm_id": "unit"}).copy()
    emergence = doy_to_date(
        frame["year"], frame["obs_emergence_doy"], autumn=True
    ) - pd.Timedelta(days=CLMS_EMERGENCE_OFFSET_DAYS)
    harvest = doy_to_date(
        frame["year"], frame["obs_harvest_doy"], autumn=False
    ) + pd.Timedelta(days=CLMS_HARVEST_OFFSET_DAYS)

    # The within-unit pixel spread is the observation's own precision: a unit
    # whose wheat emerges across six weeks is a weaker constraint than one
    # that emerges in four days, and weighting by it is the only thing that
    # says so.
    frame["w_emergence"] = _spread_weight(
        frame["emergence_doy_p90"] - frame["emergence_doy_p10"]
    )
    frame["w_harvest"] = _spread_weight(
        frame["harvest_doy_p90"] - frame["harvest_doy_p10"]
    )
    frame["emergence_date"] = emergence
    frame["harvest_date"] = harvest

    return pd.concat(
        [
            _long(frame, "emergence_date", "emergence_date", unit_kind="adm",
                  support="adm", source="clms", weight_col="w_emergence"),
            _long(frame, "harvest_date", "harvest_date", unit_kind="adm",
                  support="adm", source="clms", weight_col="w_harvest"),
        ],
        ignore_index=True,
    )


def _spread_weight(spread: pd.Series, floor_days: float = 4.0) -> pd.Series:
    """``1 / max(spread, floor)``, normalised to a mean of one.

    The floor is the product's own 6-day compositing step rounded down: below
    it the spread is quantisation, not precision, and an unfloored inverse
    would hand a single unit an unbounded weight.
    """
    inverse = 1.0 / spread.astype(float).clip(lower=floor_days)
    return inverse / inverse.mean()


def _detrend(frame: pd.DataFrame, value_col: str, group: str) -> pd.DataFrame:
    """Split a per-unit series into its mean level and a detrended anomaly.

    A least-squares line per unit, not a difference from the unit mean: the
    trend is what LINTUL-5 structurally cannot produce, and removing only the
    mean would leave it inside the anomaly for the parameters to chase.
    """
    out = frame.copy()
    grouped = out.groupby(group, observed=True)
    year = out["year"].astype(float)
    value = out[value_col].astype(float)

    year_mean = grouped["year"].transform("mean").astype(float)
    value_mean = grouped[value_col].transform("mean").astype(float)
    dx, dy = year - year_mean, value - value_mean

    covariance = (dx * dy).groupby(out[group], observed=True).transform("sum")
    variance = (dx * dx).groupby(out[group], observed=True).transform("sum")
    # A unit with fewer than four seasons, or one reporting a single year, has
    # no trend worth fitting: the "trend" would be noise and removing it would
    # take the interannual signal with it.
    fittable = (variance > 0) & (grouped[value_col].transform("size") >= 4)
    slope = np.where(fittable, covariance / variance.where(variance > 0, 1.0), 0.0)

    out["level"] = value_mean
    out["anomaly"] = dy - slope * dx
    return out


def load_observations(
    clms: pd.DataFrame | None = None,
    yields: pd.DataFrame | None = None,
    fpar: pd.DataFrame | None = None,
    pep: pd.DataFrame | None = None,
    *,
    years: tuple[int, int] | None = None,
) -> ObservationSet:
    """Pool whichever references are supplied into the long table.

    Args:
        clms: :func:`cropmodelling4eu.evaluation.clms.load_phenology` output on
            the ``adm`` support.
        yields: :func:`cropmodelling4eu.evaluation.cybench.load_yield` output.
        fpar: CyBench ``fpar`` — ``adm_id``, ``date``, ``fpar``.
        pep: :func:`data4simplace.phenology.pep725.match_to_grid` output, i.e.
            station-seasons already carrying ``SimplaceID``.
        years: Inclusive harvest-year filter applied to every source.

    Returns:
        The :class:`ObservationSet`.

    Raises:
        ValueError: If every source is ``None`` or the pool comes out empty.
    """
    parts: list[pd.DataFrame] = []

    if clms is not None and not clms.empty:
        parts.append(_clms_rows(clms))

    if yields is not None and not yields.empty:
        frame = yields.rename(columns={"adm_id": "unit", "harvest_year": "year"}).copy()
        frame = _detrend(frame, "yield", "unit")
        # Area is the unit's weight in its region's mean, and the fallback for
        # the 75 % of German rows without one is an equal share -- the same
        # rule `evaluation.aggregate.weighted_mean` documents.
        frame["w"] = frame["harvest_area"].fillna(0.0).astype(float)
        frame.loc[frame["w"] <= 0, "w"] = frame.loc[frame["w"] > 0, "w"].median()
        frame["w"] = frame["w"].fillna(1.0)
        frame["w"] /= frame["w"].mean()
        parts += [
            _long(frame, "yield_level", "level", unit_kind="adm", support="adm",
                  source="cybench", weight_col="w"),
            _long(frame, "yield_anomaly", "anomaly", unit_kind="adm", support="adm",
                  source="cybench", weight_col="w"),
        ]

    if fpar is not None and not fpar.empty:
        parts.append(_fapar_rows(fpar))

    if pep is not None and not pep.empty:
        parts.append(_pep_rows(pep))

    if not parts:
        raise ValueError("no observation source supplied")

    table = pd.concat(parts, ignore_index=True)
    if years is not None:
        lo, hi = years
        table = table[table["year"].between(lo, hi)]
    if table.empty:
        raise ValueError("the observation pool is empty after filtering")

    table = table[list(COLUMNS)].reset_index(drop=True)
    observations = ObservationSet(table)
    logger.info("%s", observations.summarise())
    return observations


def _fapar_rows(fpar: pd.DataFrame) -> pd.DataFrame:
    """Dekadal fAPAR as a per-(unit, year) seasonal series.

    Compared to the model in **fAPAR space**, never inverted into LAI:
    ``DiagnosticState.frac_intercepted`` *is* the model's own
    ``1 - exp(-k.LAI)``, so comparing there means the inversion never happens
    and ``k`` never has to be guessed.

    The observation is carried as ``(day-of-year, value)`` pairs packed into
    two rows per unit-year — the peak level and the senescence half-fall date —
    because fAPAR saturates above LAI ~ 3 and so constrains canopy *timing* far
    better than the plateau.
    """
    frame = fpar.rename(columns={"adm_id": "unit"}).copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    frame = frame[frame["date"].notna()]
    value = pd.to_numeric(frame["fpar"], errors="coerce")
    # DE111 runs 7.9-30.7 while DEB3K opens at 45.9, so the column is percent
    # with crop-mask dilution rather than a fraction. Scaling is unambiguous;
    # the dilution is not, and it stays in the residual rather than being
    # silently rescaled away.
    frame["fapar"] = np.where(value.max() > 1.5, value / 100.0, value)
    frame["doy"] = frame["date"].dt.dayofyear
    # A dekad in the first half of the calendar year belongs to the season
    # harvested that year; one in the second half to the next.
    frame["year"] = frame["date"].dt.year + (frame["doy"] >= AUTUMN_CUT_DOY).astype(int)

    grouped = frame.groupby(["unit", "year"], observed=True)
    peak = grouped["fapar"].max().rename("value").reset_index()
    peak_doy = grouped.apply(
        lambda b: b.loc[b["fapar"].idxmax(), "date"], include_groups=False
    ).rename("peak_date").reset_index()

    half = (
        frame.merge(peak.rename(columns={"value": "peak"}), on=["unit", "year"])
        .merge(peak_doy, on=["unit", "year"])
    )
    half = half[(half["date"] > half["peak_date"]) & (half["fapar"] <= 0.5 * half["peak"])]
    senescence = (
        half.groupby(["unit", "year"], observed=True)["date"].min()
        .rename("value").reset_index()
    )

    return pd.concat(
        [
            _long(peak, "fapar_peak", "value", unit_kind="adm", support="adm",
                  source="cybench"),
            _long(senescence, "fapar_half_fall_date", "value", unit_kind="adm",
                  support="adm", source="cybench"),
        ],
        ignore_index=True,
    )


def _pep_rows(pep: pd.DataFrame) -> pd.DataFrame:
    """PEP725 station-seasons as dates on the 10 km cell that contains them.

    Point-matched, not nearest-centre: on a 0.1 degree grid the nearest centre
    to an interior point *is* its own cell, so a radius only ever admits
    stations from cells they are not in.
    """
    frame = pep.rename(columns={"harvest_year": "year"}).copy()
    frame["unit"] = frame["SimplaceID"].astype("int64").astype("string")
    frame["emergence_date"] = doy_to_date(frame["year"], frame["emergence_doy"], True)
    frame["harvest_date"] = doy_to_date(frame["year"], frame["harvest_doy"], False)

    parts = [
        _long(frame, "emergence_date", "emergence_date", unit_kind="cell",
              support="cell", source="pep725"),
        _long(frame, "harvest_date", "harvest_date", unit_kind="cell",
              support="cell", source="pep725"),
    ]
    if "heading_doy" in frame.columns:
        # BBCH 51 is heading, DVS ~ 0.85-0.90 -- not anthesis (BBCH 61,
        # DVS 1.0). It is the third dated stage the identifiability rule frees
        # `tsum1` and `tsum2` separately on, and it exists only where PEP725
        # has stations: CON, ATC, NEM and parts of ALS/PAN.
        frame["heading_date"] = doy_to_date(frame["year"], frame["heading_doy"], False)
        parts.append(
            _long(frame, "heading_date", "heading_date", unit_kind="cell",
                  support="cell", source="pep725")
        )
    return pd.concat(parts, ignore_index=True)


#: PEP725 BBCH 51. Requested explicitly because
#: :data:`data4simplace.phenology.pep725.PHASES` is the set with a **CLMS
#: counterpart**, and heading has none — that module drops it on purpose. It is
#: also the only observation that separates the pre- from the post-anthesis
#: thermal sum, so without asking for it here the identifiability rule binds at
#: two dated stages everywhere and tier 2 never unlocks.
HEADING_PHASE: int = 51


def load_pep725(
    root: Path,
    grid,
    *,
    years: tuple[int, int] = (1990, 2024),
    include_heading: bool = True,
) -> pd.DataFrame:
    """PEP725 winter-wheat station-seasons, matched to the export's cells.

    A wrapper over :mod:`data4simplace.phenology.pep725`, which owns the
    calendar-year to harvest-year alignment, the containment match and the
    circular statistics. Two things are asked of it that its own defaults do
    not give, and both are the reason PEP725 is loaded at all:

    * **Heading**, via ``extra_phases``. It is the third dated stage, and the
      only one that separates ``tsum1`` from ``tsum2``.
    * **The full record**, not the 2017-2024 CLMS window that module defaults
      to. PEP725's other unique contribution is exactly the seasons CLMS does
      not cover, and a calibration has no reason to discard them — the pairing
      that needs a common window is CLMS-versus-PEP725 validation, not this.

    ``station_seasons`` pivots only the ``PHASES`` stages, so heading is
    pivoted here and merged on.
    """
    from data4simplace.phenology import pep725

    phases = (HEADING_PHASE,) if include_heading else ()
    observations = pep725.load_observations(root, years, extra_phases=phases)
    known = {phase.code for phase in pep725.PHASES}
    seasons = pep725.station_seasons(
        observations[observations["phase_id"].isin(known)]
    )

    if include_heading:
        heading = observations[observations["phase_id"] == HEADING_PHASE]
        if heading.empty:
            logger.warning(
                "no BBCH %d rows in %s: no region gains a third dated stage, so "
                "tier-2 parameters stay frozen", HEADING_PHASE, root,
            )
        else:
            # Heading is a spring stage, so it belongs to the harvest year it
            # falls in -- no autumn shift, the same rule `PHASES` gives BBCH 100.
            wide = (
                heading.pivot_table(index=["s_id", "year"], values="doy",
                                    aggfunc="median")
                .reset_index()
                .rename(columns={"doy": "heading_doy", "year": "harvest_year"})
            )
            seasons = seasons.merge(wide, on=["s_id", "harvest_year"], how="left")
            logger.info(
                "PEP725 heading: %d station-seasons on %d stations",
                int(seasons["heading_doy"].notna().sum()),
                heading["s_id"].nunique(),
            )

    return pep725.match_to_grid(seasons, grid)
