"""Loader for the combined TorchCrop Europe run.

The run is one Parquet of ``(cell, season)`` rows written by
``submit/torchcrop_maps.py``. This module reads it, derives the phenology
day-of-year columns the evaluation needs, and does nothing else — country
assignment lives in :mod:`utils.regions`, aggregation in
:mod:`utils.aggregate`.

What the run does and does not predict is worth stating once, because it bounds
every phenology result downstream:

* ``yield_g_m2`` / ``yield_t_ha`` — grain dry matter at maturity.
* ``days_to_maturity`` — days from the sowing latch to the first day at
  ``DVS >= 2``. The only phenology the Parquet carries.
* Sowing is a **constant input** (``TC_SOWING_DOY``, DOY 270), not an output.
* Flowering and harvest are not in the file at all: ``run_batch`` discards the
  DVS trajectory once it has placed the fertilizer schedule.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import HARVEST_LAG_DAYS, SIM_PARQUET, SIM_SOWING_DOY
from .doy import wrap_doy

logger = logging.getLogger(__name__)

__all__ = [
    "PHENOLOGY_COLUMNS",
    "add_phenology_columns",
    "load_simulation",
    "simulation_cells",
]

#: Columns :func:`add_phenology_columns` creates, in stage order.
PHENOLOGY_COLUMNS: tuple[str, ...] = (
    "sowing_doy",
    "maturity_doy",
    "harvest_doy",
    "season_length_days",
)


def load_simulation(
    path: Path | str = SIM_PARQUET,
    columns: list[str] | None = None,
    years: tuple[int, int] | None = None,
    with_phenology: bool = True,
    sowing_doy: int = SIM_SOWING_DOY,
    harvest_lag_days: float = HARVEST_LAG_DAYS,
) -> pd.DataFrame:
    """Read the combined TorchCrop Parquet.

    Args:
        path: Combined Parquet (``torchcrop_europe.parquet``).
        columns: Subset to read. ``None`` reads every column. Pushing the
            subset into the Parquet reader rather than dropping afterwards
            matters: the full file is ~1.7 M rows across 19 columns.
        years: Inclusive ``(first, last)`` harvest-year filter, applied after
            the read.
        with_phenology: Derive the day-of-year columns (see
            :func:`add_phenology_columns`). Requires ``days_to_maturity`` to be
            among the columns read.
        sowing_doy: The run's sowing latch. Must match the run being read.
        harvest_lag_days: Days from maturity to harvest.

    Returns:
        One row per ``(SimplaceID, year)``. ``year`` is the **harvest** year;
        the season was sown in the preceding autumn.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"TorchCrop run not found: {path}. Set TORCHCROP_RUN_DIR or "
            "SIM_PARQUET to point at a finished run."
        )

    if columns is not None and with_phenology and "days_to_maturity" not in columns:
        columns = [*columns, "days_to_maturity"]

    frame = pd.read_parquet(path, columns=columns)
    logger.info("read %s: %d rows, %d columns", path.name, len(frame), frame.shape[1])

    frame["year"] = frame["year"].astype("int16")
    if years is not None:
        first, last = years
        frame = frame[frame["year"].between(first, last)].reset_index(drop=True)
        logger.info("kept %d rows in %d-%d", len(frame), first, last)

    if with_phenology:
        frame = add_phenology_columns(
            frame, sowing_doy=sowing_doy, harvest_lag_days=harvest_lag_days
        )
    return frame


def add_phenology_columns(
    frame: pd.DataFrame,
    sowing_doy: int = SIM_SOWING_DOY,
    harvest_lag_days: float = HARVEST_LAG_DAYS,
) -> pd.DataFrame:
    """Add the simulated stage dates as days-of-year.

    ``days_to_maturity`` counts forward from the sowing latch, so maturity is
    ``sowing_doy + days_to_maturity`` wrapped back into the year — a winter
    wheat crop sown on DOY 270 and maturing 272 days later matures on DOY 177
    of the *following* calendar year, which is the harvest year the row is
    labelled with.

    **The run's own ``sowing_doy`` column wins when it is present.** Since
    data4simplace gained its site stage the sowing date is a per-cell input, so
    the runner writes the date each cell was actually sown on. Applying one
    constant on top of that would put every cell's maturity date at the same
    offset from a date most of them were not sown on — a silent error of up to
    a couple of months at the domain's edges. The ``sowing_doy`` argument is
    the fallback for a run made before that column existed.

    Args:
        frame: Simulation rows carrying ``days_to_maturity``, and ideally
            ``sowing_doy``.
        sowing_doy: Sowing latch to assume when the frame carries no
            ``sowing_doy`` column.
        harvest_lag_days: Days from maturity to harvest. Zero makes
            ``harvest_doy`` identical to ``maturity_doy``, which is the honest
            default — LINTUL-5 stops the crop at maturity.

    Returns:
        A copy of ``frame`` with :data:`PHENOLOGY_COLUMNS` added.

    Raises:
        KeyError: If ``days_to_maturity`` is missing.
    """
    if "days_to_maturity" not in frame.columns:
        raise KeyError(
            "days_to_maturity is required to derive phenology; re-read the "
            "Parquet without excluding it"
        )

    out = frame.copy()
    days = out["days_to_maturity"].astype(float)

    if "sowing_doy" in out.columns:
        sown = out["sowing_doy"].astype(float)
        n_dates = int(sown.nunique())
        logger.info(
            "using the run's own per-cell sowing date (%d distinct value%s, "
            "%.0f-%.0f)", n_dates, "" if n_dates == 1 else "s",
            sown.min(), sown.max(),
        )
    else:
        sown = pd.Series(float(sowing_doy), index=out.index)
        logger.warning(
            "the run carries no sowing_doy column; assuming a constant DOY %d "
            "for every cell. Re-run against an export with a site table to "
            "make the sowing date data rather than an assumption.", sowing_doy,
        )
        out["sowing_doy"] = sown

    out["maturity_doy"] = wrap_doy(sown + days)
    out["harvest_doy"] = wrap_doy(sown + days + harvest_lag_days)
    # Kept linear on purpose: a season length is a duration, not a date, so it
    # must not be wrapped and must not use circular statistics.
    out["season_length_days"] = days + harvest_lag_days

    # A cell that never reached DVS = 2 has no maturity date. The run reports
    # that as NaN in days_to_maturity; keep it NaN rather than letting it
    # become a spurious DOY.
    missing = days.isna()
    if missing.any():
        logger.warning(
            "%d of %d rows (%.2f%%) never reached maturity; their dates are NaN",
            int(missing.sum()), len(out), 100.0 * float(missing.mean()),
        )
    return out


def simulation_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """The unique ``(SimplaceID, lon, lat)`` cell table behind a run.

    The country join runs on this — ~69 k rows rather than the ~1.7 M of the
    full frame — and its result is merged back per cell.

    Args:
        frame: Any simulation frame carrying ``SimplaceID``, ``lon`` and
            ``lat``.

    Returns:
        One row per cell, sorted by ``SimplaceID``.

    Raises:
        ValueError: If a ``SimplaceID`` carries more than one coordinate, which
            would mean two runs with different grids had been concatenated.
    """
    cells = (
        frame[["SimplaceID", "lon", "lat"]]
        .drop_duplicates()
        .sort_values("SimplaceID")
        .reset_index(drop=True)
    )
    duplicated = cells["SimplaceID"].duplicated()
    if duplicated.any():
        ids = cells.loc[duplicated, "SimplaceID"].unique()[:5]
        raise ValueError(
            f"{int(duplicated.sum())} SimplaceIDs carry more than one "
            f"coordinate (e.g. {list(ids)}); the input mixes grids"
        )
    logger.info("%d unique cells", len(cells))
    return cells
