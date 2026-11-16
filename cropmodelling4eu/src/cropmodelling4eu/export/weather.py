"""Read the export's per-cell weather files and slice seasons out of them.

Each cell has one gzipped, tab-separated CSV covering the whole export window
(1979-2024 for the European run). A season is a slice of it.

Two properties drive the design:

* **A gzip stream has no random access.** Whichever season is wanted, the whole
  file is decoded — so it is decoded *once* and every requested season is
  sliced out of the result. That is the difference between one pass and 45.
* **Reads do not scale over threads.** Decoding releases the GIL but date
  parsing and the per-season slicing do not: 16 threads reach ~2 busy cores
  and are slower per row than 4 (3.1 vs 5.8 ms, 2026-11-10). A run whose model
  is fast enough to wait on its weather uses :class:`RowBlockReader`, which
  reads in worker processes, ahead of the model.

Unit conversions live here rather than in a model runner, because they are
properties of the export rather than of a model: MSWX ``rsds`` is exported as a
daily-mean flux in W m-2, and vapour pressure has to be rebuilt from mean
temperature and relative humidity because no vapour-pressure column exists.
"""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig
from cropmodelling4eu.export.cells import weather_path

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNELS",
    "RowBlockReader",
    "SENTINEL",
    "load_row_block",
    "load_season_block",
    "load_seasons",
    "load_windows",
    "ordered_temperatures",
    "saturation_vp",
    "season_window",
    "sowing_date",
    "to_channels",
]

#: Missing-value sentinel of the weather export.
SENTINEL = -99.9

#: Daily-mean W m-2 -> MJ m-2 d-1. MSWX publishes a flux; LINTUL wants a daily
#: total.
W_PER_M2_TO_MJ = 86_400.0 / 1e6

#: Channel order of the returned arrays, matching `torchcrop.WeatherDriver`.
CHANNELS = ("doy", "davtmp", "tmin", "tmax", "irrad", "rain", "vp", "wind")

#: Columns read from the export's weather file.
_COLUMNS = [
    "Date", "Precipitation", "TempMin", "TempMean", "TempMax",
    "Radiation", "Windspeed", "RelHumCalc",
]

#: Fallback 2 m wind speed [m/s], used only when a file's Windspeed column is
#: entirely sentinel. Exports written before `sfcWind` reached the weather
#: exporter carry -99.9 in every row; re-exporting picks the real column up
#: with no flag to change.
FALLBACK_WIND_M_S = 2.0


def saturation_vp(temp_c: np.ndarray) -> np.ndarray:
    """Tetens saturation vapour pressure [kPa] for air temperature [degC]."""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def sowing_date(harvest_year: int, sowing_doy: int) -> pd.Timestamp:
    """Calendar date of the sowing harvested in ``harvest_year``.

    A sowing day-of-year in the second half of the year is an autumn sowing, so
    the season straddles New Year and the crop goes in the ground the year
    *before* harvest (winter wheat, ``sowing_doy`` around 270-300). An earlier
    DOY is a spring sowing, harvested in the same calendar year.
    """
    year = harvest_year - 1 if sowing_doy > 182 else harvest_year
    return pd.Timestamp(year, 1, 1) + pd.Timedelta(days=sowing_doy - 1)


def season_window(
    harvest_year: int,
    sowing_doy: int,
    spinup_months: int = 6,
    sim_years: int = 2,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Simulation window for the season harvested in ``harvest_year``.

    The window opens ``spinup_months`` before sowing and runs ``sim_years``
    from there:

    * The **spin-up** carries no crop — the model latches sowing on
      ``doy >= idpl`` — so its only job is to let the water balance relax out
      of the initial profile before the crop needs it.
    * The **remainder** is deliberately generous rather than cut at a nominal
      harvest date: the model self-terminates at ``DVS = 2``, so days past
      maturity change no reported quantity, whereas a window that ends early
      truncates a late season in a cold year.

    Consecutive harvest years therefore overlap by a year. That is intended —
    each season is an independent run, not a continuous rotation.
    """
    start = sowing_date(harvest_year, sowing_doy) - pd.DateOffset(months=spinup_months)
    return start, start + pd.DateOffset(years=sim_years) - pd.Timedelta(days=1)


def _read_file(path: Path) -> pd.DataFrame:
    """One cell's whole weather record, indexed by date."""
    return pd.read_csv(
        path,
        sep="\t",
        parse_dates=["Date"],
        usecols=_COLUMNS,
        na_values=[SENTINEL, -99.0],
    ).set_index("Date")


def ordered_temperatures(
    tmin: np.ndarray, tmean: np.ndarray, tmax: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(min, middle, max)`` of the three daily temperatures, per day.

    Bias-corrected CMIP6 exports correct each variable independently and so
    invert ~1.2% of days (Tmin > Tmax, Tmean outside the pair). SIMPLACE's
    solution applies the same reordering in its weather SQL, so both models
    read one weather. Days with a missing value are left as they are, for the
    sentinel handling downstream.
    """
    stacked = np.stack([tmin, tmean, tmax]).astype(float)
    complete = np.isfinite(stacked).all(axis=0)
    ordered = np.where(complete, np.sort(stacked, axis=0), stacked)
    return ordered[0], ordered[1], ordered[2]


def to_channels(block: pd.DataFrame, has_wind: bool) -> np.ndarray:
    """One date-indexed weather slice as a ``[T, 8]`` array in :data:`CHANNELS` order.

    The unit conversions are properties of the *export*, not of a model or of a
    window, so every reader goes through here: a second implementation of
    "W m-2 to MJ m-2 d-1" or of the vapour-pressure reconstruction is a second
    chance to get one of them wrong.
    """
    tmin, tmean, tmax = ordered_temperatures(
        block["TempMin"].to_numpy(), block["TempMean"].to_numpy(), block["TempMax"].to_numpy()
    )
    frame = pd.DataFrame(
        {
            "doy": block.index.dayofyear.to_numpy(float),
            "davtmp": (tmin + tmax) / 2.0,
            "tmin": tmin,
            "tmax": tmax,
            "irrad": block["Radiation"].to_numpy() * W_PER_M2_TO_MJ,
            "rain": np.nan_to_num(block["Precipitation"].to_numpy(), nan=0.0),
            # No vapour-pressure column exists, so it is rebuilt from the
            # saturation value at mean temperature and relative humidity.
            "vp": saturation_vp(tmean)
            * block["RelHumCalc"].to_numpy()
            / 100.0,
            "wind": (
                block["Windspeed"].to_numpy() if has_wind else FALLBACK_WIND_M_S
            ),
        }
    ).interpolate(limit_direction="both")
    return frame[list(CHANNELS)].to_numpy(dtype=np.float32)


def load_windows(
    export_dir: Path,
    simplace_id: int,
    windows: dict[int, tuple[pd.Timestamp, pd.Timestamp]],
    grid: GridConfig,
) -> dict[int, np.ndarray]:
    """Read one cell's file once and slice out explicit date windows.

    :func:`load_seasons` derives its windows from a sowing day-of-year; a
    calibration run cuts a much shorter one around each season (see
    :mod:`cropmodelling4eu.calibration.dataset`) and needs to say so directly.
    A window the record cannot cover in full is omitted rather than returned
    short, so every caller can assume a fixed length.
    """
    raw = _read_file(weather_path(export_dir, simplace_id, grid))
    has_wind = raw["Windspeed"].notna().any()
    out: dict[int, np.ndarray] = {}
    for year, (start, end) in windows.items():
        block = raw.loc[start:end]
        if block.empty or block.index[0] > start or block.index[-1] < end:
            continue
        out[year] = to_channels(block, has_wind)
    return out


def load_seasons(
    export_dir: Path,
    simplace_id: int,
    years: list[int],
    sowing_doy: int,
    grid: GridConfig,
    spinup_months: int = 6,
    min_days_after_sowing: int = 365,
    raw: pd.DataFrame | None = None,
) -> dict[int, np.ndarray]:
    """Read one cell's file once and slice out every requested season.

    ``raw`` is the cell's already-decoded record (see :func:`load_season_block`'s
    ``records``); without it the file is read here.

    A harvest year is dropped when the record cannot cover its window's crop
    phase — it starts after sowing, or ends less than ``min_days_after_sowing``
    past it. Without that check the first and last requested years would run on
    a silently truncated window: a front-truncated one misses its own sowing
    entirely and the latch fires on the *next* autumn, reporting a season a
    year off.

    Returns
    -------
    dict
        ``{harvest_year: [T, 8]}`` in :data:`CHANNELS` order.
    """
    if raw is None:
        raw = _read_file(weather_path(export_dir, simplace_id, grid))

    # Exports predating the sfcWind fix carry -99.9 in every Windspeed row,
    # which read_csv turned into NaN. Fall back to the constant only then, so a
    # re-exported file is picked up with no flag to set.
    has_wind = raw["Windspeed"].notna().any()

    out: dict[int, np.ndarray] = {}
    for year in years:
        start, end = season_window(year, sowing_doy, spinup_months)
        block = raw.loc[start:end]
        sown = sowing_date(year, sowing_doy)
        if (
            block.empty
            or block.index[0] > sown
            or (block.index[-1] - sown).days < min_days_after_sowing
        ):
            continue
        out[year] = to_channels(block, has_wind)
    return out


def load_season_block(
    export_dir: Path,
    ids: np.ndarray,
    years: list[int],
    sowing_doy: int,
    grid: GridConfig,
    workers: int = 16,
    spinup_months: int = 6,
    min_days_after_sowing: int = 365,
    records: dict[int, pd.DataFrame] | None = None,
) -> dict[int, tuple[np.ndarray, int]]:
    """Weather for a batch of cells, as ``{year: ([B, T, 8], start_doy)}``.

    Every cell in a batch must share one time axis, so a year is kept only if
    every cell has a usable window and they all start on the same day; the
    series are then trimmed to the shortest. ``start_doy`` is read back off the
    day-of-year channel rather than from the nominal window, so a record
    truncated at the front still hands the model the day its sowing latch
    actually opens on.

    Batching by a **single** sowing day-of-year is what makes that invariant
    hold — see :func:`cropmodelling4eu.torchcrop.run.group_by_sowing`.

    ``records`` caches each cell's decoded file across calls. A caller batching
    one season at a time (a simulated-sowing table, see
    :func:`cropmodelling4eu.torchcrop.run.sowing_plan`) would otherwise decode
    every cell's whole record once per season: for a 165-year CMIP6 export
    that was 85 s of a 102 s batch. The cache costs ~4 MB per cell.
    """
    def cell_seasons(sid: int) -> dict[int, np.ndarray]:
        raw = None
        if records is not None:
            raw = records.get(sid)
            if raw is None:
                raw = records[sid] = _read_file(weather_path(export_dir, sid, grid))
        return load_seasons(
            export_dir, sid, years, sowing_doy, grid,
            spinup_months, min_days_after_sowing, raw,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_cell = list(pool.map(cell_seasons, ids))

    blocks: dict[int, tuple[np.ndarray, int]] = {}
    for year in years:
        arrays = [cell[year] for cell in per_cell if year in cell]
        if len(arrays) != len(ids):
            logger.warning(
                "year %d: only %d/%d cells have a usable window; skipping the year",
                year, len(arrays), len(ids),
            )
            continue
        start_doys = {int(a[0, 0]) for a in arrays}
        if len(start_doys) != 1:
            logger.warning(
                "year %d: cells disagree on the first day (%s); skipping the year",
                year, sorted(start_doys),
            )
            continue
        n_days = min(a.shape[0] for a in arrays)
        blocks[year] = (np.stack([a[:n_days] for a in arrays]), start_doys.pop())
    return blocks


def load_row_block(
    export_dir: Path,
    ids: np.ndarray,
    years: np.ndarray,
    sowing_doy: int,
    grid: GridConfig,
    records: dict[int, pd.DataFrame],
    workers: int = 16,
    spinup_months: int = 6,
    min_days_after_sowing: int = 365,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Weather for ``(cell, season)`` rows of *different* seasons sharing one anchor.

    :func:`load_season_block`'s batch is one set of cells over shared seasons;
    here row ``i`` is cell ``ids[i]`` in harvest year ``years[i]``, every
    window opened ``spinup_months`` before ``sowing_doy`` of its own year. That
    lands on one day-of-year in every year except where a leap day falls
    between window start and sowing, so rows are grouped by their actual first
    day, each group trimmed to its shortest window.

    A row without a usable window is dropped on its own, not with its season.

    Returns
    -------
    list
        ``[(row_index, [B, T, 8], start_doy)]``, ``row_index`` into ``ids``.
    """
    def row(i: int) -> np.ndarray | None:
        return _window_row(export_dir, int(ids[i]), int(years[i]), sowing_doy, grid,
                           records, spinup_months, min_days_after_sowing)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(pool.map(row, range(ids.size)))
    return _group_rows(arrays)


def _window_row(
    export_dir: Path,
    sid: int,
    year: int,
    sowing_doy: int,
    grid: GridConfig,
    records: dict[int, pd.DataFrame],
    spinup_months: int,
    min_days_after_sowing: int,
) -> np.ndarray | None:
    raw = records.get(sid)
    if raw is None:
        raw = records[sid] = _read_file(weather_path(export_dir, sid, grid))
    return load_seasons(
        export_dir, sid, [year], sowing_doy, grid,
        spinup_months, min_days_after_sowing, raw,
    ).get(year)


def _group_rows(arrays: list[np.ndarray | None]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Rows grouped by their window's first day, each group trimmed to its shortest."""
    dropped = sum(a is None for a in arrays)
    if dropped:
        logger.warning("%d of %d (cell, season) rows have no usable window", dropped, len(arrays))
    by_start: dict[int, list[int]] = {}
    for i, a in enumerate(arrays):
        if a is not None:
            by_start.setdefault(int(a[0, 0]), []).append(i)
    out = []
    for start_doy, rows in sorted(by_start.items()):
        n_days = min(arrays[i].shape[0] for i in rows)
        out.append((
            np.asarray(rows), np.stack([arrays[i][:n_days] for i in rows]), start_doy,
        ))
    return out


#: Per-process record cache of a :class:`RowBlockReader` worker.
_WORKER_RECORDS: dict[int, pd.DataFrame] = {}


def _worker_rows(
    export_dir: Path,
    sids: np.ndarray,
    years: np.ndarray,
    sowing_doy: int,
    grid: GridConfig,
    spinup_months: int,
    min_days_after_sowing: int,
) -> list[np.ndarray | None]:
    return [
        _window_row(export_dir, int(sid), int(year), sowing_doy, grid, _WORKER_RECORDS,
                    spinup_months, min_days_after_sowing)
        for sid, year in zip(sids, years)
    ]


class _PendingBlock:
    """One batch's weather being read across the workers."""

    __slots__ = ("size", "shards")

    def __init__(self, size: int, shards: list[tuple[np.ndarray, Future]]) -> None:
        self.size, self.shards = size, shards

    def result(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        arrays: list[np.ndarray | None] = [None] * self.size
        for positions, future in self.shards:
            for i, a in zip(positions, future.result()):
                arrays[i] = a
        return _group_rows(arrays)


class RowBlockReader:
    """:func:`load_row_block` in ``n_workers`` processes, submitted ahead of the model.

    Each cell is pinned to one worker (``SimplaceID % n_workers``), so its
    decoded record stays in that worker's cache across batches exactly as the
    in-process ``records`` dict keeps it, and no file is decoded twice. One
    single-process pool per worker is what makes the pinning possible.

    Workers are spawned, not forked: the parent may already hold a CUDA
    context, and a forked child inherits it in a state CUDA does not support.

    Use as a context manager; :meth:`submit` returns at once and the block's
    ``result()`` has the :func:`load_row_block` return value.
    """

    def __init__(
        self,
        export_dir: Path,
        grid: GridConfig,
        n_workers: int,
        spinup_months: int = 6,
        min_days_after_sowing: int = 365,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, not {n_workers}")
        self.export_dir, self.grid = export_dir, grid
        self.spinup_months, self.min_days_after_sowing = spinup_months, min_days_after_sowing
        context = multiprocessing.get_context("spawn")
        self._pools = [ProcessPoolExecutor(1, mp_context=context) for _ in range(n_workers)]
        # Spawning imports pandas in every worker: pay it here, not in the first batch.
        for future in [pool.submit(int, 0) for pool in self._pools]:
            future.result()

    def submit(self, ids: np.ndarray, years: np.ndarray, sowing_doy: int) -> _PendingBlock:
        owner = np.asarray(ids, dtype=np.int64) % len(self._pools)
        shards = []
        for k, pool in enumerate(self._pools):
            positions = np.flatnonzero(owner == k)
            if positions.size:
                shards.append((positions, pool.submit(
                    _worker_rows, self.export_dir, ids[positions], years[positions],
                    sowing_doy, self.grid, self.spinup_months, self.min_days_after_sowing,
                )))
        return _PendingBlock(len(ids), shards)

    def close(self) -> None:
        for pool in self._pools:
            pool.shutdown(cancel_futures=True)

    def __enter__(self) -> RowBlockReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
