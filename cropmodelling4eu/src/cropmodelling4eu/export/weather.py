"""Read the export's per-cell weather files and slice seasons out of them.

Each cell has one gzipped, tab-separated CSV covering the whole export window
(1979-2024 for the European run). A season is a slice of it.

Two properties drive the design:

* **A gzip stream has no random access.** Whichever season is wanted, the whole
  file is decoded — so it is decoded *once* and every requested season is
  sliced out of the result. That is the difference between one pass and 45.
* **Reads release the GIL**, being gzip-bound, so a thread pool scales here
  without the pickling cost of processes.

Unit conversions live here rather than in a model runner, because they are
properties of the export rather than of a model: MSWX ``rsds`` is exported as a
daily-mean flux in W m-2, and vapour pressure has to be rebuilt from mean
temperature and relative humidity because no vapour-pressure column exists.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig
from cropmodelling4eu.export.cells import weather_path

logger = logging.getLogger(__name__)

__all__ = [
    "CHANNELS",
    "SENTINEL",
    "load_season_block",
    "load_seasons",
    "saturation_vp",
    "season_window",
    "sowing_date",
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


def load_seasons(
    export_dir: Path,
    simplace_id: int,
    years: list[int],
    sowing_doy: int,
    grid: GridConfig,
    spinup_months: int = 6,
    min_days_after_sowing: int = 365,
) -> dict[int, np.ndarray]:
    """Read one cell's file once and slice out every requested season.

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
        tmin = block["TempMin"].to_numpy()
        tmax = block["TempMax"].to_numpy()
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
                "vp": saturation_vp(block["TempMean"].to_numpy())
                * block["RelHumCalc"].to_numpy()
                / 100.0,
                "wind": (
                    block["Windspeed"].to_numpy() if has_wind else FALLBACK_WIND_M_S
                ),
            }
        ).interpolate(limit_direction="both")
        out[year] = frame[list(CHANNELS)].to_numpy(dtype=np.float32)
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
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_cell = list(
            pool.map(
                lambda sid: load_seasons(
                    export_dir, sid, years, sowing_doy, grid,
                    spinup_months, min_days_after_sowing,
                ),
                ids,
            )
        )

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
