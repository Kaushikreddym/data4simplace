"""Per-season heat, drought and heavy-precipitation indices for every exported cell.

The indices are computed from the **same daily files the models ran on**
(``<export>/weather/<row>/daily_mean_RES1_C<col>R<row>.csv.gz``), so a hazard
and the yield it is set against share one forcing.

Each index is read inside a window anchored on the cell's **climatological**
phenology — the per-cell median sowing date, days to anthesis and days to
maturity of the torchcrop run — rather than on each season's simulated dates.
That keeps the hazard exogenous: a window that moved with the simulated
phenology would let a model's own response (an early maturity in a hot year)
change the exposure it is scored against, and an observed yield has no
simulated phenology at all.

Heat is counted against absolute thresholds (31 °C at flowering, 30 °C degree
days) and against the cell's own climate (TX90p): an absolute threshold is
almost never crossed north of ~55° N, where a heat wave is still a heat wave.

Windows, in days after sowing (``A`` anthesis, ``M`` maturity):

====================  ==================  =========================================
Window                Days after sowing   Why
====================  ==================  =========================================
``winter``            ``[0, A − 60)``     Frost kill before regrowth
``preanthesis``       ``[A − 60, A − 7)`` Stem elongation/booting: water demand peaks
``flowering``         ``[A − 7, A + 7)``  Heat sterility of the florets
``grainfill``         ``[A + 7, M)``      Heat and drought shorten grain filling
``season``            ``[0, M)``          Whole-season totals and maxima
====================  ==================  =========================================

``RefET`` is ``-99.9`` throughout the MSWX export, so the climatic water
balance uses Hargreaves–Samani reference ET (FAO-56 extraterrestrial radiation
from latitude and day), which needs only the temperatures the export carries.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPORT_DIR = Path(os.environ.get(
    "EXTREMES_EXPORT_DIR",
    "/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs/exports/mswx/EU_val_nfix",
))

#: Grid the export's ``SimplaceID`` is row-major on, from the NW corner.
N_LON, MIN_LON, MAX_LAT, RESOLUTION = 690, -17.0, 72.0, 0.1
MISSING = -99.9

#: Anthesis threshold for floret sterility in wheat (Porter & Gawith 1999).
HEAT_FLOWERING_C = 31.0
#: Base of the extreme degree days; yield response turns sharply negative above
#: ~30 °C in European wheat (Schlenker & Roberts 2009; Zampieri et al. 2017).
EDD_BASE_C = 30.0
#: Winter-kill proxy. The export has no snow cover, so this counts exposure
#: rather than kill; it belongs to the non-extreme control group.
FROST_C = -15.0
DRY_DAY_MM = 1.0
#: ETCCDI R95p: precipitation on days above the cell's wet-day 95th percentile.
HEAVY_QUANTILE = 0.95

#: ``feature -> (group, unit, label)``. The group decides which hazard an
#: index's importance and attribution are credited to. Drought is measured by
#: precipitation alone: P − ET₀ falls with temperature, so a water-balance
#: drought index would credit heat damage to drought. The water balance is kept
#: as a ``descriptive`` index for the trend maps and is never a model feature.
FEATURES: dict[str, tuple[str, str, str]] = {
    "heat_days_flowering": ("heat", "days", f"Days Tmax ≥ {HEAT_FLOWERING_C:.0f} °C, flowering"),
    "edd_grainfill": ("heat", "°C·d", f"Degree days > {EDD_BASE_C:.0f} °C, grain fill"),
    "tx90p_grainfill": ("heat", "days", "Days Tmax > cell's 90th percentile (TX90p), grain fill"),
    "prec_preanthesis": ("drought", "mm", "Precipitation, pre-anthesis"),
    "prec_grainfill": ("drought", "mm", "Precipitation, grain fill"),
    "cdd_spring": ("drought", "days", "Longest dry spell, pre-anthesis → maturity"),
    "rx5day_season": ("wet", "mm", "Wettest 5 days of the season"),
    "r95p_season": ("wet", "mm", "Rain on very wet days (R95p), season"),
    "r95p_grainfill": ("wet", "mm", "Rain on very wet days (R95p), grain fill"),
    "tmean_season": ("other", "°C", "Mean temperature, season"),
    "rad_grainfill": ("other", "MJ m⁻² d⁻¹", "Mean radiation, grain fill"),
    "frost_days_winter": ("other", "days", f"Days Tmin ≤ {FROST_C:.0f} °C, winter"),
    "cwb_preanthesis": ("descriptive", "mm", "Climatic water balance P − ET₀, pre-anthesis"),
    "cwb_grainfill": ("descriptive", "mm", "Climatic water balance P − ET₀, grain fill"),
}

GROUPS: dict[str, list[str]] = {
    g: [f for f, (group, *_) in FEATURES.items() if group == g]
    for g in ("heat", "drought", "wet", "other")
}
MODEL_FEATURES: list[str] = [f for cols in GROUPS.values() for f in cols]


@dataclass(slots=True, frozen=True)
class Windows:
    """Climatological phenology of one cell, in days after sowing."""

    sowing_doy: int
    anthesis: int
    maturity: int

    def bounds(self) -> dict[str, tuple[int, int]]:
        a, m = self.anthesis, self.maturity
        return {
            "winter": (0, max(a - 60, 1)),
            "preanthesis": (max(a - 60, 0), a - 7),
            "flowering": (a - 7, a + 7),
            "grainfill": (a + 7, max(m, a + 8)),
            "spring": (max(a - 60, 0), max(m, a + 8)),
            "season": (0, max(m, a + 8)),
        }


def cell_position(simplace_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(row, col)`` of a ``SimplaceID`` on the export grid."""
    index = np.asarray(simplace_id, dtype=np.int64) - 1
    return index // N_LON, index % N_LON


def weather_path(simplace_id: int, export_dir: Path = EXPORT_DIR) -> Path:
    row, col = divmod(int(simplace_id) - 1, N_LON)
    return export_dir / "weather" / str(row) / f"daily_mean_RES1_C{col}R{row}.csv.gz"


def phenology_windows(torchcrop_parquet: Path) -> pd.DataFrame:
    """Per-cell median sowing DOY, days to anthesis and days to maturity."""
    run = pd.read_parquet(torchcrop_parquet, columns=[
        "SimplaceID", "lat", "lon", "sowing_doy", "days_to_anthesis", "days_to_maturity"])
    out = run.groupby("SimplaceID").median()
    stages = ["sowing_doy", "days_to_anthesis", "days_to_maturity"]
    out[stages] = out[stages].round().astype(int)
    return out.reset_index()


def hargreaves_et0(doy: np.ndarray, lat_deg: float, tmin: np.ndarray, tmax: np.ndarray,
                   tmean: np.ndarray) -> np.ndarray:
    """FAO-56 Hargreaves–Samani reference ET (mm/d)."""
    phi = np.deg2rad(lat_deg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))
    ra = 24.0 * 60.0 / np.pi * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))
    return 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt(np.clip(tmax - tmin, 0.0, None))


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return int((edges[1::2] - edges[::2]).max())


def _rx5(p: np.ndarray) -> float:
    return float(np.convolve(p, np.ones(5), "valid").max()) if p.size >= 5 else float(p.sum())


def cell_indices(simplace_id: int, lat: float, windows: Windows,
                 years: tuple[int, int], export_dir: Path = EXPORT_DIR) -> pd.DataFrame:
    """One row per harvest year with every :data:`FEATURES` index of one cell."""
    daily = pd.read_csv(weather_path(simplace_id, export_dir), sep="\t", usecols=[
        "Date", "Precipitation", "TempMin", "TempMean", "TempMax", "Radiation"])
    dates = pd.to_datetime(daily["Date"]).to_numpy()
    values = daily.drop(columns="Date").to_numpy(dtype=float)
    values[values <= MISSING + 1e-6] = np.nan
    prec, tmin, tmean, tmax, rad = (np.nan_to_num(values[:, 0]), *values[:, 1:4].T,
                                    values[:, 4] * 0.0864)
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    cwb = prec - hargreaves_et0(doy, lat, tmin, tmax, tmean)
    wet = prec[prec >= DRY_DAY_MM]
    heavy = float(np.quantile(wet, HEAVY_QUANTILE)) if wet.size else np.inf
    excess_heat = np.clip(tmax - EDD_BASE_C, 0.0, None)

    t0 = dates[0].astype("datetime64[D]")
    bounds = windows.bounds()
    seasons = {}
    for year in range(years[0], years[1] + 1):
        sow = int((np.datetime64(f"{year - 1}-01-01") - t0).astype(int)) + windows.sowing_doy - 1
        if sow >= 0 and sow + bounds["season"][1] <= len(dates):
            seasons[year] = {k: slice(sow + a, sow + b) for k, (a, b) in bounds.items()}
    tx90 = np.nanquantile(np.concatenate([tmax[s["grainfill"]] for s in seasons.values()]), 0.9)
    rows = []
    for year, s in seasons.items():
        rows.append({
            "year": year,
            "heat_days_flowering": np.sum(tmax[s["flowering"]] >= HEAT_FLOWERING_C),
            "edd_grainfill": np.nansum(excess_heat[s["grainfill"]]),
            "tx90p_grainfill": np.sum(tmax[s["grainfill"]] > tx90),
            "prec_preanthesis": prec[s["preanthesis"]].sum(),
            "prec_grainfill": prec[s["grainfill"]].sum(),
            "cwb_preanthesis": np.nansum(cwb[s["preanthesis"]]),
            "cwb_grainfill": np.nansum(cwb[s["grainfill"]]),
            "cdd_spring": _longest_run(prec[s["spring"]] < DRY_DAY_MM),
            "rx5day_season": _rx5(prec[s["season"]]),
            "r95p_season": prec[s["season"]][prec[s["season"]] > heavy].sum(),
            "r95p_grainfill": prec[s["grainfill"]][prec[s["grainfill"]] > heavy].sum(),
            "tmean_season": np.nanmean(tmean[s["season"]]),
            "rad_grainfill": np.nanmean(rad[s["grainfill"]]),
            "frost_days_winter": np.sum(tmin[s["winter"]] <= FROST_C),
        })
    return pd.DataFrame(rows).assign(SimplaceID=simplace_id).astype(
        {f: "float32" for f in FEATURES})


def _chunk(args: tuple[pd.DataFrame, tuple[int, int], Path]) -> pd.DataFrame:
    cells, years, export_dir = args
    frames = []
    for c in cells.itertuples(index=False):
        try:
            frames.append(cell_indices(c.SimplaceID, c.lat, Windows(
                c.sowing_doy, c.days_to_anthesis, c.days_to_maturity), years, export_dir))
        except FileNotFoundError:
            logger.warning("cell %d: no weather file", c.SimplaceID)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def season_extremes(windows: pd.DataFrame, years: tuple[int, int],
                    export_dir: Path = EXPORT_DIR, workers: int | None = None,
                    chunk_size: int = 250) -> pd.DataFrame:
    """Every cell's indices for every harvest year, in parallel over cells."""
    workers = workers or min(64, os.cpu_count() or 1)
    chunks = [(windows.iloc[i:i + chunk_size], years, export_dir)
              for i in range(0, len(windows), chunk_size)]
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(_chunk, chunks))
    out = pd.concat(frames, ignore_index=True)
    return out[["SimplaceID", "year", *FEATURES]].astype({"year": "int16"})


def load_or_compute(cache: Path, torchcrop_parquet: Path, years: tuple[int, int],
                    export_dir: Path = EXPORT_DIR, workers: int | None = None,
                    refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(windows, extremes)``, read from ``cache`` or computed and written there."""
    window_file = cache.with_name(cache.stem + "_windows.parquet")
    if cache.is_file() and window_file.is_file() and not refresh:
        return pd.read_parquet(window_file), pd.read_parquet(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    windows = phenology_windows(torchcrop_parquet)
    extremes = season_extremes(windows, years, export_dir, workers)
    windows.to_parquet(window_file, index=False)
    extremes.to_parquet(cache, index=False)
    return windows, extremes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True, help="production run directory")
    parser.add_argument("--first", type=int, default=1980)
    parser.add_argument("--last", type=int, default=2024)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--refresh", action="store_true")
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    load_or_compute(a.run / "analysis" / "extremes" / "season_extremes.parquet",
                    a.run / "torchcrop" / "torchcrop_europe.parquet", (a.first, a.last),
                    workers=a.workers, refresh=a.refresh)
