"""Per-season weather indices for a sample of cells, in every model and experiment.

Windows are fixed per cell (median phenology of the ``historical`` baseline across the
models), so an index measures the forcing and not the crop's response. Thresholds that
depend on a cell's own climate (TX90p, R95p) are fixed per model on that same baseline:
computed per run, a warmer scenario would raise its own threshold and hide its heat.

Cells are drawn with probability proportional to cropland area, so an unweighted mean
over the sample estimates the cropland-weighted mean.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import scenarios as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import extremes as ex  # noqa: E402

logger = logging.getLogger(__name__)

#: Every model has seasons from here on (ACCESS-ESM1-5's forcing starts in 1950).
FIRST_YEAR = 1951
CACHE = sc.OUT / "explainability" / "season_features.parquet"

#: ``feature -> (group, unit, label)``. The group is what a contribution is credited to.
FEATURES: dict[str, tuple[str, str, str]] = {
    "heat_days_flowering": ("heat", "days", "Days Tmax ≥ 31 °C, flowering"),
    "edd_grainfill": ("heat", "°C·d", "Degree days > 30 °C, grain fill"),
    "tx90p_grainfill": ("heat", "days", "Days above the baseline TX90, grain fill"),
    "prec_preanthesis": ("water", "mm", "Precipitation, pre-anthesis"),
    "prec_grainfill": ("water", "mm", "Precipitation, grain fill"),
    "cdd_spring": ("water", "days", "Longest dry spell, pre-anthesis to maturity"),
    "vpd_grainfill": ("vpd", "kPa", "Mean vapour-pressure deficit, grain fill"),
    "vpd_preanthesis": ("vpd", "kPa", "Mean vapour-pressure deficit, pre-anthesis"),
    "rx5day_season": ("wet", "mm", "Wettest 5 days, season"),
    "r95p_season": ("wet", "mm", "Rain above the baseline wet-day P95, season"),
    "tmean_winter": ("temperature", "°C", "Mean temperature, winter"),
    "tmean_preanthesis": ("temperature", "°C", "Mean temperature, pre-anthesis"),
    "tmean_grainfill": ("temperature", "°C", "Mean temperature, grain fill"),
    "rad_preanthesis": ("radiation", "MJ m⁻² d⁻¹", "Mean radiation, pre-anthesis"),
    "rad_grainfill": ("radiation", "MJ m⁻² d⁻¹", "Mean radiation, grain fill"),
    "frost_days_winter": ("frost", "days", "Days Tmin ≤ −15 °C, winter"),
}
COLUMNS = ["Date", "Precipitation", "TempMin", "TempMean", "TempMax", "Radiation", "RelHumCalc"]


def _daily(model: str, experiment: str, simplace_id: int) -> tuple[np.ndarray, ...]:
    path = ex.weather_path(simplace_id, sc.EXPORTS / model / experiment)
    frame = pd.read_csv(path, sep="\t", usecols=COLUMNS)
    values = frame[COLUMNS[1:]].to_numpy(dtype=float)
    values[values <= ex.MISSING + 1e-6] = np.nan
    return (pd.to_datetime(frame["Date"]).to_numpy(), *values.T)


def _season_slices(dates: np.ndarray, bounds: dict[str, tuple[int, int]], sowing_doy: int,
                   years: tuple[int, int]) -> dict[int, dict[str, slice]]:
    t0 = dates[0].astype("datetime64[D]")
    seasons = {}
    for year in range(years[0], years[1] + 1):
        sow = int((np.datetime64(f"{year - 1}-01-01") - t0).astype(int)) + sowing_doy - 1
        if sow >= 0 and sow + bounds["season"][1] <= len(dates):
            seasons[year] = {k: slice(sow + a, sow + b) for k, (a, b) in bounds.items()}
    return seasons


def _vpd(tmean: np.ndarray, rh: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3)) * np.clip(1.0 - rh / 100.0, 0.0, 1.0)


def cell_features(task: tuple[str, int, int, int, int]) -> pd.DataFrame:
    """Every experiment's season indices for one model and cell."""
    model, simplace_id, sowing, anthesis, maturity = task
    bounds = ex.Windows(sowing, anthesis, maturity).bounds()
    dates, prec, _, _, tmax, _, _ = _daily(model, "historical", simplace_id)
    prec = np.nan_to_num(prec)
    base = _season_slices(dates, bounds, sowing, sc.BASELINE)
    tx90 = np.nanquantile(np.concatenate([tmax[s["grainfill"]] for s in base.values()]), 0.9)
    wet = np.concatenate([prec[s["season"]] for s in base.values()])
    heavy = float(np.quantile(wet[wet >= ex.DRY_DAY_MM], ex.HEAVY_QUANTILE)) if (wet >= ex.DRY_DAY_MM).any() else np.inf

    frames = []
    for experiment in sc.EXPERIMENTS:
        dates, prec, tmin, tmean, tmax, rad, rh = _daily(model, experiment, simplace_id)
        prec, rad, vpd = np.nan_to_num(prec), rad * 0.0864, _vpd(tmean, rh)
        excess = np.clip(tmax - ex.EDD_BASE_C, 0.0, None)
        first = FIRST_YEAR if experiment in sc.HISTORICAL else 2016
        rows = []
        for year, s in _season_slices(dates, bounds, sowing, (first, 2100)).items():
            season_rain = prec[s["season"]]
            rows.append({
                "year": year,
                "heat_days_flowering": np.sum(tmax[s["flowering"]] >= ex.HEAT_FLOWERING_C),
                "edd_grainfill": np.nansum(excess[s["grainfill"]]),
                "tx90p_grainfill": np.sum(tmax[s["grainfill"]] > tx90),
                "prec_preanthesis": prec[s["preanthesis"]].sum(),
                "prec_grainfill": prec[s["grainfill"]].sum(),
                "cdd_spring": ex._longest_run(prec[s["spring"]] < ex.DRY_DAY_MM),
                "vpd_grainfill": np.nanmean(vpd[s["grainfill"]]),
                "vpd_preanthesis": np.nanmean(vpd[s["preanthesis"]]),
                "rx5day_season": ex._rx5(season_rain),
                "r95p_season": season_rain[season_rain > heavy].sum(),
                "tmean_winter": np.nanmean(tmean[s["winter"]]),
                "tmean_preanthesis": np.nanmean(tmean[s["preanthesis"]]),
                "tmean_grainfill": np.nanmean(tmean[s["grainfill"]]),
                "rad_preanthesis": np.nanmean(rad[s["preanthesis"]]),
                "rad_grainfill": np.nanmean(rad[s["grainfill"]]),
                "frost_days_winter": np.sum(tmin[s["winter"]] <= ex.FROST_C),
            })
        frames.append(pd.DataFrame(rows).assign(experiment=experiment))
    return pd.concat(frames, ignore_index=True).assign(model=model, SimplaceID=simplace_id).astype(
        {f: "float32" for f in FEATURES} | {"year": "int16", "SimplaceID": "int32"})


def sample_cells(n: int, seed: int = 0) -> pd.DataFrame:
    """``n`` cells drawn by cropland area, with their fixed phenology windows."""
    cells = sc.load_cells()
    p = cells.weight / cells.weight.sum()
    chosen = np.sort(np.random.default_rng(seed).choice(cells.ids, n, replace=False, p=p))
    windows = {name: sc.period(sc.cube(name).sel(experiment="historical").sel(cell=chosen), sc.BASELINE)
               .median(["model", "year"]) for name in ("sowing_doy", "days_to_anthesis", "days_to_maturity")}
    frame = pd.DataFrame({k: np.rint(v.values).astype(int) for k, v in windows.items()}, index=chosen)
    frame.index.name = "SimplaceID"
    return frame


def build(n_cells: int, workers: int) -> pd.DataFrame:
    sampled = sample_cells(n_cells)
    tasks = [(m, int(c), *map(int, row)) for m in sc.MODELS
             for c, row in zip(sampled.index, sampled.to_numpy())]
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(cell_features, tasks, chunksize=8))
    out = pd.concat(frames, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    sampled.to_parquet(CACHE.with_name("season_features_windows.parquet"))
    logger.info("%d rows -> %s", len(out), CACHE)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=64)
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build(a.cells, a.workers)
