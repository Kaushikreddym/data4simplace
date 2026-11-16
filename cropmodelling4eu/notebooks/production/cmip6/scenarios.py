"""Common cells, data cubes, regional means, trends and maps for the CMIP6 runs.

Every statistic is computed on the cells **all 20 runs** simulated. Each model's
bias-corrected forcing has its own frame, and a cell outside it carries ``-99.9``
weather and a null yield in every year, so per-run means would compare different
places. Means are weighted by PROBA-V cropland area, the exported ``total_area_km2``.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import stage_indices as si  # noqa: E402

from cropmodelling4eu.evaluation.plots import EUROPE_PROJECTION  # noqa: E402
from cropmodelling4eu.evaluation.style import DIVERGING_CMAP, diverging_norm  # noqa: E402

DATA = Path("/data01/FDS/muduchuru/Data/SIMPLACE")
RUNS = Path(os.environ.get("CMIP6_RUN_DIR", DATA / "cropmodelling4eu/runs/winter_wheat/calibrated/cmip6"))
MSWX_RUN = RUNS.parent / "mswx"
EXPORTS = DATA / "cropmodelling4eu_inputs/exports/cmip6"
BASE_EXPORT = DATA / "cropmodelling4eu_inputs/exports/mswx/EU_val_nfix"
CALIBRATION = DATA / "cropmodelling4eu/winter_wheat_2000_2024_euval/calibration"
OUT = Path(os.environ.get("CMIP6_ANALYSIS_DIR", RUNS / "analysis"))

MODELS: tuple[str, ...] = ("ACCESS-ESM1-5", "IPSL-CM6A-LR", "MIROC6", "MRI-ESM2-0")
HISTORICAL: tuple[str, ...] = ("historical", "hist-GHG", "hist-aer")
FUTURE: tuple[str, ...] = ("ssp126", "ssp370")
EXPERIMENTS: tuple[str, ...] = (*HISTORICAL, *FUTURE)
YEARS = np.arange(1851, 2101)

#: The last 30 years of ``historical``; every model has them.
BASELINE: tuple[int, int] = (1985, 2014)
#: The earliest 30 years every model has: ACCESS-ESM1-5 starts in 1951.
EARLY: tuple[int, int] = (1951, 1980)
NEAR: tuple[int, int] = (2021, 2050)
FAR: tuple[int, int] = (2071, 2100)

#: Shared by the models' forcing frames; established when the BC data were linked.
COMMON_BOX = {"lat": (35.15, 69.85), "lon": (-23.55, 43.55)}
MAP_EXTENT = (-11.0, 44.5, 34.5, 66.0)
REGION_NAMES = {"Europe": "Europe (all cells)", "EU23": "EU-23 (CyBench countries)", "NEU": "Northern Europe",
                "WCE": "Western & Central Europe", "EEU": "Eastern Europe", "MED": "Mediterranean"}
N_LON, MAX_LAT, MIN_LON, RES = 690, 72.0, -17.0, 0.1
#: Below this a cell's % change explodes on a near-zero denominator.
MIN_BASE_YIELD = 1.0


def table_dir(name: str) -> tuple[Path, Path]:
    """``(figures, tables)`` of one notebook, created on first use."""
    figures, tables = OUT / name / "figures", OUT / name / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return figures, tables


def run_file(model: str, experiment: str) -> Path:
    return RUNS / model / experiment / "torchcrop" / "torchcrop_europe.parquet"


@dataclass(slots=True)
class Cells:
    """The common cell set, its weights and its region membership."""

    frame: pd.DataFrame
    masks: xr.DataArray

    @property
    def ids(self) -> np.ndarray:
        return self.frame.index.to_numpy()

    @property
    def weight(self) -> np.ndarray:
        return self.frame["cropland_km2"].to_numpy(dtype=np.float32)

    @property
    def n(self) -> int:
        return len(self.frame)

    def region_weights(self, regions: tuple[str, ...] | None = None) -> xr.DataArray:
        return self.masks.sel(region=list(regions or self.masks.region.values)) * self.weight

    def grid(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lon_edges, lat_edges, field)`` of one per-cell vector."""
        index = self.ids - 1
        row, col = index // N_LON, index % N_LON
        r0, c0 = row.min(), col.min()
        field = np.full((row.max() - r0 + 1, col.max() - c0 + 1), np.nan, dtype=float)
        field[row - r0, col - c0] = values
        lon = MIN_LON + np.arange(c0, col.max() + 2) * RES
        lat = MAX_LAT - np.arange(r0, row.max() + 2) * RES
        return lon, lat, field


def _scan(model: str, experiment: str) -> tuple[set[int], set[int]]:
    run = pd.read_parquet(run_file(model, experiment), columns=["SimplaceID", "yield_t_ha"],
                          filters=[("year", "==", 2000 if experiment in HISTORICAL else 2050)])
    return set(run["SimplaceID"]), set(run.loc[run["yield_t_ha"].isna(), "SimplaceID"])


@lru_cache(maxsize=1)
def run_inventory() -> pd.DataFrame:
    """Per run: cells held, cells with no yield, seasons, the cell set and null set."""
    keys = [(m, e) for m in MODELS for e in EXPERIMENTS]
    with ThreadPoolExecutor(10) as pool:
        scans = list(pool.map(lambda k: _scan(*k), keys))
    return pd.DataFrame([{"model": m, "experiment": e, "cells": held, "null": null}
                         for (m, e), (held, null) in zip(keys, scans)])


@lru_cache(maxsize=1)
def load_cells() -> Cells:
    """Cells every run simulated, with cropland area, CyBench unit and AR6 region."""
    inventory = run_inventory()
    held = set.intersection(*inventory["cells"])
    if not all(c == held for c in inventory["cells"]):
        raise ValueError("runs hold different cell sets")
    ids = np.array(sorted(held - set.union(*inventory["null"])), dtype=np.int64)
    index = ids - 1
    frame = pd.DataFrame({"lat": MAX_LAT - RES / 2 - (index // N_LON) * RES,
                          "lon": MIN_LON + RES / 2 + (index % N_LON) * RES}, index=pd.Index(ids, name="SimplaceID"))
    frame["cropland_km2"] = si.cropland_area(BASE_EXPORT).reindex(ids).fillna(0.0).to_numpy()
    adm = pd.read_parquet(CALIBRATION / "cell_to_adm.parquet").set_index("SimplaceID")
    frame = frame.join(adm[["adm_id", "adm_country"]])
    regions = pd.read_parquet(CALIBRATION / "regions.parquet", columns=["SimplaceID", "region_id"]).drop_duplicates("SimplaceID")
    frame = frame.join(regions.set_index("SimplaceID")["region_id"])
    ar6 = regionmask.defined_regions.ar6.land
    codes = ar6.mask(xr.DataArray(frame["lon"].to_numpy(), dims="cell"), xr.DataArray(frame["lat"].to_numpy(), dims="cell"))
    abbrev = dict(zip(ar6.numbers, ar6.abbrevs))
    frame["ar6"] = pd.Series(codes.values, index=frame.index).map(abbrev)
    masks = {"Europe": np.ones(len(frame), bool), "EU23": frame["adm_country"].notna().to_numpy(),
             **{r: (frame["ar6"] == r).to_numpy() for r in ("NEU", "WCE", "EEU", "MED")}}
    return Cells(frame, xr.DataArray(np.stack(list(masks.values())).astype(np.float32), dims=("region", "cell"),
                                     coords={"region": list(masks)}))


@lru_cache(maxsize=16)
def cube(variable: str = "yield_t_ha") -> xr.DataArray:
    """One variable on ``(model, experiment, year, cell)``; NaN where a run has no season."""
    cells = load_cells()
    ids, first = cells.ids, YEARS[0]
    data = np.full((len(MODELS), len(EXPERIMENTS), len(YEARS), cells.n), np.nan, dtype=np.float32)

    def fill(key: tuple[int, int]) -> None:
        m, e = key
        run = pd.read_parquet(run_file(MODELS[m], EXPERIMENTS[e]), columns=["SimplaceID", "year", variable])
        position = np.searchsorted(ids, run["SimplaceID"].to_numpy())
        keep = (position < len(ids)) & (ids[np.minimum(position, len(ids) - 1)] == run["SimplaceID"].to_numpy())
        data[m, e, run["year"].to_numpy()[keep] - first, position[keep]] = run[variable].to_numpy(np.float32)[keep]

    with ThreadPoolExecutor(10) as pool:
        list(pool.map(fill, [(m, e) for m in range(len(MODELS)) for e in range(len(EXPERIMENTS))]))
    return xr.DataArray(data, dims=("model", "experiment", "year", "cell"), name=variable,
                        coords={"model": list(MODELS), "experiment": list(EXPERIMENTS), "year": YEARS, "cell": ids})


@lru_cache(maxsize=4)
def mswx_cube(variable: str = "yield_t_ha") -> xr.DataArray:
    """The MSWX torchcrop run on ``(year, cell)``, restricted to the common cells."""
    run = pd.read_parquet(MSWX_RUN / "torchcrop" / "torchcrop_europe.parquet", columns=["SimplaceID", "year", variable])
    wide = run.pivot(index="year", columns="SimplaceID", values=variable).reindex(columns=load_cells().ids)
    return xr.DataArray(wide.to_numpy(np.float32), dims=("year", "cell"),
                        coords={"year": wide.index.to_numpy(), "cell": wide.columns.to_numpy()}, name=variable)


def co2_series(model: str = "MIROC6") -> pd.DataFrame:
    """Annual CO₂ (ppm) per experiment; the default model holds the full 1851–2100 span."""
    series = {e: pd.read_csv(EXPORTS / model / e / "site" / "co2.csv", comment="#").set_index("year")["co2_ppm"]
              for e in EXPERIMENTS}
    return pd.DataFrame(series).reindex(YEARS)


def period(data: xr.DataArray, span: tuple[int, int]) -> xr.DataArray:
    return data.sel(year=slice(*span))


def clim(data: xr.DataArray, span: tuple[int, int]) -> xr.DataArray:
    """Per-cell mean over ``span``, without the scalar coordinate of a selected experiment."""
    mean = period(data, span).mean("year")
    return mean.drop_vars("experiment") if "experiment" not in mean.dims and "experiment" in mean.coords else mean


def regional_mean(data: xr.DataArray, cells: Cells | None = None,
                  regions: tuple[str, ...] | None = None) -> xr.DataArray:
    """Cropland-weighted mean over ``cell``, NaN-aware, adding a ``region`` dimension."""
    cells = cells or load_cells()
    weights = cells.region_weights(regions)
    valid = data.notnull().astype(np.float32)
    num = xr.dot(data.fillna(0.0), weights, dim="cell")
    den = xr.dot(valid, weights, dim="cell")
    return (num / den.where(den > 0)).rename(data.name)


def relative_change(data: xr.DataArray, experiment: str, span: tuple[int, int],
                    reference: str = "historical", base: tuple[int, int] = BASELINE) -> xr.DataArray:
    """Per-model, per-cell ``100 · (mean / reference mean − 1)``; blank below :data:`MIN_BASE_YIELD`.

    Never average this over a region: low-yield cells dominate a mean of ratios. Use
    :func:`regional_relative_change`, the ratio of the regional means.
    """
    ref = clim(data.sel(experiment=reference), base)
    return 100.0 * (clim(data.sel(experiment=experiment), span) / ref.where(ref >= MIN_BASE_YIELD) - 1.0)


def regional_relative_change(data: xr.DataArray, experiment: str, span: tuple[int, int], reference: str = "historical",
                             base: tuple[int, int] = BASELINE, aggregate=None) -> xr.DataArray:
    """% change of the area mean, per model; ``aggregate`` defaults to :func:`regional_mean`."""
    aggregate = aggregate or regional_mean
    ref = aggregate(clim(data.sel(experiment=reference), base))
    return 100.0 * (aggregate(clim(data.sel(experiment=experiment), span)) / ref - 1.0)


def absolute_change(data: xr.DataArray, experiment: str, span: tuple[int, int],
                    reference: str = "historical", base: tuple[int, int] = BASELINE) -> xr.DataArray:
    return clim(data.sel(experiment=experiment), span) - clim(data.sel(experiment=reference), base)


def agreement(delta: xr.DataArray, need: int = 3) -> xr.DataArray:
    """True where at least ``need`` models share the sign of the multi-model mean."""
    same = (np.sign(delta) == np.sign(delta.mean("model"))).sum("model")
    return same >= need


def sen_mk(values: np.ndarray, years: np.ndarray, batch: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Theil–Sen slope (per decade) and Mann–Kendall p per column of ``(time, cell)``.

    The variance omits the tie correction: ties only occur among zero-yield seasons,
    which are rare, so a tied cell's test is slightly conservative.
    """
    n = len(years)
    i, j = np.triu_indices(n, 1)
    dt = (years[j] - years[i]).astype(np.float32)[:, None]
    variance = n * (n - 1) * (2 * n + 5) / 18.0

    def fit(start: int) -> tuple[np.ndarray, np.ndarray]:
        diff = values[j, start:start + batch] - values[i, start:start + batch]
        s = np.sign(diff).sum(0)
        z = (s - np.sign(s)) / np.sqrt(variance)
        return 10.0 * np.median(diff / dt, axis=0), 2.0 * norm.sf(np.abs(z))

    with ThreadPoolExecutor(32) as pool:
        parts = list(pool.map(fit, range(0, values.shape[1], batch)))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def fdr(p: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Benjamini–Hochberg mask at ``alpha``; Wilks (2016) advises 0.10 for correlated fields."""
    order = np.argsort(p)
    passed = p[order] <= alpha * np.arange(1, len(p) + 1) / len(p)
    mask = np.zeros(len(p), bool)
    if passed.any():
        mask[order[: np.flatnonzero(passed).max() + 1]] = True
    return mask


def map_axes(rows: int, cols: int, width: float = 3.5, height: float = 3.5):
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows + 0.5), squeeze=False,
                             subplot_kw={"projection": EUROPE_PROJECTION}, layout="constrained")
    return fig, axes


def draw_map(ax: plt.Axes, values: np.ndarray, cells: Cells | None = None, *, cmap=None, norm_=None,
             vmin: float | None = None, vmax: float | None = None, title: str = "",
             hatch: np.ndarray | None = None):
    """One per-cell field; ``hatch`` marks cells where the signal is not robust."""
    cells = cells or load_cells()
    lon, lat, field = cells.grid(np.asarray(values, dtype=float))
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2f1ec", zorder=0)
    mesh = ax.pcolormesh(lon, lat, field, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm_,
                         vmin=None if norm_ else vmin, vmax=None if norm_ else vmax, rasterized=True, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, edgecolor="#555555", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#777777", zorder=3)
    if hatch is not None:
        marks = np.asarray(hatch, bool) & (cells.ids % 7 == 0) & (((cells.ids - 1) // N_LON) % 5 == 0)
        ax.scatter(cells.frame["lon"].to_numpy()[marks], cells.frame["lat"].to_numpy()[marks], s=0.5, c="k", linewidths=0,
                   alpha=0.6, transform=ccrs.PlateCarree(), zorder=2)
    if title:
        ax.set_title(title, fontsize=9)
    return mesh


def diverging(values: xr.DataArray | np.ndarray, q: float = 0.98) -> dict:
    """Keyword arguments of :func:`draw_map` for a symmetric anomaly scale."""
    finite = np.asarray(values, dtype=float)
    limit = float(np.nanquantile(np.abs(finite), q)) or 1.0
    return {"cmap": DIVERGING_CMAP, "norm_": diverging_norm(finite, vmax=limit)}


def regional_table(fields: dict[str, xr.DataArray], decimals: int = 1) -> pd.DataFrame:
    """Multi-model mean with the model range; a field with a ``cell`` dimension is averaged first."""
    rows = {}
    for name, field in fields.items():
        reg = regional_mean(field) if "cell" in field.dims else field
        mean, low, high = reg.mean("model"), reg.min("model"), reg.max("model")
        rows[name] = {REGION_NAMES[r]: f"{float(mean.sel(region=r)):+.{decimals}f} "
                                       f"[{float(low.sel(region=r)):+.{decimals}f}, {float(high.sel(region=r)):+.{decimals}f}]"
                      for r in reg.region.values}
    return pd.DataFrame(rows).T


def country_series(data: xr.DataArray, min_km2: float = 2000.0) -> xr.DataArray:
    """Cropland-weighted mean per CyBench country holding at least ``min_km2`` of cropland."""
    cells = load_cells()
    area = cells.frame.groupby("adm_country")["cropland_km2"].sum()
    codes = [c for c in area.index if area[c] >= min_km2]
    weights = xr.DataArray(np.stack([cells.frame["adm_country"].eq(c).fillna(False).to_numpy(np.float32) * cells.weight for c in codes]),
                           dims=("country", "cell"), coords={"country": codes})
    num = xr.dot(data.fillna(0.0), weights, dim="cell")
    den = xr.dot(data.notnull().astype(np.float32), weights, dim="cell")
    return num / den.where(den > 0)
