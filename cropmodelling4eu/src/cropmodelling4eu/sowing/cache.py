"""The season cache: every (cell, harvest year) the pipeline trains on or predicts.

Rows are cell-major — row ``i * n_years + j`` is cell ``i``, harvest year
``years[j]`` — so a worker owning a run of cells owns a contiguous run of rows
and writes it straight into the shared memmaps. Daily blocks are float16:
weather to two decimals is all a sowing decision needs and it halves the
continental cache (~15 GB for 68 685 cells x 35 seasons).

Statics (location, SAGE window, topsoil) are one row per cell in
``cells.parquet``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig, load_config
from cropmodelling4eu.export.cells import id_to_lonlat, weather_ids, weather_path
from cropmodelling4eu.export.site import read_site
from cropmodelling4eu.export.soil import read_soil
from cropmodelling4eu.export.weather import _read_file, to_channels

from .config import SowingConfig
from .features import LONG_FEATURES, MEDIUM_CHANNELS, SHORT_CHANNELS, cell_features

logger = logging.getLogger(__name__)

__all__ = ["STATIC_FEATURES", "SeasonCache", "build_cache", "doy_to_index", "index_to_doy"]

STATIC_FEATURES: tuple[str, ...] = (
    "lat", "lon", "altitude_m", "sage_start", "sage_end", "sage_sow",
    "fc", "wp", "clay", "sand",
)
_FILES = {"short": "short.f16", "medium": "medium.f16", "long": "long.f32", "valid": "valid.u1"}


def doy_to_index(doy: np.ndarray, harvest_year: np.ndarray, start_doy: int) -> np.ndarray:
    """Cache index of a day-of-year belonging to the season harvested in ``harvest_year``.

    A DOY on or after 180 is in the autumn of Y-1, earlier ones in Y — the
    same cut PEP725 and the calibration use.
    """
    doy = np.asarray(doy, float)
    year = np.asarray(harvest_year) - (doy >= 180).astype(int)
    start = pd.to_datetime((np.asarray(harvest_year) - 1).astype(str)) + pd.to_timedelta(start_doy - 1, "D")
    day = pd.to_datetime(year.astype(str)) + pd.to_timedelta(np.floor(doy) - 1, "D")
    return (np.asarray((day - start).days, float) + (doy - np.floor(doy)))


def index_to_doy(index: np.ndarray, harvest_year: np.ndarray, start_doy: int) -> np.ndarray:
    """Day-of-year of an integer cache index (inverse of :func:`doy_to_index`)."""
    start = pd.to_datetime((np.asarray(harvest_year) - 1).astype(str)) + pd.to_timedelta(start_doy - 1, "D")
    return np.asarray((start + pd.to_timedelta(np.asarray(index, int), "D")).dayofyear)


@dataclass(slots=True)
class SeasonCache:
    """Read access to a built cache."""

    root: Path
    cells: pd.DataFrame
    years: np.ndarray
    short: np.memmap
    medium: np.memmap
    long: np.memmap
    valid: np.ndarray
    meta: dict

    @classmethod
    def open(cls, root: Path) -> "SeasonCache":
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        n, t = meta["rows"], meta["cache_days"]
        mm = lambda name, dtype, shape: np.memmap(root / _FILES[name], dtype=dtype, mode="r", shape=shape)
        return cls(
            root,
            pd.read_parquet(root / "cells.parquet"),
            np.asarray(meta["years"]),
            mm("short", np.float16, (n, t, len(SHORT_CHANNELS))),
            mm("medium", np.float16, (n, t, len(MEDIUM_CHANNELS))),
            mm("long", np.float32, (n, len(LONG_FEATURES))),
            np.asarray(mm("valid", np.uint8, (n,))).astype(bool),
            meta,
        )

    def rows(self, ids: np.ndarray, years: np.ndarray) -> np.ndarray:
        """Row of each (cell, year); -1 where either is not cached."""
        cell = pd.Index(self.cells["SimplaceID"]).get_indexer(np.asarray(ids, np.int64))
        year = pd.Index(self.years).get_indexer(np.asarray(years))
        ok = (cell >= 0) & (year >= 0)
        return np.where(ok, cell * len(self.years) + year, -1)

    def row_keys(self) -> pd.DataFrame:
        """``SimplaceID``, ``year``, ``row`` for every row."""
        ids = np.repeat(self.cells["SimplaceID"].to_numpy(), len(self.years))
        return pd.DataFrame({"SimplaceID": ids, "year": np.tile(self.years, len(self.cells)),
                             "row": np.arange(ids.size)})


def _statics(config: SowingConfig, ids: np.ndarray, grid: GridConfig) -> pd.DataFrame:
    """One row per cell: location, SAGE window as cache indices, topsoil water and texture."""
    lon, lat = id_to_lonlat(ids, grid)
    site = read_site(config.export_dir).frame.reindex(ids)
    to_index = lambda doy: np.asarray(doy, float) - (config.cache_start_doy - 1) - 1
    soil = read_soil(config.export_dir).select(ids)
    top = lambda stem: (soil.depth_mean(stem, 0.0, config.topsoil_m) if stem in soil
                        else np.full(ids.size, np.nan))
    frame = pd.DataFrame({
        "SimplaceID": ids, "lat": lat, "lon": lon,
        "altitude_m": site.get("altitude_m", pd.Series(np.nan, index=ids)).to_numpy(float),
        # SAGE windows open in Y-1 for every European cell; an end before the
        # start wraps past New Year.
        "sage_start": to_index(site["sowing_start_doy"]),
        "sage_end": to_index(np.where(site["sowing_end_doy"] < site["sowing_start_doy"],
                                      site["sowing_end_doy"] + 365, site["sowing_end_doy"])),
        "sage_sow": to_index(site["sowing_doy"]),
        "fc": top("soilwater_fc"), "wp": top("soilwater_wp"),
        "clay": top("clay"), "sand": top("sand"),
    })
    missing = [c for c in STATIC_FEATURES if frame[c].isna().all()]
    if missing:
        logger.warning("static feature(s) absent from the export, left NaN: %s", missing)
    return frame


def _work(args: tuple) -> int:
    root, first_row, ids, capacity, years, export_dir, grid, settings = args
    meta = json.loads((Path(root) / "meta.json").read_text())
    n, t = meta["rows"], meta["cache_days"]
    open_ = lambda name, dtype, shape: np.memmap(Path(root) / _FILES[name], dtype=dtype, mode="r+", shape=shape)
    short = open_("short", np.float16, (n, t, len(SHORT_CHANNELS)))
    medium = open_("medium", np.float16, (n, t, len(MEDIUM_CHANNELS)))
    long = open_("long", np.float32, (n, len(LONG_FEATURES)))
    valid = open_("valid", np.uint8, (n,))
    done = 0
    for k, (sid, cap) in enumerate(zip(ids, capacity)):
        raw = _read_file(weather_path(export_dir, int(sid), grid))
        raw = raw.reindex(pd.date_range(raw.index[0], raw.index[-1], freq="D"))
        channels = to_channels(raw, bool(raw["Windspeed"].notna().any()))
        feats = cell_features(raw.index, channels, years, capacity_mm=float(cap), **settings)
        rows = slice(first_row + k * len(years), first_row + (k + 1) * len(years))
        short[rows], medium[rows], long[rows] = feats.short, feats.medium, feats.long
        valid[rows] = feats.valid
        done += int(feats.valid.sum())
    for array in (short, medium, long, valid):
        array.flush()
    return done


def build_cache(config: SowingConfig, ids: np.ndarray | None = None, chunk: int = 200) -> SeasonCache:
    """Extract every (cell, season) once, in ``config.workers`` processes.

    Args:
        config: The run.
        ids: Cells to cache; default every cell with a weather file *and* a
            site row.
        chunk: Cells per task.
    """
    grid = load_config(config.run_config).grid
    available = weather_ids(config.export_dir, grid)
    site_ids = read_site(config.export_dir).frame.index.to_numpy(np.int64)
    ids = np.intersect1d(available if ids is None else np.asarray(ids, np.int64), site_ids)
    years = list(range(config.years[0], config.years[1] + 1))
    root = config.cache_dir
    root.mkdir(parents=True, exist_ok=True)

    cells = _statics(config, ids, grid)
    capacity = np.clip((cells["fc"] - cells["wp"]).fillna(0.12).to_numpy(), 0.02, None) * config.topsoil_m * 1000.0
    n, t = len(ids) * len(years), config.cache_days
    meta = {"rows": n, "cache_days": t, "years": years, "cache_start_doy": config.cache_start_doy,
            "short_channels": SHORT_CHANNELS, "medium_channels": MEDIUM_CHANNELS,
            "long_features": LONG_FEATURES, "static_features": STATIC_FEATURES,
            "normals_years": config.normals_years, "normals_min_years": config.normals_min_years,
            "export_dir": str(config.export_dir), "complete": False}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    cells.to_parquet(root / "cells.parquet", index=False)
    for name, dtype, shape in (("short", np.float16, (n, t, len(SHORT_CHANNELS))),
                               ("medium", np.float16, (n, t, len(MEDIUM_CHANNELS))),
                               ("long", np.float32, (n, len(LONG_FEATURES))),
                               ("valid", np.uint8, (n,))):
        np.memmap(root / _FILES[name], dtype=dtype, mode="w+", shape=shape).flush()

    settings = {"cache_start_doy": config.cache_start_doy, "cache_days": t,
                "normals_years": config.normals_years, "normals_min_years": config.normals_min_years}
    tasks = [
        (str(root), start * len(years), ids[start:start + chunk], capacity[start:start + chunk],
         years, config.export_dir, grid, settings)
        for start in range(0, len(ids), chunk)
    ]
    logger.info("caching %d cells x %d seasons in %d tasks on %d workers",
                len(ids), len(years), len(tasks), config.workers)
    with ProcessPoolExecutor(config.workers) as pool:
        valid = sum(pool.map(_work, tasks))
    meta["complete"] = True
    meta["valid_rows"] = int(valid)
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("cache complete: %d of %d rows valid -> %s", valid, n, root)
    return SeasonCache.open(root)
