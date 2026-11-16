"""Training targets on the cache's time axis, and the PEP725 hold-out.

Two sources, one frame (``source``, ``SimplaceID``, ``year``, ``split``,
``sow_idx``, ``em_idx``), all dates as cache indices:

* **CLMS** ``CPMCE`` on the 10 km cells, minus the PEP725-measured 6.0 d. It
  carries the *level* of emergence; its year-to-year signal is near zero
  (anomaly r 0.11 against PEP725). Split by the calibration's own
  ``phenology_split.parquet`` through each cell's CyBench unit, so a unit the
  calibration held out is held out here too.
* **PEP725** station-seasons: sowing (BBCH 0) and emergence (BBCH 10). Split
  by **spatial blocks of stations**, drawn once and written to disk, because
  the same stations also fit ``tsumem`` and later score the result — any
  scoring must be on blocks neither step saw.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .cache import doy_to_index
from .config import SowingConfig

logger = logging.getLogger(__name__)

__all__ = ["clms_targets", "load_targets", "pep725_seasons", "pep725_split"]

SPLITS = ("train", "val", "test")


def _grid(config: SowingConfig):
    from data4simplace.grid import TargetGrid

    from cropmodelling4eu.config import load_config

    g = load_config(config.run_config).grid
    return TargetGrid(resolution_deg=g.resolution_deg, min_lon=g.min_lon, max_lon=g.max_lon,
                      min_lat=g.min_lat, max_lat=g.max_lat)


def pep725_seasons(config: SowingConfig) -> pd.DataFrame:
    """PEP725 winter-wheat station-seasons on the export grid, harvest-year aligned."""
    from cropmodelling4eu.calibration.observations import load_pep725

    return load_pep725(config.pep725_root, _grid(config), years=config.years, include_heading=False)


def pep725_split(config: SowingConfig, seasons: pd.DataFrame | None = None,
                 overwrite: bool = False) -> pd.DataFrame:
    """Station -> block -> split, drawn once and reused.

    Blocks of ``pep_block_deg`` degrees are permuted with a fixed seed and
    cut at the configured shares by *station count*, so the shares hold even
    though Germany holds most stations.
    """
    path = config.pep_split_file
    if path.is_file() and not overwrite:
        return pd.read_parquet(path)
    seasons = pep725_seasons(config) if seasons is None else seasons
    stations = seasons[["s_id", "lon", "lat"]].drop_duplicates("s_id").copy()
    stations["block"] = (np.floor(stations["lat"] / config.pep_block_deg).astype(int).astype(str) + "_"
                         + np.floor(stations["lon"] / config.pep_block_deg).astype(int).astype(str))
    sizes = stations.groupby("block").size()
    order = np.random.default_rng(config.train.seed).permutation(sizes.index.to_numpy())
    share = sizes.reindex(order).cumsum() / sizes.sum()
    cuts = np.cumsum(config.pep_split)
    block_split = pd.Series(np.array(SPLITS)[np.searchsorted(cuts, share.to_numpy() - 1e-12)],
                            index=order)
    stations["split"] = stations["block"].map(block_split)
    path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(path, index=False)
    logger.info("PEP725 split: %s stations in %d blocks -> %s",
                stations["split"].value_counts().to_dict(), len(sizes), path)
    return stations


def clms_targets(config: SowingConfig) -> pd.DataFrame:
    """CLMS emergence per cell-season, corrected, with the calibration's split."""
    from cropmodelling4eu.evaluation.clms import load_phenology

    clms = load_phenology(zone="grid", years=config.clms_years)
    clms = clms.dropna(subset=["obs_emergence_doy"])
    cell_to_adm = pd.read_parquet(config.calibration_dir / "cell_to_adm.parquet")
    split = (pd.read_parquet(config.calibration_dir / "phenology_split.parquet")
             .assign(unit=lambda f: f["unit"].astype(str)).set_index(["unit", "year"])["split"])
    adm = clms["SimplaceID"].map(cell_to_adm.set_index("SimplaceID")["adm_id"]).astype("string")
    keys = pd.MultiIndex.from_arrays([adm.fillna("").astype(str), clms["year"]])
    # A cell outside every calibration unit-year is training data: no
    # calibration score is ever computed on it.
    labels = pd.Series(split.reindex(keys).to_numpy(), index=clms.index).fillna("train")
    return pd.DataFrame({
        "source": "clms",
        "SimplaceID": clms["SimplaceID"].astype(np.int64),
        "year": clms["year"].astype(int),
        "split": labels.replace({"unused": "train"}),
        "sow_idx": np.nan,
        "em_idx": doy_to_index(clms["obs_emergence_doy"].to_numpy(), clms["year"].to_numpy(),
                               config.cache_start_doy) - config.clms_emergence_offset_days,
    })


def load_targets(config: SowingConfig) -> pd.DataFrame:
    """Both sources in one frame, PEP725 one row per station-season."""
    seasons = pep725_seasons(config)
    stations = pep725_split(config, seasons)
    pep = seasons.merge(stations[["s_id", "split"]], on="s_id", how="inner")
    to_idx = lambda col: np.where(pep[col].notna(),
                                  doy_to_index(pep[col].fillna(1).to_numpy(), pep["harvest_year"].to_numpy(),
                                               config.cache_start_doy), np.nan)
    pep = pd.DataFrame({
        "source": "pep725", "SimplaceID": pep["SimplaceID"].astype(np.int64),
        "year": pep["harvest_year"].astype(int), "split": pep["split"],
        "sow_idx": to_idx("sowing_doy"), "em_idx": to_idx("emergence_doy"),
        "s_id": pep["s_id"],
    }).dropna(subset=["sow_idx", "em_idx"], how="all")
    frame = pd.concat([clms_targets(config), pep], ignore_index=True)
    logger.info("targets: %s", frame.groupby(["source", "split"]).size().to_dict())
    return frame
