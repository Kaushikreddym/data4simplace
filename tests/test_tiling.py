"""Tests for tiled execution: geometry, global identity, and output parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.tiling import (
    TileWindow,
    _global_cell_table,
    _tile_config,
    iter_windows,
    run_tiled,
)


def _cfg(**grid):
    base = {
        "flags": {},
        "grid": {"resolution_deg": 0.1, "min_lon": 0.0, "max_lon": 1.0,
                 "min_lat": 0.0, "max_lat": 1.0, **grid},
        "time": {"start": "1979-01-01", "end": "1979-01-03"},
        "paths": {"mswx_root": "/tmp"},
    }
    return PipelineConfig.model_validate(base)


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #
def test_iter_windows_tiles_cover_grid_without_overlap():
    wins = list(iter_windows(10, 10, 5))
    assert len(wins) == 4
    covered = np.zeros((10, 10), dtype=int)
    for w in wins:
        covered[w.r0:w.r1, w.c0:w.c1] += 1
    assert (covered == 1).all()  # exact partition


def test_iter_windows_handles_ragged_last_tile():
    wins = list(iter_windows(7, 7, 5))
    assert len(wins) == 4
    assert wins[-1] == TileWindow(5, 7, 5, 7)  # 2x2 remainder


def test_tile_config_is_exact_slice_of_global_grid():
    cfg = _cfg()  # 10x10
    grid = TargetGrid.from_config(cfg.grid)
    w = TileWindow(0, 5, 5, 10)  # top-right block
    tgrid = TargetGrid.from_config(_tile_config(cfg, grid, w).grid)
    assert tgrid.shape == (5, 5)
    np.testing.assert_allclose(tgrid.lon_centers, grid.lon_centers[5:10])
    np.testing.assert_allclose(tgrid.lat_centers, grid.lat_centers[0:5])


def test_global_cell_table_identity():
    cfg = _cfg()
    grid = TargetGrid.from_config(cfg.grid)
    w = TileWindow(5, 10, 5, 10)
    tgrid = TargetGrid.from_config(_tile_config(cfg, grid, w).grid)
    ct = _global_cell_table(tgrid, w, n_lon=grid.shape[1])
    # A local cell (row=2, col=3) maps to global (7, 8) with row-major id.
    r = ct[(ct["row"] == 2) & (ct["col"] == 3)].iloc[0]
    assert (r["grow"], r["gcol"]) == (7, 8)
    assert r["SimplaceID"] == 7 * 10 + 8 + 1
    # Local indices stay in range for tile-array indexing.
    assert ct["row"].max() < tgrid.shape[0] and ct["col"].max() < tgrid.shape[1]


# --------------------------------------------------------------------------- #
# Weather exporter honours global identity
# --------------------------------------------------------------------------- #
def test_weather_export_uses_global_identity(tmp_path):
    from data4simplace.exporters import WeatherExporter

    cfg = _cfg()
    climate = xr.Dataset(
        {"tas": (("time", "lat", "lon"), np.full((2, 1, 2), 5.0)),
         "pr": (("time", "lat", "lon"), np.full((2, 1, 2), 1.0))},
        coords={"time": pd.to_datetime(["1979-01-01", "1979-01-02"]),
                "lat": [0.55], "lon": [0.55, 0.65]},
    )
    ct = pd.DataFrame({
        "SimplaceID": [1, 2], "row": [0, 0], "col": [0, 1],
        "grow": [5, 5], "gcol": [5, 6], "lat": [0.55, 0.55], "lon": [0.55, 0.65],
    })
    WeatherExporter(cfg, None).export(climate, ct, tmp_path)
    names = {p.name for p in (tmp_path / "weather").glob("*.csv.gz")}
    assert names == {"daily_mean_RES1_C5R5.csv.gz", "daily_mean_RES1_C6R5.csv.gz"}
    df = pd.read_csv(tmp_path / "weather" / "daily_mean_RES1_C5R5.csv.gz", sep="\t")
    assert df["Gridcell"].iloc[0] == "C_5:R_5"


# --------------------------------------------------------------------------- #
# Integration: tiled weather output must equal a single-run over the same grid
# --------------------------------------------------------------------------- #
MSWX_ROOT = Path("/data01/FDS/muduchuru/Atmos/MSWX")


@pytest.mark.skipif(not (MSWX_ROOT / "TAS").is_dir(), reason="MSWX data not available")
def test_tiled_weather_matches_single_run(tmp_path):
    grid = {"resolution_deg": 0.1, "min_lon": 12.0, "max_lon": 12.4,
            "min_lat": 52.4, "max_lat": 52.6}
    common = {
        "flags": {"run_climate_processing": True, "export_simplace_weather": True},
        "grid": grid,
        "time": {"start": "1979-01-01", "end": "1979-01-03"},
        "paths": {"mswx_root": str(MSWX_ROOT), "output_dir": ""},
        "climate": {"variables": {"TAS": "tas", "PR": "pr"}},
    }
    from data4simplace.pipeline import Pipeline

    single_dir = tmp_path / "single"
    tiled_dir = tmp_path / "tiled"
    Pipeline(PipelineConfig.model_validate(
        {**common, "paths": {**common["paths"], "output_dir": str(single_dir)}})).run()
    run_tiled(PipelineConfig.model_validate(
        {**common, "paths": {**common["paths"], "output_dir": str(tiled_dir)}}),
        tile_deg=0.2)

    def load(d):
        return {p.name: pd.read_csv(p, sep="\t") for p in (d / "weather").glob("*.csv.gz")}

    s, t = load(single_dir), load(tiled_dir)
    assert set(s) == set(t) and len(s) > 1  # same files, actually tiled (>1 tile)
    for name in s:
        pd.testing.assert_frame_equal(
            s[name].sort_values("Date").reset_index(drop=True),
            t[name].sort_values("Date").reset_index(drop=True),
        )
