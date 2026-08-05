"""Tests for the MSWX loader: window resolution, gaps and corrupt files.

The corrupt-file cases are regressions: the MSWX mirror holds zero-byte days
(``TAS/1988171.nc``, ``HURS/1999269.nc`` and four more), and an earlier loader
let one of them abort a whole 46-year tile with an opaque
"did not find a match in any of xarray's currently installed IO backends".
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.climate.mswx_handler import MSWXHandler, _read_window
from data4simplace.config import PipelineConfig


LAT = np.arange(72.0, 33.9, -0.1)   # descending, as MSWX ships it
LON = np.arange(-17.0, 52.0, 0.1)


def _frame(path, value: float) -> None:
    """Write a one-day MSWX-shaped file."""
    ds = xr.Dataset(
        {"air_temperature": (("time", "lat", "lon"),
                             np.full((1, LAT.size, LON.size), value, "float32"))},
        coords={"time": [0], "lat": LAT, "lon": LON},
    )
    ds.to_netcdf(path, engine="h5netcdf")


@pytest.fixture()
def mswx_root(tmp_path):
    """A three-day TAS folder whose middle day is zero-byte."""
    root = tmp_path / "MSWX"
    var = root / "TAS"
    var.mkdir(parents=True)
    _frame(var / "1979001.nc", 1.0)
    (var / "1979002.nc").touch()          # the zero-byte day
    _frame(var / "1979003.nc", 3.0)
    return root


def _config(root, **over) -> PipelineConfig:
    base = {
        "flags": {"run_climate_processing": True},
        "grid": {"resolution_deg": 0.1, "min_lon": 8.0, "max_lon": 13.0,
                 "min_lat": 52.0, "max_lat": 57.0, "crs": "EPSG:4326"},
        "time": {"start": "1979-01-01", "end": "1979-01-03"},
        "paths": {"mswx_root": str(root), "output_dir": "./out"},
        "climate": {"variables": {"TAS": "tas"}, "read_workers": 1},
    }
    base.update(over)
    return PipelineConfig.model_validate(base)


def test_zero_byte_day_is_skipped_not_fatal(mswx_root):
    """A zero-byte file becomes a gap; the other days still load."""
    ds = MSWXHandler(_config(mswx_root)).load()
    assert "tas" in ds
    # Two readable days out of the three-day window.
    assert ds.sizes["time"] == 2


def test_all_days_unreadable_raises(tmp_path):
    """If nothing is readable the loader must fail loudly, not return junk."""
    var = tmp_path / "MSWX" / "TAS"
    var.mkdir(parents=True)
    for stem in ("1979001", "1979002", "1979003"):
        (var / f"{stem}.nc").touch()
    with pytest.raises(RuntimeError):
        MSWXHandler(_config(tmp_path / "MSWX")).load()


def test_read_window_returns_none_on_corrupt(tmp_path):
    """The worker swallows a bad file rather than killing the pool."""
    bad = tmp_path / "bad.nc"
    bad.touch()
    assert _read_window((str(bad), 0, 5, 0, 5)) is None


def test_window_clips_to_tile_bbox(mswx_root):
    """Only the tile's cells (plus a one-cell halo) are read."""
    h = MSWXHandler(_config(mswx_root))
    lat_vals, lon_vals = h._resolve_window([mswx_root / "TAS" / "1979001.nc"])
    # 5 degrees at 0.1 deg = 50 cells, plus one halo cell on each side.
    assert 50 <= lat_vals.size <= 53
    assert 50 <= lon_vals.size <= 53
    assert lat_vals.max() <= 57.0 + 0.11
    assert lat_vals.min() >= 52.0 - 0.11


def test_descending_latitude_preserved(mswx_root):
    """MSWX ships lat descending; the loader must not silently reorder it."""
    h = MSWXHandler(_config(mswx_root))
    lat_vals, _ = h._resolve_window([mswx_root / "TAS" / "1979001.nc"])
    assert lat_vals[0] > lat_vals[-1]


def test_worker_count_respects_config(mswx_root):
    """An explicit read_workers wins over the SLURM allocation."""
    cfg = _config(mswx_root)
    cfg = cfg.model_copy(update={"climate": cfg.climate.model_copy(
        update={"read_workers": 3})})
    assert MSWXHandler(cfg)._workers == 3


def test_worker_count_is_capped(mswx_root, monkeypatch):
    """A large allocation does not spawn an unbounded pool."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "80")
    cfg = _config(mswx_root)
    cfg = cfg.model_copy(update={"climate": cfg.climate.model_copy(
        update={"read_workers": None})})
    assert MSWXHandler(cfg)._workers == 16
