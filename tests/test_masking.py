"""Tests for the agricultural mask, including CORINE reclassification."""

from __future__ import annotations

import io

import numpy as np
import xarray as xr
from PIL import Image

from data4simplace.config import PipelineConfig
from data4simplace.spatial.corine import CLC_COLORS, CorineLandCover
from data4simplace.spatial.masking import CroplandMask


def _config(tmp_path, **mask) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "flags": {"apply_agricultural_mask": True},
            "grid": {
                "resolution_deg": 0.1,
                "min_lon": 13.0,
                "max_lon": 13.2,
                "min_lat": 52.3,
                "max_lat": 52.5,
            },
            "time": {"start": "1979-01-01", "end": "1979-01-01"},
            "paths": {"mswx_root": str(tmp_path), "output_dir": str(tmp_path / "out")},
            "mask": {"cache_dir": str(tmp_path / "cache"), **mask},
        }
    )


def _clc_png(width: int, height: int, left_code: int) -> bytes:
    """A PNG whose left half is one CLC class colour and right half urban."""
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:, : width // 2] = CLC_COLORS[left_code]
    arr[:, width // 2 :] = (230, 0, 77)  # 111 continuous urban fabric (non-agri)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_corine_reclassifies_agricultural_colours(tmp_path):
    cfg = _config(tmp_path, source="corine")
    corine = CorineLandCover(cfg)
    png = _clc_png(20, 10, left_code=211)  # arable on the left
    field = corine._agricultural_array(png, 20, 10)

    assert field.dims == ("lat", "lon")
    vals = field.values
    assert np.all(vals[:, :10] == 1.0)  # arable half -> agricultural
    assert np.all(vals[:, 10:] == 0.0)  # urban half -> not agricultural
    # Latitude axis runs north (top) -> south.
    assert field["lat"].values[0] > field["lat"].values[-1]


def test_corine_ignores_non_agricultural_class(tmp_path):
    cfg = _config(tmp_path, source="corine")
    corine = CorineLandCover(cfg)
    # 311 broad-leaved forest is not in the agricultural set.
    png = _clc_png(20, 10, left_code=231)  # pastures (agri) on the left
    field = corine._agricultural_array(png, 20, 10)
    assert field.values[:, :10].mean() == 1.0
    assert field.values[:, 10:].mean() == 0.0


def test_pixel_size_respects_max_pixels(tmp_path):
    cfg = _config(tmp_path, source="corine", resolution_m=1.0, max_pixels=256)
    width, height = CorineLandCover(cfg)._pixel_size()
    assert max(width, height) <= 256


def test_cropland_mask_from_corine_thresholds(tmp_path, monkeypatch):
    cfg = _config(tmp_path, source="corine", threshold=0.5)
    grid_da = CroplandMask(cfg)._grid.target_dataarray()
    # Half the cells above threshold, half below.
    frac = xr.ones_like(grid_da) * 0.2
    frac.values[:, : frac.shape[1] // 2] = 0.8
    monkeypatch.setattr(
        "data4simplace.spatial.corine.CorineLandCover.agricultural_fraction",
        lambda self: frac,
    )
    mask = CroplandMask(cfg).build()
    assert mask.dtype == bool
    assert bool(mask.values[:, : frac.shape[1] // 2].all())
    assert not bool(mask.values[:, frac.shape[1] // 2 :].any())


def test_mask_source_none_keeps_all(tmp_path):
    cfg = _config(tmp_path, source="none")
    mask = CroplandMask(cfg).build()
    assert bool(mask.values.all())
