"""Tests for the PROBA-V cropland filters (250 m pixels and 10 km cells)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.soil.classify import fill_missing_cells, valid_soil_cells
from data4simplace.spatial import (
    CroplandWeights,
    apply_cell_mask,
    export_cell_mask,
    keep_cells,
)

# A 2x2 target grid of 0.1 deg cells: lon 11.2-11.4, lat 52.0-52.2.
GRID = {
    "resolution_deg": 0.1,
    "min_lon": 11.2,
    "max_lon": 11.4,
    "min_lat": 52.0,
    "max_lat": 52.2,
    "crs": "EPSG:4326",
}


def _config(tmp_path, cover_path=None, **soil) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "flags": {"apply_agricultural_mask": True},
            "grid": GRID,
            "time": {"start": "1979-01-01", "end": "1979-01-01"},
            "paths": {
                "mswx_root": str(tmp_path),
                "output_dir": str(tmp_path / "out"),
                "cropland_weights_path": str(cover_path) if cover_path else None,
            },
            "soil": soil,
        }
    )


def _cover_tif(tmp_path, fraction: np.ndarray) -> str:
    """Write ``fraction`` (0-1, shape (n, n)) as a PROBA-V-style percentage GeoTIFF.

    The array spans exactly the target grid bbox, so pixel (i, j) of each
    quadrant falls in a known target cell.
    """
    n = fraction.shape[0]
    res = 0.2 / n
    lon = 11.2 + res / 2.0 + res * np.arange(n)
    lat = 52.2 - res / 2.0 - res * np.arange(n)  # north -> south
    da = xr.DataArray(
        (fraction * 100.0).astype("float32"),
        dims=("y", "x"),
        coords={"y": lat, "x": lon},
        name="crops",
    )
    da = da.rio.write_crs("EPSG:4326")
    path = tmp_path / "cover.tif"
    da.rio.to_raster(path)
    return path


def _quadrants(values: tuple[float, float, float, float], n: int = 8) -> np.ndarray:
    """An ``n x n`` cover array, one constant value per target cell (NW, NE, SW, SE)."""
    nw, ne, sw, se = values
    arr = np.empty((n, n), dtype="float32")
    h = n // 2
    arr[:h, :h], arr[:h, h:] = nw, ne
    arr[h:, :h], arr[h:, h:] = sw, se
    return arr


# ---------------------------------------------------------------------------- #
# cell_mask
# ---------------------------------------------------------------------------- #
def test_cell_mask_keeps_cells_above_the_cover_threshold(tmp_path):
    # NW and SE are solid cropland; NE and SW are below the 0.8 threshold.
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.5, 0.2, 0.85)))
    cfg = _config(tmp_path, path, cropland_min_fraction=0.8)
    grid = TargetGrid.from_config(cfg.grid)

    mask = CroplandWeights(cfg).cell_mask(grid)

    assert mask is not None
    assert mask.dtype == bool
    assert mask.dims == ("lat", "lon")
    np.testing.assert_array_equal(mask.values, [[True, False], [False, True]])
    # Latitude runs north -> south, as everywhere else in the pipeline.
    assert mask["lat"].values[0] > mask["lat"].values[-1]


def test_cell_mask_honours_min_cropland_pixels(tmp_path):
    # The NW cell has 16 qualifying pixels (a quarter of its 8x8 block).
    arr = np.zeros((16, 16), dtype="float32")
    arr[:4, :4] = 0.9
    path = _cover_tif(tmp_path, arr)

    lenient = CroplandWeights(
        _config(tmp_path, path, cropland_min_fraction=0.8, min_cropland_pixels=1)
    ).cell_mask(TargetGrid.from_config(_config(tmp_path, path).grid))
    strict = CroplandWeights(
        _config(tmp_path, path, cropland_min_fraction=0.8, min_cropland_pixels=17)
    ).cell_mask(TargetGrid.from_config(_config(tmp_path, path).grid))

    assert bool(lenient.values[0, 0]) is True
    assert not bool(strict.values.any())


def test_cell_mask_is_none_without_a_cropland_raster(tmp_path):
    cfg = _config(tmp_path, cover_path=None)
    assert CroplandWeights(cfg).cell_mask(TargetGrid.from_config(cfg.grid)) is None


def test_load_fraction_is_read_once_per_instance(tmp_path, monkeypatch):
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.9, 0.9, 0.9)))
    cfg = _config(tmp_path, path)
    weights = CroplandWeights(cfg)

    calls = {"n": 0}
    original = CroplandWeights._load_fraction

    def counted(self):
        calls["n"] += 1
        return original(self)

    monkeypatch.setattr(CroplandWeights, "_load_fraction", counted)
    weights.cell_mask(TargetGrid.from_config(cfg.grid))
    weights.load_fraction()
    assert calls["n"] == 1


# ---------------------------------------------------------------------------- #
# Unit detection
# ---------------------------------------------------------------------------- #
def _da(values, encoding_dtype=None) -> xr.DataArray:
    da = xr.DataArray(np.asarray(values, dtype="float32"), dims=("y", "x"))
    if encoding_dtype is not None:
        da.encoding["dtype"] = encoding_dtype
    return da


def test_percent_detection_follows_the_on_disk_dtype():
    # A sparse subset peaking at 1% must still be read as a percentage: the
    # values alone are indistinguishable from a 0-1 fraction.
    assert CroplandWeights._stores_percent(_da([[0.0, 1.0]], "uint8")) is True
    assert CroplandWeights._stores_percent(_da([[0.0, 90.0]], "uint8")) is True


def test_percent_detection_falls_back_to_the_range_for_float_layers():
    assert CroplandWeights._stores_percent(_da([[0.0, 0.9]])) is False
    assert CroplandWeights._stores_percent(_da([[0.0, 90.0]])) is True
    # An all-no-data subset carries no signal; scaling it is a no-op either way.
    assert CroplandWeights._stores_percent(_da([[np.nan, np.nan]])) is True


# ---------------------------------------------------------------------------- #
# Applying a cell mask
# ---------------------------------------------------------------------------- #
@pytest.fixture()
def mask_2x2() -> xr.DataArray:
    """Keep the NW and SE cells of the 2x2 target grid."""
    grid = TargetGrid(**GRID)
    return xr.DataArray(
        np.array([[True, False], [False, True]]),
        dims=("lat", "lon"),
        coords={"lat": grid.lat_centers, "lon": grid.lon_centers},
    )


def test_apply_cell_mask_nans_the_dropped_cells(mask_2x2):
    data = xr.Dataset(
        {"clay": (("lat", "lon"), np.array([[10.0, 20.0], [30.0, 40.0]]))},
        coords={"lat": mask_2x2["lat"], "lon": mask_2x2["lon"]},
    )
    out = apply_cell_mask(data, mask_2x2)
    np.testing.assert_array_equal(np.isnan(out["clay"].values), [[False, True], [True, False]])
    assert out["clay"].values[0, 0] == 10.0


def test_apply_cell_mask_passes_through_a_none_mask(mask_2x2):
    data = xr.Dataset(
        {"clay": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": mask_2x2["lat"], "lon": mask_2x2["lon"]},
    )
    assert apply_cell_mask(data, None) is data


def test_keep_cells_filters_the_cell_table_in_grid_order(mask_2x2):
    table = TargetGrid(**GRID).cell_table()
    kept = keep_cells(table, mask_2x2)
    # Row-major order: SimplaceID 1 is NW, 4 is SE.
    assert list(kept["SimplaceID"]) == [1, 4]
    assert list(kept.index) == [0, 1]  # reset


def test_keep_cells_passes_through_a_none_mask():
    table = TargetGrid(**GRID).cell_table()
    assert len(keep_cells(table, None)) == len(table)


# ---------------------------------------------------------------------------- #
# Bounded gap fill
# ---------------------------------------------------------------------------- #
def _gappy_field() -> xr.DataArray:
    """A 1x4 row: one valid cell, three gaps."""
    return xr.DataArray(
        np.array([[25.0, np.nan, np.nan, np.nan]]),
        dims=("lat", "lon"),
        coords={"lat": [52.05], "lon": [11.25, 11.35, 11.45, 11.55]},
        name="clay",
    )


def test_gap_fill_is_unbounded_without_a_mask():
    filled = fill_missing_cells(_gappy_field())
    np.testing.assert_array_equal(filled.values, [[25.0, 25.0, 25.0, 25.0]])


def test_gap_fill_stops_at_the_cropland_mask():
    field = _gappy_field()
    within = xr.DataArray(
        np.array([[True, True, False, False]]),
        dims=("lat", "lon"),
        coords={"lat": field["lat"], "lon": field["lon"]},
    )
    filled = fill_missing_cells(field, within=within)

    np.testing.assert_array_equal(filled.values[0, :2], [25.0, 25.0])
    assert np.isnan(filled.values[0, 2:]).all()
    assert filled.dims == field.dims


def test_gap_fill_keeps_valid_values_outside_the_mask():
    field = xr.DataArray(
        np.array([[25.0, np.nan, 42.0]]),
        dims=("lat", "lon"),
        coords={"lat": [52.05], "lon": [11.25, 11.35, 11.45]},
    )
    within = xr.DataArray(
        np.array([[True, False, False]]),
        dims=("lat", "lon"),
        coords={"lat": field["lat"], "lon": field["lon"]},
    )
    filled = fill_missing_cells(field, within=within)
    assert filled.values[0, 0] == 25.0
    assert np.isnan(filled.values[0, 1])  # gap outside the mask stays a gap
    assert filled.values[0, 2] == 42.0  # value outside the mask is untouched


def test_gap_fill_bounds_every_depth_slice_of_a_dataset():
    depths = ["0-5cm", "5-15cm"]
    data = np.full((2, 1, 3), np.nan)
    data[:, 0, 0] = [10.0, 20.0]
    ds = xr.Dataset(
        {"clay": (("depth", "lat", "lon"), data)},
        coords={"depth": depths, "lat": [52.05], "lon": [11.25, 11.35, 11.45]},
    )
    within = xr.DataArray(
        np.array([[True, True, False]]),
        dims=("lat", "lon"),
        coords={"lat": ds["lat"], "lon": ds["lon"]},
    )
    filled = fill_missing_cells(ds, within=within)

    assert filled["clay"].dims == ("depth", "lat", "lon")
    np.testing.assert_array_equal(filled["clay"].values[:, 0, 1], [10.0, 20.0])
    assert np.isnan(filled["clay"].values[:, 0, 2]).all()


# ---------------------------------------------------------------------------- #
# The exported cell set
# ---------------------------------------------------------------------------- #
def _soil_2x2(valid: np.ndarray, depths: int = 2) -> xr.Dataset:
    """A depth-resolved 2x2 soil dataset, NaN wherever ``valid`` is False."""
    grid = TargetGrid(**GRID)
    values = np.where(valid, 20.0, np.nan)[None, :, :].repeat(depths, axis=0)
    return xr.Dataset(
        {"clay": (("depth", "lat", "lon"), values),
         "sand": (("depth", "lat", "lon"), values)},
        coords={"depth": [f"{i}-{i + 5}cm" for i in range(depths)],
                "lat": grid.lat_centers, "lon": grid.lon_centers},
    )


def test_valid_soil_cells_needs_one_finite_value_anywhere():
    soil = _soil_2x2(np.array([[True, False], [False, True]]))
    # A cell valid at only one depth still counts.
    soil["clay"].values[1, 1, 0] = 15.0

    valid = valid_soil_cells(soil)
    np.testing.assert_array_equal(valid.values, [[True, False], [True, True]])


def test_valid_soil_cells_is_none_without_lat_lon_fields():
    assert valid_soil_cells(xr.Dataset({"x": ("depth", [1.0, 2.0])})) is None


def test_export_cell_mask_intersects_cropland_and_soil(tmp_path):
    # Cropland: NW + SE. Soil: NW + SW. Exported: NW only.
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.1, 0.1, 0.9)))
    cfg = _config(tmp_path, path, cropland_min_fraction=0.8)
    soil = _soil_2x2(np.array([[True, False], [True, False]]))

    mask = export_cell_mask(cfg, TargetGrid.from_config(cfg.grid), soil)
    np.testing.assert_array_equal(mask.values, [[True, False], [False, False]])


def test_export_cell_mask_uses_soil_alone_when_the_cropland_flag_is_off(tmp_path):
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.9, 0.9, 0.9)))
    cfg = _config(tmp_path, path)
    cfg = cfg.model_copy(
        update={"flags": cfg.flags.model_copy(update={"apply_agricultural_mask": False})}
    )
    soil = _soil_2x2(np.array([[True, False], [False, False]]))

    mask = export_cell_mask(cfg, TargetGrid.from_config(cfg.grid), soil)
    np.testing.assert_array_equal(mask.values, [[True, False], [False, False]])


def test_export_cell_mask_uses_cropland_alone_without_a_soil_stage(tmp_path):
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.1, 0.1, 0.1)))
    cfg = _config(tmp_path, path, cropland_min_fraction=0.8)

    mask = export_cell_mask(cfg, TargetGrid.from_config(cfg.grid), soil=None)
    np.testing.assert_array_equal(mask.values, [[True, False], [False, False]])


def test_export_cell_mask_is_none_when_neither_condition_applies(tmp_path):
    cfg = _config(tmp_path, cover_path=None)
    cfg = cfg.model_copy(
        update={"flags": cfg.flags.model_copy(update={"apply_agricultural_mask": False})}
    )
    assert export_cell_mask(cfg, TargetGrid.from_config(cfg.grid), soil=None) is None


def test_export_cell_mask_drops_cropland_cells_without_soil(tmp_path):
    """The point of the soil condition: no weather file without a soil profile."""
    path = _cover_tif(tmp_path, _quadrants((0.9, 0.9, 0.9, 0.9)))
    cfg = _config(tmp_path, path, cropland_min_fraction=0.8)
    grid = TargetGrid.from_config(cfg.grid)

    cropland_only = CroplandWeights(cfg).cell_mask(grid)
    assert int(cropland_only.values.sum()) == 4  # all four cells are cropland

    soil = _soil_2x2(np.array([[True, True], [False, False]]))  # SoilGrids gap in the south
    mask = export_cell_mask(cfg, grid, soil)

    assert int(mask.values.sum()) == 2
    kept = keep_cells(grid.cell_table(), mask)
    assert list(kept["SimplaceID"]) == [1, 2]


def test_fill_missing_defaults_to_off(tmp_path):
    """A borrowed profile must be opted into: it also widens the exported cells."""
    assert _config(tmp_path).soil.fill_missing is False


def test_cell_table_and_gap_fill_agree_on_the_exported_cells(mask_2x2):
    """A cell kept by the mask is filled; a dropped cell is neither filled nor exported."""
    soil = xr.DataArray(
        np.array([[30.0, np.nan], [np.nan, np.nan]]),
        dims=("lat", "lon"),
        coords={"lat": mask_2x2["lat"], "lon": mask_2x2["lon"]},
    )
    filled = fill_missing_cells(soil, within=mask_2x2)
    kept = keep_cells(TargetGrid(**GRID).cell_table(), mask_2x2)

    values = pd.Series(filled.values.ravel())[list(kept.index.map(
        lambda i: int(kept.loc[i, "row"]) * 2 + int(kept.loc[i, "col"]))
    )]
    assert values.notna().all()  # every exported cell carries a value
    assert np.isnan(filled.values[0, 1])  # a dropped cell was left alone
