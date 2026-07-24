"""Tests for the per-target-cell soil aggregation engine."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.grid import TargetGrid
from data4simplace.soil.aggregate import (
    bin_reduce,
    cell_geomean,
    cell_mean,
    cell_ph,
    fill_missing_cells,
    reducer_for,
)


@pytest.fixture()
def two_cell_grid() -> TargetGrid:
    """A 1x2 target grid: lon cells [0,0.1] and [0.1,0.2], one lat row."""
    return TargetGrid(min_lon=0.0, max_lon=0.2, min_lat=0.0, max_lat=0.1, resolution_deg=0.1)


def _fine(values, lon, lat=(0.05,)):
    return xr.DataArray(
        np.asarray(values, dtype="float64").reshape(len(lat), len(lon)),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(lat), "lon": np.asarray(lon)},
    )


def test_cell_mean_reducer():
    assert cell_mean(np.array([10.0, 20.0])) == 15.0


def test_cell_geomean_reducer():
    assert cell_geomean(np.array([1.0, 10.0, 100.0])) == pytest.approx(10.0)


def test_cell_geomean_drops_nonpositive():
    # Zero/negative pixels are undefined in log space and must be ignored.
    assert cell_geomean(np.array([0.0, 4.0, 9.0])) == pytest.approx(6.0)
    assert np.isnan(cell_geomean(np.array([0.0, -1.0])))


def test_cell_ph_uses_hydrogen_activity():
    # Mean of 10^-6 and 10^-8, back to pH: closer to the acidic (higher [H+]) end.
    result = cell_ph(np.array([6.0, 8.0]))
    assert result == pytest.approx(-np.log10((1e-6 + 1e-8) / 2), abs=1e-9)
    assert result == pytest.approx(6.2967, abs=1e-3)


def test_bin_reduce_assigns_pixels_to_correct_cells(two_cell_grid):
    # Two pixels per cell; means should be per-cell, not global.
    out = bin_reduce(
        _fine([10.0, 20.0, 100.0, 300.0], lon=[0.02, 0.05, 0.12, 0.15]),
        two_cell_grid,
        cell_mean,
    )
    np.testing.assert_allclose(out.values, [[15.0, 200.0]])


def test_bin_reduce_empty_cell_is_nan(two_cell_grid):
    # No pixels fall in the second cell -> NaN there.
    out = bin_reduce(_fine([10.0, 20.0], lon=[0.02, 0.05]), two_cell_grid, cell_mean)
    assert out.values[0, 0] == pytest.approx(15.0)
    assert np.isnan(out.values[0, 1])


def test_bin_reduce_ignores_nonfinite_pixels(two_cell_grid):
    out = bin_reduce(
        _fine([10.0, np.nan, 100.0, 300.0], lon=[0.02, 0.05, 0.12, 0.15]),
        two_cell_grid,
        cell_mean,
    )
    np.testing.assert_allclose(out.values, [[10.0, 200.0]])


def test_reducer_for_dispatch():
    assert reducer_for("clay") is cell_mean
    assert reducer_for("soc") is cell_geomean
    assert reducer_for("nitrogen") is cell_geomean
    assert reducer_for("phh2o") is cell_ph


def _grid_da(values):
    values = np.asarray(values, dtype="float64")
    n_lat, n_lon = values.shape
    return xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lat": np.arange(n_lat, 0, -1), "lon": np.arange(n_lon)},
    )


def test_fill_missing_cells_fills_from_nearest():
    da = _grid_da([[1.0, np.nan, 3.0], [np.nan, np.nan, np.nan]])
    filled = fill_missing_cells(da)
    assert np.isfinite(filled.values).all()
    # The hole at (0,1) is equidistant-ish; nearest finite neighbour is 1 or 3.
    assert filled.values[0, 1] in (1.0, 3.0)
    # Bottom row cells take their nearest top-row value.
    assert filled.values[1, 0] == 1.0
    assert filled.values[1, 2] == 3.0


def test_fill_missing_cells_preserves_finite_values():
    da = _grid_da([[5.0, 6.0], [7.0, np.nan]])
    filled = fill_missing_cells(da)
    np.testing.assert_array_equal(filled.values[[0, 0, 1], [0, 1, 0]], [5.0, 6.0, 7.0])
    assert np.isfinite(filled.values[1, 1])


def test_fill_missing_cells_over_dataset_and_depth():
    da = xr.DataArray(
        np.array([[[1.0, np.nan]], [[np.nan, 4.0]]]),
        dims=("depth", "lat", "lon"),
        coords={"depth": [0, 1], "lat": [1.0], "lon": [0, 1]},
    )
    filled = fill_missing_cells(xr.Dataset({"soc": da}))
    assert np.isfinite(filled["soc"].values).all()
    assert filled["soc"].isel(depth=0).values.tolist() == [[1.0, 1.0]]
    assert filled["soc"].isel(depth=1).values.tolist() == [[4.0, 4.0]]
