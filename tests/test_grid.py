"""Tests for the target grid and SimplaceID assignment."""

from __future__ import annotations

import numpy as np

from data4simplace.grid import TargetGrid


def _grid() -> TargetGrid:
    return TargetGrid(min_lon=11.2, max_lon=11.5, min_lat=52.0, max_lat=52.3, resolution_deg=0.1)


def test_shape_and_centers():
    g = _grid()
    assert g.shape == (3, 3)
    # centres offset by half a cell from the edges
    assert np.isclose(g.lon_centers[0], 11.25)
    assert np.isclose(g.lat_centers[0], 52.25)  # north-to-south
    assert g.lat_centers[0] > g.lat_centers[-1]


def test_cell_table_ids_are_unique_and_stable():
    g = _grid()
    table = g.cell_table()
    assert len(table) == 9
    assert table["SimplaceID"].is_unique
    assert table["SimplaceID"].tolist() == list(range(1, 10))
    # regenerating yields identical IDs for identical coordinates
    again = g.cell_table()
    assert table.equals(again)


def test_regrid_coarsens_fine_field():
    import xarray as xr

    g = _grid()
    fine_lat = np.arange(52.3, 52.0, -0.02)
    fine_lon = np.arange(11.2, 11.5, 0.02)
    da = xr.DataArray(
        np.ones((fine_lat.size, fine_lon.size), dtype="float32"),
        dims=("lat", "lon"),
        coords={"lat": fine_lat, "lon": fine_lon},
    )
    out = g.regrid(da, method="mean")
    assert out.shape == g.shape
    assert np.allclose(out.values[np.isfinite(out.values)], 1.0)
