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


def _same_resolution_source(g: TargetGrid, n_time: int = 0):
    """A source on the target's own grid, with float32 axes as MSWX ships them.

    float32 makes the measured spacing 0.09999... rather than 0.1, which is what
    used to send an equal-resolution source down the binned-reduction branch.
    """
    import xarray as xr

    lat = np.arange(g.max_lat + 0.05, g.min_lat - 0.05, -0.1, dtype="float32")
    lon = np.arange(g.min_lon - 0.05, g.max_lon + 0.05, 0.1, dtype="float32")
    if n_time:
        values = np.arange(n_time * lat.size * lon.size, dtype="float32")
        return xr.DataArray(
            values.reshape(n_time, lat.size, lon.size),
            dims=("time", "lat", "lon"),
            coords={"time": np.arange(n_time), "lat": lat, "lon": lon},
        )
    values = np.arange(lat.size * lon.size, dtype="float32")
    return xr.DataArray(
        values.reshape(lat.size, lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


def test_regrid_same_resolution_picks_nearest_source_cell():
    g = _grid()
    da = _same_resolution_source(g)
    out = g.regrid(da, method="mean")

    assert out.shape == g.shape
    # Every target cell carries the value of the source cell it coincides with,
    # not an average of it with anything else.
    expected = da.sel(lat=g.lat_centers, lon=g.lon_centers, method="nearest").values
    assert np.array_equal(out.values, expected)


def test_regrid_same_resolution_does_not_explode_the_dask_graph():
    """Regression: the binned branch emitted n_lat*n_lon tasks *per time chunk*.

    A 46-year MSWX tile chunked by 30 days reached ~8.5 million tasks per
    variable and exhausted the node inside ``dask.optimization.fuse``.
    """
    pytest = __import__("pytest")
    pytest.importorskip("dask")

    g = _grid()
    n_time = 300
    da = _same_resolution_source(g, n_time=n_time).chunk({"time": 10})
    out = g.regrid(da, method="mean")

    n_chunks = n_time // 10
    # A few tasks per chunk, not one per target cell per chunk.
    assert len(out.data.dask) < 10 * n_chunks
    assert out.compute().shape == (n_time, *g.shape)
