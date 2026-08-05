"""Tests for the latitude-aware surface areas."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.spatial.area import (
    EARTH_RADIUS_KM,
    latitude_band_area_km2,
    pixel_area_km2,
)


def _field(lats, lons) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((len(lats), len(lons))),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(lats, dtype="float64"),
                "lon": np.asarray(lons, dtype="float64")},
    )


def test_equatorial_cell_matches_the_analytic_area():
    # At the equator a 0.1 deg cell is (R * 0.1 deg)^2 to within the cosine
    # correction over its own height.
    area = latitude_band_area_km2(np.array([0.0]), 0.1, 0.1)[0]
    side = EARTH_RADIUS_KM * np.deg2rad(0.1)
    assert area == pytest.approx(side**2, rel=1e-4)


def test_area_shrinks_with_the_cosine_of_latitude():
    lats = np.array([0.0, 45.0, 60.0, 72.0])
    areas = latitude_band_area_km2(lats, 0.1, 0.1)
    assert np.all(np.diff(areas) < 0)
    # cos(60) = 0.5 exactly, so a 60 deg cell is half an equatorial one.
    assert areas[2] / areas[0] == pytest.approx(0.5, rel=1e-3)
    # The Europe grid spans ~103 km2 (Crete, 34N) to ~39 km2 (North Cape, 71.5N):
    # a 10 km cell at the north edge is under 40 % of one at the south edge.
    assert 100 < latitude_band_area_km2(np.array([34.0]), 0.1, 0.1)[0] < 105
    assert 38 < latitude_band_area_km2(np.array([71.5]), 0.1, 0.1)[0] < 41


def test_whole_sphere_sums_to_earths_surface():
    # A coarse global grid must still integrate to 4*pi*R^2.
    lats = np.arange(-89.5, 90.0, 1.0)
    per_row = latitude_band_area_km2(lats, 1.0, 1.0)
    total = per_row.sum() * 360
    assert total == pytest.approx(4 * np.pi * EARTH_RADIUS_KM**2, rel=1e-6)


def test_pixel_area_infers_the_grid_spacing():
    area = pixel_area_km2(_field([52.2, 52.1, 52.0], [11.2, 11.3]))
    assert area.dims == ("lat", "lon")
    assert area.shape == (3, 2)
    # Constant along longitude, decreasing northwards.
    assert np.allclose(area.values[:, 0], area.values[:, 1])
    assert area.values[0, 0] < area.values[-1, 0]
    assert area.attrs["units"] == "km2"


def test_single_row_subset_uses_the_other_axis_spacing():
    # A one-row window is common at a tile edge; it must not raise.
    area = pixel_area_km2(_field([52.2], [11.2, 11.3, 11.4]))
    assert area.shape == (1, 3)
    assert np.all(area.values > 0)


def test_degenerate_grid_needs_explicit_sizes():
    with pytest.raises(ValueError, match="1x1 grid"):
        pixel_area_km2(_field([52.2], [11.2]))
    explicit = pixel_area_km2(_field([52.2], [11.2]), dlat_deg=0.1, dlon_deg=0.1)
    assert explicit.values[0, 0] > 0
