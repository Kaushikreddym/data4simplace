"""Latitude-aware surface areas for geographic (lon/lat) grids.

A 0.1 degree cell covers ~103 km2 on Crete (34N) but only ~39 km2 at North Cape
(71.5N), so the per-class areas the multi-class soil workflow reports cannot be
a pixel count times a constant. Every area here is computed from the spherical
cap between the pixel's latitude edges:

``A = R^2 * dlon_rad * (sin(lat_north) - sin(lat_south))``

which is exact on a sphere and accurate to ~0.5 % against the WGS84 ellipsoid --
far below the uncertainty of the underlying land-cover and soil-class rasters.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: Earth's mean radius (IUGG), in kilometres.
EARTH_RADIUS_KM = 6371.0088

#: Hectares per square kilometre.
HECTARES_PER_KM2 = 100.0


def latitude_band_area_km2(
    lat_centers: np.ndarray, dlat_deg: float, dlon_deg: float
) -> np.ndarray:
    """Area (km2) of one pixel in each latitude row of a regular lon/lat grid.

    Parameters
    ----------
    lat_centers:
        Latitudes of the pixel centres, in degrees (any order).
    dlat_deg, dlon_deg:
        Pixel size in degrees.

    Returns
    -------
    numpy.ndarray
        One area per input latitude, in km2. Latitude edges are clipped to the
        poles so a grid running to +-90 degrees does not produce areas beyond
        the sphere.
    """
    lat = np.asarray(lat_centers, dtype="float64")
    north = np.clip(lat + abs(dlat_deg) / 2.0, -90.0, 90.0)
    south = np.clip(lat - abs(dlat_deg) / 2.0, -90.0, 90.0)
    strip = np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    return EARTH_RADIUS_KM**2 * np.deg2rad(abs(dlon_deg)) * strip


def _coord_step(values: np.ndarray) -> float:
    """Spacing of a 1-D coordinate; 0 for a degenerate single-element axis."""
    values = np.asarray(values, dtype="float64")
    if values.size < 2:
        return 0.0
    return float(np.abs(np.diff(values)).mean())


def pixel_area_km2(
    field: xr.DataArray,
    dlat_deg: float | None = None,
    dlon_deg: float | None = None,
) -> xr.DataArray:
    """Per-pixel surface area (km2) of a field's ``lat``/``lon`` grid.

    Parameters
    ----------
    field:
        Any field carrying ``lat`` and ``lon`` coordinates (e.g. the 250 m soil
        class raster). Only its coordinates are read.
    dlat_deg, dlon_deg:
        Pixel size in degrees. Inferred from the coordinate spacing when omitted,
        which is what callers want for a regular SoilGrids or target grid.

    Returns
    -------
    xarray.DataArray
        Area per pixel on ``(lat, lon)``, in km2. Constant along ``lon``, varying
        with the cosine of latitude along ``lat``.

    Raises
    ------
    ValueError
        If the grid spacing cannot be inferred (a single row *and* column with no
        explicit sizes given).
    """
    lat = np.asarray(field["lat"].values, dtype="float64")
    lon = np.asarray(field["lon"].values, dtype="float64")
    dlat = abs(dlat_deg) if dlat_deg is not None else _coord_step(lat)
    dlon = abs(dlon_deg) if dlon_deg is not None else _coord_step(lon)
    # A single-row (or single-column) subset still has a well-defined pixel size
    # in the other direction; only a 1x1 field is genuinely ambiguous.
    dlat = dlat or dlon
    dlon = dlon or dlat
    if not dlat or not dlon:
        raise ValueError(
            "Cannot infer pixel size from a 1x1 grid; pass dlat_deg/dlon_deg"
        )

    per_row = latitude_band_area_km2(lat, dlat, dlon)
    area = np.repeat(per_row[:, None], lon.size, axis=1)
    return xr.DataArray(
        area,
        dims=("lat", "lon"),
        coords={"lat": field["lat"], "lon": field["lon"]},
        name="pixel_area_km2",
        attrs={"units": "km2", "long_name": "pixel surface area"},
    )
