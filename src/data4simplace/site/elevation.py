"""Per-cell mean altitude from a digital elevation model.

SIMPLACE takes an altitude per location (``vAltitude``) and LINTUL-5 uses it in
the Penman reference evapotranspiration through the atmospheric pressure, so a
continental run over the Alps and the North German Plain cannot share one
constant. Nothing in the SoilGrids or MSWX inputs carries elevation, so this
stage reads a DEM and aggregates it to the target grid.

**Not every "elevation" file is terrain.** ``EGM96_30arcsec.nc4`` under
``Land/Elevation`` holds ``geoid_altitude`` — a WGS84-to-EGM96 vertical *offset*
of roughly +/-100 m, which is a datum correction, not a height above sea level.
Pointing ``paths.dem_path`` at it produces plausible-looking numbers that are
wrong everywhere, so :func:`load_elevation` rejects a geoid variable by name.
The terrain product on this system is GMTED2010:

``GMTED2010_15n015_00625deg.nc``
    Global 0.0625 degree ``elevation`` (a mean statistic). The default, and
    finer than the 0.1 degree target, so it aggregates by binned mean like every
    other fine source in the pipeline.

``GMTED2010_maximum_15arcsec.nc4``
    15 arcsec, but its only variable is ``surface_altitude_maximum`` — a
    per-pixel *maximum*, biased high in exactly the terrain where altitude
    matters. Usable, but say so deliberately via ``site.dem_variable``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

__all__ = ["DEM_VARIABLE_CANDIDATES", "load_elevation", "resolve_dem_variable"]

#: Variable names searched for, in order, when ``site.dem_variable`` is unset.
DEM_VARIABLE_CANDIDATES: tuple[str, ...] = (
    "elevation",
    "surface_altitude",
    "altitude",
    "height",
    "surface_altitude_maximum",
    "topo",
    "z",
)

#: Substrings that mark a variable as a datum offset rather than terrain. A
#: geoid undulation is the one wrong layer most likely to be reached for, since
#: it lives in a directory called ``Elevation`` and its units are metres.
_GEOID_TOKENS: tuple[str, ...] = ("geoid", "undulation", "vertical_offset")


def resolve_dem_variable(dataset: xr.Dataset, configured: str | None) -> str:
    """Pick the terrain variable of a DEM file.

    Parameters
    ----------
    dataset:
        The opened DEM.
    configured:
        ``site.dem_variable``. When given it is used as-is (after the geoid
        check), so an unusual product needs no code change.

    Raises
    ------
    ValueError
        If the named variable is absent, if it is a geoid/datum layer, or if no
        candidate matches. Each case names what the file actually holds.
    """
    available = sorted(str(v) for v in dataset.data_vars)

    if configured is not None:
        if configured not in dataset.data_vars:
            raise ValueError(
                f"site.dem_variable {configured!r} is not in the DEM; it holds "
                f"{available}"
            )
        name = configured
    else:
        name = next((c for c in DEM_VARIABLE_CANDIDATES if c in dataset.data_vars), "")
        if not name:
            raise ValueError(
                f"No elevation variable found in the DEM (looked for "
                f"{list(DEM_VARIABLE_CANDIDATES)}); it holds {available}. Set "
                f"site.dem_variable to choose one."
            )

    lowered = name.lower()
    if any(token in lowered for token in _GEOID_TOKENS):
        raise ValueError(
            f"DEM variable {name!r} is a geoid/datum offset, not terrain "
            f"elevation. Point paths.dem_path at a terrain DEM (e.g. GMTED2010) "
            f"rather than an EGM96 conversion grid."
        )
    return name


def _subset(layer: xr.DataArray, config: PipelineConfig) -> xr.DataArray:
    """Sort the axes and cut the global DEM to the target bbox with a halo."""
    layer = layer.sortby(["lat", "lon"])
    grid = config.grid
    halo = grid.resolution_deg
    return layer.sel(
        lon=slice(grid.min_lon - halo, grid.max_lon + halo),
        lat=slice(grid.min_lat - halo, grid.max_lat + halo),
    )


def load_elevation(config: PipelineConfig, grid: TargetGrid) -> xr.DataArray:
    """Mean terrain elevation per target cell, in metres.

    The DEM is subset to the bounding box and then aggregated by
    :meth:`~data4simplace.grid.TargetGrid.regrid`, which for a source finer than
    the target is a binned mean over the source cells whose centres fall in each
    target cell — the same reduction the NPK and cropland stages use.

    Sea-level cells are kept at their DEM value rather than masked: a coastal
    10 km cell whose cropland sits at 3 m has an altitude of 3 m, and dropping
    it would remove the cell from the export entirely.

    Returns
    -------
    xarray.DataArray
        ``altitude_m`` on ``(lat, lon)``. NaN where the DEM has no data.

    Raises
    ------
    ValueError
        If ``paths.dem_path`` is unset, or the file holds no usable terrain
        variable (see :func:`resolve_dem_variable`).
    FileNotFoundError
        If the DEM file does not exist.
    """
    path = config.paths.dem_path
    if path is None:
        raise ValueError(
            "flags.run_site_processing needs paths.dem_path to point at a "
            "terrain DEM (e.g. GMTED2010_15n015_00625deg.nc)"
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DEM not found: {path}")

    with xr.open_dataset(path) as raw:
        renames = {
            name: target
            for name, target in (("longitude", "lon"), ("latitude", "lat"),
                                 ("x", "lon"), ("y", "lat"))
            if name in raw.coords and target not in raw.coords
        }
        raw = raw.rename(renames) if renames else raw
        name = resolve_dem_variable(raw, config.site.dem_variable)
        layer = _subset(raw[name], config).load()

    logger.info(
        "DEM %s[%s]: %d x %d source pixels in the domain",
        path.name, name, layer.sizes.get("lat", 0), layer.sizes.get("lon", 0),
    )

    altitude = grid.regrid(layer, method="mean")
    assert isinstance(altitude, xr.DataArray)  # a DataArray in, a DataArray out
    altitude = altitude.rename("altitude_m")
    altitude.attrs = {
        "long_name": "mean terrain elevation",
        "units": "m",
        "source": path.name,
        "source_variable": name,
    }

    finite = np.isfinite(altitude.values)
    if finite.any():
        logger.info(
            "Altitude aligned to %d target cells (%.0f to %.0f m)",
            int(finite.sum()),
            float(np.nanmin(altitude.values)),
            float(np.nanmax(altitude.values)),
        )
    else:
        logger.warning("DEM %s covered no target cell", path.name)
    return altitude
