"""The two geometries the 10 m pixels are aggregated into.

Every pixel is labelled twice, in one pass over the rasters:

**CyBench administrative units** -- 946 polygons over 23 countries -- because
that is the support the observations are reported on, so a phenology table keyed
by ``adm_id`` joins straight onto CyBench yields and calendars.

**The 0.1 degree target grid** -- the same ``SimplaceID`` cells the export uses --
so ``site/calendar.py`` can take a ``calendar_source: clms`` without a second
pass over 80 GB.

The polygon read here duplicates a few lines of
``cropmodelling4eu.evaluation.regions.load_admin_polygons``. That is deliberate:
``data4simplace`` is the base package and ``cropmodelling4eu`` consumes its
exports, so importing upward would invert the dependency. Moving that function
down instead would drag CyBench's yield and calendar loaders with it, which
belong to the evaluation package, not to this one.

**Zone rasters are built once per tile and reused across its years.** The
geometry does not change with the season, so an array task that owns one tile
and loops its eight years pays for this once rather than eight times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import Affine

from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

__all__ = ["ZoneRasters", "admin_zone_raster", "grid_zone_raster", "load_admin_zones"]

#: Value meaning "this pixel belongs to no zone".
NO_ZONE: int = 0


def load_admin_zones(
    countries: tuple[str, ...] | list[str], polygon_dir: Path
) -> gpd.GeoDataFrame:
    """CyBench administrative polygons for a set of countries, in EPSG:4326.

    Args:
        countries: Two-letter CyBench codes (``EL`` for Greece, not ``GR``).
        polygon_dir: Root holding ``<country>/<country>.shp``.

    Returns:
        Columns ``country``, ``adm_id``, ``geometry``.

    Raises:
        ValueError: If no shapefile was found for any country.
    """
    frames = []
    for code in countries:
        path = Path(polygon_dir) / code / f"{code}.shp"
        if not path.is_file():
            logger.info("%s: no polygons at %s", code, path)
            continue
        frame = gpd.read_file(path)
        frame["country"] = code
        frames.append(frame[["country", "adm_id", "geometry"]])

    if not frames:
        raise ValueError(f"no CyBench polygons found under {polygon_dir}")

    admin = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs
    ).to_crs("EPSG:4326")
    logger.info(
        "%d admin polygons across %d countries", len(admin), admin["country"].nunique()
    )
    return admin


@dataclass(frozen=True, slots=True)
class ZoneRasters:
    """Both zone labellings of one tile, plus the lookup back to real ids.

    Attributes:
        admin: ``int16`` raster of 1-based row indices into :attr:`admin_ids`;
            :data:`NO_ZONE` outside every polygon.
        admin_ids: ``(country, adm_id)`` per index, in index order.
        grid: ``int32`` raster of ``SimplaceID``; :data:`NO_ZONE` off-grid.
    """

    admin: np.ndarray
    admin_ids: pd.DataFrame
    grid: np.ndarray


def admin_zone_raster(
    admin: gpd.GeoDataFrame,
    transform: Affine,
    shape: tuple[int, int],
    crs,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Burn the admin polygons onto a tile's grid.

    Only the polygons that intersect the tile are burned, and the returned index
    is dense over *those*, so a tile touching three units carries three indices
    rather than a sparse slice of 946.

    Args:
        admin: Output of :func:`load_admin_zones`.
        transform: The tile's affine transform.
        shape: ``(height, width)``.
        crs: The tile's CRS.

    Returns:
        ``(raster, ids)`` -- an ``int16`` raster of 1-based indices, and the
        ``country``/``adm_id`` table those indices point into.
    """
    local = admin.to_crs(crs)
    height, width = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (width, height)
    bounds = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

    hit = local[local.intersects(gpd.GeoSeries([_box(bounds)], crs=crs).iloc[0])]
    if hit.empty:
        return (
            np.zeros(shape, dtype=np.int16),
            pd.DataFrame(columns=["country", "adm_id"]),
        )

    hit = hit.reset_index(drop=True)
    # int16 caps at 32767 zones per tile; 946 exist in total, so the cast is safe
    # and halves the memory of a 100 M-pixel labelling.
    raster = rasterize(
        ((geom, i + 1) for i, geom in enumerate(hit.geometry)),
        out_shape=shape,
        transform=transform,
        fill=NO_ZONE,
        dtype="int32",
        all_touched=False,
    ).astype(np.int16)
    return raster, hit[["country", "adm_id"]].copy()


def grid_zone_raster(
    transform: Affine,
    shape: tuple[int, int],
    crs,
    grid: TargetGrid,
) -> np.ndarray:
    """Label every pixel with the ``SimplaceID`` of the 0.1 degree cell it falls in.

    Computed from pixel centres rather than by rasterizing cell polygons: the
    cells are axis-aligned in EPSG:4326 but curved quadrilaterals in the tile's
    projection, and binning centres is both exact and the convention this
    project already uses for every fine-to-coarse step.

    Args:
        transform: The tile's affine transform.
        shape: ``(height, width)``.
        crs: The tile's CRS.
        grid: The export's target grid, for resolution, origin and id scheme.

    Returns:
        ``int32`` raster of ``SimplaceID``; :data:`NO_ZONE` outside the grid.
    """
    from pyproj import Transformer

    height, width = shape
    rows = np.arange(height)
    cols = np.arange(width)
    # Pixel centres in the tile CRS, built as 1-D axes: the transform is axis
    # aligned, so the full 2-D mesh is never materialised in projected space.
    xs, _ = transform * (cols + 0.5, np.full(width, 0.5))
    _, ys = transform * (np.full(height, 0.5), rows + 0.5)

    to4326 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    lon, lat = to4326.transform(mesh_x, mesh_y)
    del mesh_x, mesh_y

    res = grid.resolution_deg
    n_lon = grid.lon_centers.size
    n_lat = grid.lat_centers.size
    col = np.floor((lon - grid.min_lon) / res).astype(np.int64)
    row = np.floor((grid.max_lat - lat) / res).astype(np.int64)
    del lon, lat

    inside = (col >= 0) & (col < n_lon) & (row >= 0) & (row < n_lat)
    ids = np.where(inside, row * n_lon + col + 1, NO_ZONE)
    return ids.astype(np.int32)


def _box(bounds: tuple[float, float, float, float]):
    """Rectangle polygon from ``(minx, miny, maxx, maxy)``."""
    from shapely.geometry import box

    return box(*bounds)
