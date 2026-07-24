"""Agricultural cropland mask application.

Restricts pipeline outputs to cells that are (predominantly) cropland. The
mask source may be a raster (fractional or binary cropland cover) or a vector
layer of agricultural polygons; either is rasterised/aligned to the target grid
and thresholded into a boolean cell mask.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio import features
from rasterio.transform import from_origin

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

_VECTOR_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}
_RASTER_SUFFIXES = {".tif", ".tiff", ".nc"}


class CroplandMask:
    """Build and apply a cropland mask on the target grid.

    The mask source is selected by ``config.mask.source``:

    * ``file``   - use the raster/vector at ``paths.cropland_mask``.
    * ``corine`` - fetch CORINE Land Cover and threshold the agricultural share.
    * ``auto``   - use the local file when set/available, else CORINE.
    * ``none``   - keep every cell.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    threshold:
        Overrides ``config.mask.threshold`` when provided (minimum agricultural
        fraction, 0-1, for a cell to be kept).
    """

    def __init__(self, config: PipelineConfig, threshold: float | None = None) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)
        self._path = config.paths.cropland_mask
        self._mask_cfg = config.mask
        self._threshold = threshold if threshold is not None else config.mask.threshold

    # ------------------------------------------------------------------ #
    # Mask construction
    # ------------------------------------------------------------------ #
    def build(self) -> xr.DataArray:
        """Return a boolean ``(lat, lon)`` mask (True = keep the cell)."""
        source = self._mask_cfg.source
        if source == "none":
            logger.info("Mask source 'none'; keeping all cells")
            return self._all_true()

        has_file = self._path is not None and Path(self._path).exists()

        if source in ("file", "auto") and has_file:
            return self._from_file(Path(self._path))  # type: ignore[arg-type]
        if source == "file":
            logger.warning("Mask source 'file' but no valid cropland_mask; keeping all cells")
            return self._all_true()

        if source in ("corine", "auto"):
            return self._from_corine()

        logger.warning("Unknown mask source %r; keeping all cells", source)
        return self._all_true()

    def _from_file(self, path: Path) -> xr.DataArray:
        """Dispatch a local mask file to the raster or vector loader."""
        suffix = path.suffix.lower()
        if suffix in _VECTOR_SUFFIXES:
            return self._from_vector(path)
        if suffix in _RASTER_SUFFIXES:
            return self._from_raster(path)
        logger.warning("Unrecognised mask format %s; keeping all cells", suffix)
        return self._all_true()

    def _from_corine(self) -> xr.DataArray:
        """Build the mask from CORINE Land Cover agricultural classes."""
        from data4simplace.spatial.corine import CorineLandCover

        logger.info("Building agricultural mask from CORINE Land Cover")
        fraction = CorineLandCover(self._config).agricultural_fraction()
        if fraction is None:
            logger.warning("CORINE fetch failed; keeping all cells")
            return self._all_true()
        mask = (fraction >= self._threshold).fillna(False)
        return mask.rename("cropland_mask")

    def _all_true(self) -> xr.DataArray:
        """A mask that keeps every target cell."""
        template = self._grid.target_dataarray()
        return xr.ones_like(template, dtype=bool).rename("cropland_mask")

    def _from_raster(self, path: Path) -> xr.DataArray:
        """Align a cropland-cover raster and threshold it into a boolean mask."""
        if path.suffix.lower() == ".nc":
            ds = xr.open_dataset(path)
            da = ds[next(iter(ds.data_vars))]
        else:
            da = rioxarray.open_rasterio(path, masked=True)
            if isinstance(da, list):
                da = da[0]
            da = da.squeeze("band", drop=True) if "band" in da.dims else da

        aligned = self._grid.regrid(da, method="mean")
        # Normalise percentage covers (0-100) to fractions.
        if float(aligned.max()) > 1.5:
            aligned = aligned / 100.0
        mask = (aligned >= self._threshold).fillna(False)
        return mask.rename("cropland_mask")

    def _from_vector(self, path: Path) -> xr.DataArray:
        """Rasterise agricultural polygons onto the target grid."""
        gdf = gpd.read_file(path)
        if gdf.crs is not None and str(gdf.crs) != self._config.grid.crs:
            gdf = gdf.to_crs(self._config.grid.crs)

        lats = self._grid.lat_centers
        lons = self._grid.lon_centers
        res = self._grid.resolution_deg
        transform = from_origin(lons.min() - res / 2.0, lats.max() + res / 2.0, res, res)
        burned = features.rasterize(
            ((geom, 1) for geom in gdf.geometry if geom is not None),
            out_shape=self._grid.shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )
        mask = xr.DataArray(
            burned.astype(bool),
            dims=("lat", "lon"),
            coords={"lat": lats, "lon": lons},
            name="cropland_mask",
        )
        return mask

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    def apply(self, data: xr.Dataset | xr.DataArray, mask: xr.DataArray | None = None) -> xr.Dataset | xr.DataArray:
        """Mask non-cropland cells to NaN, aligning the mask to ``data``."""
        if mask is None:
            mask = self.build()
        mask = mask.reindex(lat=data["lat"], lon=data["lon"], method="nearest")
        return data.where(mask)

    def keep_cells(self, cell_table, mask: xr.DataArray | None = None):
        """Filter a grid cell table (see :meth:`TargetGrid.cell_table`).

        Returns only the rows whose ``(lat, lon)`` fall on an unmasked cell.
        """
        if mask is None:
            mask = self.build()
        flat = mask.values.ravel()
        keep = np.asarray(flat, dtype=bool)
        return cell_table.loc[keep].reset_index(drop=True)
