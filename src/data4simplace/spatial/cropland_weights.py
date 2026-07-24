"""PROBA-V cropland cover fraction loader and 250 m cropland filter.

The dominant soil-type workflow (see CLAUDE.md) restricts SoilGrids 250 m pixels
to cropland *before* selecting the dominant class and aggregating. Cropland
cover comes from the Copernicus PROBA-V LC100 100 m ``Crops-CoverFraction``
layer. This module loads that layer, resamples it onto a reference (250 m
SoilGrids) grid, and thresholds it into a boolean keep-mask.

Unlike :class:`~data4simplace.spatial.masking.CroplandMask` (a boolean *10 km*
cell filter built from CORINE or a local file, applied to final outputs), this
operates at the native 250 m soil resolution and yields per-pixel weights that
drive the dominant-type selection.
"""

from __future__ import annotations

import logging
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from data4simplace.config import PipelineConfig

logger = logging.getLogger(__name__)


class CroplandWeights:
    """Load PROBA-V cropland cover fraction and build a 250 m keep-mask.

    Parameters
    ----------
    config:
        The validated pipeline configuration. Uses
        ``paths.cropland_weights_path`` (the PROBA-V GeoTIFF) and
        ``soil.cropland_min_fraction`` (the keep threshold).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._path = config.paths.cropland_weights_path
        self._min_fraction = config.soil.cropland_min_fraction

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _bbox(self) -> tuple[float, float, float, float]:
        """Target bounding box (+1-cell halo) as ``(min_lon, min_lat, max_lon, max_lat)``."""
        g = self._config.grid
        halo = g.resolution_deg
        return (g.min_lon - halo, g.min_lat - halo, g.max_lon + halo, g.max_lat + halo)

    def load_fraction(self) -> xr.DataArray | None:
        """Load the PROBA-V cover fraction, subset to the grid bbox, as 0-1.

        Returns ``None`` when no ``cropland_weights_path`` is configured or the
        file is missing, so callers can fall back to keeping all pixels.
        """
        if self._path is None:
            logger.info("No cropland_weights_path set; cropland filter disabled")
            return None
        path = Path(self._path)
        if not path.is_file():
            logger.warning("Cropland weights file not found: %s", path)
            return None

        da = rioxarray.open_rasterio(path, masked=True, chunks={"x": 2048, "y": 2048})
        if isinstance(da, list):
            da = da[0]
        da = da.squeeze("band", drop=True) if "band" in da.dims else da

        # Subset to the region of interest before any resampling.
        min_lon, min_lat, max_lon, max_lat = self._bbox()
        da = da.rio.clip_box(min_lon, min_lat, max_lon, max_lat)

        # PROBA-V stores cover as a percentage (0-100); normalise to a fraction.
        # A dedicated 255 sentinel marks non-land / no-data and is masked above.
        if float(da.max()) > 1.5:
            da = da / 100.0
        return da.rename("crops_cover_fraction")

    # ------------------------------------------------------------------ #
    # Alignment + thresholding
    # ------------------------------------------------------------------ #
    def align_to(self, reference: xr.DataArray) -> xr.DataArray | None:
        """Resample the cover fraction onto ``reference``'s 250 m grid.

        The 100 m PROBA-V fraction is reprojected/aggregated to match the
        SoilGrids pixel grid so cover and soil pixels are cell-aligned.
        """
        fraction = self.load_fraction()
        if fraction is None:
            return None
        if fraction.rio.crs is None:
            fraction = fraction.rio.write_crs(self._config.grid.crs)
        aligned = fraction.rio.reproject_match(reference)
        return aligned.rename("crops_cover_fraction")

    def keep_mask(self, reference: xr.DataArray) -> xr.DataArray | None:
        """Boolean 250 m mask: True where cropland fraction >= the threshold.

        Aligned to ``reference``. Returns ``None`` when no cropland source is
        available, letting the caller keep every pixel.
        """
        aligned = self.align_to(reference)
        if aligned is None:
            return None
        mask = (aligned >= self._min_fraction).fillna(False)
        logger.info(
            "Cropland filter: keeping 250 m pixels with cover >= %.0f%%",
            self._min_fraction * 100.0,
        )
        return mask.rename("cropland_keep")
