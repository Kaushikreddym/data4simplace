"""PROBA-V cropland cover: the single cropland definition of the pipeline.

Cropland cover comes from the Copernicus PROBA-V LC100 100 m
``Crops-CoverFraction`` layer, and this module derives both cropland filters the
pipeline applies, from that one source and one threshold
(``soil.cropland_min_fraction``):

* :meth:`CroplandWeights.keep_mask` - a **250 m** boolean pixel mask. The
  dominant soil-type workflow (see CLAUDE.md) restricts SoilGrids pixels with it
  *before* selecting the dominant class and aggregating, so it decides the
  *value* a target cell carries.
* :meth:`CroplandWeights.cell_mask` - a **10 km** boolean target-cell mask: a
  cell is kept when it holds at least ``soil.min_cropland_pixels`` qualifying
  100 m pixels. It decides which cells are *exported at all*, and so is applied
  to every product (climate, soil, hydraulics, NPK) and to the cell table.

Both return ``None`` when no cropland raster is configured, letting callers fall
back to keeping everything.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)


class CroplandWeights:
    """Load PROBA-V cropland cover and build the 250 m and 10 km cropland masks.

    Parameters
    ----------
    config:
        The validated pipeline configuration. Uses
        ``paths.cropland_weights_path`` (the PROBA-V GeoTIFF),
        ``soil.cropland_min_fraction`` (the per-pixel keep threshold) and
        ``soil.min_cropland_pixels`` (how many such pixels a target cell needs).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._path = config.paths.cropland_weights_path
        self._min_fraction = config.soil.cropland_min_fraction
        self._min_pixels = config.soil.min_cropland_pixels
        # The bbox subset is read once per instance and reused by both masks.
        self._fraction: xr.DataArray | None = None
        self._loaded = False

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

        Cached per instance, so building both the 250 m pixel mask and the 10 km
        cell mask reads the raster only once.

        Returns ``None`` when no ``cropland_weights_path`` is configured or the
        file is missing, so callers can fall back to keeping all pixels.
        """
        if self._loaded:
            return self._fraction
        self._loaded = True
        self._fraction = self._load_fraction()
        return self._fraction

    def _load_fraction(self) -> xr.DataArray | None:
        """Read and normalise the cover fraction (see :meth:`load_fraction`)."""
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

        percent = self._stores_percent(da)

        # Subset to the region of interest before any resampling.
        min_lon, min_lat, max_lon, max_lat = self._bbox()
        da = da.rio.clip_box(min_lon, min_lat, max_lon, max_lat)

        # PROBA-V stores cover as a percentage (0-100); normalise to a fraction.
        # A dedicated 255 sentinel marks non-land / no-data and is masked above.
        if percent:
            da = da / 100.0
        return da.rename("crops_cover_fraction")

    @staticmethod
    def _stores_percent(da: xr.DataArray) -> bool:
        """Whether the cover layer is a percentage (0-100) rather than a fraction.

        Decided from the raster's **on-disk dtype**, not from the values present:
        the layer is read one bbox subset at a time, and a sparse subset whose
        peak cover is 1% would look indistinguishable from a 0-1 fraction. An
        integer layer (PROBA-V ships ``uint8`` with a 255 no-data sentinel) is
        always a percentage; a float layer falls back to the value range, which
        is only reached for non-PROBA-V inputs.
        """
        dtype = da.encoding.get("dtype", da.dtype)
        if np.issubdtype(np.dtype(dtype), np.integer):
            return True
        peak = float(da.max())  # float layer: the range is the only signal left
        return not np.isfinite(peak) or peak > 1.5

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

    # ------------------------------------------------------------------ #
    # Target-cell filter
    # ------------------------------------------------------------------ #
    def cell_mask(self, grid: TargetGrid) -> xr.DataArray | None:
        """Boolean ``(lat, lon)`` target-cell mask: True = export the cell.

        A cell is kept when at least ``soil.min_cropland_pixels`` of the native
        100 m PROBA-V pixels falling inside it reach ``cropland_min_fraction``
        cover -- the same source and threshold as the 250 m pixel filter, so the
        exported cell set and the aggregated values agree on what cropland is.

        Counted on the native PROBA-V grid rather than on the aligned 250 m
        coverages, so the filter is available even when the soil stage is off.

        Parameters
        ----------
        grid:
            The target 10 km grid.

        Returns
        -------
        xarray.DataArray | None
            The cell mask, or ``None`` when no cropland raster is configured
            (the caller then keeps every cell).
        """
        from data4simplace.soil.classify import cell_assignment

        fraction = self.load_fraction()
        if fraction is None:
            return None

        qualifying = (fraction >= self._min_fraction).fillna(False)
        n_lat, n_lon = grid.shape
        cell, values = cell_assignment(qualifying.astype("float32"), grid)
        counts = np.bincount(cell, weights=values, minlength=n_lat * n_lon)

        mask = xr.DataArray(
            counts.reshape(n_lat, n_lon) >= self._min_pixels,
            dims=("lat", "lon"),
            coords={"lat": grid.lat_centers, "lon": grid.lon_centers},
            name="cropland_cells",
        )
        logger.info(
            "Cropland cell filter: %d/%d target cells hold >= %d pixel(s) with "
            "cover >= %.0f%%",
            int(mask.values.sum()), mask.size, self._min_pixels,
            self._min_fraction * 100.0,
        )
        return mask


# ---------------------------------------------------------------------------- #
# Resolving the exported cell set
# ---------------------------------------------------------------------------- #
def export_cell_mask(
    config: PipelineConfig, grid: TargetGrid, soil: xr.Dataset | None = None
) -> xr.DataArray | None:
    """The cells a run exports, as a boolean ``(lat, lon)`` mask.

    Two independent conditions, intersected:

    * **Cropland** (``flags.apply_agricultural_mask``): the cell holds
      ``soil.min_cropland_pixels`` PROBA-V pixels at or above
      ``soil.cropland_min_fraction`` cover.
    * **Valid soil** (whenever the soil stage ran): the cell carries soil values.
      Weather and management files are pointless for a cell SIMPLACE has no soil
      profile for, so the soil result is what defines the exported cell set.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    grid:
        The target grid the mask is built on.
    soil:
        The processed soil dataset, when the soil stage ran. ``None`` skips the
        valid-soil condition (nothing is known about coverage).

    Returns
    -------
    xarray.DataArray | None
        The mask, or ``None`` when neither condition applies (keep every cell).
    """
    from data4simplace.soil.classify import valid_soil_cells

    mask: xr.DataArray | None = None

    if config.flags.apply_agricultural_mask:
        mask = CroplandWeights(config).cell_mask(grid)
        if mask is None:
            logger.warning(
                "apply_agricultural_mask set but no paths.cropland_weights_path; "
                "no cropland filtering"
            )

    if soil is not None:
        has_soil = valid_soil_cells(soil)
        if has_soil is None:
            logger.warning("Soil dataset carries no lat/lon fields; no soil filtering")
        elif mask is None:
            mask = has_soil
        else:
            # Both come from the same TargetGrid, but align defensively: an
            # inner join on mismatched float coords would silently empty the run.
            mask = mask & has_soil.reindex_like(mask, method="nearest")

    if mask is not None:
        logger.info("Exporting %d/%d target cells", int(mask.values.sum()), mask.size)
    return mask


# ---------------------------------------------------------------------------- #
# Applying a cell mask
# ---------------------------------------------------------------------------- #
def apply_cell_mask(
    data: xr.Dataset | xr.DataArray, mask: xr.DataArray | None
) -> xr.Dataset | xr.DataArray:
    """Set non-cropland cells to NaN, aligning ``mask`` to ``data``'s grid.

    A ``None`` mask (no cropland raster configured) returns ``data`` unchanged.
    """
    if mask is None:
        return data
    mask = mask.reindex(lat=data["lat"], lon=data["lon"], method="nearest")
    return data.where(mask)


def keep_cells(cell_table: pd.DataFrame, mask: xr.DataArray | None) -> pd.DataFrame:
    """Keep only the cell-table rows whose ``(lat, lon)`` cell is cropland.

    The table is in row-major target-grid order (see
    :meth:`~data4simplace.grid.TargetGrid.cell_table`), so the flattened mask
    indexes it directly. A ``None`` mask returns the table unchanged.
    """
    if mask is None:
        return cell_table
    keep = np.asarray(mask.values.ravel(), dtype=bool)
    return cell_table.loc[keep].reset_index(drop=True)
