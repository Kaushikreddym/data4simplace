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
#: Candidate ``(delimiter, id column)`` pairs of an exported soil CSV. The wide
#: file keys on ``location``; the long dialects declare their own key and
#: divider, so they are read from the single definition in
#: :mod:`data4simplace.exporters.layout` rather than restated here.
def _soil_id_candidates() -> list[tuple[str, str]]:
    """``(delimiter, key column)`` pairs an exported soil CSV may use."""
    from data4simplace.exporters.layout import SOIL_DIALECTS

    seen = {(",", "location")}
    for dialect in SOIL_DIALECTS.values():
        seen.add((dialect.delimiter, dialect.key_column))
    return sorted(seen)


def _read_soil_cell_ids(path: Path) -> np.ndarray | None:
    """``SimplaceID``s of the rows of an exported soil CSV, or ``None``.

    Reads the one identifier column, never the ~150 MB of profile values beside
    it. Returns ``None`` when no candidate delimiter/key pair matches the
    header, which is the honest outcome for a file written by a dialect this
    build does not know.
    """
    header = path.open().readline()
    for delimiter, key in _soil_id_candidates():
        if key in header.rstrip("\r\n").split(delimiter):
            ids = pd.read_csv(path, sep=delimiter, usecols=[key])[key]
            return np.unique(ids.to_numpy(dtype=np.int64))
    logger.warning("%s: no known id column in its header; ignored", path.name)
    return None


def exported_soil_cells(output_dir: Path, grid: TargetGrid) -> xr.DataArray | None:
    """The cells an *already written* soil export covers, as a ``(lat, lon)`` mask.

    The on-disk counterpart of :func:`~data4simplace.soil.classify.valid_soil_cells`,
    for a run whose flags do not include the soil stage. ``SimplaceID`` is a
    1-based row-major index over the whole target grid
    (:meth:`~data4simplace.grid.TargetGrid.cell_table`), so the exported
    identifiers unflatten straight back into a mask.

    Parameters
    ----------
    output_dir:
        A run's ``paths.output_dir``, holding ``soil/soil.csv`` and/or
        ``soil/soil_long.csv``.
    grid:
        The target grid the mask is built on. Must be the grid the export was
        written from.

    Returns
    -------
    xarray.DataArray | None
        The mask, or ``None`` when no readable soil export is present.

    Raises
    ------
    ValueError
        If an identifier falls outside the grid. That means the export was
        written from a different grid definition, and every identifier in it is
        then meaningless -- silently intersecting with a shifted cell set would
        produce a plausible file describing the wrong places.
    """
    soil_dir = Path(output_dir) / "soil"
    for name in ("soil.csv", "soil_long.csv"):
        path = soil_dir / name
        if not path.is_file():
            continue
        ids = _read_soil_cell_ids(path)
        if ids is None or ids.size == 0:
            continue

        n_lat, n_lon = grid.shape
        flat = np.zeros(n_lat * n_lon, dtype=bool)
        index = ids - 1
        if index.min() < 0 or index.max() >= flat.size:
            raise ValueError(
                f"{path} carries SimplaceID {ids.min()}-{ids.max()}, outside the "
                f"{flat.size}-cell target grid: it was written from a different "
                "grid definition"
            )
        flat[index] = True

        logger.info(
            "Exported cell set read from %s: %d cells", path, int(flat.sum())
        )
        return xr.DataArray(
            flat.reshape(n_lat, n_lon),
            dims=("lat", "lon"),
            coords={"lat": grid.lat_centers, "lon": grid.lon_centers},
            name="exported_soil_cells",
        )
    return None


def export_cell_mask(
    config: PipelineConfig,
    grid: TargetGrid,
    soil: xr.Dataset | None = None,
    soil_export_fallback: bool = True,
) -> xr.DataArray | None:
    """The cells a run exports, as a boolean ``(lat, lon)`` mask.

    Two independent conditions, intersected:

    * **Cropland** (``flags.apply_agricultural_mask``): the cell holds
      ``soil.min_cropland_pixels`` PROBA-V pixels at or above
      ``soil.cropland_min_fraction`` cover.
    * **Valid soil**: the cell carries soil values. Weather and management files
      are pointless for a cell SIMPLACE has no soil profile for, so the soil
      result is what defines the exported cell set.

    The valid-soil condition is taken from ``soil`` when the soil stage ran, and
    otherwise from the soil export already in ``paths.output_dir``. **A partial
    run must not widen the cell set.** ``submit/management.sh`` runs the NPK and
    management stages alone, with ``run_soil_processing`` off; without the
    on-disk fallback that run keeps every cropland cell, and the schedule then
    covers cells that have no soil profile and no site row -- which is exactly
    what the 2026-09-02 EU run produced (72 290 scheduled locations against
    70 705 exported ones). The contract that weather, soil, site and management
    cover *the same* cells is resolved here, so it has to hold whichever stages
    a given job runs.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    grid:
        The target grid the mask is built on.
    soil:
        The processed soil dataset, when the soil stage ran. ``None`` falls back
        to the existing export, and to no soil filtering when there is none.
    soil_export_fallback:
        Allow that fallback. Only a caller holding the **whole** target grid may
        use it: ``SimplaceID`` indexes the full grid, so the identifiers in an
        export cannot be unflattened onto a sub-grid. ``data4simplace.tiling``
        passes ``False`` for exactly that reason.

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

    has_soil: xr.DataArray | None = None
    if soil is not None:
        has_soil = valid_soil_cells(soil)
        if has_soil is None:
            logger.warning("Soil dataset carries no lat/lon fields; no soil filtering")
    elif soil_export_fallback:
        has_soil = exported_soil_cells(config.paths.output_dir, grid)
        if has_soil is None:
            logger.warning(
                "The soil stage did not run and %s holds no soil export: the "
                "exported cell set is cropland only, so this run may cover cells "
                "that have no soil profile",
                config.paths.output_dir,
            )

    if has_soil is not None:
        if mask is None:
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
