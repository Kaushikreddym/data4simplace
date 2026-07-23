"""SoilGrids fetcher, scale-factor un-scaler and CRS transformer.

SoilGrids v2 provides 250 m global soil property layers in the Homolosine
projection (``EPSG:152160``). Values are stored as scaled integers; the
official conversion factors must be applied before any calculation. This module
loads layers (from a local root or the public WCS service), un-scales them,
reprojects to ``EPSG:4326`` and aggregates onto the 10 km target grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

# Official SoilGrids v2 conversion factors: mapped_value / factor -> unit.
# Reference: https://www.isric.org/explore/soilgrids/faq-soilgrids
SCALE_FACTORS: dict[str, float] = {
    "clay": 10.0,      # g/kg -> % (0.1 %)
    "silt": 10.0,      # g/kg -> %
    "sand": 10.0,      # g/kg -> %
    "bdod": 100.0,     # cg/cm3 -> kg/dm3
    "soc": 10.0,       # dg/kg -> g/kg
    "phh2o": 10.0,     # pH*10 -> pH
    "nitrogen": 100.0,  # cg/kg -> g/kg
    "cec": 10.0,       # mmol(c)/kg -> cmol(c)/kg
    "cfvo": 10.0,      # cm3/dm3 -> %
}

# WCS base per property (SoilGrids v2 Web Coverage Service, Homolosine grid).
_WCS_BASE = "https://maps.isric.org/mapserv?map=/map/{layer}.map"


class SoilGridsHandler:
    """Load, un-scale, reproject and regrid SoilGrids layers.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)
        self._soil = config.soil
        self._root = config.paths.soilgrids_root

    # ------------------------------------------------------------------ #
    # Loading a single (layer, depth) coverage
    # ------------------------------------------------------------------ #
    def _local_path(self, layer: str, depth: str) -> Path | None:
        """Resolve a local GeoTIFF for ``layer``/``depth`` if a root is set."""
        if self._root is None:
            return None
        # Accept both ``clay_0-5cm_mean.tif`` and ``clay/clay_0-5cm.tif`` layouts.
        candidates = [
            Path(self._root) / f"{layer}_{depth}_mean.tif",
            Path(self._root) / f"{layer}_{depth}.tif",
            Path(self._root) / layer / f"{layer}_{depth}_mean.tif",
            Path(self._root) / layer / f"{layer}_{depth}.tif",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    def _open_coverage(self, layer: str, depth: str) -> xr.DataArray | None:
        """Open one SoilGrids coverage from local disk (WCS fetch stubbed)."""
        path = self._local_path(layer, depth)
        if path is not None:
            da = rioxarray.open_rasterio(path, masked=True, chunks={"x": 2048, "y": 2048})
            if isinstance(da, list):  # multi-file datasets are not expected here
                da = da[0]
            da = da.squeeze("band", drop=True) if "band" in da.dims else da
            return da  # type: ignore[return-value]

        if self._root is not None:
            logger.warning("SoilGrids coverage not found locally: %s %s", layer, depth)
            return None

        # No local root: a live WCS fetch would go here. It is intentionally not
        # executed automatically to avoid unattended network calls in a pipeline
        # run; see ``fetch_wcs`` for the explicit opt-in entry point.
        logger.warning(
            "No soilgrids_root configured; skipping %s %s (use fetch_wcs to pull via WCS)",
            layer,
            depth,
        )
        return None

    def fetch_wcs(self, layer: str, depth: str) -> str:
        """Return the SoilGrids WCS coverage URL for a layer/depth.

        Provided as an explicit, documented hook. Network retrieval is left to
        the caller so that pipeline runs never make implicit outbound requests.
        """
        return (
            f"{_WCS_BASE.format(layer=layer)}&SERVICE=WCS&VERSION=2.0.1"
            f"&REQUEST=GetCoverage&COVERAGEID={layer}_{depth}_mean"
            f"&FORMAT=GEOTIFF_INT16&SUBSETTINGCRS={self._soil.homolosine_crs}"
        )

    # ------------------------------------------------------------------ #
    # Transformations
    # ------------------------------------------------------------------ #
    @staticmethod
    def unscale(da: xr.DataArray, layer: str) -> xr.DataArray:
        """Apply the official SoilGrids conversion factor for ``layer``."""
        factor = SCALE_FACTORS.get(layer)
        if factor is None:
            logger.warning("No scale factor known for layer %s; leaving unscaled", layer)
            return da
        return da / factor

    def reproject(self, da: xr.DataArray) -> xr.DataArray:
        """Reproject a Homolosine coverage to the target CRS (WGS84)."""
        if da.rio.crs is None:
            da = da.rio.write_crs(self._soil.homolosine_crs)
        return da.rio.reproject(self._soil.target_crs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load(self) -> xr.Dataset:
        """Load all configured layers as depth-resolved fields on the grid.

        Returns
        -------
        xarray.Dataset
            One variable per soil property with a ``depth`` dimension, all on
            the 10 km target grid in ``EPSG:4326``.
        """
        per_layer: dict[str, xr.DataArray] = {}

        for layer in self._soil.layers:
            depth_slices: list[xr.DataArray] = []
            loaded_depths: list[str] = []
            for depth in self._soil.depths:
                raw = self._open_coverage(layer, depth)
                if raw is None:
                    continue
                unscaled = self.unscale(raw, layer)
                wgs84 = self.reproject(unscaled)
                regridded = self._grid.regrid(wgs84, method="mean")
                depth_slices.append(regridded)
                loaded_depths.append(depth)

            if not depth_slices:
                logger.warning("No coverages loaded for layer %s", layer)
                continue

            stacked = xr.concat(depth_slices, dim="depth")
            stacked = stacked.assign_coords(depth=loaded_depths)
            per_layer[layer] = stacked
            logger.info("Loaded soil layer %s across %d depths", layer, len(loaded_depths))

        if not per_layer:
            raise RuntimeError(
                "No SoilGrids layers could be loaded. Set paths.soilgrids_root to "
                "a directory of GeoTIFF coverages or pre-fetch via fetch_wcs."
            )

        return xr.Dataset(per_layer)

    @staticmethod
    def normalise_texture(ds: xr.Dataset) -> xr.Dataset:
        """Rescale clay/silt/sand so the three fractions sum to 100 %."""
        needed = {"clay", "silt", "sand"}
        if not needed.issubset(ds.data_vars):
            return ds
        total = ds["clay"] + ds["silt"] + ds["sand"]
        total = total.where(total > 0)
        out = ds.copy()
        for frac in needed:
            out[frac] = (ds[frac] / total * 100.0).astype("float32")
        return out
