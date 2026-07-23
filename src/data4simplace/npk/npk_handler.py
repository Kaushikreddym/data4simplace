"""NPK / fertilizer dataset aligner.

Aligns global nutrient application datasets (nitrogen, phosphorus, potassium
application rates) onto the unified 10 km target grid. Sources are typically
coarse global rasters (e.g. gridded fertilizer application products), so
alignment is a resample/regrid operation onto the target cells.
"""

from __future__ import annotations

import logging
from pathlib import Path

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

# Filename stems recognised for each nutrient, in priority order.
_NUTRIENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "N": ("nitrogen", "n_app", "napp", "n_"),
    "P": ("phosphorus", "p_app", "papp", "p_"),
    "K": ("potassium", "k_app", "kapp", "k_"),
}


class NPKHandler:
    """Align NPK application rasters onto the 10 km target grid.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)
        self._root = config.paths.npk_root

    def _discover(self) -> dict[str, Path]:
        """Match nutrient rasters within the configured NPK root."""
        if self._root is None:
            return {}
        root = Path(self._root)
        if not root.is_dir():
            logger.warning("NPK root is not a directory: %s", root)
            return {}

        rasters = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".nc"})
        found: dict[str, Path] = {}
        for nutrient, patterns in _NUTRIENT_PATTERNS.items():
            for path in rasters:
                name = path.name.lower()
                if any(pat in name for pat in patterns):
                    found[nutrient] = path
                    break
        return found

    def _open(self, path: Path) -> xr.DataArray:
        """Open a nutrient raster (GeoTIFF or netCDF) as a single-band array."""
        if path.suffix.lower() == ".nc":
            ds = xr.open_dataset(path, chunks={})
            da = ds[next(iter(ds.data_vars))]
        else:
            da = rioxarray.open_rasterio(path, masked=True, chunks={"x": 2048, "y": 2048})
            if isinstance(da, list):
                da = da[0]
            da = da.squeeze("band", drop=True) if "band" in da.dims else da
        return da  # type: ignore[return-value]

    def load(self) -> xr.Dataset:
        """Load and regrid all discovered nutrient layers.

        Returns
        -------
        xarray.Dataset
            Variables ``N``, ``P``, ``K`` (whichever are found) on the 10 km
            grid. Empty dataset if no NPK root / rasters are configured.
        """
        discovered = self._discover()
        if not discovered:
            logger.warning("No NPK rasters discovered; returning empty dataset")
            return xr.Dataset()

        layers: dict[str, xr.DataArray] = {}
        for nutrient, path in discovered.items():
            da = self._open(path)
            if da.rio.crs is None:
                da = da.rio.write_crs(self._config.grid.crs)
            if str(da.rio.crs) != self._config.grid.crs:
                da = da.rio.reproject(self._config.grid.crs)
            layers[nutrient] = self._grid.regrid(da, method="mean")
            logger.info("Aligned NPK nutrient %s from %s", nutrient, path.name)

        return xr.Dataset(layers)
