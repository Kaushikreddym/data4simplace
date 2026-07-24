"""CORINE Land Cover fetcher for building agricultural masks.

CORINE Land Cover (CLC, Copernicus/EEA) classifies European land into ~44
classes at 100 m. The agricultural classes are CLC codes 211-244. The EEA
DiscoMap service exposes CLC only as a symbolised WMS raster (no WCS / class
values), so this module fetches the standard-symbology PNG for the grid bounding
box and reclassifies pixels back to agricultural / non-agricultural by matching
the official CLC colour palette (the WMS renders it without anti-aliasing, so
colours map exactly to classes).

The result is an agricultural-fraction field on the target grid: for each 10 km
cell, the share of underlying CLC pixels that are agricultural.
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from PIL import Image

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid

logger = logging.getLogger(__name__)

# Official CLC class -> RGB colour (agricultural classes 211-244).
CLC_COLORS: dict[int, tuple[int, int, int]] = {
    211: (255, 255, 168),  # Non-irrigated arable land
    212: (255, 255, 0),    # Permanently irrigated land
    213: (230, 230, 0),    # Rice fields
    221: (230, 128, 0),    # Vineyards
    222: (242, 166, 77),   # Fruit trees and berry plantations
    223: (230, 166, 0),    # Olive groves
    231: (230, 230, 77),   # Pastures
    241: (255, 230, 166),  # Annual crops associated with permanent crops
    242: (255, 230, 77),   # Complex cultivation patterns
    243: (230, 204, 77),   # Agriculture with significant natural vegetation
    244: (242, 204, 166),  # Agro-forestry areas
}


class CorineLandCover:
    """Fetch CORINE Land Cover and derive an agricultural-fraction field.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._mask = config.mask
        self._grid = TargetGrid.from_config(config.grid)
        self._cache_dir = (
            Path(self._mask.cache_dir)
            if self._mask.cache_dir is not None
            else Path(config.paths.output_dir) / "_corine_cache"
        )

    # ------------------------------------------------------------------ #
    # Request sizing
    # ------------------------------------------------------------------ #
    def _pixel_size(self) -> tuple[int, int]:
        """Image width/height (px) for the bbox at the target resolution.

        Sizing keeps pixels near ``resolution_m`` metres but caps each axis at
        ``max_pixels`` (coarsening if needed). Staying at/above ~100 m also keeps
        the request within the CLC raster layer's visible scale range.
        """
        g = self._config.grid
        mid_lat = math.radians((g.min_lat + g.max_lat) / 2.0)
        res_deg_lat = self._mask.resolution_m / 111_320.0
        res_deg_lon = self._mask.resolution_m / (111_320.0 * max(math.cos(mid_lat), 1e-6))
        width = math.ceil((g.max_lon - g.min_lon) / res_deg_lon)
        height = math.ceil((g.max_lat - g.min_lat) / res_deg_lat)
        cap = self._mask.max_pixels
        if max(width, height) > cap:
            scale = cap / max(width, height)
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))
            logger.info("CORINE request coarsened to %dx%d px to fit max_pixels", width, height)
        return width, height

    # ------------------------------------------------------------------ #
    # Fetch
    # ------------------------------------------------------------------ #
    def _fetch_png(self, width: int, height: int) -> bytes | None:
        """GET the CLC WMS GetMap PNG for the grid bounding box."""
        g = self._config.grid
        wms = self._mask.corine_wms.format(year=self._mask.corine_year)
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": self._mask.corine_layer,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": f"{g.min_lon},{g.min_lat},{g.max_lon},{g.max_lat}",
            "WIDTH": str(width),
            "HEIGHT": str(height),
            "FORMAT": "image/png",
        }
        try:
            resp = requests.get(wms, params=params, timeout=self._mask.timeout)
            resp.raise_for_status()
            if "png" not in resp.headers.get("Content-Type", ""):
                logger.warning("CORINE WMS returned non-image response: %s", resp.text[:300])
                return None
            return resp.content
        except requests.RequestException as exc:
            logger.warning("CORINE WMS fetch failed: %s", exc)
            return None

    def _cached_png(self, width: int, height: int) -> bytes | None:
        """Return the CLC PNG for the bbox, using an on-disk cache."""
        g = self._config.grid
        key = (
            f"clc{self._mask.corine_year}_{g.min_lon:.3f}_{g.min_lat:.3f}"
            f"_{g.max_lon:.3f}_{g.max_lat:.3f}_{width}x{height}.png"
        )
        cache_file = self._cache_dir / key
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            logger.debug("Using cached CORINE tile %s", cache_file)
            return cache_file.read_bytes()

        content = self._fetch_png(width, height)
        if content is None:
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(content)
        return content

    # ------------------------------------------------------------------ #
    # Reclassify
    # ------------------------------------------------------------------ #
    def _agricultural_array(self, png: bytes, width: int, height: int) -> xr.DataArray:
        """Reclassify a CLC PNG into an agricultural boolean field on lon/lat."""
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        agri = np.zeros(rgb.shape[:2], dtype=bool)
        for code in self._mask.agricultural_codes:
            color = CLC_COLORS.get(code)
            if color is None:
                logger.warning("No CLC colour known for agricultural code %s", code)
                continue
            agri |= np.all(rgb == np.array(color, dtype=rgb.dtype), axis=-1)

        g = self._config.grid
        # Pixel centres; row 0 is the northern edge (max_lat).
        lon = np.linspace(g.min_lon, g.max_lon, width, endpoint=False) + (g.max_lon - g.min_lon) / width / 2
        lat = np.linspace(g.max_lat, g.min_lat, height, endpoint=False) - (g.max_lat - g.min_lat) / height / 2
        return xr.DataArray(
            agri.astype("float32"),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
            name="agricultural",
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def agricultural_fraction(self) -> xr.DataArray | None:
        """Agricultural fraction (0-1) per target cell, or ``None`` on failure."""
        width, height = self._pixel_size()
        png = self._cached_png(width, height)
        if png is None:
            return None
        fine = self._agricultural_array(png, width, height)
        fraction = self._grid.regrid(fine, method="mean")
        return fraction.rename("agricultural_fraction")
