"""Spatial utilities: PROBA-V cropland masks (250 m pixels, 10 km cells) and
latitude-aware surface areas."""

from data4simplace.spatial.area import (
    EARTH_RADIUS_KM,
    HECTARES_PER_KM2,
    latitude_band_area_km2,
    pixel_area_km2,
)
from data4simplace.spatial.cropland_weights import (
    CroplandWeights,
    apply_cell_mask,
    export_cell_mask,
    keep_cells,
)

__all__ = [
    "CroplandWeights",
    "EARTH_RADIUS_KM",
    "HECTARES_PER_KM2",
    "apply_cell_mask",
    "export_cell_mask",
    "keep_cells",
    "latitude_band_area_km2",
    "pixel_area_km2",
]
