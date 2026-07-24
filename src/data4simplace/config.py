"""Configuration parsing and validation for the ``data4simplace`` pipeline.

The public entry point is :func:`load_config`, which reads a YAML file and
returns a validated :class:`PipelineConfig`. All downstream modules receive
this immutable, type-checked object rather than raw dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionFlags(BaseModel):
    """Boolean switches controlling which pipeline stages run."""

    model_config = ConfigDict(extra="forbid")

    run_climate_processing: bool = False
    run_soil_processing: bool = False
    compute_ptf: bool = False
    run_npk_processing: bool = False
    apply_agricultural_mask: bool = False
    export_simplace_weather: bool = False
    export_simplace_soil: bool = False
    export_simplace_management: bool = False


class GridConfig(BaseModel):
    """Definition of the unified target grid in geographic coordinates."""

    model_config = ConfigDict(extra="forbid")

    resolution_deg: float = Field(0.1, gt=0.0, description="Cell size in degrees (~10 km).")
    min_lon: float = Field(..., ge=-180.0, le=180.0)
    max_lon: float = Field(..., ge=-180.0, le=180.0)
    min_lat: float = Field(..., ge=-90.0, le=90.0)
    max_lat: float = Field(..., ge=-90.0, le=90.0)
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def _check_bounds(self) -> "GridConfig":
        if self.min_lon >= self.max_lon:
            raise ValueError("grid.min_lon must be < grid.max_lon")
        if self.min_lat >= self.max_lat:
            raise ValueError("grid.min_lat must be < grid.max_lat")
        return self


class TimeConfig(BaseModel):
    """Temporal window for climate processing (ISO ``YYYY-MM-DD``)."""

    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class PathsConfig(BaseModel):
    """Input data locations and the output directory."""

    model_config = ConfigDict(extra="forbid")

    mswx_root: Path
    soilgrids_root: Optional[Path] = None
    npk_root: Optional[Path] = None
    cropland_mask: Optional[Path] = None
    # PROBA-V LC100 Crops-CoverFraction GeoTIFF, used to filter SoilGrids 250 m
    # pixels to cropland before the dominant-soil-type aggregation.
    cropland_weights_path: Optional[Path] = None
    output_dir: Path = Path("./output")


class ReferenceConfig(BaseModel):
    """Locations of the SIMPLACE reference files (output structure source)."""

    model_config = ConfigDict(extra="forbid")

    weather_dir: Optional[Path] = None
    soil_dir: Optional[Path] = None
    management_file: Optional[Path] = None


class ClimateConfig(BaseModel):
    """MSWX variable mapping and dask chunk sizes."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, str] = Field(default_factory=dict)
    chunks: dict[str, int] = Field(default_factory=lambda: {"time": 30, "lat": 512, "lon": 512})


class SoilConfig(BaseModel):
    """SoilGrids layer, depth, projection and WCS-fetch configuration."""

    model_config = ConfigDict(extra="forbid")

    layers: list[str] = Field(
        default_factory=lambda: ["clay", "silt", "sand", "bdod", "soc", "phh2o", "nitrogen"]
    )
    depths: list[str] = Field(
        default_factory=lambda: [
            "0-5cm",
            "5-15cm",
            "15-30cm",
            "30-60cm",
            "60-100cm",
            "100-200cm",
        ]
    )
    homolosine_crs: str = "EPSG:152160"
    target_crs: str = "EPSG:4326"

    # Dominant soil-type aggregation (see CLAUDE.md workflow). ``none`` keeps the
    # legacy plain-mean regrid; the others filter 250 m pixels to cropland, pick
    # the dominant class per target cell and aggregate only those pixels:
    #   usda          - 12 USDA texture classes of the topsoil layer
    #   usda_profile  - 12 x 12 composite (topsoil, rooting-zone) texture classes
    #   wrb           - SoilGrids WRB Reference Soil Group
    dominant_mode: Literal["usda", "usda_profile", "wrb", "none"] = "usda"
    # Bottom of the rooting zone (cm) whose thickness-weighted texture gives the
    # second key of the ``usda_profile`` composite class; its top is the bottom
    # of the topsoil layer (5 cm for the standard SoilGrids depths).
    rootzone_bottom_cm: float = Field(100.0, gt=0.0)
    # Minimum PROBA-V cropland cover fraction (0-1) for a 250 m pixel to be kept.
    cropland_min_fraction: float = Field(0.8, ge=0.0, le=1.0)
    # WRB Reference Soil Group coverage id (SoilGrids ``wrb`` map) for ``wrb`` mode.
    wrb_layer: str = "MostProbable"
    # Fill target cells left empty (coastal/islands) by nearest-neighbour search.
    fill_missing: bool = True

    # WCS fallback: fetch coverages from the ISRIC service when a local tile is
    # not found under ``paths.soilgrids_root``.
    use_wcs: bool = True
    wcs_stat: str = "mean"
    wcs_crs: str = "EPSG:4326"  # request CRS; 4326 fetches directly in lon/lat
    wcs_cache_dir: Optional[Path] = None
    wcs_timeout: int = 180


class MaskConfig(BaseModel):
    """Agricultural (cropland) mask source and CORINE fetch configuration."""

    model_config = ConfigDict(extra="forbid")

    # ``auto`` uses paths.cropland_mask when set, else falls back to CORINE.
    source: str = "auto"  # auto | file | corine | none
    threshold: float = Field(0.5, ge=0.0, le=1.0)

    # CORINE Land Cover (Copernicus/EEA) WMS settings.
    corine_year: int = 2018
    corine_layer: str = "12"  # WMS layer id of the mainland CLC raster
    corine_wms: str = (
        "https://image.discomap.eea.europa.eu/arcgis/services/"
        "Corine/CLC{year}_WM/MapServer/WMSServer"
    )
    resolution_m: float = 100.0
    max_pixels: int = 2048
    agricultural_codes: list[int] = Field(
        default_factory=lambda: [211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244]
    )
    cache_dir: Optional[Path] = None
    timeout: int = 120


class PipelineConfig(BaseModel):
    """Fully validated configuration for a pipeline run."""

    model_config = ConfigDict(extra="forbid")

    flags: ExecutionFlags
    grid: GridConfig
    time: TimeConfig
    paths: PathsConfig
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    soil: SoilConfig = Field(default_factory=SoilConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    missing_value: float = -99.0


def load_config(path: str | Path) -> PipelineConfig:
    """Read and validate a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        A validated, immutable configuration object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the YAML is malformed or fails schema validation.
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping, got {type(raw).__name__}")

    return PipelineConfig.model_validate(raw)
