"""Configuration parsing and validation for the ``data4simplace`` pipeline.

The public entry point is :func:`load_config`, which reads a YAML file and
returns a validated :class:`PipelineConfig`. All downstream modules receive
this immutable, type-checked object rather than raw dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
    """SoilGrids layer, depth and projection configuration."""

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
