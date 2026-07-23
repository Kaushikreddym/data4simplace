"""data4simplace — climate, soil and nutrient data preparation for SIMPLACE.

The package ingests MSWX climate, SoilGrids soil and NPK nutrient datasets,
harmonises them onto a unified 10 km grid, and exports point-based CSV files
that conform to the SIMPLACE crop-model input conventions.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from data4simplace.config import PipelineConfig, load_config

try:
    __version__ = version("data4simplace")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["PipelineConfig", "load_config", "__version__"]
