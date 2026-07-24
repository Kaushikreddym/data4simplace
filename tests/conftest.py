"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig


@pytest.fixture()
def config_dict(tmp_path) -> dict:
    """A minimal but complete configuration dictionary."""
    return {
        "flags": {
            "run_climate_processing": True,
            "run_soil_processing": False,
            "compute_ptf": False,
            "run_npk_processing": False,
            "apply_agricultural_mask": False,
            "export_simplace_weather": True,
            "export_simplace_soil": False,
            "export_simplace_management": False,
        },
        "grid": {
            "resolution_deg": 0.1,
            "min_lon": 11.2,
            "max_lon": 11.5,
            "min_lat": 52.0,
            "max_lat": 52.3,
            "crs": "EPSG:4326",
        },
        "time": {"start": "1979-01-01", "end": "1979-01-03"},
        "paths": {
            "mswx_root": "/data01/FDS/muduchuru/Atmos/MSWX",
            "output_dir": str(tmp_path / "output"),
        },
        "reference": {},
        "climate": {"variables": {"TAS": "tas"}},
        "missing_value": -99,
    }


@pytest.fixture()
def config(config_dict) -> PipelineConfig:
    return PipelineConfig.model_validate(config_dict)


@pytest.fixture()
def soil_dataset() -> xr.Dataset:
    """A depth-resolved soil dataset on a 2x2 grid with SoilGrids depths."""
    lat = np.array([52.25, 52.15])
    lon = np.array([11.25, 11.35])
    depth = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
    shape = (len(depth), len(lat), len(lon))

    def layer(value):
        return (("depth", "lat", "lon"), np.full(shape, value, dtype="float32"))

    return xr.Dataset(
        {"clay": layer(20.0), "silt": layer(40.0), "sand": layer(40.0),
         "bdod": layer(1.4), "soc": layer(15.0), "phh2o": layer(6.5), "nitrogen": layer(1.2)},
        coords={"depth": depth, "lat": lat, "lon": lon},
    )
