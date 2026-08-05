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
         "bdod": layer(1.4), "soc": layer(15.0), "phh2o": layer(6.5), "nitrogen": layer(1.2),
         # Volumetric water contents in vol%, as un-scaled by SoilGridsHandler.
         "wv0010": layer(42.0), "wv0033": layer(31.0), "wv1500": layer(14.0)},
        coords={"depth": depth, "lat": lat, "lon": lon},
    )


@pytest.fixture()
def top_classes(soil_dataset):
    """A three-rank :class:`TopClassAggregation` over the 2x2 soil fixture.

    Cell (0, 0) holds three classes (60/30/10 % of its area); cell (1, 1) is
    uniform, so it has nothing at ranks 2 and 3.
    """
    from data4simplace.soil.multiclass import TopClassAggregation

    ranks = np.array([1, 2, 3])
    # Each rank carries a different texture, so a test can tell them apart.
    properties = xr.concat(
        [soil_dataset + offset for offset in (0.0, 15.0, 30.0)], dim="rank"
    ).assign_coords(rank=ranks)

    def grid(values):
        return (("rank", "lat", "lon"), np.asarray(values, dtype="float64"))

    classes = xr.Dataset(
        {
            "class_code": (
                ("rank", "lat", "lon"),
                np.array(
                    [[[7, 7], [7, 10]], [[10, 0], [0, 0]], [[12, 0], [0, 0]]],
                    dtype="int16",
                ),
            ),
            "pixels": (
                ("rank", "lat", "lon"),
                np.array(
                    [[[6, 6], [6, 10]], [[3, 0], [0, 0]], [[1, 0], [0, 0]]],
                    dtype="int32",
                ),
            ),
            "share_percent": grid(
                [[[60, 60], [60, 100]], [[30, np.nan], [np.nan, np.nan]],
                 [[10, np.nan], [np.nan, np.nan]]]
            ),
            "area_km2": grid(
                [[[60, 60], [60, 100]], [[30, np.nan], [np.nan, np.nan]],
                 [[10, np.nan], [np.nan, np.nan]]]
            ),
            "area_fraction": grid(
                [[[0.6, 0.6], [0.6, 1.0]], [[0.3, np.nan], [np.nan, np.nan]],
                 [[0.1, np.nan], [np.nan, np.nan]]]
            ),
        },
        coords={"rank": ranks, "lat": soil_dataset["lat"], "lon": soil_dataset["lon"]},
    )

    uncertainty = xr.Dataset(
        {
            "n_classes": (("lat", "lon"), np.array([[3, 1], [1, 1]], dtype="int16")),
            "cropland_pixels": (
                ("lat", "lon"), np.array([[10, 6], [6, 10]], dtype="int32")
            ),
            "total_area_km2": (("lat", "lon"), np.array([[100.0, 60.0], [60.0, 100.0]])),
            "dominance_ratio": (("lat", "lon"), np.array([[0.4, 0.0], [0.0, 0.0]])),
            "shannon_entropy": (("lat", "lon"), np.array([[0.83, 0.0], [0.0, 0.0]])),
        },
        coords={"lat": soil_dataset["lat"], "lon": soil_dataset["lon"]},
    )

    return TopClassAggregation(
        properties=properties, classes=classes, uncertainty=uncertainty, mode="usda"
    )
