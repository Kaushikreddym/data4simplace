"""Tests for configuration parsing and validation."""

from __future__ import annotations

import pytest
import yaml

from data4simplace.config import PipelineConfig, load_config


def test_valid_config(config_dict):
    cfg = PipelineConfig.model_validate(config_dict)
    assert cfg.flags.run_climate_processing is True
    assert cfg.grid.resolution_deg == 0.1
    assert cfg.climate.variables == {"TAS": "tas"}


def test_invalid_bounds(config_dict):
    config_dict["grid"]["min_lon"] = 15.0  # min >= max
    with pytest.raises(ValueError):
        PipelineConfig.model_validate(config_dict)


def test_end_before_start_is_allowed_at_config_but_flagged_later(config_dict):
    # Config itself does not enforce time ordering; the handler does.
    config_dict["time"] = {"start": "1979-01-05", "end": "1979-01-01"}
    cfg = PipelineConfig.model_validate(config_dict)
    assert cfg.time.start == "1979-01-05"


def test_extra_keys_rejected(config_dict):
    config_dict["unexpected"] = 1
    with pytest.raises(ValueError):
        PipelineConfig.model_validate(config_dict)


def test_load_config_roundtrip(tmp_path, config_dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict))
    cfg = load_config(path)
    assert isinstance(cfg, PipelineConfig)


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_statistics_defaults(config_dict):
    cfg = PipelineConfig.model_validate(config_dict)
    assert cfg.flags.write_soil_statistics is False
    assert cfg.soil.n_primary_classes == 3
    assert cfg.soil.export_statistic == "mean"
    assert cfg.soil.rootzone_bottom_cm == 100.0


@pytest.mark.parametrize(
    "key,value",
    [
        ("export_statistic", "mode"),      # only mean | median
        ("n_primary_classes", 0),          # at least the dominant class
        ("n_primary_classes", 13),         # at most the 12 USDA classes
        ("dominant_mode", "profile"),      # usda | usda_profile | wrb | none
        ("rootzone_bottom_cm", 0),         # must be a positive depth
    ],
)
def test_invalid_soil_settings_rejected(config_dict, key, value):
    config_dict.setdefault("soil", {})[key] = value
    with pytest.raises(ValueError):
        PipelineConfig.model_validate(config_dict)


def test_usda_profile_settings_accepted(config_dict):
    config_dict.setdefault("soil", {}).update(
        {"dominant_mode": "usda_profile", "export_statistic": "median",
         "n_primary_classes": 5, "rootzone_bottom_cm": 60}
    )
    cfg = PipelineConfig.model_validate(config_dict)
    assert cfg.soil.dominant_mode == "usda_profile"
    assert cfg.soil.export_statistic == "median"
    assert cfg.soil.n_primary_classes == 5
    assert cfg.soil.rootzone_bottom_cm == 60
