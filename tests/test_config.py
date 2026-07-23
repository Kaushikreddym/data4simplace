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
