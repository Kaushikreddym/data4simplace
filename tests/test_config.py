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


def test_top3_csv_export_requires_the_multiclass_method(config_dict):
    config_dict["flags"]["run_soil_processing"] = True
    config_dict["flags"]["export_top3_soil_csvs"] = True
    with pytest.raises(ValueError, match="aggregation_method"):
        PipelineConfig.model_validate(config_dict)

    config_dict["soil"] = {"aggregation_method": "top3"}
    assert PipelineConfig.model_validate(config_dict).flags.export_top3_soil_csvs


def test_top3_csv_export_requires_the_soil_stage(config_dict):
    config_dict["flags"]["export_top3_soil_csvs"] = True
    config_dict["flags"]["run_soil_processing"] = False
    config_dict["soil"] = {"aggregation_method": "top3"}
    with pytest.raises(ValueError, match="run_soil_processing"):
        PipelineConfig.model_validate(config_dict)


def test_multiclass_needs_a_classification(config_dict):
    config_dict["soil"] = {"aggregation_method": "top3", "dominant_mode": "none"}
    with pytest.raises(ValueError, match="dominant_mode"):
        PipelineConfig.model_validate(config_dict)


def test_aggregation_method_defaults_to_the_legacy_single_profile(config_dict):
    config = PipelineConfig.model_validate(config_dict)
    assert config.soil.aggregation_method == "dominant"
    assert config.flags.export_top3_soil_csvs is False


def test_npk_defaults_to_npkgrids_wheat(config_dict):
    config = PipelineConfig.model_validate(config_dict)
    assert config.npk.source == "npkgrids"
    assert config.npk.crop == "wheat"
    assert config.npk.simplace_crop == "winter_wheat"
    assert (config.npk.n_fertilizer, config.npk.p_fertilizer, config.npk.k_fertilizer) == (
        "KAS", "P", "K",
    )
    assert config.npk.include_zero_rate is False


def test_management_export_requires_the_npk_stage(config_dict):
    config_dict["flags"]["export_simplace_management"] = True
    config_dict["flags"]["run_npk_processing"] = False
    with pytest.raises(ValueError, match="run_npk_processing"):
        PipelineConfig.model_validate(config_dict)


def test_npkgrids_management_export_requires_a_root(config_dict):
    config_dict["flags"]["export_simplace_management"] = True
    config_dict["flags"]["run_npk_processing"] = True
    with pytest.raises(ValueError, match="npk_root"):
        PipelineConfig.model_validate(config_dict)

    config_dict["paths"]["npk_root"] = config_dict["paths"]["mswx_root"]
    assert PipelineConfig.model_validate(config_dict).flags.export_simplace_management


def test_npk_split_must_be_usable(config_dict):
    config_dict["npk"] = {"n_split": [0.5, -0.5]}
    with pytest.raises(ValueError, match="n_split"):
        PipelineConfig.model_validate(config_dict)

    config_dict["npk"] = {"n_split": []}
    with pytest.raises(ValueError, match="n_split"):
        PipelineConfig.model_validate(config_dict)

    config_dict["npk"] = {"n_split": [0.5, 0.25, 0.25]}
    assert PipelineConfig.model_validate(config_dict).npk.n_split == [0.5, 0.25, 0.25]
