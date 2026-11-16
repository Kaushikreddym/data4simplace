"""Crop profiles: wheat reproduces the old constants, maize is self-consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cropmodelling4eu.crops import CROP_PROFILES, profile


def test_wheat_profile_reproduces_the_previous_constants():
    from cropmodelling4eu.calibration import config as cal
    from cropmodelling4eu.evaluation import config as ev

    wheat = CROP_PROFILES["wheat"]
    assert (wheat.clms_harvest_offset_days, wheat.clms_emergence_offset_days) == (18.7, 6.0)
    assert wheat.autumn_cut_doy == 180
    assert (wheat.cybench, wheat.clms, wheat.sage, wheat.gdhy) == (
        "wheat", "wheat_winter", "Wheat.Winter", "wheat_winter"
    )
    assert (wheat.yield_moisture_content, wheat.min_plausible_season_days) == (0.135, 150.0)
    assert wheat.sim_sowing_doy == 270 and wheat.simplace_block == "WW"
    # The default process runs wheat, so the module constants are its values.
    assert cal.CLMS_HARVEST_OFFSET_DAYS == 18.7 and cal.AUTUMN_CUT_DOY == 180
    assert ev.CLMS_CROP == "wheat_winter" and ev.CLMS_REQUIRE_SPLIT


def test_maize_profile_never_moves_a_stage():
    maize = profile("maize")
    assert maize.autumn_cut_doy > 366
    assert not maize.clms_require_split
    # CLMS - PEP725 was +4.6 d (harvest) and +16.4 d (emergence).
    assert (maize.clms_harvest_offset_days, maize.clms_emergence_offset_days) == (-4.6, 16.4)
    assert maize.simplace_block == "MAIZ" and maize.clms == "maize"


def test_unknown_crop_is_refused():
    with pytest.raises(ValueError, match="known crops"):
        profile("barley")


@pytest.mark.parametrize("name", sorted(CROP_PROFILES))
def test_stage_parameter_files_exist_and_parse(name):
    root = Path(__file__).parents[1] / "src/cropmodelling4eu/calibration"
    for file in CROP_PROFILES[name].stage_parameters.values():
        raw = yaml.safe_load((root / file).read_text())
        assert raw["parameters"], file


def test_season_crop_must_match_the_process_profile():
    from cropmodelling4eu.config import SeasonConfig

    with pytest.raises(ValueError, match="CM4EU_CROP"):
        SeasonConfig(crop="maize")


@pytest.mark.parametrize("name", sorted(CROP_PROFILES))
def test_default_spinup_never_crosses_new_year_before_sowing(name):
    # The torchcrop latch is `doy >= idpl`: a window opening in the previous
    # calendar year sows on day 0. --export-dir runs take this default.
    from cropmodelling4eu.export.weather import season_window, sowing_date

    crop = CROP_PROFILES[name]
    earliest = crop.sim_sowing_doy if crop.autumn_cut_doy <= 366 else 57
    start, _ = season_window(2010, earliest, crop.spinup_months)
    assert start.year == sowing_date(2010, earliest).year
    assert CROP_PROFILES["wheat"].spinup_months == 6
