"""SIMPLACE Optuna calibration: parameter space, crop edits and torchcrop's loss terms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.calibration.config import LossConfig
from cropmodelling4eu.simplace.optuna_calibration import (
    SimplaceScorer,
    _sample_cells,
    apply_values,
    heading_dates,
    load_space,
    read_value,
    season_dates,
    starting_values,
)

TEMPLATE = Path("/data01/FDS/muduchuru/Data/SIMPLACE/TEMPLATES/Rotation_monocrop/data/crop/LINTUL5_crop.xml")
CROP = {
    "tsumem": 60.0, "tsum1": 1125.0, "tsum2": 1000.0, "versat": 70.0, "vbase": 14.0,
    "ruetb": [[0.0, 2.6], [1.0, 2.5], [1.3, 2.0], [2.0, 0.9]],
    "fotb": [[0.0, 0.0], [0.99, 0.0], [1.0, 1.0], [2.0, 1.0]],
}


def test_yield_space_is_the_three_simplace_live_parameters() -> None:
    names = [p.name for p in load_space("yield").parameters]
    assert names == ["crop.scale_factor_rue", "crop.ruetb@1.3", "crop.fotb@0.99"]
    # versat's lower bound is the fixed vbase + 1: SIMPLACE divides by versat - vbase.
    versat = next(p for p in load_space("phenology").parameters if p.name == "crop.versat")
    assert versat.low > CROP["vbase"]


def test_apply_values_edits_knots_and_scalars_without_touching_the_source() -> None:
    space = load_space("yield")
    out = apply_values(CROP, space, {"crop.ruetb@1.3": 1.5, "crop.fotb@0.99": 0.3,
                                     "crop.scale_factor_rue": 0.9})
    assert out["ruetb"][2] == [1.3, 1.5] and out["fotb"][1] == [0.99, 0.3]
    assert out["scale_factor_rue"] == 0.9
    assert CROP["ruetb"][2] == [1.3, 2.0] and "scale_factor_rue" not in CROP
    assert read_value(CROP, space.parameters[0]) == 1.0


def test_starting_values_use_init_then_crop_clipped_to_bounds() -> None:
    start = starting_values(load_space("phenology"), CROP)
    # tsumem: torchcrop's init 150, not the crop file's 60; versat 70 is in range.
    assert start["crop.tsumem"] == 150.0 and start["crop.versat"] == 70.0


def test_prior_penalty_matches_torchcrop_form() -> None:
    space = load_space("phenology")
    values = {"crop.tsum1": 1060.0, "crop.tsum2": 794.0, "crop.versat": 54.0, "crop.tsumem": 150.0}
    ratio = 1060.0 / (1060.0 + 794.0)
    expected = 0.05 * (1.0 + ((ratio - 0.52) / 0.06) ** 2)
    assert space.penalty(values, 0.05) == pytest.approx(expected)


def test_season_dates_put_autumn_emergence_in_the_previous_year() -> None:
    sim = season_dates(pd.DataFrame({"year": [2010, 2010], "maturity_doy": [200.0, 200.0],
                                     "emergence_doy": [300.0, 20.0]}))
    epoch = pd.Timestamp(0)
    assert sim["emergence_d"].iloc[0] == (pd.Timestamp("2009-10-27") - epoch).days
    assert sim["emergence_d"].iloc[1] == (pd.Timestamp("2010-01-20") - epoch).days
    assert sim["maturity_d"].iloc[0] == (pd.Timestamp("2010-07-19") - epoch).days


def test_heading_is_the_linear_dvs_crossing() -> None:
    dates = pd.date_range("2010-05-01", periods=4)
    daily = pd.DataFrame({"SimplaceID": 1, "year": 2010, "date": dates, "DVS": [0.80, 0.85, 0.89, 0.95]})
    out = heading_dates(daily)
    expected = (dates[1] - pd.Timestamp(0)).days + 0.5
    assert out["heading_d"].iloc[0] == pytest.approx(expected)


def test_sample_cells_spreads_evenly_within_units() -> None:
    pool = pd.DataFrame({"unit": ["A"] * 9 + ["B"] * 2, "SimplaceID": list(range(9)) + [20, 21]})
    assert _sample_cells(pool, 3).tolist() == [0, 4, 8, 20, 21]
    assert len(_sample_cells(pool, 0)) == 11


def _scorer(stage: str, split: str = "train") -> SimplaceScorer:
    epoch_day = lambda s: pd.Timestamp(s).value
    pool = pd.DataFrame({"unit": ["U"] * 4, "year": [2001, 2001, 2002, 2002],
                         "SimplaceID": [1, 2, 1, 2], "region_id": 1})
    observations = pd.DataFrame({
        "unit": "U", "year": [2001, 2002, 2001, 2002],
        "variable": ["harvest_date", "harvest_date", "yield_level", "yield_level"],
        "value": [epoch_day("2001-07-20"), epoch_day("2002-07-25"), 7.0, 6.0],
        "weight": 1.0,
    })
    split_table = pd.DataFrame({"unit": "U", "year": [2001, 2002], "region_id": 1, "split": split})
    return SimplaceScorer(pool, observations, split_table, LossConfig(), stage)


def _sim(maturity: list[str], yield_g_m2: list[float]) -> pd.DataFrame:
    epoch = pd.Timestamp(0)
    days = [(pd.Timestamp(d) - epoch).days for d in maturity]
    return pd.DataFrame({"SimplaceID": [1, 2, 1, 2], "year": [2001, 2001, 2002, 2002],
                         "maturity_d": days, "emergence_d": np.nan, "yield_g_m2": yield_g_m2})


def test_scorer_reproduces_torchcrop_terms_by_hand() -> None:
    # Unit-year means: maturity 2001-07-18 and 2002-07-27 against observed
    # 07-20 and 07-25 (errors -2, +2 d); yield 7.5 and 5.0 against 7 and 6 t/ha.
    sim = _sim(["2001-07-17", "2001-07-19", "2002-07-26", "2002-07-28"], [700, 800, 500, 500])
    loss, report = _scorer("yield").score(sim, "train")
    harvest_level = 4.0      # mean(2^2, 2^2)
    harvest_anomaly = 4.0    # sim anomalies -187/+187 vs observed -185/+185
    hinge = 2.0 * (np.mean([0.0, 2.0**2]) + 0.1 * 0.0)  # +2 d late once; 2 d early is inside drydown
    phenology = harvest_level + harvest_anomaly + hinge
    yield_level = float(np.mean([0.5**2, 1.0**2]))
    assert report["phenology/harvest_date_rmse_d"] == pytest.approx(2.0)
    assert report["phenology/harvest_date_anomaly_rmse_d"] == pytest.approx(2.0)
    assert report["phenology/loss"] == pytest.approx(phenology)
    assert report["yield/yield_rmse_t_ha"] == pytest.approx(np.sqrt(yield_level))
    assert report["yield/loss"] == pytest.approx(yield_level)   # no yield_anomaly observation
    assert loss == pytest.approx(yield_level + 0.5 * phenology)
    assert report["never_matured_frac"] == 0.0


def test_scorer_drops_unmatured_cells_and_reports_them() -> None:
    sim = _sim(["2001-07-20", "2001-07-20", "2002-07-25", "2002-07-25"], [700, 700, 600, 600])
    sim.loc[1, "maturity_d"] = np.nan
    _, report = _scorer("phenology").score(sim, "train")
    assert report["never_matured_frac"] == pytest.approx(0.25)
    assert report["phenology/harvest_date_rmse_d"] == pytest.approx(0.0)
    assert _scorer("phenology", split="val").score(sim, "train")[1]["unit_years"] == 0


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="SUSTAg LINTUL5_crop.xml not on this system")
def test_written_crop_xml_carries_the_trial_values(tmp_path: Path) -> None:
    from cropmodelling4eu.torchcrop.params import load_simplace_crop, write_simplace_crop_xml

    crop = apply_values(CROP, load_space("yield"),
                        {"crop.scale_factor_rue": 0.9, "crop.ruetb@1.3": 1.5, "crop.fotb@0.99": 0.3})
    crop = apply_values(crop, load_space("phenology"), {"crop.tsum1": 900.0})
    out = write_simplace_crop_xml(tmp_path / "crop.xml", crop, TEMPLATE, "WW")
    written = load_simplace_crop(out, "WW")
    assert written["TSUM1"] == pytest.approx(900.0)
    # scale_factor_rue is folded into every RUETB ordinate, the edited knot included.
    assert written["RUETB"] == pytest.approx([0.0, 2.34, 1.0, 2.25, 1.3, 1.35, 2.0, 0.81])
    assert written["FOTB"][3] == pytest.approx(0.3)


def test_anomaly_survives_years_simulated_but_not_observed() -> None:
    # A third season is simulated with no observation. On absolute days its
    # 365 d offset would leak into the simulated climatology; within-season
    # days keep the anomaly at the 2 d it is.
    scorer = _scorer("phenology")
    scorer.pool = pd.concat([scorer.pool, pd.DataFrame(
        {"unit": "U", "year": [2003, 2003], "SimplaceID": [1, 2], "region_id": 1})])
    scorer.split = pd.concat([scorer.split, pd.DataFrame(
        {"unit": "U", "year": [2003], "region_id": 1, "split": "train"})])
    sim = pd.concat([
        _sim(["2001-07-18", "2001-07-18", "2002-07-27", "2002-07-27"], [700] * 4),
        _sim(["2003-07-22"] * 4, [700] * 4).iloc[:2].assign(year=2003),
    ], ignore_index=True)
    _, report = scorer.score(sim, "train")
    # Simulated within-season days 199, 208, 203 (mean 203.33); observed 201, 206 (mean 203.5).
    expected = np.sqrt(np.mean([((199 - 203 - 1 / 3) - (201 - 203.5)) ** 2,
                                ((208 - 203 - 1 / 3) - (206 - 203.5)) ** 2]))
    assert report["phenology/harvest_date_anomaly_rmse_d"] == pytest.approx(expected)


def test_knots_match_through_float32_round_off() -> None:
    crop = {**CROP, "ruetb": [[0.0, 2.6], [1.0, 2.5], [1.2999999523162842, 1.56], [2.0, 0.9]]}
    space = load_space("yield")
    assert read_value(crop, space.parameters[1]) == pytest.approx(1.56)
    assert apply_values(crop, space, {"crop.ruetb@1.3": 1.2})["ruetb"][2][1] == 1.2
