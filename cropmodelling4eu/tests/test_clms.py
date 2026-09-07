"""Tests for the CLMS phenology reference.

Two seams matter and neither is exercised by the notebook running green:
the quality gates must actually drop what they claim to, and the skill split
must treat a day-of-year as circular. A winter crop's emergence straddles New
Year, so a plain anomaly turns a 10-day shift into a 355-day one — the failure
is silent, and it inflates the observed spread rather than raising an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.evaluation import clms


def _reference(tmp_path, rows):
    frame = pd.DataFrame(rows)
    path = tmp_path / "phenology_adm.parquet"
    frame.to_parquet(path)
    return path


def _row(**kw):
    base = dict(
        zone_kind="adm", country="DE", adm_id="DE111", crop="wheat_winter",
        year=2020, emergence_doy_median=300.0, harvest_doy_median=200.0,
        season_length_days_median=265.0,
        emergence_doy_p10=290.0, emergence_doy_p90=310.0,
        harvest_doy_p10=195.0, harvest_doy_p90=205.0,
        season_length_days_p10=255.0, season_length_days_p90=275.0,
        n_pixels=500_000, area_km2=50.0, season_share=0.8, split_validated=True,
    )
    return base | kw


def test_gates_drop_thin_unvalidated_and_minority_rows(tmp_path, caplog):
    path = _reference(tmp_path, [
        _row(adm_id="KEEP"),
        _row(adm_id="THIN", n_pixels=100),
        _row(adm_id="UNVALIDATED", split_validated=False),
        _row(adm_id="MINORITY", season_share=0.05),
        _row(adm_id="OTHERCROP", crop="wheat_spring"),
    ])
    with caplog.at_level("INFO"):
        out = clms.load_phenology(path=path)

    assert out["adm_id"].tolist() == ["KEEP"]
    # Every gate reports what it cost, so the survivor count is auditable.
    assert "1 rows dropped (split not validated)" in caplog.text
    assert "1 rows dropped (< 10,000 pixels)" in caplog.text


def test_duplicate_keys_are_refused(tmp_path):
    """A fan-out join would make every metric under it wrong, silently."""
    path = _reference(tmp_path, [_row(), _row()])
    with pytest.raises(ValueError, match="duplicated"):
        clms.load_phenology(path=path)


def test_missing_file_names_the_stage_that_writes_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="validation_reduce"):
        clms.load_phenology(path=tmp_path / "absent.parquet")


def test_split_skill_handles_a_season_crossing_new_year():
    """DOY 360 and DOY 5 are ten days apart, not 355."""
    paired = pd.DataFrame({
        "adm_id": ["A"] * 4,
        # Observed emergence straddles New Year; the model tracks it exactly,
        # offset by a constant 20 days.
        "obs": [360.0, 5.0, 362.0, 3.0],
        "sim": [340.0, 350.0, 342.0, 348.0],
    })
    circular = clms.split_skill(paired, "obs", "sim", circular=True)
    linear = clms.split_skill(paired, "obs", "sim", circular=False)

    # The true observed swing is ~8 days about the unit's circular mean.
    assert circular["obs_anomaly_sd"] < 10
    # Treated linearly, the New Year wrap manufactures a ~200-day spread.
    assert linear["obs_anomaly_sd"] > 100
    # And the model, which follows the observation, is correlated with it.
    assert circular["anomaly_r"] > 0.9


def test_split_skill_leaves_durations_linear():
    """A 250-day season must not be folded under half a year."""
    paired = pd.DataFrame({
        "adm_id": ["A", "A", "B", "B"],
        "obs": [250.0, 260.0, 200.0, 210.0],
        "sim": [255.0, 265.0, 205.0, 215.0],
    })
    out = clms.split_skill(paired, "obs", "sim", circular=False)
    assert out["units"] == 2
    assert out["spatial_r"] == pytest.approx(1.0)
    assert out["anomaly_r"] == pytest.approx(1.0)


def test_calendar_offset_reports_the_level_shift():
    observed = pd.DataFrame({
        "adm_id": ["A", "A", "B"],
        "obs_harvest_doy": [180.0, 182.0, 190.0],
    })
    calendar = pd.DataFrame({"adm_id": ["A", "B"], "eos": [210.0, 220.0]})
    out = clms.calendar_offset(observed, calendar)

    assert out.loc["A", "difference"] == pytest.approx(-29.0)
    assert out.loc["B", "difference"] == pytest.approx(-30.0)
    assert not np.isnan(out["difference"]).any()
