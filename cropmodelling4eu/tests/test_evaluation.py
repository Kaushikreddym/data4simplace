"""Tests for the evaluation library, focused on what the move changed.

The library's metrics and loaders are exercised by the four notebooks; what is
tested here is the seam between a run and the evaluation — that a per-cell
sowing date reaches the phenology derivation instead of being overwritten by
the old continental constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.evaluation.config import SIM_SOWING_DOY
from cropmodelling4eu.evaluation.metrics import (
    agreement_metrics,
    index_of_agreement,
    mean_normalized_bias,
)
from cropmodelling4eu.evaluation.torchcrop import add_phenology_columns


def test_per_cell_sowing_date_is_used_when_present():
    """The run's own column must win over the fallback constant."""
    frame = pd.DataFrame(
        {
            "SimplaceID": [1, 2],
            "days_to_maturity": [272.0, 272.0],
            # A German and a Spanish cell, sown 52 days apart.
            "sowing_doy": [280.0, 332.0],
        }
    )
    out = add_phenology_columns(frame)

    assert out["sowing_doy"].tolist() == [280.0, 332.0]
    # Equal season lengths from different sowing dates give different maturity
    # dates -- which is exactly what a constant would have flattened.
    assert out["maturity_doy"].iloc[1] - out["maturity_doy"].iloc[0] == 52.0
    assert out["season_length_days"].tolist() == [272.0, 272.0]


def test_missing_sowing_column_falls_back_and_warns(caplog):
    frame = pd.DataFrame({"days_to_maturity": [272.0]})
    out = add_phenology_columns(frame)

    assert out["sowing_doy"].tolist() == [float(SIM_SOWING_DOY)]
    assert "no sowing_doy column" in caplog.text
    # 270 + 272 = 542 -> wrapped into the harvest year.
    assert out["maturity_doy"].iloc[0] == 177.0


def test_explicit_sowing_doy_is_only_a_fallback():
    """The argument must not override a run that carries its own dates."""
    frame = pd.DataFrame({"days_to_maturity": [272.0], "sowing_doy": [300.0]})
    out = add_phenology_columns(frame, sowing_doy=270)
    assert out["sowing_doy"].iloc[0] == 300.0


def test_cells_that_never_matured_keep_nan_dates(caplog):
    frame = pd.DataFrame(
        {"days_to_maturity": [272.0, np.nan], "sowing_doy": [280.0, 280.0]}
    )
    out = add_phenology_columns(frame)

    assert np.isnan(out["maturity_doy"].iloc[1])
    assert not np.isnan(out["maturity_doy"].iloc[0])
    assert "never reached maturity" in caplog.text


def test_days_to_maturity_is_required():
    with pytest.raises(KeyError, match="days_to_maturity"):
        add_phenology_columns(pd.DataFrame({"yield_g_m2": [500.0]}))


def test_evaluation_library_imports_as_a_subpackage():
    """The move must not have broken the relative imports inside the library."""
    from cropmodelling4eu import evaluation

    for name in evaluation.__all__:
        assert hasattr(evaluation, name), f"{name} did not import"


def test_index_of_agreement_matches_willmott():
    obs = np.array([2.0, 4.0, 6.0])
    sim = np.array([3.0, 4.0, 8.0])
    # mean(obs) = 4: sum (S-O)^2 = 1 + 0 + 4 = 5; sum (|S-4| + |O-4|)^2 = 9 + 0 + 36 = 45
    assert index_of_agreement(obs, sim) == pytest.approx(1.0 - 5.0 / 45.0)
    assert index_of_agreement(obs, obs) == pytest.approx(1.0)


def test_mean_normalized_bias_skips_zero_observations():
    obs = np.array([2.0, 4.0, 0.0])
    sim = np.array([3.0, 3.0, 1.0])
    assert mean_normalized_bias(obs, sim) == pytest.approx((0.5 - 0.25) / 2 * 100.0)


def test_agreement_metrics_takes_dates_the_short_way_round():
    obs = np.array([360.0, 200.0, 100.0])
    sim = np.array([3.0, 205.0, 95.0])
    scores = agreement_metrics(obs, sim, circular=True)
    assert scores["bias"] == pytest.approx((8.0 + 5.0 - 5.0) / 3)
    assert scores["rmse"] == pytest.approx(np.sqrt((64.0 + 25.0 + 25.0) / 3))
    assert np.isnan(scores["mnb"])
    assert scores["ia"] > 0.99



def test_agreement_r2_is_squared_pearson_and_blind_to_offset():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    sim = 2.0 * obs + 5.0
    scores = agreement_metrics(obs, sim)
    assert scores["pearson_r2"] == pytest.approx(1.0)
    assert scores["ia"] < 0.9
