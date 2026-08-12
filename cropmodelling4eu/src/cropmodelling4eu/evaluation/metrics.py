"""Skill metrics for continuous quantities and for dates.

Two families, because a date is not a number:

* :func:`yield_metrics` — RMSE, MAE, Bias, MAPE, R² and Pearson r on plain
  values.
* :func:`phenology_metrics` — the same error statistics built on the circular
  difference :func:`utils.doy.doy_difference`, so an error across New Year is 5
  days rather than 360.

Sign convention throughout: **error = simulated - observed**, so a positive
bias means the model is too high (yield) or too late (phenology).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .config import DAYS_IN_YEAR
from .doy import circular_mean_doy, doy_difference, unwrap_doy

logger = logging.getLogger(__name__)

__all__ = [
    "METRIC_LABELS",
    "PHENOLOGY_METRIC_ORDER",
    "YIELD_METRIC_ORDER",
    "metrics_by_group",
    "phenology_metrics",
    "rank_countries",
    "to_anomalies",
    "yield_metrics",
]

#: Column order for a yield summary table.
YIELD_METRIC_ORDER: tuple[str, ...] = (
    "n", "obs_mean", "sim_mean", "bias", "mae", "rmse", "mape", "r2", "pearson_r",
)

#: Column order for a phenology summary table.
PHENOLOGY_METRIC_ORDER: tuple[str, ...] = (
    "n", "obs_mean", "sim_mean", "bias", "mae", "rmse", "pearson_r",
)

#: Display names, including units. Used by every table and figure so a metric
#: is never labelled two different ways.
METRIC_LABELS: dict[str, str] = {
    "n": "n",
    "obs_mean": "Observed mean",
    "sim_mean": "Simulated mean",
    "bias": "Bias",
    "mae": "MAE",
    "rmse": "RMSE",
    "mape": "MAPE (%)",
    "r2": "R²",
    "pearson_r": "Pearson r",
    "pearson_p": "p",
}


def _clean(observed: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is missing or non-finite."""
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"shape mismatch: observed {obs.shape}, simulated {sim.shape}")
    keep = np.isfinite(obs) & np.isfinite(sim)
    return obs[keep], sim[keep]


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and its p-value, NaN when it is undefined.

    scipy raises on fewer than two points and warns when a side is constant.
    Both are normal here — a country with one paired year, or the simulated
    sowing date, which is a constant by construction — so both return NaN.
    """
    if x.size < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan"), float("nan")
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def yield_metrics(observed, simulated) -> dict[str, float]:
    """Skill of a simulated continuous quantity against observations.

    Args:
        observed: Reference values.
        simulated: Model values, aligned with ``observed``.

    Returns:
        ``n``, ``obs_mean``, ``sim_mean``, ``bias``, ``mae``, ``rmse``,
        ``mape``, ``r2``, ``pearson_r``, ``pearson_p``. Every value is NaN when
        no pair survives.

        * ``bias`` — mean(sim - obs); positive means over-prediction.
        * ``mape`` — mean absolute percentage error over pairs with a non-zero
          observation. Rows with ``obs == 0`` are excluded rather than
          returning infinity.
        * ``r2`` — the coefficient of determination
          ``1 - SS_res / SS_tot``, **not** the square of Pearson r. It is
          negative when the simulation predicts worse than the observed mean,
          which is the informative case for a process model with a systematic
          offset; ``pearson_r`` carries the correlation separately.
    """
    obs, sim = _clean(observed, simulated)
    if obs.size == 0:
        return {k: float("nan") for k in (*YIELD_METRIC_ORDER, "pearson_p")} | {"n": 0}

    error = sim - obs
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    nonzero = obs != 0.0
    r, p = _pearson(obs, sim)

    return {
        "n": int(obs.size),
        "obs_mean": float(obs.mean()),
        "sim_mean": float(sim.mean()),
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": (
            float(np.abs(error[nonzero] / obs[nonzero]).mean() * 100.0)
            if nonzero.any() else float("nan")
        ),
        "r2": 1.0 - float(np.sum(error**2)) / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": r,
        "pearson_p": p,
    }


def phenology_metrics(
    observed, simulated, days_in_year: float = DAYS_IN_YEAR
) -> dict[str, float]:
    """Skill of simulated dates against observed dates, on the circle.

    Errors use :func:`utils.doy.doy_difference`. The means reported are
    circular means, and the correlation is computed on both sides unwrapped
    about the *observed* circular mean — a shared linear origin, without which
    a Mediterranean country whose dates straddle New Year would correlate
    against an artefact of the wrap.

    Args:
        observed: Observed days-of-year.
        simulated: Simulated days-of-year, aligned with ``observed``.
        days_in_year: Circle length.

    Returns:
        ``n``, ``obs_mean``, ``sim_mean`` (both days-of-year), ``bias``,
        ``mae``, ``rmse`` (all in days), ``pearson_r`` and ``pearson_p``.
        Positive bias means the simulation is late.
    """
    obs, sim = _clean(observed, simulated)
    if obs.size == 0:
        return {
            k: float("nan") for k in (*PHENOLOGY_METRIC_ORDER, "pearson_p")
        } | {"n": 0}

    error = doy_difference(sim, obs, days_in_year)
    centre = circular_mean_doy(obs, days_in_year=days_in_year)
    r, p = _pearson(
        unwrap_doy(obs, centre, days_in_year), unwrap_doy(sim, centre, days_in_year)
    )

    return {
        "n": int(obs.size),
        "obs_mean": centre,
        "sim_mean": circular_mean_doy(sim, days_in_year=days_in_year),
        "bias": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": r,
        "pearson_p": p,
    }


def to_anomalies(
    frame: pd.DataFrame,
    columns: list[str],
    by: str | list[str] = "grid_id",
    suffix: str = "_anom",
) -> pd.DataFrame:
    """Subtract each group's own mean from the given columns.

    A pooled correlation over a field mixes two questions: does the model put
    the high-yielding regions in the right place (spatial), and does it move
    with the weather from year to year (temporal). The spatial term is far the
    larger of the two, so it hides the temporal one entirely. Removing each
    cell's own long-term mean leaves only the interannual signal, which is the
    part a reference like GDHY was built to carry.

    Args:
        frame: Paired rows.
        columns: Columns to convert.
        by: Group whose mean is removed — the cell, for an interannual anomaly.
        suffix: Appended to each column name.

    Returns:
        A copy of ``frame`` with the anomaly columns added. A group with one
        row yields an anomaly of exactly 0 on both sides, which contributes
        nothing to a correlation and is left in rather than silently dropped.
    """
    out = frame.copy()
    grouped = out.groupby(by, observed=True)
    for column in columns:
        out[f"{column}{suffix}"] = out[column] - grouped[column].transform("mean")
    return out


def metrics_by_group(
    frame: pd.DataFrame,
    observed_col: str,
    simulated_col: str,
    group_col: str | list[str] | None = "country",
    metric_fn: Callable[..., dict[str, float]] = yield_metrics,
) -> pd.DataFrame:
    """Run a metric function per group, plus an ``ALL`` row over everything.

    The pooled ``ALL`` row is computed over the raw pairs, not by averaging the
    per-group metrics — averaging RMSEs is not an RMSE, and a country with two
    paired years would otherwise weigh as much as one with twenty.

    Args:
        frame: Paired rows.
        observed_col: Observation column.
        simulated_col: Simulation column.
        group_col: Grouping key(s). ``None`` returns only the pooled row.
        metric_fn: :func:`yield_metrics` or :func:`phenology_metrics`.

    Returns:
        One row per group, index named after the grouping key, with an ``ALL``
        row appended last.
    """
    pooled = metric_fn(frame[observed_col], frame[simulated_col])

    if group_col is None:
        return pd.DataFrame([pooled], index=pd.Index(["ALL"], name="country"))

    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    rows = {
        (name if len(keys) == 1 else name): metric_fn(
            block[observed_col], block[simulated_col]
        )
        for name, block in frame.groupby(keys[0] if len(keys) == 1 else keys,
                                         sort=True, observed=True)
    }
    rows["ALL"] = pooled

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = keys[0] if len(keys) == 1 else "group"
    return out


def rank_countries(
    metrics: pd.DataFrame,
    by: str = "rmse",
    ascending: bool = True,
    min_n: int = 3,
) -> pd.DataFrame:
    """Order a per-country metric table, best first, with the pooled row last.

    Args:
        metrics: Output of :func:`metrics_by_group`.
        by: Column to rank on. ``rmse`` by default: it is the metric that
            penalises both the offset and the scatter, so it is the single
            column that best summarises "how usable is this country".
        ascending: ``True`` for error-like metrics, ``False`` for r or R².
        min_n: Countries with fewer paired points are still listed, but at the
            bottom and flagged — a two-year RMSE is not comparable with a
            twenty-year one.

    Returns:
        A copy with a ``rank`` column, sorted. ``ALL`` keeps rank NaN and stays
        last.
    """
    table = metrics.copy()
    pooled = table.loc[["ALL"]] if "ALL" in table.index else None
    if pooled is not None:
        table = table.drop(index="ALL")

    table["sparse"] = table["n"] < min_n
    table = table.sort_values(["sparse", by], ascending=[True, ascending])
    table["rank"] = np.where(table["sparse"], np.nan, np.arange(1, len(table) + 1))

    if pooled is not None:
        pooled = pooled.assign(sparse=False, rank=np.nan)
        table = pd.concat([table, pooled])
    return table
