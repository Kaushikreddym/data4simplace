"""Probability of extreme yields against each model's own 1985–2014 climate.

A threshold is set **per model and per cell** on the ``historical`` baseline, so a
model's mean bias cannot read as a change in risk. An in-sample baseline probability
is exact by construction, whereas a future one carries the threshold's sampling error,
so the reference probability is estimated leave-one-year-out (:func:`loo_probability`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from scenarios import BASELINE, period


@dataclass(slots=True, frozen=True)
class Event:
    """A yield event defined against the baseline: a quantile or a share of the mean."""

    key: str
    label: str
    quantile: float | None = None
    fraction_of_mean: float | None = None
    upper: bool = False


EVENTS: tuple[Event, ...] = (
    Event("p10", "Low yield: below the baseline 10th percentile (1-in-10)", quantile=0.10),
    Event("p05", "Very low yield: below the baseline 5th percentile (1-in-20)", quantile=0.05),
    Event("half", "Failure: below 50 % of the baseline mean", fraction_of_mean=0.5),
    Event("p90", "High yield: above the baseline 90th percentile", quantile=0.90, upper=True),
)
EVENT = {e.key: e for e in EVENTS}


def threshold(data: xr.DataArray, event: Event, span: tuple[int, int] = BASELINE,
              reference: str = "historical") -> xr.DataArray:
    """Per-model, per-cell threshold of ``event`` on the reference climate."""
    base = period(data.sel(experiment=reference), span)
    if event.quantile is not None:
        return base.quantile(event.quantile, dim="year", skipna=False).drop_vars("quantile")
    return event.fraction_of_mean * base.mean("year")


def hits(data: xr.DataArray, limit: xr.DataArray, event: Event) -> xr.DataArray:
    """Whether each season crossed ``limit``; NaN where the run has no season."""
    crossed = data > limit if event.upper else data < limit
    return crossed.astype(np.float32).where(data.notnull())


def probability(data: xr.DataArray, limit: xr.DataArray, event: Event, span: tuple[int, int]) -> xr.DataArray:
    """Per-cell frequency of the event over ``span``."""
    return hits(period(data, span), limit, event).mean("year")


def loo_hits(data: xr.DataArray, event: Event, span: tuple[int, int] = BASELINE,
             reference: str = "historical") -> xr.DataArray:
    """Baseline hits with each season scored on a threshold from the other seasons.

    This is what a future window experiences: seasons the threshold has not seen.
    """
    base = period(data.sel(experiment=reference), span)
    folds = []
    for year in base.year.values:
        rest = base.drop_sel(year=year)
        limit = (rest.quantile(event.quantile, dim="year", skipna=False).drop_vars("quantile")
                 if event.quantile is not None else event.fraction_of_mean * rest.mean("year"))
        folds.append(hits(base.sel(year=year), limit, event))
    return xr.concat(folds, "year")


def loo_probability(data: xr.DataArray, event: Event, span: tuple[int, int] = BASELINE,
                    reference: str = "historical") -> xr.DataArray:
    return loo_hits(data, event, span, reference).mean("year")


def scaled_baseline_probability(data: xr.DataArray, limit: xr.DataArray, event: Event, experiment: str,
                                span: tuple[int, int], reference: str = "historical",
                                base_span: tuple[int, int] = BASELINE) -> xr.DataArray:
    """Frequency if only the mean moved: baseline seasons rescaled by the mean's ratio.

    Scaling rather than adding keeps the coefficient of variation fixed, because a
    yield's spread grows with its mean. The remainder of the change is variability.
    """
    base = period(data.sel(experiment=reference), base_span)
    ratio = period(data.sel(experiment=experiment), span).mean("year") / base.mean("year").where(lambda m: m > 0)
    return hits(base * ratio, limit, event).mean("year")


def bootstrap_ci(series: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of a mean over seasons, resampling whole seasons (spatial structure intact)."""
    finite = series[np.isfinite(series)]
    draws = np.random.default_rng(seed).integers(0, len(finite), (n, len(finite)))
    return tuple(np.quantile(finite[draws].mean(1), [0.025, 0.975]))


def coefficient_of_variation(data: xr.DataArray, span: tuple[int, int]) -> xr.DataArray:
    """Per-cell interannual CV (%) over ``span``, linearly detrended first.

    Without the detrending a window's trend would read as a rise in variability.
    """
    window = period(data, span)
    fit = window.polyfit("year", deg=1)["polyfit_coefficients"]
    residual = window - xr.polyval(window["year"], fit)
    return 100.0 * residual.std("year") / window.mean("year").where(lambda m: m > 0)
