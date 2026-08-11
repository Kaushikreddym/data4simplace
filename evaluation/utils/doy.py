"""Circular day-of-year arithmetic.

A day-of-year lives on a circle, and every routine that treats it as a plain
number is wrong at the wrap. CyBench puts Spanish wheat's start-of-season at
DOY 363 in one region and DOY 0.7 in its neighbour: their arithmetic mean is
DOY 182 (July), their circular mean is DOY 364.35 (30 December). The same
applies to the error — a simulated DOY 3 against an observed DOY 363 is a
5-day error, not a 360-day one.

Every function here takes and returns day-of-year in ``(0, DAYS_IN_YEAR]``,
where 1.0 is 1 January.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYS_IN_YEAR

__all__ = [
    "circular_mean_doy",
    "doy_difference",
    "doy_in_window",
    "doy_to_month_day",
    "window_position",
    "unwrap_doy",
    "wrap_doy",
]

ArrayLike = np.ndarray | pd.Series | float


def wrap_doy(doy: ArrayLike, days_in_year: float = DAYS_IN_YEAR) -> ArrayLike:
    """Wrap a day-of-year into ``(0, days_in_year]``.

    Args:
        doy: Day-of-year, possibly outside the year (e.g. sowing DOY 270 plus a
            272-day season gives 542).
        days_in_year: Length of the year the wrap is taken modulo.

    Returns:
        The same type as ``doy``, wrapped. DOY 365.0 stays 365.0 rather than
        collapsing to 0.

    Examples:
        >>> float(wrap_doy(542.0))
        177.0
        >>> float(wrap_doy(0.0))
        365.0
    """
    if isinstance(doy, pd.Series):
        return (doy - 1.0) % days_in_year + 1.0
    return (np.asarray(doy, dtype=float) - 1.0) % days_in_year + 1.0


def doy_difference(
    a: ArrayLike, b: ArrayLike, days_in_year: float = DAYS_IN_YEAR
) -> np.ndarray:
    """Signed shortest difference ``a - b`` in days, in ``[-Y/2, +Y/2)``.

    This is the error definition every phenology metric in this project uses.
    Positive means ``a`` is *later* than ``b`` by the short way round.

    Args:
        a: Day-of-year (conventionally the simulation).
        b: Day-of-year (conventionally the observation).
        days_in_year: Circle length.

    Returns:
        Float array of signed differences. NaN propagates.

    Examples:
        >>> float(doy_difference(3.0, 363.0))
        5.0
        >>> float(doy_difference(363.0, 3.0))
        -5.0
    """
    half = days_in_year / 2.0
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (diff + half) % days_in_year - half


def circular_mean_doy(
    doy: ArrayLike,
    weights: ArrayLike | None = None,
    days_in_year: float = DAYS_IN_YEAR,
) -> float:
    """Weighted circular mean of a set of days-of-year.

    Each date is mapped to a unit vector, the vectors are averaged, and the
    result is read back as an angle. NaN values — and, when weights are given,
    entries with a NaN or non-positive weight — are dropped.

    Args:
        doy: Days-of-year to average.
        weights: Non-negative weights, aligned with ``doy``. ``None`` averages
            unweighted.
        days_in_year: Circle length.

    Returns:
        The mean day-of-year, or NaN if nothing survives the drop. Returns NaN
        when the resultant vector is degenerate (dates spread evenly round the
        circle), because no mean direction exists.

    Examples:
        >>> round(circular_mean_doy([363.0, 0.7]), 2)
        364.35
    """
    values = np.asarray(doy, dtype=float)
    w = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != values.shape:
        raise ValueError(f"weights {w.shape} do not match doy {values.shape}")

    valid = np.isfinite(values) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return float("nan")
    values, w = values[valid], w[valid]

    angle = 2.0 * np.pi * (values - 1.0) / days_in_year
    x = float(np.sum(w * np.cos(angle)))
    y = float(np.sum(w * np.sin(angle)))
    # A resultant this short means the dates have no mean direction; returning
    # the arctan of numerical noise would invent a date.
    if np.hypot(x, y) < 1e-9 * float(np.sum(w)):
        return float("nan")
    return float(wrap_doy(np.arctan2(y, x) * days_in_year / (2.0 * np.pi) + 1.0))


def unwrap_doy(
    doy: ArrayLike, centre: float, days_in_year: float = DAYS_IN_YEAR
) -> np.ndarray:
    """Shift days-of-year by whole years so they lie within Y/2 of ``centre``.

    Correlation, regression and a scatter axis are all linear operations, so
    they need a linear representation of the dates. Unwrapping about the
    observed circular mean gives one: DOY 363 near a centre of 5 becomes -2, so
    it plots and correlates beside DOY 3 instead of at the far end of the axis.

    Args:
        doy: Days-of-year to unwrap.
        centre: The day-of-year to unwrap about, typically the circular mean of
            the *observations* so both sides share one origin.
        days_in_year: Circle length.

    Returns:
        Float array in ``[centre - Y/2, centre + Y/2)``. Values may be negative
        or exceed ``days_in_year`` — that is the point.
    """
    return centre + doy_difference(doy, centre, days_in_year)


def doy_in_window(
    doy: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
    days_in_year: float = DAYS_IN_YEAR,
) -> np.ndarray:
    """Whether each day-of-year lies inside a circular ``[start, end]`` window.

    The window is walked **forwards from start to end the short way**, which is
    the only reading that works for a northern-hemisphere planting window
    running from late September to early December *and* for one that crosses
    New Year. A window is at most a year long, so ``end`` before ``start`` in
    plain numbers means it wraps.

    Args:
        doy: Days-of-year to test.
        start: Window start day-of-year.
        end: Window end day-of-year.
        days_in_year: Circle length.

    Returns:
        Boolean array, ``False`` wherever any input is NaN. Both endpoints are
        inclusive.

    Examples:
        >>> bool(doy_in_window(270.0, 250.0, 300.0))
        True
        >>> bool(doy_in_window(5.0, 340.0, 20.0))
        True
        >>> bool(doy_in_window(200.0, 340.0, 20.0))
        False
    """
    d = np.asarray(doy, dtype=float)
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    finite = np.isfinite(d) & np.isfinite(s) & np.isfinite(e)
    # Both offsets measured forwards from the window start, so the test is a
    # plain comparison on the unrolled circle.
    into = (d - s) % days_in_year
    span = (e - s) % days_in_year
    return np.asarray(finite & (into <= span), dtype=bool)


def window_position(
    doy: ArrayLike,
    start: ArrayLike,
    end: ArrayLike,
    days_in_year: float = DAYS_IN_YEAR,
) -> np.ndarray:
    """Signed distance in days from a circular window, 0 inside it.

    The quantity :func:`doy_in_window` reduces to a boolean. Keeping the
    distance says *how far* outside a date falls, which is what separates a
    sowing latch that misses the window by three days from one that misses it
    by six weeks.

    Args:
        doy: Days-of-year to place.
        start: Window start day-of-year.
        end: Window end day-of-year.
        days_in_year: Circle length.

    Returns:
        Float array: ``0`` inside the window, negative by the number of days
        before ``start``, positive by the number of days after ``end``. NaN
        propagates. A date outside the window is measured to whichever edge is
        nearer, so the sign is the side it falls on.

    Examples:
        >>> float(window_position(270.0, 250.0, 300.0))
        0.0
        >>> float(window_position(240.0, 250.0, 300.0))
        -10.0
        >>> float(window_position(310.0, 250.0, 300.0))
        10.0
    """
    d = np.asarray(doy, dtype=float)
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)

    before = doy_difference(d, s, days_in_year)   # < 0 when earlier than start
    after = doy_difference(d, e, days_in_year)    # > 0 when later than end
    inside = doy_in_window(d, s, e, days_in_year)

    # Outside the window exactly one edge is the near one; take the smaller
    # absolute displacement so the sign reports the side.
    outside = np.where(np.abs(before) <= np.abs(after), before, after)
    result = np.where(inside, 0.0, outside)
    return np.where(np.isfinite(d) & np.isfinite(s) & np.isfinite(e), result, np.nan)


def doy_to_month_day(doy: float, days_in_year: float = DAYS_IN_YEAR) -> str:
    """Format a day-of-year as ``"12 Aug"`` for axis ticks and tables.

    Uses 2001, a non-leap year, so DOY 60 is 1 March as in the 365-day
    convention the model and CyBench both use.

    Args:
        doy: Day-of-year; wrapped first, so 542 formats as 26 June.
        days_in_year: Circle length.

    Returns:
        Day and abbreviated month, or ``""`` for NaN.
    """
    if not np.isfinite(doy):
        return ""
    wrapped = float(wrap_doy(float(doy), days_in_year))
    stamp = pd.Timestamp("2001-01-01") + pd.Timedelta(days=wrapped - 1.0)
    return f"{stamp.day} {stamp.strftime('%b')}"
