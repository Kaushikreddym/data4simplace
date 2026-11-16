"""Per-cell features for the three branches, from one decoded weather record.

=========  ====================  ==============================================
Branch     Shape per season      Content
=========  ====================  ==============================================
short      ``[T, 7]`` daily      the weather itself (:data:`SHORT_CHANNELS`)
medium     ``[T, 6]`` daily      causal state: topsoil bucket, the SIMPLACE
                                 rule's own indicators, dry spell, degree days
long       ``[9]``               normals over autumns ``Y-31 .. Y-2``
=========  ====================  ==============================================

Every quantity at cache day ``t`` is a function of days ``<= t`` only. The
bucket runs continuously over the whole record, so its state on the cache's
first day carries the summer before it (the plan's "spin-up from 1 April" is
the special case of a record that starts then).
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd

__all__ = [
    "LONG_FEATURES",
    "MEDIUM_CHANNELS",
    "SHORT_CHANNELS",
    "CellFeatures",
    "cell_features",
]

SHORT_CHANNELS: tuple[str, ...] = ("tmin", "tmax", "davtmp", "rain", "irrad", "vp", "wind")
MEDIUM_CHANNELS: tuple[str, ...] = ("bucket", "tmin7", "rain3", "rain30", "dry_days", "gdd")
LONG_FEATURES: tuple[str, ...] = (
    "tmin_sep", "tmin_oct", "tmean_aso", "rain_sep", "rain_oct", "wetday_so",
    "onset_tmin10", "onset_tmean12", "bucket_sep",
)

#: Index of each channel in ``export.weather.CHANNELS``.
_SRC = {"doy": 0, "davtmp": 1, "tmin": 2, "tmax": 3, "irrad": 4, "rain": 5, "vp": 6, "wind": 7}
#: A day with at least this much rain ends a dry spell [mm].
_WET_MM = 5.0
#: Onsets are searched from 1 August.
_ONSET_FROM_DOY = 213


@dataclass(slots=True)
class CellFeatures:
    """One cell's seasons, aligned with the requested harvest years."""

    short: np.ndarray   # [Y, T, 7] float16
    medium: np.ndarray  # [Y, T, 6] float16
    long: np.ndarray    # [Y, 9] float32
    valid: np.ndarray   # [Y] bool


@numba.njit(cache=True)
def _bucket(rain: np.ndarray, et0: np.ndarray, capacity: float) -> np.ndarray:
    """Topsoil water fraction: fills with rain, drains by ET0 scaled by fullness."""
    out = np.empty(rain.size, np.float32)
    w = 0.5 * capacity
    for t in range(rain.size):
        w = w + rain[t] - et0[t] * (w / capacity)
        w = min(max(w, 0.0), capacity)
        out[t] = w / capacity
    return out


@numba.njit(cache=True)
def _dry_days(rain: np.ndarray, wet: float, cap: float) -> np.ndarray:
    out = np.empty(rain.size, np.float32)
    run = cap
    for t in range(rain.size):
        run = 0.0 if rain[t] >= wet else min(run + 1.0, cap)
        out[t] = run
    return out


def _rolling(x: np.ndarray, n: int, how: str) -> np.ndarray:
    """Trailing n-day mean/sum including today; the first days use what exists."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(1, x.size + 1)
    total = c[idx] - c[np.maximum(idx - n, 0)]
    return (total / np.minimum(idx, n) if how == "mean" else total).astype(np.float32)


def _autumn_table(dates: pd.DatetimeIndex, ch: np.ndarray, bucket: np.ndarray,
                  tmin7: np.ndarray, tmean7: np.ndarray) -> pd.DataFrame:
    """One row per calendar year: the autumn statistics :data:`LONG_FEATURES` average."""
    frame = pd.DataFrame({
        "year": dates.year, "month": dates.month, "doy": dates.dayofyear,
        "tmin": ch[:, _SRC["tmin"]], "tmean": ch[:, _SRC["davtmp"]],
        "rain": ch[:, _SRC["rain"]], "bucket": bucket, "tmin7": tmin7, "tmean7": tmean7,
    })
    by = frame.groupby("year")
    month = lambda m, col, agg: frame[frame["month"] == m].groupby("year")[col].agg(agg)
    so = frame[frame["month"].isin((9, 10))].groupby("year")
    late = frame[frame["doy"] >= _ONSET_FROM_DOY]

    def onset(col: str, limit: float) -> pd.Series:
        hit = late[late[col] <= limit].groupby("year")["doy"].min()
        return hit.reindex(by.size().index).fillna(366.0)

    table = pd.DataFrame({
        "tmin_sep": month(9, "tmin", "mean"),
        "tmin_oct": month(10, "tmin", "mean"),
        "tmean_aso": frame[frame["month"].isin((8, 9, 10))].groupby("year")["tmean"].mean(),
        "rain_sep": month(9, "rain", "sum"),
        "rain_oct": month(10, "rain", "sum"),
        "wetday_so": so["rain"].apply(lambda r: float((r >= 1.0).mean())),
        "onset_tmin10": onset("tmin7", 10.0),
        "onset_tmean12": onset("tmean7", 12.0),
        "bucket_sep": month(9, "bucket", "mean"),
    })
    # An autumn the record does not cover in full (the last year, a gap) is
    # not a normal.
    complete = by["month"].apply(lambda m: m.isin((8, 9, 10, 11)).sum() >= 120)
    return table[complete.reindex(table.index).fillna(False)]


def cell_features(
    dates: pd.DatetimeIndex,
    channels: np.ndarray,
    years: list[int],
    *,
    capacity_mm: float,
    cache_start_doy: int,
    cache_days: int,
    normals_years: int,
    normals_min_years: int,
) -> CellFeatures:
    """All seasons of one cell.

    Args:
        dates: Daily index of the record, contiguous.
        channels: ``[D, 8]`` in ``export.weather.CHANNELS`` order.
        years: Harvest years wanted.
        capacity_mm: Plant-available water of the topsoil bucket.
        cache_start_doy, cache_days: The season window, opening in Y-1.
        normals_years, normals_min_years: Window and minimum for the normals.
    """
    ch = np.asarray(channels, np.float32)
    rain, temp = ch[:, _SRC["rain"]], ch[:, _SRC["davtmp"]]
    et0 = np.maximum(0.0135 * (temp + 17.8) * ch[:, _SRC["irrad"]] * 0.408, 0.0).astype(np.float32)
    bucket = _bucket(rain.astype(np.float64), et0.astype(np.float64), max(capacity_mm, 5.0))
    tmin7 = _rolling(ch[:, _SRC["tmin"]], 7, "mean")
    tmean7 = _rolling(temp, 7, "mean")
    medium_all = np.stack([
        bucket, tmin7, _rolling(rain, 3, "sum"), _rolling(rain, 30, "sum"),
        _dry_days(rain, _WET_MM, 60.0),
    ], axis=1)
    short_all = ch[:, [_SRC[c] for c in SHORT_CHANNELS]]

    autumns = _autumn_table(dates, ch, bucket, tmin7, tmean7)
    first = dates[0]
    n_y = len(years)
    short = np.full((n_y, cache_days, len(SHORT_CHANNELS)), np.nan, np.float16)
    medium = np.full((n_y, cache_days, len(MEDIUM_CHANNELS)), np.nan, np.float16)
    long = np.full((n_y, len(LONG_FEATURES)), np.nan, np.float32)
    valid = np.zeros(n_y, bool)

    for j, year in enumerate(years):
        start = pd.Timestamp(year - 1, 1, 1) + pd.Timedelta(days=cache_start_doy - 1)
        i0 = (start - first).days
        prior = autumns.loc[(autumns.index >= year - 1 - normals_years) & (autumns.index <= year - 2)]
        if i0 < 0 or i0 + cache_days > len(dates) or len(prior) < normals_min_years:
            continue
        window = slice(i0, i0 + cache_days)
        short[j] = short_all[window]
        gdd = np.cumsum(np.maximum(temp[window], 0.0))
        medium[j] = np.column_stack([medium_all[window], gdd])
        long[j] = prior[list(LONG_FEATURES)].mean().to_numpy(np.float32)
        valid[j] = np.isfinite(short_all[window]).all()
    return CellFeatures(short, medium, long, valid)
