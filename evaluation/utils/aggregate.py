"""Country aggregation for both sides of the comparison.

Both sides need the same treatment of two problems, so both live here:

**Weights.** A national yield is the production-weighted mean of its regions,
not the plain mean of them — a 200 000 ha region and a 2 000 ha one do not
carry equal weight. CyBench supplies ``harvest_area``; where it is missing (75 %
of the German rows) the aggregation falls back to unweighted and *records that
it did*, so a country whose method changes between years is visible rather than
a step in the time series nobody can explain.

**Circularity.** Averaging a day-of-year is not averaging a number. Anything
carrying a date goes through :func:`utils.doy.circular_mean_doy`; durations
(season length) stay linear.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import MIN_WEIGHT_COVERAGE
from .doy import circular_mean_doy

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_observed_calendar",
    "aggregate_observed_yield",
    "aggregate_simulated",
    "pair_observations",
    "to_dry_matter",
    "weighted_mean",
]


def weighted_mean(
    values: pd.Series,
    weights: pd.Series | None = None,
    min_coverage: float = MIN_WEIGHT_COVERAGE,
    circular: bool = False,
) -> tuple[float, str, int]:
    """Mean of ``values``, weighted where the weights are trustworthy.

    Args:
        values: The quantity to average. NaN entries are dropped.
        weights: Aligned non-negative weights, or ``None`` for unweighted.
        min_coverage: Fraction of valid values that must carry a positive,
            finite weight before weighting is used. Below it the weighted mean
            would describe only the subset that happens to be weighted, which
            is worse than an honest unweighted mean over everything.
        circular: Treat ``values`` as days-of-year.

    Returns:
        ``(mean, method, n)`` where ``method`` is ``"weighted"``,
        ``"unweighted"`` or ``"none"``, and ``n`` is the number of values the
        mean was taken over. ``mean`` is NaN when ``n`` is 0.
    """
    valid = values.notna()
    n = int(valid.sum())
    if n == 0:
        return float("nan"), "none", 0

    v = values[valid]
    w = None
    if weights is not None:
        candidate = weights[valid]
        usable = candidate.notna() & np.isfinite(candidate) & (candidate > 0)
        if usable.mean() >= min_coverage and usable.any():
            v, w = v[usable], candidate[usable]
            n = int(usable.sum())

    if circular:
        return (
            circular_mean_doy(v.to_numpy(), None if w is None else w.to_numpy()),
            "weighted" if w is not None else "unweighted",
            n,
        )
    if w is None:
        return float(v.mean()), "unweighted", n
    return float(np.average(v.to_numpy(), weights=w.to_numpy())), "weighted", n


def _group_means(
    frame: pd.DataFrame,
    by: list[str],
    value_cols: dict[str, bool],
    weight_col: str | None,
    min_coverage: float,
    method_col: str,
) -> pd.DataFrame:
    """Apply :func:`weighted_mean` to every value column of every group.

    Args:
        frame: Rows to aggregate.
        by: Grouping keys.
        value_cols: ``{column: is_circular}``.
        weight_col: Weight column, or ``None``.
        min_coverage: Passed through.
        method_col: Name for the column recording which method was used. The
            method is taken from the *first* value column, which is why the
            caller lists the headline quantity first.

    Returns:
        One row per group, with the group keys, the means, ``method_col``, and
        ``n_regions``/``n_cells`` supplied by the caller as ``count_col``.
    """
    records = []
    for keys, block in frame.groupby(by, sort=True, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(by, keys))
        weights = block[weight_col] if weight_col else None
        for i, (col, circular) in enumerate(value_cols.items()):
            mean, method, n = weighted_mean(
                block[col], weights, min_coverage, circular=circular
            )
            record[col] = mean
            if i == 0:
                record[method_col] = method
                record["n"] = n
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_simulated(
    frame: pd.DataFrame,
    value_cols: dict[str, bool],
    by: list[str] | None = None,
    weight_col: str | None = None,
    min_coverage: float = MIN_WEIGHT_COVERAGE,
) -> pd.DataFrame:
    """Average simulated 10 km cells up to country level.

    Args:
        frame: Simulation rows carrying ``country`` and the value columns.
            Cells outside the CyBench footprint must already be dropped.
        value_cols: ``{column: is_circular}``, headline quantity first.
        by: Grouping keys; ``["country", "year"]`` by default.
        weight_col: Per-cell weight. ``None`` — the default — gives every
            simulated cropland cell equal weight. That is the honest choice
            here: the pipeline's cell set is already restricted to cropland
            (``flags.apply_agricultural_mask``), but nothing in the export
            carries a per-cell *wheat* area, so any weight would have to be
            invented. It does mean the national mean is an average over
            cropland cells, not over the wheat crop, and a country whose wheat
            concentrates in its best cells will read low.
        min_coverage: Passed to :func:`weighted_mean`.

    Returns:
        One row per group with the means, ``sim_method`` and ``n_cells``.

    Raises:
        KeyError: If ``country`` or a value column is missing.
    """
    by = by or ["country", "year"]
    missing = ({"country", *value_cols} | set(by)) - set(frame.columns)
    if missing:
        raise KeyError(f"simulation frame lacks {sorted(missing)}")

    out = _group_means(
        frame, by, value_cols, weight_col, min_coverage, "sim_method"
    ).rename(columns={"n": "n_cells"})
    logger.info(
        "simulated: %d country-groups, %d-%d cells each",
        len(out), int(out["n_cells"].min()), int(out["n_cells"].max()),
    )
    return out


def aggregate_observed_yield(
    frame: pd.DataFrame,
    min_coverage: float = MIN_WEIGHT_COVERAGE,
) -> pd.DataFrame:
    """Average CyBench sub-national yields up to country level.

    Weighted by ``harvest_area`` wherever enough regions carry one, which makes
    the result the national yield ``sum(production) / sum(area)`` rather than
    the mean of regional yields.

    Args:
        frame: Output of :func:`utils.cybench.load_yield`.
        min_coverage: Passed to :func:`weighted_mean`.

    Returns:
        Columns ``country``, ``year``, ``obs_yield`` [t/ha], ``obs_method``,
        ``n_regions``.
    """
    renamed = frame.rename(columns={"harvest_year": "year", "yield": "obs_yield"})
    out = _group_means(
        renamed,
        ["country", "year"],
        {"obs_yield": False},
        "harvest_area",
        min_coverage,
        "obs_method",
    ).rename(columns={"n": "n_regions"})

    share = (out["obs_method"] == "weighted").mean()
    logger.info(
        "observed yields: %d country-years, %.0f%% area-weighted",
        len(out), 100.0 * share,
    )
    return out


def aggregate_observed_calendar(
    calendar: pd.DataFrame,
    crop_mask: pd.DataFrame | None = None,
    min_coverage: float = MIN_WEIGHT_COVERAGE,
) -> pd.DataFrame:
    """Average the static CyBench crop calendar up to country level.

    ``sos`` and ``eos`` are days-of-year, so both are averaged circularly —
    without which Spain, whose regional ``sos`` straddles New Year (DOY 0.7 to
    363), would come out at the start of July.

    Args:
        calendar: Output of :func:`utils.cybench.load_calendar`.
        crop_mask: Output of :func:`utils.cybench.load_crop_mask`, supplying
            ``crop_area`` as the weight. ``None`` averages unweighted.
        min_coverage: Passed to :func:`weighted_mean`.

    Returns:
        Columns ``country``, ``sos``, ``eos``, ``obs_method``, ``n_regions``.
        No ``year``: the calendar is climatological.
    """
    frame = calendar.copy()
    weight_col = None
    if crop_mask is not None and not crop_mask.empty:
        frame = frame.merge(
            crop_mask[["country", "adm_id", "crop_area"]],
            on=["country", "adm_id"],
            how="left",
        )
        weight_col = "crop_area"

    out = _group_means(
        frame,
        ["country"],
        {"sos": True, "eos": True},
        weight_col,
        min_coverage,
        "obs_method",
    ).rename(columns={"n": "n_regions"})
    logger.info("observed calendars: %d countries", len(out))
    return out


def pair_observations(
    simulated: pd.DataFrame,
    observed: pd.DataFrame,
    on: list[str],
) -> pd.DataFrame:
    """Inner-join the two sides and report what the join dropped.

    An inner join is the right default — a metric can only be computed where
    both sides exist — but a silent one hides a country that fell out entirely.

    Args:
        simulated: Aggregated simulation.
        observed: Aggregated reference.
        on: Join keys, e.g. ``["country", "year"]`` or ``["country"]``.

    Returns:
        The joined frame, sorted by ``on``.
    """
    paired = simulated.merge(observed, on=on, how="inner").sort_values(on)

    sim_only = set(map(tuple, simulated[on].to_numpy())) - set(
        map(tuple, paired[on].to_numpy())
    )
    obs_only = set(map(tuple, observed[on].to_numpy())) - set(
        map(tuple, paired[on].to_numpy())
    )
    logger.info(
        "paired %d rows; %d simulated-only and %d observed-only dropped",
        len(paired), len(sim_only), len(obs_only),
    )
    if "country" in on:
        lost = sorted(
            set(simulated["country"]) ^ set(observed["country"])
        )
        if lost:
            logger.warning("countries present on only one side: %s", lost)
    return paired.reset_index(drop=True)


def to_dry_matter(
    yield_value: pd.Series | np.ndarray, moisture: float
) -> pd.Series | np.ndarray:
    """Convert a market-moisture yield to dry matter.

    **Not applied by default.** The notebooks compare raw values, so the
    simulated grain dry matter and CyBench's market-moisture statistics stay on
    their own bases and the resulting offset is visible in Bias and MAPE
    instead of being absorbed. Call this explicitly to put both sides on dry
    matter.

    Args:
        yield_value: Yield at ``moisture`` water content.
        moisture: Gravimetric water content as a fraction, e.g. 0.135.

    Returns:
        ``yield_value * (1 - moisture)``.

    Raises:
        ValueError: If ``moisture`` is not in ``[0, 1)``.
    """
    if not 0.0 <= moisture < 1.0:
        raise ValueError(f"moisture must be in [0, 1), got {moisture}")
    return yield_value * (1.0 - moisture)
