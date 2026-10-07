"""Train / validation / test splits, and why the two terms cannot share one.

Two splits are made and **both** are reported, because they answer different
questions:

*temporal*
    held-out **year blocks** — can the parameters extrapolate to a season they
    were not fitted on?
*spatial*
    held-out ``adm_id`` **within each region** — do the region's parameters
    transfer to units inside it that the fit never saw?

A model can pass one and fail the other, and a single 70/15/15 that mixes them
hides which.

**The phenology term cannot have the yield term's split.** CLMS covers eight
seasons, so a 15 % year block is one year and a 70/15/15 leaves six for
training — enough to fit but not enough to say anything about the held-out
one. Phenology therefore uses leave-one-year-out; yield, with 25 years, keeps
the 70/15/15.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["Split", "make_split"]


def _protect_window(
    years: np.ndarray, n_val: int, n_test: int, protect: tuple[int, int]
) -> tuple[int, int]:
    """Shrink the held-out block until ``protect`` keeps half its years to train on.

    A year-block split holds out the *last* block, which is the right shape for
    testing temporal extrapolation. It is the wrong shape when one reference
    only exists in those last years — the split then reads as ordinary while
    silently removing that reference from training altogether.
    """
    first, last = protect
    covered = years[(years >= first) & (years <= last)]
    if covered.size == 0:
        return n_val, n_test

    keep = covered.size - covered.size // 2          # at least half, rounded up
    while n_val + n_test > 1:
        held = set(years[-(n_val + n_test):].tolist())
        if len(set(covered.tolist()) - held) >= keep:
            break
        # Take from validation first: the test block is the harder question and
        # is what the acceptance numbers are quoted on.
        if n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    trained = sorted(set(covered.tolist()) - set(years[-(n_val + n_test):].tolist()))
    logger.info(
        "protected window %d-%d keeps %d of %d years in training (%s); "
        "hold-out is %d validation + %d test years",
        first, last, len(trained), covered.size,
        ", ".join(str(y) for y in trained) or "none", n_val, n_test,
    )
    return n_val, n_test


@dataclass(frozen=True, slots=True)
class Split:
    """Train / validation / test frames of ``(unit, year)`` rows.

    Attributes:
        train: Rows the gradient is taken on.
        val: Rows early stopping reads. Held out **temporally**.
        test: Rows scored once, at the end. Held out **spatially**.
        held_out_years: The years no training row comes from.
        held_out_units: The units no training row comes from.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    held_out_years: tuple[int, ...]
    held_out_units: tuple[str, ...]

    def summarise(self) -> str:
        return (
            f"split: {len(self.train)} train / {len(self.val)} val / "
            f"{len(self.test)} test unit-years; years held out "
            f"{list(self.held_out_years)}; {len(self.held_out_units)} units held out"
        )


def make_split(
    unit_years: pd.DataFrame,
    mode: str = "block",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    protect: tuple[int, int] | None = None,
) -> Split:
    """Split unit-years temporally and spatially at once.

    Args:
        unit_years: ``unit``, ``region_id``, ``year``.
        mode: ``"block"`` for the 70/15/15 year-block split (yield, 25 years),
            or ``"loyo"`` for leave-one-year-out (phenology, 8 seasons).
        val_fraction: Share of years held out for validation in ``block`` mode.
        test_fraction: Share of units held out per region, both modes.
        seed: Draw seed for the spatial hold-out.
        protect: Inclusive ``(first, last)`` window that must keep **at least
            half its years in training**. The references do not share a window:
            CLMS covers 2017-2024 only, and a 70/15/15 block split of 25 years
            holds out the last eight — which is precisely CLMS's whole record,
            leaving the primary phenology reference in validation and test with
            nothing to train on. The hold-out is shrunk until that cannot
            happen. ``None`` disables the guard.

    Returns:
        The :class:`Split`.

    Raises:
        ValueError: On an unknown ``mode`` or a pool too small to split.
    """
    if mode not in ("block", "loyo"):
        raise ValueError(f"mode must be 'block' or 'loyo', not {mode!r}")

    years = np.sort(unit_years["year"].unique())
    if years.size < 3:
        raise ValueError(f"need at least 3 harvest years to split; got {years.size}")

    if mode == "loyo":
        # The last two seasons, not a random pair: a temporal split whose
        # hold-out sits inside the training period tests interpolation, and the
        # question is extrapolation.
        val_years, test_years = years[-2:-1], years[-1:]
    else:
        n_val = max(1, int(round(val_fraction * years.size)))
        n_test = max(1, int(round(test_fraction * years.size)))
        if protect is not None:
            n_val, n_test = _protect_window(years, n_val, n_test, protect)
        val_years, test_years = years[-(n_val + n_test) : -n_test], years[-n_test:]

    rng = np.random.default_rng(seed)
    del years  # the year blocks are fixed above; what follows is spatial
    held_units: list[str] = []
    for _, block in unit_years.groupby("region_id", observed=True):
        units = np.sort(block["unit"].unique())
        take = int(np.floor(test_fraction * units.size))
        if take and units.size > 2:
            held_units += list(rng.choice(units, size=take, replace=False))

    is_val_year = unit_years["year"].isin(val_years)
    is_test_year = unit_years["year"].isin(test_years)
    is_held_unit = unit_years["unit"].isin(held_units)

    split = Split(
        train=unit_years[~is_val_year & ~is_test_year & ~is_held_unit].reset_index(drop=True),
        val=unit_years[is_val_year & ~is_held_unit].reset_index(drop=True),
        # The test set carries both hold-outs, so the number reported at the
        # end is the harder of the two questions rather than an average of them.
        test=unit_years[is_test_year | is_held_unit].reset_index(drop=True),
        held_out_years=tuple(int(y) for y in np.concatenate([val_years, test_years])),
        held_out_units=tuple(str(u) for u in held_units),
    )
    if split.train.empty:
        raise ValueError("the split left no training rows — the pool is too small")
    logger.info("%s", split.summarise())
    return split
