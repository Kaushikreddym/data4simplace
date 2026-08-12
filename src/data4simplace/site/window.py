"""The planting window as a pair of day-of-year thresholds per cell.

A rule-based SIMPLACE solution does not sow on a date — it evaluates weather
rules from ``vSowWindowStartDOY`` and sows on the first day one holds, forcing a
sowing on ``vSowWindowEndDOY`` if none does. So what it needs from the calendar
is the *window*, which is exactly what SAGE publishes (``plant.start`` …
``plant.end``) and what collapsing the calendar to ``sowing_doy`` throws away.

Three corrections stand between SAGE's fields and a pair the solution can use:

* **A missing window is centred on the sowing date**, ``site.window_min_days``
  either side, rather than dropped: a cell with a date but no window must still
  sow. Where even the date is missing the site stage's own fallback has already
  filled it, and ``site.csv``'s ``calendar_source`` column is where that stays
  visible — the schedule carries no provenance field of its own.
* **A window that crosses New Year is re-wrapped** (``end`` below ``start``
  means the length is ``end - start + 365``). The solution's test is
  ``DOY >= start and DOY <= end`` with no wrap, so the window is then clamped
  to end at DOY 365 rather than silently inverting: an inverted pair would make
  the test false on every day of the year and the cell would never sow.
* **The length is clamped** to ``[window_min_days, window_max_days]``. SAGE's
  windows run from a few days to over three months; a zero-length window turns
  a rule back into a fixed date, which is the failure this whole path exists to
  remove.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = ["SowingWindow"]


@dataclass(frozen=True, slots=True)
class SowingWindow:
    """Per-cell planting window on the target grid.

    Attributes
    ----------
    start, end:
        ``(n_lat, n_lon)`` day-of-year thresholds, ``1 <= start <= end <= 365``.
        NaN only where the cell has no calendar at all.
    n_from_product:
        Cells whose window came from the calendar product's own start/end pair,
        as opposed to being centred on the sowing date. The rest are an
        assumption, and this count is what makes that visible in a run log.
    """

    start: np.ndarray
    end: np.ndarray
    n_from_product: int
    n_cells: int

    @classmethod
    def from_site(cls, site: xr.Dataset, config: PipelineConfig) -> "SowingWindow":
        """Derive the window grid from the site stage's calendar fields."""
        settings = config.site
        sowing = _grid(site, "sowing_doy")
        start = _grid(site, "sowing_start_doy")
        end = _grid(site, "sowing_end_doy")

        length = end - start
        # A window crossing New Year comes back negative; re-wrap it.
        length = np.where(length < 0, length + 365.0, length)

        from_product = np.isfinite(start) & np.isfinite(length)
        missing = ~from_product & np.isfinite(sowing)
        if missing.any():
            start = np.where(missing, sowing - settings.window_min_days, start)
            length = np.where(missing, 2 * settings.window_min_days, length)

        length = np.clip(length, settings.window_min_days, settings.window_max_days)
        start = np.clip(start, 1.0, 365.0 - length)
        end = start + length

        window = cls(
            start=start,
            end=end,
            n_from_product=int(from_product.sum()),
            n_cells=int(np.isfinite(start).sum()),
        )
        logger.info(
            "Sowing window: %d of %d cells from the calendar product's own "
            "start/end, %d centred on the sowing date (+/- %d d); lengths "
            "clamped to [%d, %d] d",
            window.n_from_product, window.n_cells,
            window.n_cells - window.n_from_product, settings.window_min_days,
            settings.window_min_days, settings.window_max_days,
        )
        return window

    def columns(self, cell_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(start, end)`` for every row of a cell table, as whole days.

        The window is already on the target grid, so the table's ``row``/``col``
        index it directly — one vectorised gather rather than a per-cell lookup,
        which matters at Europe's ~10^5 cells.
        """
        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)
        return (
            np.rint(self.start[rows, cols]).astype("int64"),
            np.rint(self.end[rows, cols]).astype("int64"),
        )


def _grid(site: xr.Dataset, name: str) -> np.ndarray:
    """A site variable as a float grid, all-NaN when the stage did not write it."""
    if name not in site:
        shape = tuple(site.sizes[d] for d in ("lat", "lon"))
        return np.full(shape, np.nan)
    return np.asarray(site[name].values, dtype="float64")
