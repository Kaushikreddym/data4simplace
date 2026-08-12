"""The site stage: per-cell sowing calendar and altitude on the target grid.

Composes :mod:`~data4simplace.site.calendar` and
:mod:`~data4simplace.site.elevation` into the one dataset
:class:`~data4simplace.exporters.site_export.SiteExporter` writes. CO2 is a
global series rather than a field, so it is handled separately by
:mod:`~data4simplace.site.co2` and lands in its own file.

Gap filling is deliberately split in two, because the two gaps mean different
things:

* A cell **outside the crop's calendar mask** but inside cropland is a coastal
  or fringe cell whose nearest reporting unit is a short distance away.
  :func:`fill_calendar_gaps` copies the nearest classified cell's dates in,
  bounded by the exported cell mask so the search never walks out to sea.
* A cell **no fill can reach** falls back to ``site.fallback_sowing_doy``. That
  is an assumption, not data, so it is recorded in the ``calendar_source``
  column rather than blended into the rest.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.grid import TargetGrid
from data4simplace.site.calendar import load_calendar
from data4simplace.site.elevation import load_elevation
from data4simplace.soil.classify import fill_missing_cells

logger = logging.getLogger(__name__)

__all__ = ["CALENDAR_SOURCE_CODES", "SiteHandler", "fill_calendar_gaps"]

#: ``calendar_source`` code -> label written to the CSV. Kept as an integer on
#: the grid because an xarray field of strings does not survive regridding,
#: masking or a NetCDF round-trip cleanly.
CALENDAR_SOURCE_CODES: dict[int, str] = {
    0: "fallback",   # site.fallback_sowing_doy — an assumption, not data
    1: "product",    # the calendar product covers this cell
    2: "nearest",    # copied from the nearest covered cell within the mask
}

#: Fields the nearest-neighbour fill carries across together. They describe one
#: season, so filling them independently could pair one cell's sowing date with
#: another's harvest.
_FILLABLE = (
    "sowing_doy",
    "sowing_start_doy",
    "sowing_end_doy",
    "harvest_doy",
    "season_length_days",
    "calendar_filled",
)


def fill_calendar_gaps(
    site: xr.Dataset, within: xr.DataArray | None = None
) -> xr.Dataset:
    """Fill calendar gaps from the nearest covered cell, bounded by ``within``.

    Parameters
    ----------
    site:
        The site dataset, carrying ``calendar_present`` and the date fields.
    within:
        Boolean ``(lat, lon)`` mask bounding the search — normally the exported
        cell mask. Without it the fill is unbounded and would carry a land
        calendar far out over the sea, which is the same failure
        :func:`~data4simplace.soil.classify.fill_missing_cells` guards against
        for soil profiles.

    Returns
    -------
    xarray.Dataset
        A copy with the gaps filled and ``calendar_source`` set to 2 wherever a
        value was borrowed.
    """
    if "calendar_present" not in site:
        return site

    present = site["calendar_present"] == 1.0
    fillable = [name for name in _FILLABLE if name in site]
    if not fillable:
        return site

    filled = fill_missing_cells(site[fillable], within=within)
    assert isinstance(filled, xr.Dataset)

    out = site.copy()
    for name in fillable:
        out[name] = filled[name]

    borrowed = (~present) & out["sowing_doy"].notnull()
    out["calendar_source"] = xr.where(present, 1, xr.where(borrowed, 2, 0)).astype("int16")
    n = int(borrowed.sum())
    if n:
        logger.info("Calendar: %d cell(s) filled from the nearest covered cell", n)
    return out


class SiteHandler:
    """Build the per-cell site dataset (calendar + altitude).

    Parameters
    ----------
    config:
        The validated pipeline configuration. Reads ``paths.calendar_root``,
        ``paths.dem_path`` and the ``site`` block.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)

    def load(self) -> xr.Dataset:
        """Calendar and altitude on the target grid.

        Returns
        -------
        xarray.Dataset
            ``sowing_doy``, ``sowing_start_doy``, ``sowing_end_doy``,
            ``harvest_doy``, ``season_length_days``, ``calendar_filled``,
            ``calendar_present``, ``calendar_source`` and ``altitude_m`` on
            ``(lat, lon)``. Gaps are still NaN here — filling needs the exported
            cell mask, which the pipeline resolves after every processing stage,
            so it happens in :func:`fill_calendar_gaps`.
        """
        site = load_calendar(self._config, self._grid)
        site["altitude_m"] = load_elevation(self._config, self._grid)

        # Provisional: 1 where the product covers the cell, 0 elsewhere. The
        # gap fill upgrades the zeros it reaches to 2 (nearest).
        site["calendar_source"] = xr.where(
            site["calendar_present"] == 1.0, 1, 0
        ).astype("int16")

        covered = float(site["calendar_present"].mean()) * 100.0
        logger.info(
            "Site stage: %.0f%% of cells carry a %s calendar date; "
            "altitude from %s",
            covered,
            site.attrs.get("calendar_source", "?"),
            site["altitude_m"].attrs.get("source", "?"),
        )
        return site

    def summarise(self, site: xr.Dataset) -> dict[str, float]:
        """Small diagnostic summary, for logs and tests."""
        sowing = site["sowing_doy"].values
        finite = np.isfinite(sowing)
        return {
            "cells": float(sowing.size),
            "with_calendar": float(finite.sum()),
            "sowing_doy_min": float(np.nanmin(sowing)) if finite.any() else float("nan"),
            "sowing_doy_max": float(np.nanmax(sowing)) if finite.any() else float("nan"),
        }
