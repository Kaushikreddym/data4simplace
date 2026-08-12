"""SIMPLACE site file generator (``site/site.csv``).

One row per exported cell, carrying what a run needs about the *place* rather
than about its soil or its weather: where it is, how high it is and when the
crop goes in the ground. SIMPLACE reads these as the project-file columns
``vWGS84_lat``/``vWGS84_lon``, ``vAltitude`` and ``vSowingDOY``; torchcrop reads
them as ``site.latitude``, ``site.altitude`` and ``site.idpl``.

The file is keyed on ``location`` (the ``SimplaceID``) exactly like
``soil.csv``, so the three exports join without a lookup table.

Unlike the weather, soil and management exporters this one has **no SIMPLACE
reference file** to inspect: the reference project carries the same information
split across ``location.csv`` (latitude, altitude) and the solution's
``vSowingDOY`` variable, neither of which is a per-cell table. The schema is
therefore defined here, in :data:`SITE_COLUMNS`, and the class still derives
from :class:`~data4simplace.exporters.base_exporter.BaseExporter` so it shares
the delimiter, sentinel and conformance handling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import BaseExporter, ReferenceSpec
from data4simplace.site.handler import CALENDAR_SOURCE_CODES

logger = logging.getLogger(__name__)

__all__ = ["SITE_COLUMNS", "SiteExporter"]

#: Column order of ``site.csv``.
SITE_COLUMNS: tuple[str, ...] = (
    "location",
    "latitude",
    "longitude",
    "altitude_m",
    "sowing_doy",
    "sowing_start_doy",
    "sowing_end_doy",
    "harvest_doy",
    "season_length_days",
    "calendar_filled",
    "calendar_source",
    "calendar_product",
)

#: Grid variable -> output column, for the fields taken straight off the grid.
_GRID_COLUMNS: dict[str, str] = {
    "altitude_m": "altitude_m",
    "sowing_doy": "sowing_doy",
    "sowing_start_doy": "sowing_start_doy",
    "sowing_end_doy": "sowing_end_doy",
    "harvest_doy": "harvest_doy",
    "season_length_days": "season_length_days",
}

#: Columns rounded to whole days on the way out. A day-of-year sampled from a
#: 0.5 degree field is not accurate to six decimals, and SIMPLACE's
#: ``vSowingDOY`` is an INT.
_DAY_COLUMNS: tuple[str, ...] = (
    "sowing_doy",
    "sowing_start_doy",
    "sowing_end_doy",
    "harvest_doy",
    "season_length_days",
)


class SiteExporter(BaseExporter):
    """Export the per-cell site table."""

    kind = "site"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None = None) -> None:
        super().__init__(config, reference_path)

    @property
    def spec(self) -> ReferenceSpec:
        """The schema.

        Overridden to skip the base class' "no reference available" warning:
        for this file there is no reference to be missing, so the built-in
        schema is the intended path rather than a degraded one.
        """
        if self._reference_path is not None:
            return super().spec
        if self._spec is None:
            self._spec = self.fallback_spec()
        return self._spec

    def fallback_spec(self) -> ReferenceSpec:
        """The schema, which for this file is always the built-in one."""
        return ReferenceSpec(
            delimiter=",",
            columns=list(SITE_COLUMNS),
            missing_value=str(self._config.missing_value),
        )

    def build_frame(self, site: xr.Dataset, cell_table: pd.DataFrame) -> pd.DataFrame:
        """Build the site table (one row per exported cell).

        Parameters
        ----------
        site:
            The site dataset from :class:`~data4simplace.site.SiteHandler`,
            already gap-filled and masked to the exported cells.
        cell_table:
            Grid cell table (``SimplaceID``, ``row``, ``col``, ``lat``, ``lon``).

        Returns
        -------
        pandas.DataFrame
            Rows in ``cell_table`` order. Cells with no sowing date after the
            gap fill take ``site.fallback_sowing_doy`` and are marked
            ``calendar_source = "fallback"``, so an assumed date is never
            indistinguishable from a sampled one.
        """
        columns = self.spec.columns or list(SITE_COLUMNS)
        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)

        data: dict[str, object] = {
            "location": np.asarray(cell_table["SimplaceID"], dtype=np.int64),
            "latitude": np.round(np.asarray(cell_table["lat"], dtype=float), 6),
            "longitude": np.round(np.asarray(cell_table["lon"], dtype=float), 6),
        }
        for var, col in _GRID_COLUMNS.items():
            data[col] = (
                np.asarray(site[var].values, dtype="float64")[rows, cols]
                if var in site
                else np.full(rows.size, np.nan)
            )

        data["calendar_filled"] = (
            np.asarray(site["calendar_filled"].values, dtype="float64")[rows, cols]
            if "calendar_filled" in site
            else np.full(rows.size, np.nan)
        )
        source_code = (
            np.asarray(site["calendar_source"].values, dtype="int64")[rows, cols]
            if "calendar_source" in site
            else np.zeros(rows.size, dtype="int64")
        )

        frame = pd.DataFrame(data)
        frame = self._apply_fallback(frame, source_code)

        for col in _DAY_COLUMNS:
            if col in frame:
                frame[col] = frame[col].round(0)
        frame["altitude_m"] = frame["altitude_m"].round(1)
        # A NaN flag would be indistinguishable from "not extrapolated" once the
        # sentinel is written, so an unknown flag stays 0 = reported.
        frame["calendar_filled"] = frame["calendar_filled"].fillna(0.0).astype(int)
        frame["calendar_product"] = str(site.attrs.get("calendar_crop", ""))

        missing = [c for c in columns if c not in frame.columns]
        for col in missing:
            frame[col] = self.spec.missing_value
        return frame[list(columns)]

    def _apply_fallback(
        self, frame: pd.DataFrame, source_code: np.ndarray
    ) -> pd.DataFrame:
        """Substitute the configured sowing DOY where no date could be sampled."""
        fallback = self._config.site.fallback_sowing_doy
        gaps = frame["sowing_doy"].isna().to_numpy()
        if gaps.any():
            frame.loc[gaps, "sowing_doy"] = float(fallback)
            source_code = np.where(gaps, 0, source_code)
            logger.warning(
                "%d of %d cells had no calendar date after the gap fill and took "
                "the site.fallback_sowing_doy of %d; they are marked "
                "calendar_source=fallback",
                int(gaps.sum()), len(frame), fallback,
            )
        frame["calendar_source"] = [
            CALENDAR_SOURCE_CODES.get(int(code), "fallback") for code in source_code
        ]
        return frame

    def export(
        self,
        site: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
    ) -> Path:
        """Write ``site/site.csv``; return its path."""
        frame = self.build_frame(site, cell_table)
        return self.write_csv(frame, Path(output_dir) / "site" / "site.csv")
