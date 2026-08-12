"""Read the export's per-cell site table and its CO2 series.

``site/site.csv`` carries what a run needs about the *place* — where it is, how
high it is and when the crop goes in the ground — and ``site/co2.csv`` the
annual global CO2. Both come from data4simplace's site stage; before it existed
each runner substituted a continental constant (DOY 270 everywhere, altitude 0,
a hard-coded CO2 table).

An export written before that stage has neither file. Rather than fail, this
module returns an explicitly-labelled constant fallback, so an old export still
runs and the run's own log says which inputs were assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["FALLBACK_CO2_PPM", "SiteTable", "read_co2", "read_site"]

#: Global-mean CO2 [ppm] at decadal anchors, interpolated between. Used only
#: when the export has no ``co2.csv``.
FALLBACK_CO2_PPM: dict[int, float] = {
    1979: 336.8, 1990: 354.4, 2000: 369.6, 2010: 389.9, 2020: 414.2, 2024: 422.7,
}

#: Sowing day-of-year assumed when the export carries no site table. This is
#: the constant the old runner used for all of Europe, and the single largest
#: error in that run.
FALLBACK_SOWING_DOY = 270

#: Altitude [m] assumed when the export carries no site table.
FALLBACK_ALTITUDE_M = 0.0


@dataclass
class SiteTable:
    """Per-cell site attributes, indexed by ``SimplaceID``.

    Attributes
    ----------
    frame:
        The table, indexed by ``location``. Always carries ``sowing_doy`` and
        ``altitude_m``; the calendar window and provenance columns are present
        only when the export supplied them.
    source:
        ``"export"`` or ``"fallback"`` -- which is the difference between a
        data-driven sowing date and an assumed one.
    """

    frame: pd.DataFrame
    source: str

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"

    def sowing_doy(self, ids: np.ndarray) -> np.ndarray:
        """Sowing day-of-year for ``ids``, in that order."""
        return self._column(ids, "sowing_doy", FALLBACK_SOWING_DOY).astype(int)

    def altitude_m(self, ids: np.ndarray) -> np.ndarray:
        """Altitude [m] for ``ids``, in that order."""
        return self._column(ids, "altitude_m", FALLBACK_ALTITUDE_M)

    def sowing_window(
        self, ids: np.ndarray, min_days: int = 7, max_days: int = 120
    ) -> tuple[np.ndarray, np.ndarray]:
        """Planting window as ``(start_doy, length_days)`` for ``ids``.

        This is what a rule-based sowing solution consumes: it evaluates its
        weather rules from ``start_doy`` and sows on the first day one holds,
        forcing a sowing on the last day if none does. The window is therefore
        a *constraint*, not a date — which is exactly the shape the SAGE
        calendar publishes (``plant.start`` … ``plant.end``), and a better use
        of that product than collapsing it to its midpoint.

        The length is clamped: SAGE's windows run from a few days to over three
        months, and a window of 0 days turns a rule into a fixed date while one
        that wraps the year end breaks the solution's ``start + length`` test.
        Where no window was exported, the sowing date itself is used with a
        symmetric default width.

        Returns:
            ``(start_doy, length_days)``, both integer arrays.
        """
        sowing = self.sowing_doy(ids)
        start = self._column(ids, "sowing_start_doy", np.nan)
        end = self._column(ids, "sowing_end_doy", np.nan)

        length = end - start
        # A window crossing New Year comes back negative; re-wrap it.
        length = np.where(length < 0, length + 365.0, length)

        missing = ~np.isfinite(start) | ~np.isfinite(length)
        if missing.any():
            # No window: centre a default one on the sowing date itself.
            start = np.where(missing, sowing - min_days, start)
            length = np.where(missing, 2 * min_days, length)

        length = np.clip(length, min_days, max_days)
        # The solution tests `DOY <= start + length`, so the window must not
        # run past the year end.
        start = np.clip(start, 1, 365 - length)
        return start.astype(int), length.astype(int)

    def _column(self, ids: np.ndarray, name: str, default: float) -> np.ndarray:
        if name not in self.frame.columns:
            return np.full(len(ids), default, dtype="float64")
        return (
            self.frame[name]
            .reindex(np.asarray(ids, dtype=np.int64))
            .fillna(default)
            .to_numpy(dtype="float64")
        )

    def summarise(self) -> str:
        """One line for a run log: what the site inputs actually are."""
        if self.is_fallback:
            return (
                f"site: FALLBACK -- sowing DOY {FALLBACK_SOWING_DOY} and "
                f"altitude {FALLBACK_ALTITUDE_M:.0f} m assumed for every cell "
                f"(the export carries no site.csv)"
            )
        sowing = self.frame.get("sowing_doy")
        altitude = self.frame.get("altitude_m")
        parts = [f"site: {len(self.frame)} cells from the export"]
        if sowing is not None:
            parts.append(
                f"sowing DOY {sowing.min():.0f}-{sowing.max():.0f} "
                f"(median {sowing.median():.0f})"
            )
        if altitude is not None:
            parts.append(f"altitude {altitude.min():.0f}-{altitude.max():.0f} m")
        if "calendar_source" in self.frame:
            counts = self.frame["calendar_source"].value_counts().to_dict()
            parts.append(f"calendar sources {counts}")
        return ", ".join(parts)


def read_site(export_dir: str | Path) -> SiteTable:
    """Read ``site/site.csv``, or fall back to the documented constants."""
    path = Path(export_dir) / "site" / "site.csv"
    if not path.is_file():
        logger.warning(
            "No site table at %s. Falling back to a constant sowing DOY %d and "
            "altitude %.0f m for every cell -- re-export with "
            "flags.run_site_processing to make these data instead of "
            "assumptions.", path, FALLBACK_SOWING_DOY, FALLBACK_ALTITUDE_M,
        )
        return SiteTable(frame=pd.DataFrame().rename_axis("location"), source="fallback")

    frame = pd.read_csv(path).set_index("location")
    frame.index = frame.index.astype(np.int64)
    table = SiteTable(frame=frame, source="export")
    logger.info(table.summarise())
    return table


def read_co2(export_dir: str | Path, years: list[int] | None = None) -> pd.Series:
    """Annual CO2 [ppm] from ``site/co2.csv``, or the built-in table.

    Years outside the series are carried from its nearest end rather than
    dropped: losing a season because its CO2 is missing is a worse failure than
    reusing a neighbouring year's value, which moves a yield by well under a
    percent.
    """
    path = Path(export_dir) / "site" / "co2.csv"
    if path.is_file():
        # The file leads with a '# ... source: ...' provenance comment.
        frame = pd.read_csv(path, comment="#")
        series = frame.set_index("year")["co2_ppm"].astype("float64")
        logger.info(
            "CO2 from %s: %d years (%d-%d)",
            path.name, len(series), int(series.index.min()), int(series.index.max()),
        )
    else:
        series = pd.Series(FALLBACK_CO2_PPM, dtype="float64")
        logger.warning(
            "No CO2 series at %s; using the built-in global-mean table", path
        )
    series.index = series.index.astype(int)
    series = series.sort_index()

    if years is None:
        return series
    full = range(
        min(min(years), int(series.index.min())),
        max(max(years), int(series.index.max())) + 1,
    )
    return series.reindex(full).interpolate(limit_direction="both").reindex(years)
