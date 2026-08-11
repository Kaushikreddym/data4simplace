"""Loaders for the CyBench reference tables.

CyBench publishes one directory per country under ``cybench-data/<crop>/``.
Three files are used here:

============================  ===========================================
``yield_<crop>_<c>.csv``      per ``(adm_id, harvest_year)``: ``yield``
                              [t/ha], ``harvest_area`` [ha], ``production``
``crop_calendar_<crop>_<c>``  per ``adm_id``: ``sos``, ``eos`` — fractional
                              day-of-year, **static**, no year dimension
``crop_mask_<crop>_<c>.csv``  per ``adm_id``: ``crop_area``, the weight the
                              static calendar is averaged with
============================  ===========================================

The yield schema is not uniform across countries: Germany ships four extra
columns (``season_name``, ``planting_year``, ``planting_date``,
``harvest_date``, ``planted_area``) that are empty and absent everywhere else,
so the loader projects every country onto the common set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import (
    CROP,
    CYBENCH_CALENDAR_TEMPLATE,
    CYBENCH_CROP_DIR,
    CYBENCH_CROP_MASK_TEMPLATE,
    CYBENCH_YIELD_TEMPLATE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CALENDAR_COLUMNS",
    "YIELD_COLUMNS",
    "load_calendar",
    "load_crop_mask",
    "load_yield",
]

#: The columns every country's yield file has. Anything else is dropped.
YIELD_COLUMNS: tuple[str, ...] = (
    "country_code",
    "adm_id",
    "harvest_year",
    "yield",
    "harvest_area",
    "production",
)

CALENDAR_COLUMNS: tuple[str, ...] = ("adm_id", "sos", "eos")


def _read_country_table(
    country: str,
    template: str,
    crop: str,
    crop_dir: Path,
    required: tuple[str, ...],
) -> pd.DataFrame | None:
    """Read one country's CSV, or return ``None`` if it is absent.

    A missing file is a normal outcome — CyBench covers 23 of the 31 EU and
    Schengen countries — so it is logged and skipped rather than raised. A file
    that exists but lacks a required column *is* an error: silently dropping it
    would leave a country out of the results with no trace.
    """
    path = crop_dir / country / template.format(crop=crop, country=country)
    if not path.is_file():
        logger.info("%s: no %s", country, path.name)
        return None

    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required column(s) {sorted(missing)}")

    frame = frame[list(required)].copy()
    frame["country"] = country
    return frame


def load_yield(
    countries: list[str],
    crop: str = CROP,
    crop_dir: Path | None = None,
    years: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Sub-national observed yields for a list of countries.

    Args:
        countries: Two-letter CyBench codes (``EL`` for Greece, not ``GR``).
        crop: CyBench crop sub-tree.
        crop_dir: Root holding ``<country>/`` directories.
        years: Inclusive ``(first, last)`` harvest-year filter.

    Returns:
        Columns ``country``, ``adm_id``, ``harvest_year``, ``yield``
        [t/ha], ``harvest_area`` [ha], ``production``, plus
        ``has_area_weight``. Rows with a missing or non-positive yield are
        dropped — a reported zero would drag a country average down without
        meaning a zero harvest.

    Raises:
        ValueError: If no country yielded any rows.
    """
    crop_dir = crop_dir or CYBENCH_CROP_DIR
    frames = [
        f
        for f in (
            _read_country_table(c, CYBENCH_YIELD_TEMPLATE, crop, crop_dir, YIELD_COLUMNS)
            for c in countries
        )
        if f is not None
    ]
    if not frames:
        raise ValueError(f"no CyBench yield files found under {crop_dir}")

    frame = pd.concat(frames, ignore_index=True)
    frame["harvest_year"] = frame["harvest_year"].astype(int)

    before = len(frame)
    frame = frame[frame["yield"].notna() & (frame["yield"] > 0)]
    if before != len(frame):
        logger.info("dropped %d rows with a missing or non-positive yield",
                    before - len(frame))

    if years is not None:
        first, last = years
        frame = frame[frame["harvest_year"].between(first, last)]

    # Recorded per row rather than inferred later: whether a country-year can
    # be area-weighted is a property of the data, and the aggregation reports
    # which method it fell back to.
    frame["has_area_weight"] = frame["harvest_area"].notna() & (
        frame["harvest_area"] > 0
    )

    frame = frame.reset_index(drop=True)
    logger.info(
        "observed yields: %d rows, %d countries, %d-%d, %.0f%% area-weighted",
        len(frame), frame["country"].nunique(), frame["harvest_year"].min(),
        frame["harvest_year"].max(), 100.0 * frame["has_area_weight"].mean(),
    )
    return frame


def load_calendar(
    countries: list[str],
    crop: str = CROP,
    crop_dir: Path | None = None,
) -> pd.DataFrame:
    """Observed crop calendars (``sos``/``eos``) for a list of countries.

    The calendar is **static** — one start and end of season per administrative
    unit, with no year dimension — so every simulated season is compared
    against the same reference date. Interannual scatter in the simulation
    therefore cannot be rewarded, only penalised; the per-country metrics say
    how far the long-term position is off, not how well the model tracks a
    warm year.

    Args:
        countries: Two-letter CyBench codes.
        crop: CyBench crop sub-tree.
        crop_dir: Root holding ``<country>/`` directories.

    Returns:
        Columns ``country``, ``adm_id``, ``sos``, ``eos`` (fractional
        day-of-year).

    Raises:
        ValueError: If no country yielded any rows.
    """
    crop_dir = crop_dir or CYBENCH_CROP_DIR
    frames = [
        f
        for f in (
            _read_country_table(
                c, CYBENCH_CALENDAR_TEMPLATE, crop, crop_dir, CALENDAR_COLUMNS
            )
            for c in countries
        )
        if f is not None
    ]
    if not frames:
        raise ValueError(f"no CyBench crop calendars found under {crop_dir}")

    frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    logger.info(
        "observed calendars: %d regions, %d countries", len(frame),
        frame["country"].nunique(),
    )
    return frame


def load_crop_mask(
    countries: list[str],
    crop: str = CROP,
    crop_dir: Path | None = None,
) -> pd.DataFrame:
    """Per-region cropped area, the weight for averaging the static calendar.

    A calendar averaged over administrative units would give a 200 ha alpine
    district the same say as a 200 000 ha plain. ``crop_area`` fixes that.

    Args:
        countries: Two-letter CyBench codes.
        crop: CyBench crop sub-tree.
        crop_dir: Root holding ``<country>/`` directories.

    Returns:
        Columns ``country``, ``adm_id``, ``crop_area``,
        ``crop_area_percentage``. Empty (with those columns) if no country
        ships a mask, which lets the caller fall back to unweighted averaging
        without a special case.
    """
    crop_dir = crop_dir or CYBENCH_CROP_DIR
    required = ("adm_id", "crop_area", "crop_area_percentage")
    frames = [
        f
        for f in (
            _read_country_table(
                c, CYBENCH_CROP_MASK_TEMPLATE, crop, crop_dir, required
            )
            for c in countries
        )
        if f is not None
    ]
    if not frames:
        logger.warning("no crop masks found; calendars will average unweighted")
        return pd.DataFrame(columns=["country", *required])

    frame = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    logger.info("crop masks: %d regions", len(frame))
    return frame
