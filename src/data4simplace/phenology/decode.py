"""Decode the CLMS HRL Croplands date and crop-type rasters.

The three layers this stage reads are all categorical in the sense that matters:
``CTY`` is a class code, ``CPMCE``/``CPMCH`` pack a year and a day-of-year into
one ``YYDOY`` integer and reserve the high values as flags. Neither survives
interpolation -- averaging class ``1110`` with ``1130`` gives ``1120``, "barley",
and averaging ``22300`` with ``23100`` gives a date in the wrong year -- so every
read in this package is nearest-neighbour and every aggregate is a median or a
count, never a mean.

**Dates are returned as days since an anchor, not as a day-of-year.** A raw DOY
wraps at New Year, which makes "the mean of DOY 300 and DOY 40" meaningless and
a median across the boundary wrong. Anchored on 1 August of ``season_year - 1``,
a winter-wheat emergence lands near +30..+120 and its harvest near +330..+380 --
both monotone, so an ordinary median is valid and no circular statistics are
needed. Conversion back to a day-of-year happens once, at write time, in
:mod:`data4simplace.phenology.reduce`.

**The class table lives here rather than in a file.** It is the product's own
``clms_hrlvlcc_cty.clr`` legend, which ships beside every granule -- in the FME
zip, and as one of the seven members of each S3 tile prefix. Reading it at
runtime would still be wrong: the fetch keeps only the rasters, so the copy on
disk is incidental, and a class map that can go missing is a class map that will.
The 21 entries below are transcribed from it, and
``tests/test_phenology.py`` pins the ones the split depends on.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ANCHOR_MONTH",
    "ANCHOR_DAY",
    "CTY_CLASSES",
    "DATE_FLAGS",
    "MAX_DATE_VALUE",
    "N_BINS",
    "anchor_for",
    "days_to_doy",
    "decode_yydoy",
]

#: The days-since axis is anchored on 1 August of the season year minus one.
ANCHOR_MONTH: int = 8
ANCHOR_DAY: int = 1

#: Histogram length. The axis must reach from the anchor to **31 December of the
#: season year**, because that is the last date the product can encode: 1 August
#: of ``Y-1`` to 31 December of ``Y`` is 517 days, 518 across a leap year.
#:
#: 512 was tried first and is six bins short. It silently clipped 6 987 pixels of
#: a Brandenburg tile, every one of them at day 514 -- 28 December 2022, a real
#: end-of-season harvest, not corruption. 544 leaves headroom above the leap-year
#: maximum while keeping one histogram per (zone, crop, year, variable) free.
#: Values outside are counted and reported, never silently folded in.
N_BINS: int = 544

#: ``CPMCE``/``CPMCH`` values at or above this are flags, not dates.
MAX_DATE_VALUE: int = 65526

#: CTY class code -> name, from the product's ``clms_hrlvlcc_cty.clr`` legend.
CTY_CLASSES: dict[int, str] = {
    0: "no cropland",
    1110: "wheat",
    1120: "barley",
    1130: "maize",
    1140: "rice",
    1150: "other cereals",
    1210: "fresh vegetables",
    1220: "dry pulses",
    1310: "potatoes",
    1320: "sugar beet",
    1410: "sunflower",
    1420: "soybeans",
    1430: "rapeseed",
    1440: "flax cotton and hemp",
    2100: "grapes",
    2200: "olives",
    2310: "fruits",
    2320: "nuts",
    3100: "unclassified annual crop",
    3200: "unclassified permanent crop",
    65535: "outside area",
}

#: Non-date values of the ``CPMCE``/``CPMCH`` layers. Aggregating any of these as
#: if it were a day-of-year is the obvious way to get a plausible wrong answer.
DATE_FLAGS: dict[int, str] = {
    0: "no annual cropland",
    65526: "fallow land",
    65527: "no field geometry",
    65531: "insufficient observations",
    65532: "no season delineation",
    65533: "season outside timeframe",
    65535: "outside area",
}


def anchor_for(season_year: int) -> date:
    """The days-since origin for a season: 1 August of the preceding year.

    Args:
        season_year: The season's harvest year, the year CLMS labels the file
            with.

    Returns:
        The anchor date.
    """
    return date(season_year - 1, ANCHOR_MONTH, ANCHOR_DAY)


def decode_yydoy(values: np.ndarray, season_year: int) -> np.ndarray:
    """``YYDOY`` integers to days since the season's anchor.

    ``YY = value // 1000`` and ``DOY = value % 1000``. The year is part of the
    encoding precisely because a season labelled 2022 can emerge in 2021, which
    is what makes the winter/spring split an observation rather than an
    assumption (see :mod:`data4simplace.phenology.seasons`).

    Args:
        values: Raw ``CPMCE`` or ``CPMCH`` pixels, any integer dtype.
        season_year: The season's harvest year.

    Returns:
        ``float32`` days since :func:`anchor_for`, with ``NaN`` wherever the
        input was a flag rather than a date.
    """
    v = np.asarray(values, dtype=np.int32)
    is_date = (v > 0) & (v < MAX_DATE_VALUE)
    if not is_date.any():
        return np.full(v.shape, np.nan, dtype=np.float32)

    anchor = anchor_for(season_year)
    yy = v // 1000
    doy = v % 1000

    # Offset from the anchor to 1 January of each encoded year, resolved per
    # distinct year rather than per pixel: a tile holds two years at most, so
    # this is two subtractions instead of a hundred million.
    offset = np.zeros(v.shape, dtype=np.int32)
    for y in np.unique(yy[is_date]):
        jan1 = date(2000 + int(y), 1, 1)
        offset[yy == y] = (jan1 - anchor).days

    days = np.where(is_date, offset + doy - 1, np.nan).astype(np.float32)
    return days


def days_to_doy(days: float, season_year: int) -> float:
    """Days since the anchor back to a calendar day-of-year.

    Applied once, at write time. Keeping the whole pipeline on the anchored axis
    until then is what lets medians and quantiles be taken with ordinary
    arithmetic.

    Args:
        days: Days since :func:`anchor_for`; may be fractional.
        season_year: The season's harvest year.

    Returns:
        Day-of-year in ``1..366``, or ``NaN`` if ``days`` is ``NaN``.
    """
    if days is None or not np.isfinite(days):
        return float("nan")
    when = anchor_for(season_year) + timedelta(days=float(days))
    return float(when.timetuple().tm_yday)
