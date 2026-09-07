"""Winter/spring separation and the crop naming it produces.

``CTY`` class ``1110`` is just "wheat" -- the product does not separate winter
from spring. But ``CPMCE`` encodes the emergence *year* in its ``YYDOY`` value,
precisely because a season labelled 2022 can emerge in 2021, so the split is
recoverable: wheat emerging in the autumn of ``Y-1`` is winter wheat, wheat
emerging inside ``Y`` is spring. Verified on Brandenburg season 2022 -- 74.4 %
winter against 25.6 % spring, with season lengths of 251 d and 101 d that do not
overlap even at the p90/p10 boundary, and whole field parcels falling on one side
or the other.

**Where the cut goes is measured, not assumed.** The emergence histogram is
bimodal and the trough between its modes *is* the boundary. At 52 degN that
trough sits in January -- 11 440 px against 47 769 in December and 138 355 in
February -- so a fixed 1 January cut is nearly right there. It will not be right
everywhere: in Iberia an autumn-sown crop routinely emerges in December or
January, and a fixed cut would label it spring exactly where winter wheat
dominates. :func:`antimode_cut` therefore locates the trough per country, and
:class:`SeasonCut` records whether it succeeded, so an assumed cut is never
indistinguishable from a measured one -- the discipline ``site/calendar.py``
already applies with ``calendar_source``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np

from data4simplace.phenology.decode import CTY_CLASSES, N_BINS, anchor_for

logger = logging.getLogger(__name__)

__all__ = [
    "CROP_CODES",
    "SPLIT_CROPS",
    "SeasonCut",
    "antimode_cut",
    "crop_label",
    "crop_labels",
    "default_cut_day",
    "slug",
]

#: Classes that carry a distinct autumn-sown and spring-sown form in Europe, and
#: so are emitted as ``<crop>_winter`` / ``<crop>_spring``. Everything else is
#: single-season. Only wheat's split is validated; barley and rapeseed inherit
#: the method and must pass the same season-length bimodality check before their
#: rows are trusted -- until then they carry ``split_validated = False``.
SPLIT_CROPS: dict[int, str] = {1110: "wheat", 1120: "barley", 1430: "rapeseed"}

#: Non-crop codes: nothing to aggregate.
_NOT_CROPS: frozenset[int] = frozenset({0, 65535})

#: Every class this stage produces rows for.
CROP_CODES: tuple[int, ...] = tuple(
    sorted(c for c in CTY_CLASSES if c not in _NOT_CROPS)
)


def slug(name: str) -> str:
    """Class name to an identifier usable in a filename or a column."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def crop_label(code: int, season: Literal["winter", "spring", None] = None) -> str:
    """Output name for a class code, with the season suffix where one applies.

    Follows the convention already used in this project by ``GDHY_CROP``, whose
    tree holds ``wheat``, ``wheat_winter`` and ``wheat_spring``.

    Args:
        code: A ``CTY`` class code.
        season: ``"winter"`` / ``"spring"`` for a split crop; ``None`` otherwise.

    Returns:
        e.g. ``"wheat_winter"``, ``"maize"``.

    Raises:
        ValueError: If a season is given for a crop that does not split, or
            omitted for one that does.
    """
    if code not in CTY_CLASSES:
        raise ValueError(f"unknown CTY class {code}")
    splits = code in SPLIT_CROPS
    if splits and season is None:
        raise ValueError(f"{CTY_CLASSES[code]!r} splits by season; pass one")
    if not splits and season is not None:
        raise ValueError(f"{CTY_CLASSES[code]!r} has one season; got {season!r}")
    base = slug(CTY_CLASSES[code])
    return f"{base}_{season}" if splits else base


@dataclass(frozen=True, slots=True)
class CropSlot:
    """One output crop: a class code, an optional season, and its label."""

    index: int
    code: int
    season: Literal["winter", "spring"] | None
    label: str


def crop_slots() -> tuple[CropSlot, ...]:
    """Every output crop in **canonical order**, winter immediately before spring.

    The adjacency is load-bearing, not cosmetic: :mod:`~.zonal` classifies a
    pixel by looking up its class code once and then adding 1 where the crop
    splits and emergence falls on the spring side of the cut. That is a single
    add over a hundred million pixels instead of a second lookup.
    """
    slots: list[CropSlot] = []
    for code in CROP_CODES:
        if code in SPLIT_CROPS:
            slots.append(CropSlot(len(slots), code, "winter", crop_label(code, "winter")))
            slots.append(CropSlot(len(slots), code, "spring", crop_label(code, "spring")))
        else:
            slots.append(CropSlot(len(slots), code, None, crop_label(code)))
    return tuple(slots)


def crop_labels() -> tuple[str, ...]:
    """Every crop name the stage can emit, sorted."""
    return tuple(sorted(s.label for s in crop_slots()))


def crop_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables from a raw ``CTY`` code to a crop slot.

    Returns:
        ``(slot_of_code, splits_of_code)`` -- both indexed directly by the class
        code, so a 100 M-pixel classification is one fancy-index rather than a
        dictionary walk. ``slot_of_code`` is ``-1`` for non-crop codes and gives
        the **winter** slot for a split crop; ``splits_of_code`` marks the codes
        where the spring slot is ``slot + 1``.
    """
    size = max(CTY_CLASSES) + 1
    slot_of = np.full(size, -1, dtype=np.int16)
    splits = np.zeros(size, dtype=bool)
    for s in crop_slots():
        if s.season in (None, "winter"):
            slot_of[s.code] = s.index
        if s.code in SPLIT_CROPS:
            splits[s.code] = True
    return slot_of, splits


def default_cut_day(season_year: int) -> int:
    """Days from the anchor to 1 January of the season year (the fallback cut)."""
    return (date(season_year, 1, 1) - anchor_for(season_year)).days


#: The cut is searched between 1 December and 1 March. Outside that band a
#: "trough" is not a season boundary but the tail of a single mode.
_SEARCH_FROM = (12, 1)
_SEARCH_TO = (3, 1)


@dataclass(frozen=True, slots=True)
class SeasonCut:
    """Where winter ends and spring begins, and how confidently.

    Attributes:
        day: Days since the anchor. Emergence strictly below this is winter.
        source: ``"antimode"`` if measured from the histogram's trough,
            ``"fallback"`` if the histogram did not support one.
        reason: Why a fallback was taken; empty when measured.
        winter_peak: Height of the autumn mode, for the audit table.
        spring_peak: Height of the spring mode.
        trough: Height at the chosen cut.
    """

    day: int
    source: Literal["antimode", "fallback"]
    reason: str = ""
    winter_peak: int = 0
    spring_peak: int = 0
    trough: int = 0

    @property
    def is_measured(self) -> bool:
        return self.source == "antimode"


def antimode_cut(
    histogram: np.ndarray,
    season_year: int,
    min_mode_share: float = 0.05,
    max_trough_ratio: float = 0.5,
    smooth_days: int = 7,
) -> SeasonCut:
    """Locate the winter/spring boundary as the trough between the two modes.

    Args:
        histogram: Emergence counts over **all** pixels of one crop, on the
            days-since-anchor axis, length :data:`~.decode.N_BINS`.
        season_year: The season's harvest year.
        min_mode_share: Each mode must hold at least this fraction of the
            counts inside the search band, else the distribution is treated as
            unimodal and the fallback is taken. A region that grows only winter
            wheat has no boundary to find, and inventing one would split a
            single mode down the middle.
        max_trough_ratio: The trough must fall to at most this fraction of the
            smaller mode. A shallow dip between two shoulders of one broad mode
            is not a season boundary.
        smooth_days: Width of the moving average applied before searching, so a
            single noisy bin cannot be mistaken for the trough.

    Returns:
        The :class:`SeasonCut`, measured or fallen back.
    """
    fallback = default_cut_day(season_year)
    counts = np.asarray(histogram, dtype=np.int64)
    if counts.size != N_BINS:
        raise ValueError(f"expected {N_BINS} bins, got {counts.size}")

    lo = (date(season_year - 1, *_SEARCH_FROM) - anchor_for(season_year)).days
    hi = (date(season_year, *_SEARCH_TO) - anchor_for(season_year)).days

    if smooth_days > 1:
        kernel = np.ones(smooth_days) / smooth_days
        smooth = np.convolve(counts.astype(float), kernel, mode="same")
    else:
        smooth = counts.astype(float)

    band = smooth[: hi + 1]
    if band[: lo + 1].size == 0 or band[lo:].size == 0:
        return SeasonCut(fallback, "fallback", "search band empty")

    # The two modes are sought on either side of the search band: the autumn
    # mode anywhere from the anchor to 1 December, the spring mode from 1 March
    # onward, so the trough between them is what lies inside the band.
    autumn = smooth[: lo + 1]
    spring = smooth[hi:]
    if autumn.size == 0 or spring.size == 0:
        return SeasonCut(fallback, "fallback", "search band empty")

    winter_peak = float(autumn.max())
    spring_peak = float(spring.max())
    total = float(smooth[: len(spring) + hi].sum()) or 1.0

    if winter_peak <= 0 or spring_peak <= 0:
        return SeasonCut(fallback, "fallback", "one mode is empty",
                         int(winter_peak), int(spring_peak))

    # Share is measured against the peaks rather than integrated area: a long
    # flat spring tail can out-total a sharp autumn mode without being one.
    smaller, larger = sorted((winter_peak, spring_peak))
    if smaller / (larger or 1.0) < min_mode_share:
        return SeasonCut(fallback, "fallback",
                         f"unimodal: weaker mode is {smaller / larger:.3f} of the stronger",
                         int(winter_peak), int(spring_peak))

    inner = smooth[lo : hi + 1]
    idx = int(np.argmin(inner)) + lo
    trough = float(smooth[idx])
    if trough > max_trough_ratio * smaller:
        return SeasonCut(fallback, "fallback",
                         f"trough {trough:.0f} is {trough / smaller:.2f} of the weaker mode",
                         int(winter_peak), int(spring_peak), int(trough))

    return SeasonCut(idx, "antimode", "", int(winter_peak), int(spring_peak),
                     int(trough))
