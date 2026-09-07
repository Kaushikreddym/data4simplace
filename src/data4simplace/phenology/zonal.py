"""Accumulate 10 m pixels into per-zone date histograms.

**The whole stage rests on one choice: accumulate histograms, not values.**
Dates live on a bounded integer axis -- days since 1 August of ``Y-1``, roughly
0..500 -- so each ``(zone, crop, variable)`` gets a fixed-length bin count. That
buys three things at once:

* the median is **exact**, and every other quantile is free;
* memory is O(1) per zone, where raw values for one 20 M-pixel administrative
  unit would be 80 MB per variable;
* **histograms merge by addition across tiles.** Most administrative units
  straddle a tile boundary, and averaging per-tile medians is simply wrong.
  Summing the counts and taking one median at the end is not an approximation of
  the right answer, it *is* the right answer.

Classification and binning are done with a single ``bincount`` over a packed key
rather than a loop over zones and crops: a tile is a hundred million pixels, and
946 x 22 nested passes over it would dominate the run.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data4simplace.phenology.decode import N_BINS, decode_yydoy
from data4simplace.phenology.seasons import crop_lookup, crop_slots

logger = logging.getLogger(__name__)

__all__ = ["VARIABLES", "accumulate_tile_year", "histogram_frame"]

#: The three distributions kept per zone and crop. ``season_length`` is
#: accumulated in its own right because ``median(harvest) - median(emergence)``
#: is **not** ``median(harvest - emergence)``.
VARIABLES: tuple[str, ...] = ("emergence", "harvest", "season_length")


def accumulate_tile_year(
    cty: np.ndarray,
    cpmce: np.ndarray,
    cpmch: np.ndarray,
    zone: np.ndarray,
    n_zones: int,
    season_year: int,
    cut_days: np.ndarray,
    block_rows: int = 2048,
) -> dict[str, np.ndarray]:
    """Bin one tile-year's pixels into ``(zone, crop, bin)`` counts.

    Args:
        cty: Crop-type codes.
        cpmce: Raw ``YYDOY`` emergence.
        cpmch: Raw ``YYDOY`` harvest.
        zone: Zone index per pixel; ``0`` means "no zone" and is dropped.
        n_zones: Number of real zones, so indices run ``1..n_zones``.
        season_year: The season's harvest year.
        cut_days: Winter/spring boundary **per pixel**, in days since the
            anchor. Passed as an array rather than a scalar because the cut is
            a per-country property and a tile can span a border.
        block_rows: Rows per pass, bounding peak memory independently of tile
            size.

    Returns:
        ``{variable: counts}``, each ``(n_zones + 1, n_crops, N_BINS)`` int64.
        Row 0 is the no-zone bucket and is retained rather than dropped so the
        totals still reconcile against the raster.
    """
    slots = crop_slots()
    n_crops = len(slots)
    slot_of_code, splits_code = crop_lookup()

    shape = (n_zones + 1, n_crops, N_BINS)
    out = {v: np.zeros(shape, dtype=np.int64) for v in VARIABLES}
    flat_len = int(np.prod(shape))
    dropped = {v: 0 for v in VARIABLES}
    kept = {v: 0 for v in VARIABLES}

    height = cty.shape[0]
    for r0 in range(0, height, block_rows):
        r1 = min(r0 + block_rows, height)
        codes = np.asarray(cty[r0:r1])
        zon = np.asarray(zone[r0:r1]).astype(np.int64)

        # Codes outside the legend would index past the lookup table; clip them
        # into the "not a crop" slot rather than raising, since an unexpected
        # value is data, not a bug in this loop.
        safe = np.where(codes < slot_of_code.size, codes, 0)
        slot = slot_of_code[safe].astype(np.int64)
        is_crop = (slot >= 0) & (zon > 0)
        if not is_crop.any():
            continue

        emergence = decode_yydoy(cpmce[r0:r1], season_year)
        harvest = decode_yydoy(cpmch[r0:r1], season_year)

        # The winter/spring decision, and the reason crop_slots() puts spring
        # immediately after winter: one add, no second lookup.
        spring = splits_code[safe] & np.isfinite(emergence) & (
            emergence >= cut_days[r0:r1]
        )
        slot = slot + spring.astype(np.int64)

        season_length = harvest - emergence
        series = {
            "emergence": emergence,
            "harvest": harvest,
            "season_length": season_length,
        }

        base = (zon * n_crops + slot) * N_BINS
        for name, values in series.items():
            good = is_crop & np.isfinite(values)
            if not good.any():
                continue
            binned = values[good].astype(np.int64)
            inside = (binned >= 0) & (binned < N_BINS)
            dropped[name] += int((~inside).sum())
            kept[name] += int(inside.sum())
            keys = base[good][inside] + binned[inside]
            counts = np.bincount(keys, minlength=flat_len)
            out[name] += counts[:flat_len].reshape(shape)

    # Reported once per tile-year rather than once per block: the axis is sized
    # to hold every date the product can encode (see decode.N_BINS), so anything
    # here is a real anomaly worth seeing, not routine spill.
    for name, n in dropped.items():
        if n:
            logger.warning(
                "%s: %d of %d values fell outside 0..%d and were dropped",
                name, n, n + kept[name], N_BINS - 1,
            )

    return out


def histogram_frame(
    counts: dict[str, np.ndarray],
    zone_ids: pd.DataFrame,
    zone_kind: str,
    season_year: int,
    tile: str,
) -> pd.DataFrame:
    """Flatten the count cubes to the long, sparse form written to disk.

    Only non-zero bins are kept. A dense cube is mostly zeros -- a tile holds a
    handful of crops, not all 22 -- and the long form is what merges across
    tiles with a single ``groupby``.

    Args:
        counts: Output of :func:`accumulate_tile_year`.
        zone_ids: Index-ordered identifiers; row *i* describes zone ``i + 1``.
        zone_kind: ``"adm"`` or ``"grid"``, carried into the output.
        season_year: The season's harvest year.
        tile: Tile identifier, so a bad tile can be found and re-run.

    Returns:
        Columns ``tile, zone_kind, zone, crop, year, variable, bin, count``.
    """
    slots = crop_slots()
    labels = np.array([s.label for s in slots], dtype=object)

    frames = []
    for variable, cube in counts.items():
        # Drop the no-zone row here rather than in the accumulator: it is wanted
        # for reconciliation, not for the product.
        real = cube[1:]
        zi, ci, bi = np.nonzero(real)
        if zi.size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "zone": zi,
                    "crop": labels[ci],
                    "variable": variable,
                    "bin": bi.astype(np.int16),
                    "count": real[zi, ci, bi].astype(np.int64),
                }
            )
        )
    if not frames:
        return pd.DataFrame(
            columns=["tile", "zone_kind", "zone", "crop", "year", "variable",
                     "bin", "count"]
        )

    frame = pd.concat(frames, ignore_index=True)
    ids = zone_ids.reset_index(drop=True)
    for column in ids.columns:
        frame[column] = ids[column].to_numpy()[frame["zone"].to_numpy()]
    frame["tile"] = tile
    frame["zone_kind"] = zone_kind
    frame["year"] = np.int16(season_year)
    return frame.drop(columns=["zone"])
