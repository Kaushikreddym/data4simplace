"""Merge per-tile histograms and turn them into the phenology table.

Two things happen here, and the order matters. Histograms from every tile a zone
touches are **summed first**, and only then is a quantile taken. Doing it the
other way -- a median per tile, averaged -- would be wrong for the majority of
administrative units, which straddle a tile boundary. Summing counts and
reducing once gives the same answer a single pass over the whole zone would.

Quantiles are computed from the counts directly and reproduce
``numpy.quantile(..., method="linear")`` exactly, so the histogram is a
compression of the sample, not an approximation of it.

Areas are exact without a latitude correction: the tiles are EPSG:3035, an
equal-area projection, so every 10 m pixel is 100 m^2 anywhere from Crete to
North Cape. That is not true of the 0.1 degree grid the rest of this project
works in, where ``spatial/area.py`` exists precisely because a cell's area
varies with latitude.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data4simplace.phenology.decode import N_BINS, days_to_doy
from data4simplace.phenology.seasons import SPLIT_CROPS, crop_label

logger = logging.getLogger(__name__)

__all__ = [
    "PERCENTILES",
    "PIXEL_AREA_KM2",
    "merge_histograms",
    "quantiles_from_counts",
    "reduce_to_table",
]

#: Quantiles reported for every variable. The histogram makes each one free, so
#: the spread travels with the median rather than being a second pass.
PERCENTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

#: A 10 m pixel in EPSG:3035, in km^2.
PIXEL_AREA_KM2: float = 100.0 / 1e6


def quantiles_from_counts(
    counts: np.ndarray, qs: tuple[float, ...] = PERCENTILES
) -> np.ndarray:
    """Exact quantiles of the sample a histogram represents.

    Reproduces ``numpy.quantile(values, q, method="linear")`` for the sample
    that would expand from ``counts``: the rank position ``q * (n - 1)`` is
    located and interpolated between the values on either side of it.

    Args:
        counts: Bin counts over ``0..len(counts) - 1``.
        qs: Quantiles in ``[0, 1]``.

    Returns:
        One value per entry of ``qs``; all ``NaN`` when the histogram is empty.
    """
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.full(len(qs), np.nan)

    cum = np.cumsum(counts)
    values = np.nonzero(counts)[0]
    # Rank -> value: the first bin whose cumulative count exceeds the rank.
    def at(rank: int) -> float:
        return float(np.searchsorted(cum, rank, side="right"))

    out = np.empty(len(qs), dtype=float)
    for i, q in enumerate(qs):
        pos = q * (total - 1)
        lo = int(np.floor(pos))
        hi = int(np.ceil(pos))
        v_lo = at(lo)
        out[i] = v_lo if lo == hi else v_lo + (pos - lo) * (at(hi) - v_lo)
    if values.size == 1:
        out[:] = float(values[0])
    return out


def merge_histograms(paths: list[Path] | list[str]) -> pd.DataFrame:
    """Sum the per-tile histogram parquets into one frame.

    Args:
        paths: Per-(tile, year) parquet files written by the array tasks.

    Returns:
        Long counts keyed by ``zone_kind``, the zone identifier columns,
        ``crop``, ``year``, ``variable`` and ``bin`` -- with ``tile`` summed
        away, which is the step that makes a zone spanning several tiles whole
        again.
    """
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        raise ValueError("no histogram files to merge")
    frame = pd.concat(frames, ignore_index=True)

    id_columns = [
        c
        for c in frame.columns
        if c not in {"tile", "count", "bin", "crop", "year", "variable", "zone_kind"}
    ]
    keys = ["zone_kind", *id_columns, "crop", "year", "variable", "bin"]
    merged = frame.groupby(keys, observed=True, sort=False)["count"].sum().reset_index()
    logger.info(
        "merged %d tile files -> %d (zone, crop, year, variable, bin) rows",
        len(frames), len(merged),
    )
    return merged


def reduce_to_table(
    merged: pd.DataFrame,
    cuts: pd.DataFrame | None = None,
    min_pixels: int = 1,
) -> pd.DataFrame:
    """Quantiles per zone, crop and year, from the merged histograms.

    Args:
        merged: Output of :func:`merge_histograms`.
        cuts: Optional per ``(country, year)`` cut table with ``cut_doy`` and
            ``cut_source``, joined on so every row says where its winter/spring
            boundary came from.
        min_pixels: Drop groups thinner than this. Kept at 1 by default: a thin
            group is reported with its ``n_pixels`` rather than silently
            removed, and the calibration weights by it.

    Returns:
        One row per ``(zone, crop, year)`` with ``n_pixels``, ``area_km2``, the
        percentiles of each variable as day-of-year (or days, for
        ``season_length``), and the provenance columns.
    """
    id_columns = [
        c
        for c in merged.columns
        if c not in {"crop", "year", "variable", "bin", "count", "zone_kind"}
    ]
    group_keys = ["zone_kind", *id_columns, "crop", "year"]

    records: list[dict] = []
    for key, block in merged.groupby(group_keys, observed=True, sort=False):
        row = dict(zip(group_keys, key))
        n_pixels = 0
        for variable, sub in block.groupby("variable", observed=True, sort=False):
            counts = np.zeros(N_BINS, dtype=np.int64)
            counts[sub["bin"].to_numpy()] = sub["count"].to_numpy()
            values = quantiles_from_counts(counts)
            year = int(row["year"])
            for q, value in zip(PERCENTILES, values):
                tag = "median" if q == 0.50 else f"p{int(q * 100):02d}"
                # season_length is a duration, so it stays in days; the two
                # dates become a day-of-year, which is what every downstream
                # table speaks.
                out = value if variable == "season_length" else days_to_doy(value, year)
                unit = "days" if variable == "season_length" else "doy"
                row[f"{variable}_{unit}_{tag}"] = out
            if variable == "emergence":
                n_pixels = int(counts.sum())
        row["n_pixels"] = n_pixels
        row["area_km2"] = n_pixels * PIXEL_AREA_KM2
        records.append(row)

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table

    # Season share: what fraction of the parent class this row's season holds.
    # 1.0 for a single-season crop, so the column is always meaningful.
    parents = {crop_label(c, s): _base(c) for c in SPLIT_CROPS for s in ("winter", "spring")}
    table["parent_crop"] = table["crop"].map(parents).fillna(table["crop"])
    totals = table.groupby([*group_keys[:-2], "parent_crop", "year"], observed=True)[
        "n_pixels"
    ].transform("sum")
    table["season_share"] = np.where(totals > 0, table["n_pixels"] / totals, np.nan)
    table["split_validated"] = table["crop"].str.startswith("wheat_")

    if cuts is not None and "country" in table.columns:
        table = table.merge(cuts, on=["country", "year", "crop"], how="left")

    return table.drop(columns=["parent_crop"]).sort_values(
        [*group_keys[:-1], "crop"]
    ).reset_index(drop=True)


def _base(code: int) -> str:
    """Unsplit label of a split crop code."""
    from data4simplace.phenology.decode import CTY_CLASSES
    from data4simplace.phenology.seasons import slug

    return slug(CTY_CLASSES[code])
