"""Putting the 10 km run and a 0.5° reference field on one grid.

CyBench is a table of national statistics, so :mod:`utils.aggregate` averages
the run up to countries. GDHY and the SAGE calendar are *fields*, so the
comparison unit is a grid cell — which needs a different join and a different
set of caveats, both of which live here.

**The run is aggregated up; the reference is never interpolated down.** A 0.5°
cell holds ~15-20 of the run's 10 km cropland cells at European latitudes, so
binning them is a mean over values that exist. Interpolating GDHY onto the
10 km grid would invent structure inside a cell it has no information about,
and would make neighbouring comparison points near-duplicates of each other.

**Every comparison point is a mean over cropland cells, unweighted.** Same
argument as :func:`utils.aggregate.aggregate_simulated`: the SIMPLACE export
carries no per-cell wheat area, so any weight would have to be invented.

**Comparison points are not independent.** Both references are smooth on scales
of hundreds of kilometres — GDHY because it downscales census statistics with a
satellite proxy, SAGE because it interpolates a few thousand reporting units —
so a p-value computed as though 40 000 cell-years were 40 000 independent
samples is meaningless. Quote the effect sizes; treat the significance of a
pooled correlation as decoration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import EUROPE_BBOX, GRID_RES_DEG, MIN_CELLS_PER_GRIDCELL
from .doy import circular_mean_doy

logger = logging.getLogger(__name__)

__all__ = [
    "bin_cells",
    "crop_to_bbox",
    "grid_cell_id",
    "pair_gridded",
    "snap_to_grid",
]


def snap_to_grid(
    lon, lat, resolution: float = GRID_RES_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates to the **centre** of the 0.5° cell containing them.

    Both references are cell-centred on the half-degree half-step (0.25, 0.75,
    …), so flooring to a multiple of the resolution and adding a half step
    reproduces their coordinates exactly. That exactness is the point: the join
    downstream is an equality merge on floats, and a centre computed any other
    way would miss by a rounding error.

    Args:
        lon: Longitudes in degrees east, either convention.
        lat: Latitudes in degrees north.
        resolution: Grid spacing in degrees.

    Returns:
        ``(lon_centre, lat_centre)``, rounded to 4 decimals so float32 inputs
        from the Parquet and float64 coordinates from the NetCDF land on the
        same value.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    # Wrap into [-180, 180) first: the run is already there, GDHY is not, and
    # a cell at 359.75 must land on the same centre as one at -0.25.
    lon = (lon + 180.0) % 360.0 - 180.0
    half = resolution / 2.0
    return (
        np.round(np.floor(lon / resolution) * resolution + half, 4),
        np.round(np.floor(lat / resolution) * resolution + half, 4),
    )


def grid_cell_id(lon, lat, resolution: float = GRID_RES_DEG) -> np.ndarray:
    """A stable integer id per 0.5° cell, for caching and grouping.

    Args:
        lon: Cell-centre longitudes.
        lat: Cell-centre latitudes.
        resolution: Grid spacing in degrees.

    Returns:
        ``int64`` array, unique per cell and reproducible across runs.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n_lon = int(round(360.0 / resolution))
    i = np.floor((lon + 180.0) / resolution).astype(np.int64)
    j = np.floor((lat + 90.0) / resolution).astype(np.int64)
    return j * n_lon + i


def crop_to_bbox(
    frame: pd.DataFrame,
    bbox: tuple[float, float, float, float] = EUROPE_BBOX,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """Keep the rows inside ``(lon_min, lon_max, lat_min, lat_max)``.

    Args:
        frame: Any table carrying coordinates.
        bbox: Inclusive bounds in degrees.
        lon_col: Longitude column.
        lat_col: Latitude column.

    Returns:
        The cropped frame, re-indexed.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    keep = (
        frame[lon_col].between(lon_min, lon_max)
        & frame[lat_col].between(lat_min, lat_max)
    )
    return frame[keep].reset_index(drop=True)


def bin_cells(
    frame: pd.DataFrame,
    value_cols: dict[str, bool],
    by: list[str] | None = None,
    resolution: float = GRID_RES_DEG,
    min_cells: int = MIN_CELLS_PER_GRIDCELL,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> pd.DataFrame:
    """Average 10 km cells into the 0.5° cell their centre falls in.

    A binned mean, not a conservative remap: a 10 km cell never straddles a
    0.5° boundary by more than its own width, and the run's cell set is already
    an irregular cropland subset, so there is no area to conserve — the target
    quantity is a mean, not a total.

    Args:
        frame: Simulation rows carrying ``lon``, ``lat`` and the value columns.
        value_cols: ``{column: is_circular}``. Circular columns are averaged
            with :func:`utils.doy.circular_mean_doy`; everything else with a
            plain mean. A duration (season length) is **not** circular.
        by: Extra grouping keys applied alongside the cell, e.g. ``["year"]``.
            ``None`` collapses every row of a cell into one.
        resolution: Grid spacing in degrees.
        min_cells: Drop 0.5° cells backed by fewer than this many 10 km cells.
        lon_col: Longitude column of ``frame``.
        lat_col: Latitude column of ``frame``.

    Returns:
        One row per (cell [, extra keys]) with ``lon``/``lat`` at the 0.5° cell
        centre, ``grid_id``, the means, and ``n_cells`` — the number of
        *distinct* 10 km cells behind the value, which is not the row count
        once ``by`` spans several years.

    Raises:
        KeyError: If a value column is missing.
    """
    missing = set(value_cols) - set(frame.columns)
    if missing:
        raise KeyError(f"frame lacks {sorted(missing)}")

    out = frame.copy()
    out["lon_bin"], out["lat_bin"] = snap_to_grid(
        out[lon_col], out[lat_col], resolution
    )
    keys = ["lon_bin", "lat_bin", *(by or [])]

    linear = [c for c, circular in value_cols.items() if not circular]
    circular = [c for c, circular in value_cols.items() if circular]

    grouped = out.groupby(keys, sort=True, observed=True)
    parts: list[pd.DataFrame] = []
    if linear:
        parts.append(grouped[linear].mean())
    if circular:
        parts.append(
            grouped[circular].agg(lambda s: circular_mean_doy(s.to_numpy(dtype=float)))
        )
    result = pd.concat(parts, axis=1)

    id_col = "SimplaceID" if "SimplaceID" in out.columns else lon_col
    result["n_cells"] = grouped[id_col].nunique()
    result = result.reset_index().rename(columns={"lon_bin": "lon", "lat_bin": "lat"})

    before = len(result)
    result = result[result["n_cells"] >= min_cells].reset_index(drop=True)
    logger.info(
        "binned %d rows into %d 0.5-degree groups (%d dropped below %d cells)",
        len(frame), before, before - len(result), min_cells,
    )

    result["grid_id"] = grid_cell_id(result["lon"], result["lat"], resolution)
    ordered = ["grid_id", "lon", "lat", *(by or []), *value_cols, "n_cells"]
    return result[ordered]


def pair_gridded(
    simulated: pd.DataFrame,
    observed: pd.DataFrame,
    on: list[str] | None = None,
) -> pd.DataFrame:
    """Inner-join the binned run and a reference field, reporting the losses.

    Args:
        simulated: Output of :func:`bin_cells`.
        observed: Reference on the same 0.5° grid, carrying ``grid_id`` and the
            same extra keys.
        on: Join keys; ``["grid_id"]`` by default, or add ``"year"`` for a
            time-varying reference.

    Returns:
        The joined frame. ``lon``/``lat`` come from the simulated side, which
        is identical to the observed side by construction — both are cell
        centres of the same grid.
    """
    on = on or ["grid_id"]
    right = observed.drop(columns=[c for c in ("lon", "lat") if c in observed.columns])
    paired = simulated.merge(right, on=on, how="inner").sort_values(on)

    logger.info(
        "paired %d rows; %d simulated-only and %d observed-only dropped",
        len(paired), len(simulated) - len(paired), len(observed) - len(paired),
    )
    if len(paired) == 0:
        raise ValueError(
            "no overlap between the run and the reference; check the bounding "
            "box, the year range and the longitude convention"
        )
    return paired.reset_index(drop=True)
