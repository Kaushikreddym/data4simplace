"""Cell identity: ``SimplaceID`` <-> grid position <-> weather file.

The data4simplace export needs no cell lookup table. ``SimplaceID`` is 1-based
and row-major from the **north-west** corner of the target grid, so an id is a
position, and a position is a weather filename::

    row = (id - 1) // n_lon          lon = min_lon + res/2 + col * res
    col = (id - 1) %  n_lon          lat = max_lat - res/2 - row * res

    weather/{row}/daily_mean_RES1_C{col}R{row}.csv.gz

Every function here takes the grid explicitly rather than reading a module
constant. The previous runner hard-coded one grid, which silently produced
wrong coordinates for any export written with different bounds — and the
failure is invisible, because wrong coordinates still look like coordinates.

**Two layouts exist.** data4simplace now nests the weather one directory per
grid row, which is what a SIMPLACE solution reads directly; exports written
before that change are flat in ``weather/``. Both are read here, resolved once
per export directory by :func:`_is_nested` rather than per cell, so a 70 000
cell run does not pay a stat per file for the answer.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import GridConfig

logger = logging.getLogger(__name__)

__all__ = [
    "id_to_lonlat",
    "id_to_rowcol",
    "rowcol_to_id",
    "shard_cells",
    "weather_ids",
    "weather_path",
]

#: ``daily_mean_RES1_C<col>R<row>.csv.gz`` -- the weather exporter's naming.
_WEATHER_NAME = re.compile(r"daily_mean_RES1_C(\d+)R(\d+)\.csv\.gz$")


def id_to_rowcol(simplace_id, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """``SimplaceID`` -> ``(row, col)``, both 0-based."""
    zero = np.asarray(simplace_id, dtype=np.int64) - 1
    return zero // grid.n_lon, zero % grid.n_lon


def rowcol_to_id(row, col, grid: GridConfig) -> np.ndarray:
    """``(row, col)`` -> ``SimplaceID``. The inverse of :func:`id_to_rowcol`."""
    return (
        np.asarray(row, dtype=np.int64) * grid.n_lon
        + np.asarray(col, dtype=np.int64)
        + 1
    )


def id_to_lonlat(simplace_id, grid: GridConfig) -> tuple[np.ndarray, np.ndarray]:
    """``SimplaceID`` -> cell-centre ``(lon, lat)`` in EPSG:4326."""
    row, col = id_to_rowcol(simplace_id, grid)
    res = grid.resolution_deg
    return (
        grid.min_lon + res / 2.0 + col * res,
        grid.max_lat - res / 2.0 - row * res,
    )


@lru_cache(maxsize=None)
def _is_nested(weather_dir: Path) -> bool:
    """Whether an export nests its weather one directory per grid row.

    Decided from the directory itself, not from configuration: an export is
    whatever it is on disk, and both layouts will exist side by side until the
    pre-2026-08 exports are rewritten. A directory holding neither shape reads
    as flat, so the caller fails on the missing file it names rather than here.
    """
    return any(entry.is_dir() and entry.name.isdigit()
               for entry in weather_dir.iterdir()) if weather_dir.is_dir() else False


def weather_path(export_dir: Path, simplace_id: int, grid: GridConfig) -> Path:
    """Path of one cell's gzipped weather file, in whichever layout it uses."""
    row, col = id_to_rowcol(int(simplace_id), grid)
    directory = Path(export_dir) / "weather"
    name = f"daily_mean_RES1_C{int(col)}R{int(row)}.csv.gz"
    return directory / str(int(row)) / name if _is_nested(directory) else directory / name


def weather_ids(export_dir: Path, grid: GridConfig) -> np.ndarray:
    """Every ``SimplaceID`` that has a weather file on disk, sorted.

    Read from the directory listing rather than from a manifest: the weather
    stage writes one file per cell and keeps no index, and a tiled run's files
    arrive from many independent tasks.
    """
    directory = Path(export_dir) / "weather"
    if not directory.is_dir():
        raise FileNotFoundError(f"No weather directory in the export: {directory}")
    # rglob, so a nested and a flat export enumerate the same way. The id comes
    # from the filename either way -- never from the row directory, which would
    # trust the tree's shape over the file's own identity.
    entries = directory.rglob("*.csv.gz") if _is_nested(directory) else directory.iterdir()
    ids = [
        int(match.group(2)) * grid.n_lon + int(match.group(1)) + 1
        for entry in entries
        if (match := _WEATHER_NAME.match(entry.name))
    ]
    if not ids:
        raise FileNotFoundError(f"No weather files matching the export naming in {directory}")
    return np.sort(np.asarray(ids, dtype=np.int64))


def shard_cells(ids: np.ndarray, shard: int, n_shards: int) -> np.ndarray:
    """Round-robin slice of ``ids`` for one task.

    Round-robin, not contiguous: ``SimplaceID`` runs north-to-south, so a
    contiguous split hands one task all of Scandinavia and another all of
    Iberia — very different weather-file sizes and very different wall times.
    Dealing the cells makes every shard a domain-wide sample, so the tasks
    finish together and a missing shard speckles the map rather than blanking a
    corner of it.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} out of range 0..{n_shards - 1}")
    return ids[shard::n_shards]


def cell_frame(ids: np.ndarray, grid: GridConfig) -> pd.DataFrame:
    """``SimplaceID``, ``row``, ``col``, ``lon``, ``lat`` for a set of cells."""
    row, col = id_to_rowcol(ids, grid)
    lon, lat = id_to_lonlat(ids, grid)
    return pd.DataFrame(
        {
            "SimplaceID": np.asarray(ids, dtype=np.int64),
            "row": row,
            "col": col,
            "lon": lon.astype("float64"),
            "lat": lat.astype("float64"),
        }
    )
