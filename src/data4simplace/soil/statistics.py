"""Per-class soil statistics for the three-primary-class workflow.

The export writes **one** profile per 10 km cell, taken from the dominant soil
class. That single number hides two things a crop-model study needs to report:
how variable the class is inside the cell, and how much of the cell the class
actually covers. This module carries both, as intermediate products written
alongside the SIMPLACE CSVs:

``soil_class_statistics.nc``
    ``<layer>_<statistic>`` (mean, median, std, kurtosis, pixel count) with dims
    ``(rank, depth, lat, lon)`` -- every property, every depth, for each of the
    ``soil.n_primary_classes`` most frequent classes per cell.
``soil_class_shares.nc`` / ``soil_class_shares.csv``
    The class code, pixel count and **percent of the cell's classified pixels**
    per rank, with the class name resolved (``usda``: texture class;
    ``usda_profile``: ``topsoil/rooting-zone``). Rank 1's share is the weight of
    the exported profile; the remaining ranks quantify the inter-class spread
    that the single-profile export leaves out.

The final CSVs are unaffected: they keep using rank 1 with either the mean rules
or the median, per ``soil.export_statistic``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.grid import TargetGrid
from data4simplace.soil.dominant import class_name

logger = logging.getLogger(__name__)

STATS_SUBDIR = "soil"
CLASS_STATS_FILE = "soil_class_statistics.nc"
CLASS_SHARES_FILE = "soil_class_shares.nc"
CLASS_SHARES_TABLE = "soil_class_shares.csv"


@dataclass
class PrimaryClassStatistics:
    """Statistics of the ``n`` most frequent soil classes per target cell.

    Attributes
    ----------
    classes:
        ``class_code``/``pixels``/``share_percent`` on ``(rank, lat, lon)``, as
        returned by :func:`~data4simplace.soil.dominant.rank_classes_per_cell`.
    stats:
        ``<layer>_<statistic>`` on ``(rank, depth, lat, lon)``.
    mode:
        The ``soil.dominant_mode`` the codes were built with, used to resolve
        class names.
    """

    classes: xr.Dataset
    stats: xr.Dataset
    mode: str

    # ------------------------------------------------------------------ #
    # Naming
    # ------------------------------------------------------------------ #
    def code_lookup(self) -> dict[int, str]:
        """``{class code: name}`` for every code present (excluding ``0``)."""
        codes = np.unique(self.classes["class_code"].values)
        return {int(c): class_name(int(c), self.mode) for c in codes if c > 0}

    def class_table(self, grid: TargetGrid) -> pd.DataFrame:
        """Long table of the per-cell class shares, one row per (cell, rank).

        Columns: ``SimplaceID``, ``row``, ``col``, ``lat``, ``lon``, ``rank``,
        ``class_code``, ``class_name``, ``pixels``, ``share_percent``. Ranks a
        cell does not have (fewer classes than ranks) are dropped.
        """
        cells = grid.cell_table()
        frames = []
        for rank in self.classes["rank"].values:
            per_rank = self.classes.sel(rank=rank)
            frame = cells.copy()
            frame["rank"] = int(rank)
            frame["class_code"] = per_rank["class_code"].values[
                frame["row"].to_numpy(), frame["col"].to_numpy()
            ]
            frame["pixels"] = per_rank["pixels"].values[
                frame["row"].to_numpy(), frame["col"].to_numpy()
            ]
            frame["share_percent"] = per_rank["share_percent"].values[
                frame["row"].to_numpy(), frame["col"].to_numpy()
            ]
            frames.append(frame)

        table = pd.concat(frames, ignore_index=True)
        table = table[table["class_code"] > 0].copy()
        table["class_name"] = [
            class_name(int(c), self.mode) for c in table["class_code"]
        ]
        table = table.sort_values(["SimplaceID", "rank"]).reset_index(drop=True)
        return table[
            ["SimplaceID", "row", "col", "lat", "lon", "rank", "class_code",
             "class_name", "pixels", "share_percent"]
        ]

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def write(self, output_dir: str | Path, grid: TargetGrid) -> list[Path]:
        """Write the NetCDF statistics and the class-share table.

        Parameters
        ----------
        output_dir:
            Pipeline output directory; files land in its ``soil/`` subdirectory.
        grid:
            The target grid, used to key the share table by ``SimplaceID``.

        Returns
        -------
        list[pathlib.Path]
            The files written, in order: statistics NetCDF, shares NetCDF,
            shares CSV.
        """
        out_dir = Path(output_dir) / STATS_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        lookup = self.code_lookup()

        stats = self.stats.copy()
        stats.attrs.update(
            {
                "title": "Per-class soil property statistics on the target grid",
                "dominant_mode": self.mode,
                "rank_description": (
                    "1 = dominant class (the one exported), 2..n = next most "
                    "frequent classes in the cell"
                ),
                "statistics": (
                    "mean, median, std (sample), kurt (excess kurtosis), count "
                    "(pixels)"
                ),
                "class_code_names": json.dumps(lookup),
            }
        )
        stats_path = out_dir / CLASS_STATS_FILE
        stats.to_netcdf(stats_path)

        shares = self.classes.copy()
        shares["class_code"].attrs.update(
            {"long_name": f"{self.mode} class code", "missing_value": 0}
        )
        shares["share_percent"].attrs.update(
            {"long_name": "percent of the cell's classified pixels", "units": "%"}
        )
        shares["pixels"].attrs.update({"long_name": "250 m pixels in the class"})
        shares.attrs.update(
            {
                "title": "Primary soil classes per target cell with pixel shares",
                "dominant_mode": self.mode,
                "class_code_names": json.dumps(lookup),
            }
        )
        shares_path = out_dir / CLASS_SHARES_FILE
        shares.to_netcdf(shares_path)

        table_path = out_dir / CLASS_SHARES_TABLE
        self.class_table(grid).to_csv(table_path, index=False)

        logger.info(
            "Wrote per-class statistics for %d rank(s): %s, %s, %s",
            self.classes.sizes["rank"], stats_path.name, shares_path.name,
            table_path.name,
        )
        return [stats_path, shares_path, table_path]
