"""Per-class soil products: the Method B aggregation and the class statistics.

Both containers here answer the same question from different angles -- what does
the *single* exported profile leave out? -- so they share the class-code naming
and the "write the gridded state before any CSV" rule:

:class:`TopClassAggregation` (Method B, ``soil.aggregation_method: top3``)
    Method A aggregates one profile per 10 km cell, from the cell's dominant
    soil class. That profile is all a SIMPLACE run consumes, but it silently
    discards the rest of the cell: a cell that is 41 % loamy sand and 39 %
    sandy loam exports as if it were uniform. Method B keeps the
    ``soil.n_primary_classes`` most frequent classes per cell, aggregates each
    **over its own pixels**, and carries the area each one covers plus two
    cell-level heterogeneity metrics. Rank 1 is the dominant class, so
    ``properties.sel(rank=1)`` is exactly the Method A dataset -- the two
    methods never disagree about the primary profile.

:class:`PrimaryClassStatistics` (``flags.write_soil_statistics``)
    Describes the same ranked classes by mean, median, spread and shape rather
    than reducing each to one number, and writes the per-cell class-share table.

Both are produced by
:meth:`~data4simplace.soil.soilgrids.SoilGridsHandler.load_dominant`;
:class:`TopClassAggregation` is additionally consumed by
:class:`~data4simplace.exporters.soil_export.TopSoilExporter`, which writes one
SIMPLACE CSV per rank under ``flags.export_top3_soil_csvs``.
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
from data4simplace.soil.classify import class_name

logger = logging.getLogger(__name__)

#: Subdirectory (under ``<output_dir>/soil/``) holding the gridded intermediates.
NETCDF_SUBDIR = "netcdf_tiles"

PROPERTIES_FILE = "intermediate_soil_properties.nc"
TOP_CLASSES_FILE = "intermediate_top3_classes.nc"
UNCERTAINTY_FILE = "intermediate_soil_uncertainty.nc"

#: Output subdirectory of the per-class statistics, and their filenames.
STATS_SUBDIR = "soil"
CLASS_STATS_FILE = "soil_class_statistics.nc"
CLASS_SHARES_FILE = "soil_class_shares.nc"
CLASS_SHARES_TABLE = "soil_class_shares.csv"

#: Metadata columns appended to every per-rank SIMPLACE CSV, in order.
METADATA_COLUMNS = [
    "SimplaceID",
    "latitude",
    "longitude",
    "soil_class_id",
    "class_name",
    "area_km2",
    "area_fraction",
    "cell_shannon_entropy",
    "cell_dominance_ratio",
]


@dataclass
class TopClassAggregation:
    """Per-rank soil properties, class composition and cell heterogeneity.

    Attributes
    ----------
    properties:
        Soil properties on ``(rank, depth, lat, lon)``: each rank aggregated over
        the 250 m pixels of *that* class only, with the same variable-specific
        rules Method A uses (arithmetic / geometric / pH H+ mean, or the median
        under ``soil.export_statistic: median``).
    classes:
        ``class_code``, ``pixels``, ``share_percent``, ``area_km2`` and
        ``area_fraction`` on ``(rank, lat, lon)``, from
        :func:`~data4simplace.soil.classify.class_composition`.
    uncertainty:
        ``n_classes``, ``cropland_pixels``, ``total_area_km2``,
        ``dominance_ratio`` and ``shannon_entropy`` on ``(lat, lon)``.
    mode:
        The ``soil.dominant_mode`` the codes were built with, used to name them.
    """

    properties: xr.Dataset
    classes: xr.Dataset
    uncertainty: xr.Dataset
    mode: str

    @property
    def ranks(self) -> np.ndarray:
        """The 1-based rank coordinate (1 = dominant class)."""
        return np.asarray(self.classes["rank"].values)

    # ------------------------------------------------------------------ #
    # Naming
    # ------------------------------------------------------------------ #
    def code_lookup(self) -> dict[int, str]:
        """``{class code: name}`` for every code present (excluding ``0``)."""
        codes = np.unique(self.classes["class_code"].values)
        return {int(c): class_name(int(c), self.mode) for c in codes if c > 0}

    def _name_variables(self) -> dict[str, xr.DataArray]:
        """Code -> name lookup as 1-D variables, so names live *in* the NetCDF.

        A per-cell string field would be three strings per cell for no extra
        information; the lookup is at most 144 entries.
        """
        lookup = self.code_lookup()
        codes = np.array(sorted(lookup), dtype="int16")
        names = np.array([lookup[int(c)] for c in codes], dtype=object)
        index = xr.DataArray(
            np.arange(codes.size, dtype="int32"), dims="class_index", name="class_index"
        )
        return {
            "class_code_value": xr.DataArray(
                codes,
                dims="class_index",
                coords={"class_index": index},
                attrs={"long_name": f"{self.mode} class code"},
            ),
            "class_code_name": xr.DataArray(
                names,
                dims="class_index",
                coords={"class_index": index},
                attrs={"long_name": f"{self.mode} class name"},
            ),
        }

    # ------------------------------------------------------------------ #
    # Cell masking
    # ------------------------------------------------------------------ #
    def mask_cells(self, mask: xr.DataArray | None) -> None:
        """Restrict every product to the exported cell set, in place.

        Not :func:`~data4simplace.spatial.cropland_weights.apply_cell_mask`: that
        blanks with NaN, which would promote the integer class codes and pixel
        counts to floats. Counts are zeroed instead (an excluded cell has no
        pixels), so the codes stay integers all the way into the NetCDF.
        """
        if mask is None:
            return

        def apply(source: xr.Dataset) -> xr.Dataset:
            aligned = mask.reindex(lat=source["lat"], lon=source["lon"], method="nearest")
            out = source.copy()
            for name, da in source.data_vars.items():
                if np.issubdtype(da.dtype, np.integer):
                    out[name] = da.where(aligned, 0).astype(da.dtype)
                else:
                    out[name] = da.where(aligned)
            return out

        self.properties = apply(self.properties)
        self.classes = apply(self.classes)
        self.uncertainty = apply(self.uncertainty)

    # ------------------------------------------------------------------ #
    # Per-rank export metadata
    # ------------------------------------------------------------------ #
    def metadata_frame(self, rank: int, cell_table: pd.DataFrame) -> pd.DataFrame:
        """Per-cell metadata columns for one rank's SIMPLACE CSV.

        Parameters
        ----------
        rank:
            1-based rank (1 = dominant class).
        cell_table:
            The exported cell table (``SimplaceID``, ``row``, ``col``, ``lat``,
            ``lon``), already restricted to the exported cell set.

        Returns
        -------
        pandas.DataFrame
            One row per cell with :data:`METADATA_COLUMNS`. Cells that have no
            class at this rank carry ``soil_class_id == 0`` and are dropped by
            the exporter.
        """
        rows = np.asarray(cell_table["row"], dtype=int)
        cols = np.asarray(cell_table["col"], dtype=int)
        per_rank = self.classes.sel(rank=rank)

        def at_cells(source: xr.Dataset, name: str) -> np.ndarray:
            if name not in source.data_vars:
                return np.full(rows.size, np.nan)
            return np.asarray(source[name].values)[rows, cols]

        codes = at_cells(per_rank, "class_code").astype("int64")
        frame = pd.DataFrame(
            {
                "SimplaceID": np.asarray(cell_table["SimplaceID"], dtype=np.int64),
                "latitude": np.asarray(cell_table["lat"], dtype="float64"),
                "longitude": np.asarray(cell_table["lon"], dtype="float64"),
                "soil_class_id": codes,
                "class_name": [class_name(int(c), self.mode) for c in codes],
                "area_km2": at_cells(per_rank, "area_km2"),
                "area_fraction": at_cells(per_rank, "area_fraction"),
                "cell_shannon_entropy": at_cells(self.uncertainty, "shannon_entropy"),
                "cell_dominance_ratio": at_cells(self.uncertainty, "dominance_ratio"),
            }
        )
        return frame[METADATA_COLUMNS]

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def write(
        self,
        output_dir: str | Path,
        soil: xr.Dataset | None = None,
        hydraulic: xr.Dataset | None = None,
        suffix: str = "",
    ) -> list[Path]:
        """Write the three intermediate NetCDFs to ``soil/netcdf_tiles/``.

        Written **before** the CSV export so the gridded state survives an
        exporter failure.

        Parameters
        ----------
        output_dir:
            Pipeline output directory.
        soil:
            The exported 10 km property stack (Method A / rank 1). Saved as
            ``intermediate_soil_properties.nc``; skipped when ``None``.
        hydraulic:
            Optional PTF outputs, merged into the property file.
        suffix:
            Appended to each filename stem, so a tiled run can write one set per
            tile (e.g. ``"_tile_r000_c000"``) without tiles overwriting one
            another.

        Returns
        -------
        list[pathlib.Path]
            The files written, in order: properties, top classes, uncertainty.
        """
        out_dir = Path(output_dir) / "soil" / NETCDF_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        lookup = self.code_lookup()
        common = {"dominant_mode": self.mode, "class_code_names": json.dumps(lookup)}
        written: list[Path] = []

        def stem(name: str) -> Path:
            path = Path(name)
            return out_dir / f"{path.stem}{suffix}{path.suffix}"

        if soil is not None:
            stack = soil if hydraulic is None else xr.merge([soil, hydraulic])
            stack = stack.copy()
            stack.attrs.update(
                {
                    "title": "Aggregated 10 km soil properties (dominant class)",
                    "description": (
                        "The property stack the SIMPLACE soil.csv is written "
                        "from: rank 1 (dominant class) per target cell"
                    ),
                    **common,
                }
            )
            path = stem(PROPERTIES_FILE)
            stack.to_netcdf(path)
            written.append(path)

        top = xr.merge([self.properties, self.classes])
        top = top.assign(self._name_variables())
        top.attrs.update(
            {
                "title": (
                    f"Top-{self.ranks.size} soil classes per target cell with "
                    "per-class properties"
                ),
                "rank_description": (
                    "1 = dominant class (exported as soil.csv), 2..n = next most "
                    "frequent classes, each aggregated over its own 250 m pixels"
                ),
                "class_names": (
                    "class_code_value/class_code_name give the code -> name map"
                ),
                **common,
            }
        )
        path = stem(TOP_CLASSES_FILE)
        top.to_netcdf(path)
        written.append(path)

        shares = self.classes[
            [v for v in ("share_percent", "area_km2", "area_fraction")
             if v in self.classes.data_vars]
        ]
        uncertainty = xr.merge([self.uncertainty, shares])
        uncertainty.attrs.update(
            {
                "title": "Within-cell soil class heterogeneity",
                "dominance_ratio": (
                    "1 - area fraction of the dominant class: the share of the "
                    "cell the exported profile does not represent"
                ),
                "shannon_entropy": (
                    "Shannon entropy over all classes in the cell, normalised by "
                    "ln(n_classes): 0 = uniform, 1 = all classes equally frequent"
                ),
                **common,
            }
        )
        path = stem(UNCERTAINTY_FILE)
        uncertainty.to_netcdf(path)
        written.append(path)

        logger.info(
            "Wrote %d intermediate soil raster(s) to %s: %s",
            len(written),
            out_dir,
            ", ".join(p.name for p in written),
        )
        return written


@dataclass
class PrimaryClassStatistics:
    """Statistics of the ``n`` most frequent soil classes per target cell.

    Attributes
    ----------
    classes:
        ``class_code``/``pixels``/``share_percent`` on ``(rank, lat, lon)``, as
        returned by :func:`~data4simplace.soil.classify.rank_classes_per_cell`.
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
