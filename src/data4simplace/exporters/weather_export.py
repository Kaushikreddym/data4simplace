"""SIMPLACE weather file generator.

Produces one daily weather file per grid cell from the regridded MSWX climate
dataset, conforming to the SIMPLACE weather reference. The reference files are
tab-delimited, gzipped (``daily_mean_RES1_C{col}R{row}.csv.gz``) and carry the
columns ``Date, Precipitation, TempMin, TempMean, TempMax, Radiation,
Windspeed, RefET, Gridcell, RelHumCalc`` with a ``-99.9`` sentinel. Column
order, delimiter and sentinel are taken from the reference when available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import BaseExporter, ReferenceSpec

logger = logging.getLogger(__name__)

# Map canonical MSWX names -> SIMPLACE weather column names.
_CANONICAL_TO_SIMPLACE = {
    "pr": "Precipitation",
    "tasmin": "TempMin",
    "tas": "TempMean",
    "tasmax": "TempMax",
    "rsds": "Radiation",
    "hurs": "RelHumCalc",
}
# Reference columns MSWX cannot provide (Windspeed, RefET) are left for
# conform() to fill with the reference sentinel.


class WeatherExporter(BaseExporter):
    """Export SIMPLACE daily weather files from climate data."""

    kind = "weather"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)

    def fallback_spec(self) -> ReferenceSpec:
        """Documented default matching the SIMPLACE weather layout."""
        return ReferenceSpec(
            delimiter="\t",
            columns=[
                "Date",
                "Precipitation",
                "TempMin",
                "TempMean",
                "TempMax",
                "Radiation",
                "Windspeed",
                "RefET",
                "Gridcell",
                "RelHumCalc",
            ],
            missing_value="-99.9",
        )

    @staticmethod
    def _identity(cell: pd.Series) -> tuple[int, int]:
        """Global ``(col, row)`` for naming: ``gcol``/``grow`` if present, else local.

        Under tiled execution the cell table carries local ``row``/``col`` (for
        indexing the tile's data arrays) plus global ``grow``/``gcol`` so that
        output filenames and ``Gridcell`` ids stay consistent across tiles.
        """
        col = int(cell["gcol"]) if "gcol" in cell.index else int(cell["col"])
        row = int(cell["grow"]) if "grow" in cell.index else int(cell["row"])
        return col, row

    def _gridcell_id(self, cell: pd.Series) -> str:
        """SIMPLACE ``C_<col>:R_<row>`` identifier from (global) grid indices."""
        col, row = self._identity(cell)
        return f"C_{col}:R_{row}"

    def build_frame(self, cell: pd.Series, climate: xr.Dataset) -> pd.DataFrame:
        """Build a daily weather table for a single grid cell.

        Parameters
        ----------
        cell:
            One row of :meth:`TargetGrid.cell_table` (``SimplaceID``, row, col,
            lat, lon).
        climate:
            Regridded climate dataset with dims ``(time, lat, lon)``. For a
            single cell this materialises only that column; :meth:`export`
            materialises the whole (small) grid once for bulk output.
        """
        dates = pd.to_datetime(climate["time"].values).strftime("%Y-%m-%d")
        point = climate.isel(lat=int(cell["row"]), lon=int(cell["col"]))
        frame = pd.DataFrame({"Date": dates})

        for canonical, column in _CANONICAL_TO_SIMPLACE.items():
            if canonical in point.data_vars:
                values = np.asarray(point[canonical].values)
                frame[column] = np.round(values, 2)

        frame["Gridcell"] = self._gridcell_id(cell)
        return frame

    def export(
        self,
        climate: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
    ) -> list[Path]:
        """Write one gzipped weather file per cell; return the written paths.

        The regridded dataset is materialised **once** up front so per-cell
        extraction is pure in-memory indexing — the whole 10 km grid is small
        (``n_time × n_lat × n_lon``) even for a full domain, whereas computing
        lazily per cell would re-run the regrid graph for every cell.
        """
        out_dir = Path(output_dir) / "weather"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Materialising regridded climate grid before per-cell export")
        climate = climate.load()

        # Precompute the shared date column and the per-variable value cubes.
        dates = pd.to_datetime(climate["time"].values).strftime("%Y-%m-%d")
        present = {c: v for c, v in _CANONICAL_TO_SIMPLACE.items() if c in climate.data_vars}
        cubes = {col: np.round(np.asarray(climate[canon].values), 2)
                 for canon, col in present.items()}  # each shaped (time, lat, lon)
        written: list[Path] = []

        for _, cell in cell_table.iterrows():
            # Local row/col index the tile's data cube; global (g)col/(g)row name
            # the output so files stay unique and consistent across tiles.
            row, col = int(cell["row"]), int(cell["col"])
            gcol, grow = self._identity(cell)
            columns = {name: cube[:, row, col] for name, cube in cubes.items()}
            # Skip cells with no valid climate data anywhere (e.g. masked out).
            if not columns or all(np.isnan(v).all() for v in columns.values()):
                continue
            frame = pd.DataFrame({"Date": dates, **columns})
            frame["Gridcell"] = self._gridcell_id(cell)
            fname = f"daily_mean_RES1_C{gcol}R{grow}.csv.gz"
            written.append(self.write_csv(frame, out_dir / fname))

        logger.info("Exported %d weather files to %s", len(written), out_dir)
        return written
