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
# Reference columns that cannot be derived from MSWX inputs (filled with the
# sentinel via conform()): wind speed and reference evapotranspiration.
_UNAVAILABLE = ("Windspeed", "RefET")


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
    def _gridcell_id(cell: pd.Series) -> str:
        """SIMPLACE ``C_<col>:R_<row>`` identifier from grid indices."""
        return f"C_{int(cell['col'])}:R_{int(cell['row'])}"

    def build_frame(self, cell: pd.Series, climate: xr.Dataset) -> pd.DataFrame:
        """Build a daily weather table for a single grid cell.

        Parameters
        ----------
        cell:
            One row of :meth:`TargetGrid.cell_table` (``SimplaceID``, row, col,
            lat, lon).
        climate:
            Regridded climate dataset with dims ``(time, lat, lon)``.
        """
        point = climate.sel(lat=cell["lat"], lon=cell["lon"], method="nearest")
        times = pd.to_datetime(point["time"].values)
        frame = pd.DataFrame({"Date": times.strftime("%Y-%m-%d")})

        for canonical, column in _CANONICAL_TO_SIMPLACE.items():
            if canonical in point.data_vars:
                values = point[canonical].compute().values
                frame[column] = pd.Series(values).round(2).to_numpy()

        frame["Gridcell"] = self._gridcell_id(cell)
        return frame

    def export(
        self,
        climate: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
    ) -> list[Path]:
        """Write one gzipped weather file per cell; return the written paths."""
        out_dir = Path(output_dir) / "weather"
        out_dir.mkdir(parents=True, exist_ok=True)
        weather_cols = [c for c in _CANONICAL_TO_SIMPLACE.values()]
        written: list[Path] = []

        for _, cell in cell_table.iterrows():
            frame = self.build_frame(cell, climate)
            present = [c for c in weather_cols if c in frame.columns]
            if not present or frame[present].dropna(how="all").empty:
                continue  # no valid climate data for this cell (e.g. masked)
            fname = f"daily_mean_RES1_C{int(cell['col'])}R{int(cell['row'])}.csv.gz"
            written.append(self.write_csv(frame, out_dir / fname))

        logger.info("Exported %d weather files to %s", len(written), out_dir)
        return written
