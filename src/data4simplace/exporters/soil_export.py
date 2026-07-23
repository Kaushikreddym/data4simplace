"""SIMPLACE soil profile file generator.

Produces a depth-resolved soil profile table on the target grid from the
regridded SoilGrids dataset (optionally enriched with pedotransfer-derived
hydraulic parameters). Column order, delimiter, missing sentinel and depth
horizons are taken from the SIMPLACE soil reference when available.
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

# Map SoilGrids layer names -> common SIMPLACE soil column names.
_LAYER_TO_SIMPLACE = {
    "clay": "CLAY",
    "silt": "SILT",
    "sand": "SAND",
    "bdod": "BULKDENSITY",
    "soc": "ORGANICCARBON",
    "phh2o": "PH",
    "nitrogen": "NITROGEN",
}
_PTF_TO_SIMPLACE = {
    "theta_wp": "WILTINGPOINT",
    "theta_fc": "FIELDCAPACITY",
    "theta_sat": "SATURATION",
    "ksat": "KSAT",
}


class SoilExporter(BaseExporter):
    """Export a SIMPLACE soil profile file from soil data."""

    kind = "soil"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)

    def fallback_spec(self) -> ReferenceSpec:
        """Documented default matching common SIMPLACE soil layouts."""
        return ReferenceSpec(
            delimiter=",",
            columns=[
                "SimplaceID",
                "DEPTH_TOP",
                "DEPTH_BOTTOM",
                "CLAY",
                "SILT",
                "SAND",
                "BULKDENSITY",
                "ORGANICCARBON",
                "PH",
                "NITROGEN",
            ],
            missing_value=str(self._config.missing_value),
        )

    @staticmethod
    def _depth_bounds(depth: str) -> tuple[int, int]:
        """Parse ``"5-15cm"`` -> ``(5, 15)``."""
        token = depth.lower().replace("cm", "")
        top, bottom = token.split("-")
        return int(top), int(bottom)

    def build_frame(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        hydraulic: xr.Dataset | None = None,
    ) -> pd.DataFrame:
        """Build the long-format soil profile table (one row per cell x depth).

        Parameters
        ----------
        soil:
            Regridded soil dataset; variables carry a ``depth`` dimension.
        cell_table:
            Grid cell table (``SimplaceID``, lat, lon).
        hydraulic:
            Optional PTF-derived dataset aligned to ``soil``.
        """
        if "depth" not in soil.dims:
            raise ValueError("Soil dataset must carry a 'depth' dimension")

        depths = [str(d) for d in soil["depth"].values]
        merged = soil if hydraulic is None else xr.merge([soil, hydraulic])
        rows: list[dict[str, object]] = []

        for _, cell in cell_table.iterrows():
            point = merged.sel(lat=cell["lat"], lon=cell["lon"], method="nearest")
            for i, depth in enumerate(depths):
                top, bottom = self._depth_bounds(depth)
                record: dict[str, object] = {
                    "SimplaceID": int(cell["SimplaceID"]),
                    "DEPTH_TOP": top,
                    "DEPTH_BOTTOM": bottom,
                }
                for layer, column in {**_LAYER_TO_SIMPLACE, **_PTF_TO_SIMPLACE}.items():
                    if layer in point.data_vars:
                        val = point[layer].isel(depth=i).compute().item()
                        record[column] = None if np.isnan(val) else round(float(val), 4)
                rows.append(record)

        frame = pd.DataFrame(rows)
        # Drop cells that are entirely missing (e.g. masked-out ocean/non-crop).
        value_cols = [c for c in frame.columns if c not in {"SimplaceID", "DEPTH_TOP", "DEPTH_BOTTOM"}]
        frame = frame[~frame[value_cols].isna().all(axis=1)].reset_index(drop=True)
        return frame

    def export(
        self,
        soil: xr.Dataset,
        cell_table: pd.DataFrame,
        output_dir: str | Path,
        hydraulic: xr.Dataset | None = None,
    ) -> Path:
        """Write the soil profile file; return its path."""
        frame = self.build_frame(soil, cell_table, hydraulic)
        out_path = Path(output_dir) / "soil" / "soil_profiles.csv"
        return self.write_csv(frame, out_path)
