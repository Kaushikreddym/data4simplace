"""SIMPLACE management / fertilizer schedule exporter.

Generates a fertilizer application schedule per grid cell from the regridded
NPK dataset, conforming to the SIMPLACE management reference file. When a
reference schedule is present, its columns, delimiter, application dates and
missing sentinel are reused; the per-cell N/P/K amounts are filled from data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.exporters.base_exporter import BaseExporter, ReferenceSpec, parse_reference_csv

logger = logging.getLogger(__name__)

# Recognised nutrient -> SIMPLACE management column names.
_NUTRIENT_TO_SIMPLACE = {"N": "N_AMOUNT", "P": "P_AMOUNT", "K": "K_AMOUNT"}


class ManagementExporter(BaseExporter):
    """Export a SIMPLACE fertilizer schedule from NPK data."""

    kind = "management"

    def __init__(self, config: PipelineConfig, reference_path: str | Path | None) -> None:
        super().__init__(config, reference_path)
        self._template: Optional[pd.DataFrame] = None

    def fallback_spec(self) -> ReferenceSpec:
        """Documented default matching common SIMPLACE fertilizer layouts."""
        return ReferenceSpec(
            delimiter=",",
            columns=["SimplaceID", "DATE", "N_AMOUNT", "P_AMOUNT", "K_AMOUNT"],
            missing_value=str(self._config.missing_value),
        )

    def _reference_template(self) -> Optional[pd.DataFrame]:
        """Load the reference fertilizer schedule rows, if a file exists."""
        if self._template is not None:
            return self._template
        ref_file = self._resolve_reference_file()
        if ref_file is None:
            return None
        spec = parse_reference_csv(ref_file, default_missing=str(self._config.missing_value))
        self._template = pd.read_csv(ref_file, sep=spec.delimiter)
        return self._template

    def build_frame(self, npk: xr.Dataset, cell_table: pd.DataFrame) -> pd.DataFrame:
        """Build the fertilizer schedule table.

        If a reference schedule exists, its application-date rows are replicated
        per cell and the nutrient amounts overwritten with cell values. Without
        a reference, a single application row per cell is emitted.
        """
        template = self._reference_template()
        rows: list[dict[str, object]] = []

        for _, cell in cell_table.iterrows():
            amounts: dict[str, float] = {}
            if npk.data_vars:
                point = npk.sel(lat=cell["lat"], lon=cell["lon"], method="nearest")
                for nutrient, column in _NUTRIENT_TO_SIMPLACE.items():
                    if nutrient in point.data_vars:
                        val = point[nutrient].compute().item()
                        if not np.isnan(val):
                            amounts[column] = round(float(val), 3)
            if not amounts:
                continue

            if template is not None:
                for _, tmpl_row in template.iterrows():
                    record = tmpl_row.to_dict()
                    record["SimplaceID"] = int(cell["SimplaceID"])
                    record.update(amounts)
                    rows.append(record)
            else:
                record = {"SimplaceID": int(cell["SimplaceID"]), "DATE": self._config.time.start}
                record.update(amounts)
                rows.append(record)

        return pd.DataFrame(rows)

    def export(self, npk: xr.Dataset, cell_table: pd.DataFrame, output_dir: str | Path) -> Path:
        """Write the fertilizer schedule file; return its path."""
        frame = self.build_frame(npk, cell_table)
        out_path = Path(output_dir) / "management" / "fertilizer.csv"
        return self.write_csv(frame, out_path)
