"""End-to-end pipeline orchestration.

Wires the handlers and exporters together according to the execution flags in
the configuration. Each stage is guarded by its flag so the pipeline runs only
the requested work. Results from processing stages are kept in memory and
passed to the exporters that need them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

from data4simplace.climate import MSWXHandler
from data4simplace.config import PipelineConfig
from data4simplace.exporters import ManagementExporter, SoilExporter, WeatherExporter
from data4simplace.grid import TargetGrid
from data4simplace.npk import NPKHandler
from data4simplace.soil import SoilGridsHandler
from data4simplace.soil.ptf import saxton_rawls
from data4simplace.spatial import CroplandMask

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Artifacts produced by a pipeline run."""

    climate: Optional[xr.Dataset] = None
    soil: Optional[xr.Dataset] = None
    hydraulic: Optional[xr.Dataset] = None
    npk: Optional[xr.Dataset] = None
    cell_table: Optional[pd.DataFrame] = None
    written: list[Path] = field(default_factory=list)


class Pipeline:
    """Run the configured data-preparation stages.

    Parameters
    ----------
    config:
        The validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._grid = TargetGrid.from_config(config.grid)

    def run(self) -> PipelineResult:
        """Execute all enabled stages and return the produced artifacts."""
        flags = self._config.flags
        result = PipelineResult(cell_table=self._grid.cell_table())

        # --- Processing stages ------------------------------------------------
        if flags.run_climate_processing:
            logger.info("Stage: climate processing (MSWX)")
            result.climate = MSWXHandler(self._config).load()

        if flags.run_soil_processing:
            logger.info("Stage: soil processing (SoilGrids)")
            soil = SoilGridsHandler(self._config).load()
            soil = SoilGridsHandler.normalise_texture(soil)
            result.soil = soil

            if flags.compute_ptf and {"sand", "clay"}.issubset(soil.data_vars):
                logger.info("Stage: pedotransfer functions (Saxton-Rawls)")
                om = None
                if "soc" in soil.data_vars:
                    om = soil["soc"] * 0.1 * 1.724  # g/kg SOC -> % organic matter
                result.hydraulic = saxton_rawls(soil["sand"], soil["clay"], om)

        if flags.run_npk_processing:
            logger.info("Stage: NPK processing")
            result.npk = NPKHandler(self._config).load()

        # --- Cropland masking -------------------------------------------------
        if flags.apply_agricultural_mask:
            logger.info("Stage: agricultural mask")
            masker = CroplandMask(self._config)
            mask = masker.build()
            if result.climate is not None:
                result.climate = masker.apply(result.climate, mask)
            if result.soil is not None:
                result.soil = masker.apply(result.soil, mask)
            if result.hydraulic is not None:
                result.hydraulic = masker.apply(result.hydraulic, mask)
            if result.npk is not None:
                result.npk = masker.apply(result.npk, mask)
            result.cell_table = masker.keep_cells(result.cell_table, mask)

        # --- Export stages ----------------------------------------------------
        out_dir = self._config.paths.output_dir
        assert result.cell_table is not None  # set above

        if flags.export_simplace_weather:
            if result.climate is None:
                logger.warning("export_simplace_weather set but no climate data; skipping")
            else:
                exporter = WeatherExporter(self._config, self._config.reference.weather_dir)
                result.written.extend(
                    exporter.export(result.climate, result.cell_table, out_dir)
                )

        if flags.export_simplace_soil:
            if result.soil is None:
                logger.warning("export_simplace_soil set but no soil data; skipping")
            else:
                exporter = SoilExporter(self._config, self._config.reference.soil_dir)
                result.written.append(
                    exporter.export(
                        result.soil, result.cell_table, out_dir, hydraulic=result.hydraulic
                    )
                )

        if flags.export_simplace_management:
            if result.npk is None:
                logger.warning("export_simplace_management set but no NPK data; skipping")
            else:
                exporter = ManagementExporter(
                    self._config, self._config.reference.management_file
                )
                result.written.append(
                    exporter.export(result.npk, result.cell_table, out_dir)
                )

        logger.info("Pipeline complete: %d output file(s)", len(result.written))
        return result
