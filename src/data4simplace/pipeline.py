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
from data4simplace.exporters import (
    ManagementExporter,
    SoilExporter,
    TopSoilExporter,
    WeatherExporter,
)
from data4simplace.grid import TargetGrid
from data4simplace.management import IrrigationClassification, IrrigationClassifier
from data4simplace.npk import NPKHandler
from data4simplace.soil import SoilGridsHandler
from data4simplace.soil.multiclass import PrimaryClassStatistics, TopClassAggregation
from data4simplace.spatial import apply_cell_mask, export_cell_mask, keep_cells

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Artifacts produced by a pipeline run."""

    climate: Optional[xr.Dataset] = None
    soil: Optional[xr.Dataset] = None
    hydraulic: Optional[xr.Dataset] = None
    npk: Optional[xr.Dataset] = None
    irrigation: Optional[IrrigationClassification] = None
    soil_statistics: Optional["PrimaryClassStatistics"] = None
    top_classes: Optional["TopClassAggregation"] = None
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
            logger.info(
                "Stage: soil processing (SoilGrids, mode=%s, export=%s)",
                self._config.soil.dominant_mode,
                self._config.soil.export_statistic,
            )
            soil_handler = SoilGridsHandler(self._config)
            result.soil, result.hydraulic = soil_handler.load_processed()
            result.soil_statistics = soil_handler.class_statistics
            result.top_classes = soil_handler.top_classes
            if flags.write_soil_statistics and result.soil_statistics is None:
                logger.warning(
                    "write_soil_statistics set but dominant_mode=%s produces no "
                    "class field; no statistics written",
                    self._config.soil.dominant_mode,
                )

        if flags.run_npk_processing:
            logger.info(
                "Stage: NPK processing (source=%s, crop=%s)",
                self._config.npk.source,
                self._config.npk.crop,
            )
            npk = NPKHandler(self._config).load()
            # An empty dataset (no rasters found) carries no lat/lon; treat it as
            # absent so masking and management export skip it cleanly.
            result.npk = npk if len(npk.data_vars) else None
            if result.npk is None:
                logger.warning("NPK processing produced no layers; downstream NPK steps will skip")

        if flags.run_irrigation_classification:
            classifier = IrrigationClassifier(self._config)
            logger.info(
                "Stage: irrigation classification (source=%s, crop=%s, threshold=%g)",
                self._config.irrigation.source,
                classifier.crop_group,
                self._config.irrigation.threshold,
            )
            result.irrigation = classifier.classify()

        # --- Exported cell set ------------------------------------------------
        # PROBA-V cropland (when the flag is set) intersected with the cells the
        # soil stage actually produced values for, so weather, soil and
        # management all cover exactly the same cells.
        logger.info("Stage: resolving the exported cell set")
        mask = export_cell_mask(self._config, self._grid, result.soil)
        if result.climate is not None:
            result.climate = apply_cell_mask(result.climate, mask)
        if result.soil is not None:
            result.soil = apply_cell_mask(result.soil, mask)
        if result.hydraulic is not None:
            result.hydraulic = apply_cell_mask(result.hydraulic, mask)
        if result.npk is not None:
            result.npk = apply_cell_mask(result.npk, mask)
        if result.top_classes is not None:
            # The per-class products follow the same cell set as soil.csv, so a
            # rank-2 profile is never exported for a cell weather skipped.
            result.top_classes.mask_cells(mask)
        result.cell_table = keep_cells(result.cell_table, mask)

        # --- Export stages ----------------------------------------------------
        out_dir = self._config.paths.output_dir
        assert result.cell_table is not None  # set above

        # Intermediate per-class statistics (NetCDF + share table). Written
        # before the CSVs so they survive an exporter failure.
        if flags.write_soil_statistics and result.soil_statistics is not None:
            result.written.extend(result.soil_statistics.write(out_dir, self._grid))

        # Intermediate gridded rasters of the multi-class stage (same rule: the
        # NetCDFs land before any CSV is written).
        if flags.write_soil_statistics and result.top_classes is not None:
            result.written.extend(
                result.top_classes.write(
                    out_dir, soil=result.soil, hydraulic=result.hydraulic
                )
            )

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

        if flags.export_top3_soil_csvs:
            if result.top_classes is None:
                logger.warning(
                    "export_top3_soil_csvs set but the soil stage produced no "
                    "per-class aggregation; skipping"
                )
            else:
                exporter = TopSoilExporter(self._config, self._config.reference.soil_dir)
                result.written.extend(
                    exporter.export(
                        result.top_classes,
                        result.cell_table,
                        out_dir,
                        hydraulic=result.hydraulic,
                    )
                )

        # The gridded classification lands before the schedule, so a failure in
        # the exporter does not lose a stage that re-reads two whole products.
        if result.irrigation is not None and self._config.irrigation.write_netcdf:
            nc_path = (
                Path(out_dir)
                / "management"
                / f"irrigation_class_{result.irrigation.crop_group}.nc"
            )
            nc_path.parent.mkdir(parents=True, exist_ok=True)
            result.irrigation.to_dataset(self._grid).to_netcdf(nc_path)
            result.written.append(nc_path)
            logger.info("Wrote %s", nc_path)

        if flags.export_simplace_management:
            if result.npk is None:
                logger.warning("export_simplace_management set but no NPK data; skipping")
            else:
                exporter = ManagementExporter(
                    self._config, self._config.reference.management_file
                )
                result.written.append(
                    exporter.export(
                        result.npk,
                        result.cell_table,
                        out_dir,
                        irrigation=result.irrigation,
                    )
                )

        logger.info("Pipeline complete: %d output file(s)", len(result.written))
        return result
