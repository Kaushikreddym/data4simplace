"""Tiled, restartable execution for large domains (e.g. all Europe).

A monolithic run over a continental grid and multi-decade window exhausts RAM
(the weather export materialises ``time × lat × lon`` for every variable) and
exceeds what the SoilGrids WCS will serve in one request. This module runs the
pipeline **tile by tile** over the target grid, keeping per-tile memory bounded
and each WCS request small, while preserving a single **global** cell identity
so outputs mosaic seamlessly.

Design
------
* The global :class:`~data4simplace.grid.TargetGrid` defines cell identity
  (``SimplaceID`` and the ``C<col>R<row>`` weather key). Tiles are aligned blocks
  of global cells, so a tile's local grid is an exact slice of the global one.
* Each tile is processed with a copy of the config whose ``grid`` bounds are the
  tile bounds. The resulting datasets use **local** ``row``/``col`` indices; the
  tile cell table additionally carries global ``grow``/``gcol``/``SimplaceID`` so
  the exporters name files and rows globally.
* Weather files are written directly (one per cell, globally named). Soil is
  written as one CSV **shard** per tile, concatenated into ``soil/soil.csv`` at
  the end. A per-tile marker under ``output/.tiles`` makes the run restartable.
* The multi-class stage (``soil.aggregation_method: top3``) shards the same way:
  one shard per (tile, rank), mosaicked into ``soil/soil_<rank>.csv``. Its
  intermediate rasters stay **per tile** under ``soil/netcdf_tiles/``, named with
  the tile suffix -- concatenating them is a job for xarray, not this module.
* The fertilizer schedule shards the same way, into
  ``management/fertilizer_<crop>.csv``. Its rows are per (cell, event), so the
  mosaic sorts stably to keep each cell's events in DVS order.
* The site table shards into ``site/site.csv``. Its CO2 companion is a single
  global series, so it is written once at combine time rather than by every
  tile writing the same file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from data4simplace.climate import MSWXHandler
from data4simplace.config import PipelineConfig
from data4simplace.exporters import (
    LongManagementExporter,
    LongSoilExporter,
    ManagementExporter,
    SiteExporter,
    SoilExporter,
    TopSoilExporter,
    WeatherExporter,
)
from data4simplace.grid import TargetGrid
from data4simplace.management import IrrigationClassifier
from data4simplace.npk import NPKHandler
from data4simplace.site import (
    SiteHandler,
    fill_calendar_gaps,
    load_co2_series,
    write_co2_series,
)
from data4simplace.soil import SoilGridsHandler
from data4simplace.spatial import apply_cell_mask, export_cell_mask, keep_cells

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileWindow:
    """A block of the global grid: rows ``[r0, r1)`` × cols ``[c0, c1)``."""

    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def name(self) -> str:
        return f"tile_{self.r0}_{self.c0}"


def iter_windows(n_lat: int, n_lon: int, step: int) -> Iterator[TileWindow]:
    """Yield aligned tile windows covering the ``n_lat × n_lon`` global grid."""
    for r0 in range(0, n_lat, step):
        for c0 in range(0, n_lon, step):
            yield TileWindow(r0, min(r0 + step, n_lat), c0, min(c0 + step, n_lon))


def _tile_step(config: PipelineConfig, tile_deg: float) -> int:
    """Cells per tile side for ``tile_deg`` on the config's grid resolution."""
    return max(1, round(tile_deg / config.grid.resolution_deg))


def list_windows(config: PipelineConfig, tile_deg: float) -> list[TileWindow]:
    """All tile windows for a tiled run, in the deterministic index order.

    The order is stable, so a SLURM array task can select the tile it owns by
    its task id: ``list_windows(config, tile_deg)[task_id]``.
    """
    grid = TargetGrid.from_config(config.grid)
    n_lat, n_lon = grid.shape
    return list(iter_windows(n_lat, n_lon, _tile_step(config, tile_deg)))


def count_tiles(config: PipelineConfig, tile_deg: float) -> int:
    """Number of tiles a tiled run would produce (the array size to request)."""
    return len(list_windows(config, tile_deg))


def _tile_config(config: PipelineConfig, grid: TargetGrid, w: TileWindow) -> PipelineConfig:
    """Config copy whose grid bounds cover exactly the tile's global cells."""
    res = config.grid.resolution_deg
    lat = grid.lat_centers  # north -> south (descending)
    lon = grid.lon_centers  # west -> east (ascending)
    tile_grid = config.grid.model_copy(
        update={
            "min_lon": float(lon[w.c0] - res / 2.0),
            "max_lon": float(lon[w.c1 - 1] + res / 2.0),
            "min_lat": float(lat[w.r1 - 1] - res / 2.0),
            "max_lat": float(lat[w.r0] + res / 2.0),
        }
    )
    return config.model_copy(update={"grid": tile_grid})


def _global_cell_table(tgrid: TargetGrid, w: TileWindow, n_lon: int) -> pd.DataFrame:
    """Tile cell table with local ``row``/``col`` plus global identity columns.

    ``row``/``col`` index the tile's data arrays; ``grow``/``gcol`` and
    ``SimplaceID`` carry the global identity used for output naming/keying.
    """
    ct = tgrid.cell_table()
    ct["grow"] = w.r0 + ct["row"]
    ct["gcol"] = w.c0 + ct["col"]
    ct["SimplaceID"] = (ct["grow"] * n_lon + ct["gcol"] + 1).astype("int64")
    return ct


def _run_tile(
    config: PipelineConfig,
    grid: TargetGrid,
    w: TileWindow,
    out_dir: Path,
    shard_dir: Path,
) -> int:
    """Process and export a single tile. Returns the number of exported cells."""
    _, n_lon = grid.shape
    flags = config.flags
    tcfg = _tile_config(config, grid, w)
    tgrid = TargetGrid.from_config(tcfg.grid)
    if tgrid.shape != (w.r1 - w.r0, w.c1 - w.c0):  # alignment guard
        raise RuntimeError(
            f"Tile {w.name} grid {tgrid.shape} != window {(w.r1 - w.r0, w.c1 - w.c0)}"
        )

    climate = MSWXHandler(tcfg).load() if flags.run_climate_processing else None
    soil = hydraulic = None
    top_classes = None
    if flags.run_soil_processing:
        handler = SoilGridsHandler(tcfg)
        soil, hydraulic = handler.load_processed()
        top_classes = handler.top_classes

    site = SiteHandler(tcfg).load() if flags.run_site_processing else None

    npk = None
    if flags.run_npk_processing:
        loaded = NPKHandler(tcfg).load()
        npk = loaded if len(loaded.data_vars) else None

    # Classified on the tile's own grid, so `row`/`col` below index it directly.
    irrigation = None
    if flags.run_irrigation_classification:
        irrigation = IrrigationClassifier(tcfg).classify()

    ct = _global_cell_table(tgrid, w, n_lon)

    mask = export_cell_mask(tcfg, tgrid, soil)
    climate = apply_cell_mask(climate, mask) if climate is not None else None
    soil = apply_cell_mask(soil, mask) if soil is not None else None
    hydraulic = apply_cell_mask(hydraulic, mask) if hydraulic is not None else None
    if site is not None:
        # Bounded by the tile's own export mask, so the fill never reaches
        # beyond the tile -- a fringe cell on a tile edge whose only covered
        # neighbour is in the next tile keeps the fallback instead of a
        # neighbour's calendar, which is the conservative side of that trade.
        if tcfg.site.fill_calendar_gaps:
            site = fill_calendar_gaps(site, within=mask)
        site = apply_cell_mask(site, mask)
    npk = apply_cell_mask(npk, mask) if npk is not None else None
    if top_classes is not None:
        top_classes.mask_cells(mask)
    ct = keep_cells(ct, mask)

    if ct.empty:
        logger.info("Tile %s: no exportable cells; skipping export", w.name)
        return 0

    # Gridded intermediates first, so they survive an exporter failure. One set
    # per tile: mosaicking rasters is left to the user (xr.open_mfdataset).
    if flags.write_soil_statistics and top_classes is not None:
        top_classes.write(out_dir, soil=soil, hydraulic=hydraulic, suffix=f"_{w.name}")

    if irrigation is not None and tcfg.irrigation.write_netcdf:
        # One raster per tile, as for the soil intermediates; mosaicking is left
        # to the user (xr.open_mfdataset).
        nc_dir = out_dir / "management" / "netcdf_tiles"
        nc_dir.mkdir(parents=True, exist_ok=True)
        irrigation.to_dataset(tgrid).to_netcdf(
            nc_dir / f"irrigation_class_{irrigation.crop_group}_{w.name}.nc"
        )

    if flags.export_simplace_weather and climate is not None:
        WeatherExporter(tcfg, tcfg.reference.weather_dir).export(climate, ct, out_dir)

    if flags.export_simplace_soil and soil is not None:
        if tcfg.export.writes("soil", "wide"):
            frame = SoilExporter(tcfg, tcfg.reference.soil_dir).build_frame(
                soil, ct, hydraulic=hydraulic
            )
            if not frame.empty:
                frame.to_csv(shard_dir / f"{w.name}.csv", index=False)
        if tcfg.export.writes("soil", "long"):
            exporter = LongSoilExporter(tcfg, tcfg.reference.soil_dir)
            frame = exporter.build_frame(soil, ct, hydraulic=hydraulic)
            if not frame.empty:
                long_dir = _long_soil_shard_dir(out_dir)
                long_dir.mkdir(parents=True, exist_ok=True)
                exporter.conform(frame).to_csv(long_dir / f"{w.name}.csv", index=False)

    if flags.export_simplace_site and site is not None:
        exporter = SiteExporter(tcfg)
        frame = exporter.build_frame(site, ct)
        if not frame.empty:
            site_dir = _site_shard_dir(out_dir)
            site_dir.mkdir(parents=True, exist_ok=True)
            exporter.conform(frame).to_csv(site_dir / f"{w.name}.csv", index=False)

    if flags.export_top3_soil_csvs and top_classes is not None:
        exporter = TopSoilExporter(tcfg, tcfg.reference.soil_dir)
        for rank in top_classes.ranks:
            rank = int(rank)
            frame = exporter.build_rank_frame(
                top_classes, rank, ct, hydraulic if rank == 1 else None
            )
            if frame.empty:
                continue
            rank_dir = _rank_shard_dir(shard_dir.parent, rank)
            rank_dir.mkdir(parents=True, exist_ok=True)
            # Conform here, not at concat time: the shard then already has the
            # SIMPLACE column order plus the metadata block.
            exporter.conform(frame).to_csv(rank_dir / f"{w.name}.csv", index=False)

    if flags.export_simplace_management and npk is not None:
        for layout, cls, shard_fn in (
            ("wide", ManagementExporter, _management_shard_dir),
            ("long", LongManagementExporter, _long_management_shard_dir),
        ):
            if not tcfg.export.writes("management", layout):
                continue
            exporter = cls(tcfg, tcfg.reference.management_file)
            frame = exporter.build_frame(npk, ct, irrigation=irrigation)
            if frame.empty:
                continue
            mgmt_dir = shard_fn(out_dir)
            mgmt_dir.mkdir(parents=True, exist_ok=True)
            exporter.conform(frame).to_csv(mgmt_dir / f"{w.name}.csv", index=False)

    return len(ct)


def _rank_shard_dir(soil_dir: Path, rank: int) -> Path:
    """Shard directory for one primary-class rank."""
    return soil_dir / f"_shards_rank{rank}"


def _management_shard_dir(out_dir: Path) -> Path:
    """Shard directory for the per-tile fertilizer schedules."""
    return out_dir / "management" / "_shards"


def _site_shard_dir(out_dir: Path) -> Path:
    """Shard directory for the per-tile site tables."""
    return out_dir / "site" / "_shards"


def _long_soil_shard_dir(out_dir: Path) -> Path:
    """Shard directory for the per-tile long-layout soil tables."""
    return out_dir / "soil" / "_shards_long"


def _long_management_shard_dir(out_dir: Path) -> Path:
    """Shard directory for the per-tile long-layout fertilizer schedules."""
    return out_dir / "management" / "_shards_long"


def _soil_long_key(config: PipelineConfig) -> str:
    """The cell-identifier column of the long soil dialect this run writes."""
    return LongSoilExporter(config, config.reference.soil_dir).dialect.key_column


def _management_long_key(config: PipelineConfig) -> str:
    """The cell-identifier column of the long management dialect."""
    return LongManagementExporter(
        config, config.reference.management_file
    ).dialect.key_column


def _concat_shards(
    shard_dir: Path, out_path: Path, label: str = "soil", key: str = "location"
) -> None:
    """Mosaic per-tile CSV shards into a single file (global rows).

    The sort is **stable**, so a product with several rows per cell (the
    fertilizer schedule one row per event, the long soil table one row per
    depth) keeps its within-cell ordering.

    ``key`` names the cell-identifier column, which the long dialects spell
    differently (``Location``, ``soiltype``). An unrecognised key leaves the
    shard order untouched rather than raising: concatenated-but-unsorted is a
    usable file, and every shard is internally ordered already.
    """
    shards = sorted(shard_dir.glob("tile_*.csv"))
    if not shards:
        logger.warning("No %s shards to concatenate under %s", label, shard_dir)
        return
    frames = [pd.read_csv(p) for p in shards]
    combined = pd.concat(frames, ignore_index=True)
    if key in combined.columns:
        combined = combined.sort_values(key, kind="stable").reset_index(drop=True)
    else:
        logger.warning(
            "%s shards have no %r column to sort on (columns: %s); keeping "
            "shard order", label, key, list(combined.columns)[:6],
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Wrote mosaicked %s (%d rows) -> %s", label, len(combined), out_path)


def combine_tiles(
    config: PipelineConfig, tile_deg: float | None = None
) -> dict[str, int]:
    """Combine tile outputs into final files, without reprocessing.

    Use this after tiles were produced by separate jobs, or to re-mosaic soil
    after an interrupted run. Weather files are already per-cell and globally
    named, so they need no merging; this concatenates the soil and fertilizer
    shards into ``soil/soil.csv`` / ``management/fertilizer_<crop>.csv`` and
    reports coverage.

    Parameters
    ----------
    config:
        The **global** pipeline configuration used for the run.
    tile_deg:
        If given, the tile size used for the run, so completeness can be checked
        (expected tiles vs. done-markers). Omit to just combine whatever exists.

    Returns
    -------
    dict
        ``{"weather_files", "soil_rows", "top3_rows", "management_rows",
        "tiles_expected", "tiles_done", "tiles_missing"}``.
    """
    out_dir = Path(config.paths.output_dir)
    shard_dir = out_dir / "soil" / "_shards"
    markers = out_dir / ".tiles"

    done = {p.stem for p in markers.glob("*.done")} if markers.is_dir() else set()
    expected = missing = 0
    if tile_deg is not None:
        grid = TargetGrid.from_config(config.grid)
        n_lat, n_lon = grid.shape
        step = max(1, round(tile_deg / config.grid.resolution_deg))
        names = {w.name for w in iter_windows(n_lat, n_lon, step)}
        expected = len(names)
        missing_names = sorted(names - done)
        missing = len(missing_names)
        if missing:
            logger.warning(
                "%d/%d tiles have no completion marker; soil.csv will be "
                "incomplete. First missing: %s",
                missing, expected, ", ".join(missing_names[:10]),
            )

    soil_rows = 0
    if config.flags.export_simplace_soil:
        if config.export.writes("soil", "wide"):
            out_path = out_dir / "soil" / "soil.csv"
            _concat_shards(shard_dir, out_path, label="soil")
            if out_path.is_file():
                soil_rows = sum(1 for _ in out_path.open()) - 1  # minus header
        if config.export.writes("soil", "long"):
            _concat_shards(
                _long_soil_shard_dir(out_dir),
                out_dir / "soil" / "soil_long.csv",
                label="soil (long)",
                key=_soil_long_key(config),
            )

    top3_rows = 0
    if config.flags.export_top3_soil_csvs:
        for rank in range(1, config.soil.n_primary_classes + 1):
            rank_dir = _rank_shard_dir(out_dir / "soil", rank)
            if not rank_dir.is_dir():
                continue
            rank_path = out_dir / "soil" / f"soil_{rank}.csv"
            _concat_shards(rank_dir, rank_path, label=f"soil rank {rank}")
            if rank_path.is_file():
                top3_rows += sum(1 for _ in rank_path.open()) - 1

    management_rows = 0
    if config.flags.export_simplace_management:
        crop = config.npk.simplace_crop
        if config.export.writes("management", "wide"):
            mgmt_path = out_dir / "management" / f"fertilizer_{crop}.csv"
            _concat_shards(_management_shard_dir(out_dir), mgmt_path, label="management")
            if mgmt_path.is_file():
                management_rows = sum(1 for _ in mgmt_path.open()) - 1
        if config.export.writes("management", "long"):
            _concat_shards(
                _long_management_shard_dir(out_dir),
                out_dir / "management" / f"fertilizer_{crop}_long.csv",
                label="management (long)",
                key=_management_long_key(config),
            )

    site_rows = 0
    if config.flags.export_simplace_site:
        site_path = out_dir / "site" / "site.csv"
        _concat_shards(_site_shard_dir(out_dir), site_path, label="site")
        if site_path.is_file():
            site_rows = sum(1 for _ in site_path.open()) - 1
        # The CO2 series is global, so it is written once at combine time
        # rather than sharded and re-written identically by every tile.
        series, source = load_co2_series(config.paths.co2_file, _simulated_years(config))
        write_co2_series(series, source, out_dir)

    weather_files = len(list((out_dir / "weather").glob("*.csv.gz")))
    logger.info(
        "Combined: %d weather files (already global), soil.csv %d rows, "
        "per-class CSVs %d rows, site %d rows, fertilizer %d rows, "
        "%d tile markers present",
        weather_files, soil_rows, top3_rows, site_rows, management_rows, len(done),
    )
    return {
        "weather_files": weather_files,
        "soil_rows": soil_rows,
        "top3_rows": top3_rows,
        "site_rows": site_rows,
        "management_rows": management_rows,
        "tiles_expected": expected,
        "tiles_done": len(done),
        "tiles_missing": missing,
    }


def _simulated_years(config: PipelineConfig) -> list[int]:
    """Calendar years the run's time window spans, widened by a year each side.

    A winter crop is sown the year before it is harvested, so a season that
    straddles New Year needs the CO2 of both its years.
    """
    start = int(str(config.time.start)[:4])
    end = int(str(config.time.end)[:4])
    return list(range(start - 1, end + 2))


def _tile_dirs(config: PipelineConfig) -> tuple[Path, Path, Path]:
    """Create and return ``(out_dir, markers_dir, shard_dir)`` for a tiled run."""
    out_dir = Path(config.paths.output_dir)
    markers = out_dir / ".tiles"
    shard_dir = out_dir / "soil" / "_shards"
    markers.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, markers, shard_dir


def run_one_tile(
    config: PipelineConfig,
    tile_index: int,
    tile_deg: float = 5.0,
    resume: bool = True,
) -> dict[str, int]:
    """Process exactly one tile, selected by its index in :func:`list_windows`.

    This is the unit of work for a SLURM (or any) array job: launch
    ``count_tiles`` tasks, each calling this with its task id. Every task writes
    its own weather files (globally named) and its own soil shard plus a
    ``.done`` marker, so tasks never contend for the same file. Combine the
    shards afterwards with :func:`combine_tiles` (one dependent job).

    Parameters
    ----------
    config:
        The **global** pipeline configuration.
    tile_index:
        0-based tile index; must be ``< count_tiles(config, tile_deg)``.
    tile_deg:
        Tile edge length in degrees; must match across all tasks and the combine.
    resume:
        Skip the tile if its completion marker already exists.

    Returns
    -------
    dict
        ``{"tile_index", "cells", "skipped"}`` (``skipped`` is 1 when resumed).
    """
    grid = TargetGrid.from_config(config.grid)
    windows = list_windows(config, tile_deg)
    if not 0 <= tile_index < len(windows):
        raise IndexError(
            f"tile_index {tile_index} out of range 0..{len(windows) - 1} "
            f"for tile_deg={tile_deg}"
        )

    w = windows[tile_index]
    out_dir, markers, shard_dir = _tile_dirs(config)
    marker = markers / f"{w.name}.done"
    if resume and marker.exists():
        logger.info("Tile %s (index %d) already done; skipping", w.name, tile_index)
        return {"tile_index": tile_index, "cells": 0, "skipped": 1}

    logger.info(
        "Tile %s (index %d/%d) rows[%d:%d] cols[%d:%d]",
        w.name, tile_index, len(windows) - 1, w.r0, w.r1, w.c0, w.c1,
    )
    n = _run_tile(config, grid, w, out_dir, shard_dir)
    marker.write_text("ok\n")
    logger.info("Tile %s done: %d cells exported", w.name, n)
    return {"tile_index": tile_index, "cells": n, "skipped": 0}


def run_tiled(
    config: PipelineConfig, tile_deg: float = 5.0, resume: bool = True
) -> dict[str, int]:
    """Run the pipeline tile by tile over the whole target grid.

    Parameters
    ----------
    config:
        Validated pipeline configuration (its ``grid`` is the **global** grid).
    tile_deg:
        Approximate tile edge length in degrees. Smaller tiles use less memory
        and keep SoilGrids WCS requests within service limits.
    resume:
        Skip tiles whose completion marker already exists (restartable runs).

    Returns
    -------
    dict
        ``{"tiles": total, "done": processed, "cells": exported_cells}``.
    """
    grid = TargetGrid.from_config(config.grid)
    n_lat, n_lon = grid.shape
    step = _tile_step(config, tile_deg)

    out_dir, markers, shard_dir = _tile_dirs(config)

    windows = list(iter_windows(n_lat, n_lon, step))
    logger.info(
        "Tiled run: %d tiles of ~%.1f deg (%d cells/side) over %dx%d grid",
        len(windows), tile_deg, step, n_lat, n_lon,
    )

    processed = cells = 0
    for i, w in enumerate(windows, 1):
        marker = markers / f"{w.name}.done"
        if resume and marker.exists():
            logger.info("Tile %s (%d/%d) already done; skipping", w.name, i, len(windows))
            continue
        logger.info("Tile %s (%d/%d) rows[%d:%d] cols[%d:%d]",
                    w.name, i, len(windows), w.r0, w.r1, w.c0, w.c1)
        try:
            n = _run_tile(config, grid, w, out_dir, shard_dir)
        except Exception:  # noqa: BLE001 - one tile must not abort the whole run
            logger.exception("Tile %s failed; leaving unmarked for a later retry", w.name)
            continue
        cells += n
        processed += 1
        marker.write_text("ok\n")

    if config.flags.export_simplace_soil:
        if config.export.writes("soil", "wide"):
            _concat_shards(shard_dir, out_dir / "soil" / "soil.csv", label="soil")
        if config.export.writes("soil", "long"):
            _concat_shards(
                _long_soil_shard_dir(out_dir),
                out_dir / "soil" / "soil_long.csv",
                label="soil (long)",
                key=_soil_long_key(config),
            )
    if config.flags.export_simplace_site:
        _concat_shards(_site_shard_dir(out_dir), out_dir / "site" / "site.csv", label="site")
        series, source = load_co2_series(config.paths.co2_file, _simulated_years(config))
        write_co2_series(series, source, out_dir)
    if config.flags.export_simplace_management:
        crop = config.npk.simplace_crop
        if config.export.writes("management", "wide"):
            _concat_shards(
                _management_shard_dir(out_dir),
                out_dir / "management" / f"fertilizer_{crop}.csv",
                label="management",
            )
        if config.export.writes("management", "long"):
            _concat_shards(
                _long_management_shard_dir(out_dir),
                out_dir / "management" / f"fertilizer_{crop}_long.csv",
                label="management (long)",
                key=_management_long_key(config),
            )

    logger.info("Tiled run complete: %d/%d tiles, %d cells exported",
                processed, len(windows), cells)
    return {"tiles": len(windows), "done": processed, "cells": cells}
