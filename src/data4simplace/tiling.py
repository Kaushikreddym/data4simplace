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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from data4simplace.climate import MSWXHandler
from data4simplace.config import PipelineConfig
from data4simplace.exporters import SoilExporter, WeatherExporter
from data4simplace.grid import TargetGrid
from data4simplace.soil import SoilGridsHandler
from data4simplace.spatial import CroplandMask

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
    if flags.run_soil_processing:
        soil, hydraulic = SoilGridsHandler(tcfg).load_processed()

    ct = _global_cell_table(tgrid, w, n_lon)

    if flags.apply_agricultural_mask:
        masker = CroplandMask(tcfg)
        mask = masker.build()
        if climate is not None:
            climate = masker.apply(climate, mask)
        if soil is not None:
            soil = masker.apply(soil, mask)
        if hydraulic is not None:
            hydraulic = masker.apply(hydraulic, mask)
        ct = masker.keep_cells(ct, mask)

    if ct.empty:
        logger.info("Tile %s: no cells after masking; skipping export", w.name)
        return 0

    if flags.export_simplace_weather and climate is not None:
        WeatherExporter(tcfg, tcfg.reference.weather_dir).export(climate, ct, out_dir)

    if flags.export_simplace_soil and soil is not None:
        frame = SoilExporter(tcfg, tcfg.reference.soil_dir).build_frame(
            soil, ct, hydraulic=hydraulic
        )
        if not frame.empty:
            frame.to_csv(shard_dir / f"{w.name}.csv", index=False)

    return len(ct)


def _concat_soil_shards(shard_dir: Path, out_path: Path) -> None:
    """Mosaic per-tile soil shards into a single ``soil.csv`` (global rows)."""
    shards = sorted(shard_dir.glob("tile_*.csv"))
    if not shards:
        logger.warning("No soil shards to concatenate under %s", shard_dir)
        return
    frames = [pd.read_csv(p) for p in shards]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("location").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    logger.info("Wrote mosaicked soil (%d rows) -> %s", len(combined), out_path)


def combine_tiles(
    config: PipelineConfig, tile_deg: float | None = None
) -> dict[str, int]:
    """Combine tile outputs into final files, without reprocessing.

    Use this after tiles were produced by separate jobs, or to re-mosaic soil
    after an interrupted run. Weather files are already per-cell and globally
    named, so they need no merging; this concatenates the soil shards into
    ``soil/soil.csv`` and reports coverage.

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
        ``{"weather_files", "soil_rows", "tiles_expected", "tiles_done", "tiles_missing"}``.
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
        out_path = out_dir / "soil" / "soil.csv"
        _concat_soil_shards(shard_dir, out_path)
        if out_path.is_file():
            soil_rows = sum(1 for _ in out_path.open()) - 1  # minus header

    weather_files = len(list((out_dir / "weather").glob("*.csv.gz")))
    logger.info(
        "Combined: %d weather files (already global), soil.csv %d rows, "
        "%d tile markers present",
        weather_files, soil_rows, len(done),
    )
    return {
        "weather_files": weather_files,
        "soil_rows": soil_rows,
        "tiles_expected": expected,
        "tiles_done": len(done),
        "tiles_missing": missing,
    }


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
    step = max(1, round(tile_deg / config.grid.resolution_deg))

    out_dir = Path(config.paths.output_dir)
    markers = out_dir / ".tiles"
    shard_dir = out_dir / "soil" / "_shards"
    markers.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

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
        _concat_soil_shards(shard_dir, out_dir / "soil" / "soil.csv")

    logger.info("Tiled run complete: %d/%d tiles, %d cells exported",
                processed, len(windows), cells)
    return {"tiles": len(windows), "done": processed, "cells": cells}
