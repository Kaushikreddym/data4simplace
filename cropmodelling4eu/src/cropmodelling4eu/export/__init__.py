"""Readers for a finished data4simplace export — shared by both models.

Everything here is model-agnostic on purpose. SIMPLACE and torchcrop differ in
how they simulate a cell, not in which cells exist, where those cells are, what
soil they have or when the crop goes in; sharing that layer is what makes the
two models' results comparable rather than merely adjacent.

:class:`ExportBundle` is the entry point: it resolves the runnable cell set and
holds every table a runner needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.export.cells import (
    cell_frame,
    id_to_lonlat,
    id_to_rowcol,
    rowcol_to_id,
    shard_cells,
    weather_ids,
    weather_path,
)
from cropmodelling4eu.export.management import (
    irrigation_flags,
    load_composition,
    read_fertilizer_plans,
)
from cropmodelling4eu.export.site import SiteTable, read_co2, read_site
from cropmodelling4eu.export.soil import SoilProfiles, read_soil
from cropmodelling4eu.export.weather import (
    load_season_block,
    load_seasons,
    season_window,
    sowing_date,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExportBundle",
    "SiteTable",
    "SoilProfiles",
    "cell_frame",
    "id_to_lonlat",
    "id_to_rowcol",
    "irrigation_flags",
    "load_composition",
    "load_season_block",
    "load_seasons",
    "read_co2",
    "read_fertilizer_plans",
    "read_site",
    "read_soil",
    "rowcol_to_id",
    "season_window",
    "shard_cells",
    "sowing_date",
    "weather_ids",
    "weather_path",
]


@dataclass
class ExportBundle:
    """Everything a run needs from one export, over one cell set.

    Attributes
    ----------
    config:
        The run configuration the bundle was resolved against.
    ids:
        The runnable cells, ascending.
    soil:
        Profiles for exactly those cells, in ``ids`` order.
    plans:
        ``{SimplaceID: [n_events, 4]}`` of ``(DVS, N, P, K)`` in g/m².
    site:
        Per-cell sowing date and altitude.
    irrigated:
        Per-cell irrigated flag; cells absent from it run rainfed.
    co2:
        Annual CO2 [ppm] covering the configured seasons.
    """

    config: RunConfig
    ids: np.ndarray
    soil: SoilProfiles
    plans: dict[int, np.ndarray]
    site: SiteTable
    irrigated: pd.Series
    co2: pd.Series

    @property
    def export_dir(self) -> Path:
        return Path(self.config.paths.export_dir)

    def cells(self) -> pd.DataFrame:
        """The runnable cells as a table with coordinates."""
        return cell_frame(self.ids, self.config.grid)

    def shard(self, shard: int, n_shards: int) -> np.ndarray:
        """The round-robin slice of the runnable cells for one task."""
        return shard_cells(self.ids, shard, n_shards)


def resolve_export(
    config: RunConfig,
    require_management: bool = True,
    soil_layout: str = "auto",
) -> ExportBundle:
    """Resolve the cells that can be run end to end, and load their inputs.

    Runnable means **weather and soil and a fertilizer plan** — the same
    intersection for both models, so a cell either appears in every result or
    in none. A cell with weather but no plan would otherwise fall back to some
    default schedule and quietly report a yield that is not the one the export
    describes.

    Site is deliberately *not* part of the intersection: a cell missing from
    ``site.csv`` takes the documented fallback sowing date rather than dropping
    out of the run, which keeps the cell set identical to what the models could
    run before the site stage existed.

    Parameters
    ----------
    require_management:
        Set ``False`` for a potential-yield run (``iopt=1``), which applies no
        fertilizer and so needs no schedule.
    soil_layout:
        Passed through to :func:`~cropmodelling4eu.export.soil.read_soil`.

    Raises
    ------
    FileNotFoundError
        If the export lacks weather or soil.
    RuntimeError
        If the intersection is empty, with the per-input counts, since "0 cells"
        alone does not say which input is the short one.
    """
    export_dir = Path(config.paths.export_dir)
    grid = config.grid

    with_weather = weather_ids(export_dir, grid)
    soil = read_soil(export_dir, layout=soil_layout)
    ids = np.intersect1d(with_weather, np.asarray(soil.ids, dtype=np.int64))

    plans: dict[int, np.ndarray] = {}
    irrigated = pd.Series(dtype="int64")
    schedule = _find_schedule(export_dir, config.season.crop)
    if schedule is not None:
        plans = read_fertilizer_plans(
            schedule, _find_composition(config, export_dir)
        )
        irrigated = irrigation_flags(schedule)
        if require_management:
            ids = np.intersect1d(ids, np.fromiter(plans, dtype=np.int64, count=len(plans)))
    elif require_management:
        raise FileNotFoundError(
            f"No fertilizer schedule under {export_dir / 'management'}. Run with "
            f"require_management=False for a potential-yield run."
        )

    if ids.size == 0:
        raise RuntimeError(
            f"No cell is runnable end to end: {with_weather.size} have weather, "
            f"{len(soil.ids)} have soil, {len(plans)} have a fertilizer plan. "
            f"The intersection is empty -- check that the export's stages ran "
            f"over the same cell set."
        )

    bundle = ExportBundle(
        config=config,
        ids=ids,
        soil=soil.select(ids),
        plans=plans,
        site=read_site(export_dir),
        irrigated=irrigated,
        co2=read_co2(export_dir, config.season.years),
    )
    logger.info(
        "Export %s: %d runnable cells (weather %d, soil %d, management %d)",
        export_dir.name, ids.size, with_weather.size, len(soil.ids), len(plans),
    )
    return bundle


def _find_composition(config: RunConfig, export_dir: Path) -> Optional[Path]:
    """``fertilizer_composition.xml``: the config's, else one in the export.

    The composition table is not part of what data4simplace writes — it lives
    next to the SIMPLACE reference the exporter was built from — so it is
    normally configured. An export that ships a copy is preferred over nothing,
    since a copy travelling with the data cannot drift out of step with it.
    """
    if config.paths.composition_file is not None:
        return Path(config.paths.composition_file)
    local = export_dir / "management" / "fertilizer_composition.xml"
    if local.is_file():
        logger.info("Using the composition file shipped with the export: %s", local)
        return local
    return None


def _find_schedule(export_dir: Path, crop: str) -> Optional[Path]:
    """The fertilizer schedule, preferring the wide layout.

    The wide file carries ``vType``, so its amounts can be split into N, P and
    K; the long one carries a single unattributed nutrient amount. When both
    exist the wide one is therefore the more informative read.
    """
    directory = export_dir / "management"
    if not directory.is_dir():
        return None
    candidates = [
        *sorted(directory.glob(f"fertilizer_*{crop}*.csv")),
        *sorted(directory.glob("fertilizer_*.csv")),
    ]
    wide = [p for p in candidates if not p.stem.endswith("_long")]
    chosen = next(iter(wide), None) or next(iter(candidates), None)
    if chosen is not None:
        logger.info("Fertilizer schedule: %s", chosen.name)
    return chosen
