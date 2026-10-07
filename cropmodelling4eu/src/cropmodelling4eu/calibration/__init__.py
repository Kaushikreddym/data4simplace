"""Calibrating torchcrop (LINTUL-5) against observed phenology, yield and LAI.

Backpropagation through the model's own day loop, over Europe, staged
phenology -> yield -> LAI. The design and every number behind it are in
``cropmodelling4eu/CALIBRATION.md``; this package is that document made
executable.

:func:`prepare` is the whole assembly in one call — regions, observations,
pool, weather cache, calibrator — because the pieces have to agree about the
cell set, and building them separately is how they stop agreeing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.calibration.config import (
    CLMS_HARVEST_OFFSET_DAYS,
    CalibrationConfig,
    DataConfig,
    LossConfig,
    OptimConfig,
    RegionConfig,
)
from cropmodelling4eu.calibration.dataset import (
    CellYearDataset,
    RegionBatch,
    StratifiedFractionSampler,
    WeatherCache,
    build_pool,
)
from cropmodelling4eu.calibration.losses import CanopyLoss, PhenologyLoss, YieldLoss
from cropmodelling4eu.calibration.observations import (
    ObservationSet,
    load_observations,
    load_pep725,
)
from cropmodelling4eu.calibration.problem import (
    CalibrationProblem,
    SimplexGroup,
    load_stage_specs,
)
from cropmodelling4eu.calibration.regions import CalibrationRegions, build_regions
from cropmodelling4eu.calibration.splits import make_split
from cropmodelling4eu.calibration.trainer import Calibrator, StageResult

logger = logging.getLogger(__name__)

__all__ = [
    "CLMS_HARVEST_OFFSET_DAYS",
    "CalibrationConfig",
    "CalibrationProblem",
    "CalibrationRegions",
    "Calibrator",
    "CanopyLoss",
    "CellYearDataset",
    "DataConfig",
    "LossConfig",
    "ObservationSet",
    "OptimConfig",
    "PhenologyLoss",
    "RegionBatch",
    "RegionConfig",
    "SimplexGroup",
    "StageResult",
    "StratifiedFractionSampler",
    "WeatherCache",
    "YieldLoss",
    "build_pool",
    "build_regions",
    "load_observations",
    "load_pep725",
    "load_stage_specs",
    "make_split",
    "prepare",
]


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything a calibration run needs, built against one cell set."""

    calibrator: Calibrator
    regions: CalibrationRegions
    observations: ObservationSet
    dataset: CellYearDataset

    def summarise(self) -> str:
        return "\n".join(
            [
                self.regions.summarise(),
                self.observations.summarise(),
                f"{len(self.dataset.pool)} pooled (unit, year, cell) rows, "
                f"{len(self.dataset.weather.index)} cached cell-years "
                f"at T = {self.dataset.weather.window_days} d",
            ]
        )


def prepare(
    config,
    settings: CalibrationConfig | None = None,
    *,
    countries: list[str] | None = None,
    crop_file: Path | None = None,
    sowing_file: Path | None = None,
    pep725_root: Path | None = None,
    with_yield: bool = True,
    with_fpar: bool = False,
    out_dir: Path | None = None,
    overwrite_cache: bool = False,
    workers: int = 1,
) -> Prepared:
    """Build regions, observations, the pool, the weather cache and the calibrator.

    Args:
        config: A :class:`cropmodelling4eu.config.RunConfig`.
        settings: Calibration settings; defaults are the plan's.
        countries: CyBench country codes. ``None`` uses the evaluation module's
            target set.
        crop_file: The workspace ``crop_<crop>.yaml`` the run starts from.
            ``None`` falls back to torchcrop's bundled preset, which is **not**
            the harmonised SUSTAg crop and will silently calibrate a different
            model — so it warns.
        sowing_file: ``sowing_from_simplace.csv`` from a **finished** SIMPLACE
            run — a static file, read once. ``None`` falls back to the export's
            site calendar, which sows a median 22 d later and gives every cell
            one fixed date for all seasons; see ``DataConfig.sowing_file``.
            Defaults to ``settings.data.sowing_file``.
        pep725_root: PEP725 export directory. ``None`` skips the station data,
            which costs the heading stage and the emergence anomaly target.
        with_yield: Load CyBench yields (stage 2).
        with_fpar: Load CyBench fAPAR (stage 3). Off by default because the
            dekadal series is large and the units question is unresolved.
        out_dir: Run outputs. ``None`` puts them under
            ``<output_dir>/<run_name>/calibration``.
        overwrite_cache: Rebuild the weather cache even if one covers the pool.
        workers: Processes computing region-batches concurrently. The ceiling
            is the region count (~29 here), which one node's 80 cores already
            covers; see :mod:`cropmodelling4eu.calibration.parallel`.

    Returns:
        The :class:`Prepared` bundle.
    """
    from cropmodelling4eu.evaluation import clms, config as eval_config, cybench, regions as eval_regions
    from cropmodelling4eu.export import resolve_export
    from cropmodelling4eu.torchcrop.workspace import load_crop_parameters

    settings = settings or CalibrationConfig()
    countries = list(countries or eval_config.TARGET_COUNTRIES)
    out_dir = Path(
        out_dir
        or settings.out_dir
        or Path(config.paths.output_dir) / config.run_name / "calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = resolve_export(config, require_management=config.torchcrop.iopt != 1)
    cells = bundle.cells()
    cells = eval_regions.assign_cells_to_regions(
        cells, countries, cache=out_dir / "cell_to_adm.parquet"
    ).rename(columns={"adm_country": "country"})

    calendar = cybench.load_calendar(countries)
    yields = cybench.load_yield(countries, years=(
        config.season.start_year, config.season.end_year
    ))
    regions = build_regions(
        cells, calendar, yields, settings.regions, cache=out_dir / "regions.parquet"
    )

    phenology = clms.load_phenology(countries, zone="adm", years=settings.phenology_years)
    pep = None
    if pep725_root is not None:
        from data4simplace.grid import TargetGrid

        grid = TargetGrid(
            resolution_deg=config.grid.resolution_deg,
            min_lon=config.grid.min_lon, max_lon=config.grid.max_lon,
            min_lat=config.grid.min_lat, max_lat=config.grid.max_lat,
        )
        pep = load_pep725(
            Path(pep725_root), grid,
            years=(config.season.start_year, config.season.end_year),
        )

    observations = load_observations(
        clms=phenology,
        yields=yields if with_yield else None,
        fpar=_load_fpar(countries) if with_fpar else None,
        pep=pep,
        years=(config.season.start_year, config.season.end_year),
    )

    sowing_file = sowing_file or settings.data.sowing_file
    sowing = pd.read_csv(sowing_file) if sowing_file else None
    if sowing is not None:
        logger.info("sowing table: %s (%d rows)", sowing_file, len(sowing))
    pool = build_pool(observations, regions, bundle, settings.data, sowing)
    cache = WeatherCache.build(
        pool, config, settings.data,
        settings.data.cache_dir or out_dir / "weather_cache",
        workers=config.torchcrop.io_workers,
        overwrite=overwrite_cache,
    )
    dataset = CellYearDataset(pool, cache)

    if crop_file is None:
        logger.warning(
            "no crop file given: calibrating torchcrop's bundled preset, not "
            "the harmonised SUSTAg WW crop the production run uses. Every "
            "number in CALIBRATION.md is against the harmonised crop"
        )
    crop_params = load_crop_parameters(crop_file, config.season.crop)

    calibrator = Calibrator(
        config, settings, bundle, dataset, observations, regions,
        crop_params, out_dir, crop_file, workers,
    )
    prepared = Prepared(calibrator, regions, observations, dataset)
    logger.info("\n%s", prepared.summarise())
    return prepared


def _load_fpar(countries: list[str]) -> pd.DataFrame:
    """CyBench ``fpar_wheat_<c>.csv`` for a country list."""
    from cropmodelling4eu.evaluation import config as eval_config
    from cropmodelling4eu.evaluation.cybench import _read_country_table

    frames = [
        frame
        for frame in (
            _read_country_table(
                country, "fpar_{crop}_{country}.csv", eval_config.CROP,
                eval_config.CYBENCH_CROP_DIR, ("adm_id", "date", "fpar"),
            )
            for country in countries
        )
        if frame is not None
    ]
    if not frames:
        raise FileNotFoundError(
            f"no CyBench fpar files under {eval_config.CYBENCH_CROP_DIR}"
        )
    return pd.concat(frames, ignore_index=True)
