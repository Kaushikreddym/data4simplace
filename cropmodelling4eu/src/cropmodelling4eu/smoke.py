"""The German smoke test: one config, one cell list, both models, one score.

The smoke test only means anything if SIMPLACE and torchcrop are given the
**same cells over the same seasons**, so every piece of it has one
implementation here, driven by ``cm4eu smoke <step>`` from both submit paths
(``submit_simplace.sh --smoke``, ``submit_torchcrop.sh --smoke`` and the
chained ``submit_cropmodelling.sh --smoke``):

==============  ==========================================================
``config``      :func:`derive_config` — the main run config, narrowed
``cells``       :func:`select_cells` — a spread of German cropland cells
``simplace``    :func:`run_simplace_cells` — build, run, collect those cells
``torchcrop``   :func:`run_torchcrop_cells` — the same cells through torchcrop
``validate``    :func:`report` — both against CyBench yield and PEP725
==============  ==========================================================

The scoring logic and its caveats live in
:mod:`cropmodelling4eu.evaluation.germany`, which
``evaluation/germany_smoke_evaluation.ipynb`` calls too, so the notebook and
the command line cannot drift apart.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cropmodelling4eu.config import RunConfig

logger = logging.getLogger(__name__)

__all__ = [
    "derive_config",
    "report",
    "run_simplace_cells",
    "run_torchcrop_cells",
    "select_cells",
]


def derive_config(
    source: Path,
    out: Path,
    start_year: int,
    end_year: int,
    run_name: str = "de_smoke",
    iopt: int | None = None,
) -> Path:
    """Write the smoke-test config: the main config narrowed to a few seasons.

    Derived rather than shipped as a second file, so the smoke test cannot
    drift away from the run it is a smoke test of.

    Args:
        source: The main run config.
        out: Where the derived config is written.
        start_year: First harvest year.
        end_year: Last harvest year.
        run_name: The derived run's name.
        iopt: Override both models' ``iopt``. SIMPLACE has no CLI override for
            it the way torchcrop has, so an IOPT sweep needs a config per value.

    Returns:
        ``out``.
    """
    config = yaml.safe_load(Path(source).read_text())
    config["run_name"] = run_name
    config["season"] |= {"start_year": start_year, "end_year": end_year}
    if "simplace" in config:
        # A season needs the year before it, and SIMPLACE's own window must
        # cover both. lines_per_task collapses the array to a single task.
        config["simplace"] |= {
            "start_date": f"{start_year - 2}-01-01",
            "end_date": f"{end_year + 1}-12-31",
            "lines_per_task": 10_000,
        }
    if iopt is not None:
        for model in ("simplace", "torchcrop"):
            if model in config:
                config[model]["iopt"] = iopt
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def select_cells(config: RunConfig, n_cells: int = 40) -> pd.DataFrame:
    """A spread of runnable German cropland cells, each tagged with its NUTS-3.

    A smoke test wants few cells but not a clustered few: yield and phenology
    both vary strongly across Germany, so cells are binned on **both** axes and
    the one nearest each bin's centre is kept. Banding on latitude alone walks
    down the western border, because SimplaceID runs west-to-east within a row.
    Each cell is tagged with the CyBench NUTS-3 polygon containing it, so the
    yield comparison joins without a nearest-neighbour guess.

    Returns:
        ``SimplaceID, row, col, lon, lat, adm_id``.

    Raises:
        ValueError: If no runnable cell falls inside Germany.
    """
    import geopandas as gpd

    from cropmodelling4eu.evaluation.config import CYBENCH_ROOT
    from cropmodelling4eu.export import resolve_export

    cells = resolve_export(config, require_management=True).cells()
    regions = gpd.read_file(CYBENCH_ROOT / "polygons" / "DE" / "DE.shp").to_crs("EPSG:4326")
    id_column = next(
        c for c in ("adm_id", "ADM_ID", "NUTS_ID", "nuts_id") if c in regions.columns
    )
    points = gpd.GeoDataFrame(
        cells, geometry=gpd.points_from_xy(cells["lon"], cells["lat"]), crs="EPSG:4326"
    )
    joined = (
        gpd.sjoin(points, regions[[id_column, "geometry"]], how="inner", predicate="within")
        .rename(columns={id_column: "adm_id"})
        .drop(columns=["index_right"])
    )
    logger.info(
        "%d of %d runnable cells fall inside a German NUTS-3 polygon (%d regions)",
        len(joined), len(cells), joined["adm_id"].nunique(),
    )
    if joined.empty:
        raise ValueError("no runnable cell falls inside Germany")

    n_lat = max(1, int(round(np.sqrt(n_cells * 1.3))))
    n_lon = max(1, int(np.ceil(n_cells / n_lat)))
    joined["_lat_bin"] = pd.qcut(joined["lat"], q=n_lat, duplicates="drop")
    joined["_lon_bin"] = pd.qcut(joined["lon"], q=n_lon, duplicates="drop")

    def central(group: pd.DataFrame) -> pd.DataFrame:
        distance = (group["lon"] - group["lon"].median()) ** 2 + (
            group["lat"] - group["lat"].median()
        ) ** 2
        return group.loc[[distance.idxmin()]]

    picked = (
        joined.groupby(["_lat_bin", "_lon_bin"], observed=True, sort=False)
        .apply(central, include_groups=False)
        .reset_index(drop=True)
        .sort_values("SimplaceID")
        .head(n_cells)
        .reset_index(drop=True)
    )
    return picked[["SimplaceID", "row", "col", "lon", "lat", "adm_id"]]


def run_simplace_cells(config: RunConfig, ids: np.ndarray, out: Path, debug: bool = False) -> pd.DataFrame:
    """Build, run and collect SIMPLACE over an explicit cell list.

    Raises:
        RuntimeError: If any line range exits non-zero.
    """
    from cropmodelling4eu.export import resolve_export
    from cropmodelling4eu.simplace import build_workspace, collect_run, line_ranges, run_lines

    bundle = resolve_export(config, require_management=config.simplace.iopt != 1)
    workspace = build_workspace(config, bundle, cells=ids)
    for first, last in line_ranges(workspace.n_lines, config.simplace.lines_per_task):
        if run_lines(workspace, config, first, last, debug=debug) != 0:
            raise RuntimeError(f"SIMPLACE failed on lines {first}-{last}")

    frame = pd.read_parquet(collect_run(workspace.out_dir, Path(out).parent, config.grid))
    frame["yield_t_ha"] = frame["yield_g_m2"] / 100.0
    return frame


def run_torchcrop_cells(
    config: RunConfig,
    ids: np.ndarray,
    years: list[int] | None = None,
    sowing: pd.DataFrame | None = None,
    crop_file: Path | None = None,
    mode: str = "summary",
    variables: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """torchcrop over an explicit cell list, the same cells SIMPLACE gets.

    ``run_shard`` deals cells round-robin out of the whole runnable set, which
    is right for a production array and wrong for a comparison; this drives the
    same batching and model over a given list.

    Args:
        sowing: ``(SimplaceID, year, sowing_doy)`` to sow on — SIMPLACE's
            simulated dates in the chained smoke test — instead of the export's.
        crop_file: A workspace ``crop_<crop>.yaml``; ``None`` is torchcrop's
            bundled preset.
        mode: ``summary``, ``daily`` or ``both`` (one simulation, two outputs).
        variables: Daily series to keep; ``None`` is all of them.

    Returns:
        ``(summary, daily)``, either ``None`` when ``mode`` does not produce it.
        ``iopt`` is written on both, since it is otherwise visible only in the
        file path.
    """
    from cropmodelling4eu.torchcrop.run import run_cells
    from cropmodelling4eu.torchcrop.workspace import load_crop_parameters

    if sowing is not None:
        logger.info("sowing on supplied dates (%d cell-seasons)", len(sowing))
    result = run_cells(
        config, ids, years or config.season.years, mode=mode,
        sowing=sowing, crop_params=load_crop_parameters(crop_file, config.season.crop),
        **({"variables": variables} if variables else {}),
    )
    summary, daily = {
        "both": lambda: result,
        "summary": lambda: (result, None),
        "daily": lambda: (None, result),
    }[mode]()
    for frame in (summary, daily):
        if frame is not None:
            frame["iopt"] = config.torchcrop.iopt
    if summary is not None:
        summary["model"] = "torchcrop"
        summary["yield_t_ha"] = summary["yield_g_m2"] / 100.0
    return summary, daily


def report(cells: pd.DataFrame, paths: dict[str, Path], out_dir: Path) -> None:
    """Score the runs against CyBench yield and PEP725 phenology, print and save.

    Args:
        cells: The smoke cell list (:func:`select_cells`).
        paths: ``{"simplace": parquet, "torchcrop": parquet}``; torchcrop optional.
        out_dir: Where ``yield_vs_cybench.csv`` / ``phenology_vs_pep725.csv`` go.
    """
    from cropmodelling4eu.evaluation import germany

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = germany.load_runs(**paths)
    lo, hi = germany.common_window(runs)
    runs = {name: frame[frame["year"].between(lo, hi)] for name, frame in runs.items()}
    logger.info("common window: %d-%d", lo, hi)
    rule = "=" * 78

    print(f"\n{rule}\nYIELD vs CyBench (NUTS-3, t/ha)   {lo}-{hi}, {len(cells)} cells\n{rule}")
    yields = germany.validate_yield(runs, cells)
    if not yields.empty:
        columns = ["model", "n", "regions", "years", "sim_mean", "obs_mean",
                   "bias", "rmse", "mae", "r"]
        print(yields[columns].round(2).to_string(index=False))
        yields.to_csv(out_dir / "yield_vs_cybench.csv", index=False)

    print(f"\n{rule}\nPHENOLOGY vs PEP725 (day of year), stations within "
          f"{germany.MATCH_KM:.0f} km\n{rule}")
    phenology = germany.validate_phenology(runs, germany.load_pep725(cells, (lo, hi)))
    if phenology.empty:
        print("no stage could be paired")
        return
    columns = ["stage", "model", "n", "sim_mean", "obs_mean", "bias", "rmse", "mae", "r"]
    print(phenology[columns].round(1).to_string(index=False))
    phenology.to_csv(out_dir / "phenology_vs_pep725.csv", index=False)
    print("\nCaveats:")
    for stage, caveat in phenology[["stage", "caveat"]].drop_duplicates().values:
        print(f"  {stage:24s} {caveat}")
    print(f"\nTables written to {out_dir}")
