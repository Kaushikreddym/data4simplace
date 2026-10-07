"""Command-line entry point for ``cropmodelling4eu`` / ``cm4eu``.

Usage
-----
    cm4eu inspect --export <dir>            # what the export holds, and gaps
    cm4eu inspect --config run.yaml

    cm4eu simplace build --config run.yaml --out-dir <dir>
    cm4eu simplace collect --config run.yaml --out-dir <dir>

``build`` produces a directory that is self-contained: the rewritten XML, the
project CSV, symlinks to every input, and the run's own ``submit.sh`` /
``run_task.sh``. From then on the run is driven from there and needs neither
this CLI nor a config — ``cm4eu simplace run`` remains for a one-off range in
the foreground.

    cm4eu export-regions --calibration-dir <calibrate out-dir> \\
        --template <template>/data/crop/LINTUL5_crop.xml --simplace-crop WW \\
        --out-dir <dir>

Turns a finished ``cm4eu calibrate`` run into what a *per-region* SIMPLACE
build needs: one ``cells_region_<NN>.csv`` (for ``simplace build
--cells-file``) and one ``crop_region_<NN>.xml`` per calibrated region, the
template with that region's torchcrop fit written back in. torchcrop's own
side of a per-region run takes the calibration directory directly (``cm4eu
calibrate``'s ``<stage>/crops/``) via ``--region-crop-dir`` /
``--regions-file`` on ``cropmodelling4eu.torchcrop.run``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cropmodelling4eu import __version__
from cropmodelling4eu.config import RunConfig, load_config


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_config(args: argparse.Namespace) -> RunConfig:
    """The run config, from a file or built around a bare export directory."""
    if args.config is not None:
        config = load_config(args.config)
        if args.composition is not None:
            config = config.model_copy(
                update={
                    "paths": config.paths.model_copy(
                        update={"composition_file": args.composition}
                    )
                }
            )
        return config
    if args.export is None:
        raise SystemExit("Pass either --config or --export")
    paths: dict = {"export_dir": str(args.export)}
    if args.composition is not None:
        paths["composition_file"] = str(args.composition)
    return RunConfig.from_export(args.export, paths=paths)


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Report what an export holds and which inputs are assumed rather than read."""
    from cropmodelling4eu.export import resolve_export

    config = _resolve_config(args)
    bundle = resolve_export(config, require_management=not args.no_management)
    soil = bundle.soil

    print(f"\nExport   : {bundle.export_dir}")
    print(f"Grid     : {config.grid.n_lon} x {config.grid.n_lat} cells at "
          f"{config.grid.resolution_deg} deg "
          f"({config.grid.min_lon}..{config.grid.max_lon} E, "
          f"{config.grid.min_lat}..{config.grid.max_lat} N)")
    print(f"Runnable : {bundle.ids.size} cells "
          f"(SimplaceID {bundle.ids.min()}..{bundle.ids.max()})")
    print(f"Seasons  : {config.season.start_year}-{config.season.end_year} "
          f"({len(config.season.years)} harvest years)")
    print(f"Soil     : {soil.n_layers} layers to {soil.bottoms_m[-1]:.2f} m, "
          f"{len(soil.values)} properties ({', '.join(sorted(soil.values))})")
    print(f"Fertiliser: {len(bundle.plans)} cells with a plan")
    print(f"{bundle.site.summarise()}")
    print(f"CO2      : {bundle.co2.iloc[0]:.1f} ppm in {bundle.co2.index[0]} -> "
          f"{bundle.co2.iloc[-1]:.1f} ppm in {bundle.co2.index[-1]}")
    if bundle.site.is_fallback:
        print(
            "\nWARNING: this export predates the site stage, so the sowing date "
            "and altitude below are assumptions, not data. Re-export with "
            "flags.run_site_processing to fix the largest single error in a run."
        )
    return 0


def _read_cells(path: Path | None) -> "np.ndarray | None":
    """The ``SimplaceID`` column of a cell list, or ``None``."""
    if path is None:
        return None
    import pandas as pd

    frame = pd.read_csv(path)
    if "SimplaceID" not in frame:
        raise SystemExit(f"{path} has no SimplaceID column (it has {list(frame.columns)})")
    return frame["SimplaceID"].to_numpy()


def _cmd_simplace(args: argparse.Namespace) -> int:
    """Build, run or collect a SIMPLACE run."""
    from cropmodelling4eu.export import resolve_export
    from cropmodelling4eu.simplace import (
        build_workspace,
        collect_run,
        line_ranges,
        run_lines,
        write_run_scripts,
    )

    config = _resolve_config(args)
    if args.lines_per_task:
        config = config.model_copy(
            update={
                "simplace": config.simplace.model_copy(
                    update={"lines_per_task": args.lines_per_task}
                )
            }
        )

    root = Path(args.out_dir) if args.out_dir else config.run_dir / "simplace"

    if args.step == "collect":
        parquet = collect_run(root / "out", root, config.grid)
        print(f"\nCollected -> {parquet}")
        return 0

    bundle = resolve_export(config, require_management=config.simplace.iopt != 1)
    workspace = build_workspace(
        config,
        bundle,
        max_cells=args.cells,
        validate=not args.no_validate,
        strict=args.strict,
        root=root,
        weather_cache=args.weather_cache,
        cells=_read_cells(args.cells_file),
    )
    scripts = write_run_scripts(workspace, config)

    ranges = line_ranges(workspace.n_lines, config.simplace.lines_per_task)
    print(f"\nBuilt in  : {workspace.root}")
    print(f"Workspace : {workspace.work_dir}")
    print(f"Project   : {workspace.project.name} ({workspace.n_lines} lines)")
    print(f"Tasks     : {len(ranges)} of <= {config.simplace.lines_per_task} lines")
    print(f"Outputs   : {workspace.out_dir}")
    if workspace.weather_cache is not None:
        print(f"Weather   : converted ({config.simplace.weather_contract}), "
              f"cached in {workspace.weather_cache}, linked into the workspace")
    else:
        print(f"Weather   : symlinked from {workspace.export_dir}/weather")

    if args.step == "build":
        first, last = ranges[0] if ranges else (1, 1)
        print(
            f"\nEverything below is in {workspace.root} and needs neither this "
            f"package nor a config:\n"
            f"  {scripts['submit']}                  # the whole run, as a SLURM array\n"
            f"  {scripts['submit']} --status         # what is finished\n"
            f"  {scripts['submit']} --retry          # only the unfinished tasks\n"
            f"  {scripts['task']} {first}-{last}     # one range, right here\n"
        )
        return 0

    # step == "run"
    if args.lines:
        first, last = (int(part) for part in args.lines.split("-", 1))
    elif args.task is not None:
        first, last = ranges[args.task]
    else:
        first, last = 1, workspace.n_lines
    return run_lines(
        workspace, config, first, last, debug=args.debug, dry_run=args.dry_run
    )


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit torchcrop's crop parameters against the observations, by stage."""
    from cropmodelling4eu.calibration import CalibrationConfig, prepare

    config = _resolve_config(args)
    settings = CalibrationConfig()
    if args.fraction is not None:
        settings = settings.model_copy(
            update={"data": settings.data.model_copy(update={"fraction": args.fraction})}
        )
    if args.epochs is not None:
        settings = settings.model_copy(
            update={"optim": settings.optim.model_copy(update={"max_epochs": args.epochs})}
        )
    if args.batch_size is not None:
        settings = settings.model_copy(
            update={"data": settings.data.model_copy(update={"batch_size": args.batch_size})}
        )
    if args.threads is not None:
        config = config.model_copy(
            update={"torchcrop": config.torchcrop.model_copy(
                update={"torch_threads": args.threads}
            )}
        )
    if args.stage != "all":
        settings = settings.model_copy(
            update={"stage_order": (args.stage,), "joint_finetune": False}
        )

    prepared = prepare(
        config,
        settings,
        crop_file=args.crop_file,
        sowing_file=args.sowing_file,
        pep725_root=args.pep725,
        with_yield="yield" in settings.stage_order,
        with_fpar="lai" in settings.stage_order,
        out_dir=args.out_dir,
        overwrite_cache=args.rebuild_cache,
        workers=args.workers,
    )
    if args.prepare_only:
        print(prepared.summarise())
        return 0

    for result in prepared.calibrator.run(free_blocked=args.free_blocked):
        print(
            f"\n{result.stage}: {result.epochs} epochs, best validation "
            f"{result.best_val:.4f}"
        )
        for key, value in sorted(result.test.items()):
            print(f"  test {key:<32} {value:.4f}")
        if result.blocked:
            print("  not freed:")
            for name, reason in sorted(result.blocked.items()):
                print(f"    {name:<34} {reason}")
    return 0


def _cmd_export_regions(args: argparse.Namespace) -> int:
    """Per-region SIMPLACE crop.xml + cell lists, from a finished calibration run."""
    import pandas as pd

    from cropmodelling4eu.torchcrop.params import write_simplace_crop_xml

    calibration_dir = Path(args.calibration_dir)
    regions = pd.read_parquet(calibration_dir / "regions.parquet")
    crops_dir = calibration_dir / args.stage / "crops"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[int] = []
    fallback: list[int] = []
    for region_id, group in regions.groupby("region_id", sort=True):
        region_id = int(region_id)
        name = group["region"].iloc[0]
        group[["SimplaceID"]].to_csv(
            out_dir / f"cells_region_{region_id:02d}.csv", index=False
        )
        crop_yaml = crops_dir / f"crop_region_{region_id:02d}.yaml"
        if not crop_yaml.is_file():
            fallback.append(region_id)
            print(f"region {region_id:02d} {name:<10} not calibrated (no {crop_yaml.name}) "
                  f"-- its cells run on --template unchanged")
            continue
        write_simplace_crop_xml(
            out_dir / f"crop_region_{region_id:02d}.xml",
            crop_yaml, args.template, args.simplace_crop,
        )
        written.append(region_id)

    print(f"\n{len(written)} region crop.xml file(s) written, {len(fallback)} uncalibrated "
          f"region(s) fall back to {args.template}, {len(written) + len(fallback)} cell list(s) "
          f"-- all under {out_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cm4eu",
        description="Run and evaluate SIMPLACE and torchcrop over a data4simplace export.",
    )
    parser.add_argument("--version", action="version", version=f"cropmodelling4eu {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")

    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "inspect", help="Report what an export holds and which inputs are assumed."
    )
    inspect.add_argument("--config", type=Path, help="Run configuration YAML.")
    inspect.add_argument(
        "--export", type=Path, help="Export directory (grid read from its frozen config)."
    )
    inspect.add_argument(
        "--composition",
        type=Path,
        help="fertilizer_composition.xml, needed to read a wide schedule's "
        "product amounts as nutrients.",
    )
    inspect.add_argument(
        "--no-management",
        action="store_true",
        help="Do not require a fertilizer plan (potential-yield runs).",
    )
    inspect.set_defaults(func=_cmd_inspect)

    simplace = sub.add_parser(
        "simplace", help="Build, run or collect a SIMPLACE (Singularity) run."
    )
    simplace.add_argument(
        "step",
        choices=["build", "run", "collect"],
        help="build: materialise the workspace, project CSV and run scripts, "
        "and validate the solution against them. run: invoke the container "
        "over a line range. collect: gather the per-cell outputs into one "
        "Parquet.",
    )
    simplace.add_argument("--config", type=Path, help="Run configuration YAML.")
    simplace.add_argument("--export", type=Path, help="Export directory.")
    simplace.add_argument("--composition", type=Path)
    simplace.add_argument(
        "--out-dir", type=Path,
        help="Where the run directory is created (default: "
        "<paths.output_dir>/<run_name>/simplace). Everything the run needs "
        "lands here: the XML, the project CSV, the linked inputs and the "
        "submit/retry scripts.",
    )
    simplace.add_argument(
        "--weather-cache", type=Path,
        help="Where converted weather is kept when simplace.weather_contract "
        "is set (default: <paths.output_dir>/weather_cache/<contract>). "
        "Shared between runs, so a rebuild converts nothing.",
    )
    simplace.add_argument(
        "--cells", type=int, default=0,
        help="Cap the project file to this many cells (an even subsample, for "
        "a smoke test).",
    )
    simplace.add_argument(
        "--cells-file", type=Path,
        help="CSV with a SimplaceID column: run exactly these cells. What "
        "makes a model comparison like-for-like, since both models then get "
        "the same set rather than two samples of one.",
    )
    simplace.add_argument(
        "--lines", help="Inclusive 1-based line range, e.g. 1-2000.",
    )
    simplace.add_argument(
        "--task", type=int,
        help="Run the Nth line range instead (0-based) — what a SLURM array "
        "task passes.",
    )
    simplace.add_argument(
        "--lines-per-task", type=int,
        help="Override simplace.lines_per_task, so the SLURM driver and the "
        "config cannot disagree about how the work is split.",
    )
    simplace.add_argument("--no-validate", action="store_true")
    simplace.add_argument(
        "--strict", action="store_true",
        help="Fail the build on a declared-but-absent column, not only on "
        "a missing file. SIMPLACE tolerates some, so this is off by default.",
    )
    simplace.add_argument("--debug", action="store_true", help="Keep SIMPLACE's full log level.")
    simplace.add_argument("--dry-run", action="store_true", help="Print the command, run nothing.")
    simplace.set_defaults(func=_cmd_simplace, no_management=False)

    calibrate = sub.add_parser(
        "calibrate",
        help="Fit torchcrop's crop parameters against CLMS phenology, CyBench "
        "yield and CyBench fAPAR, by stage.",
    )
    calibrate.add_argument("--config", type=Path, help="Run configuration YAML.")
    calibrate.add_argument("--export", type=Path, help="Export directory.")
    calibrate.add_argument("--composition", type=Path)
    calibrate.add_argument(
        "--stage", default="all", choices=["all", "phenology", "yield", "lai"],
        help="Run one stage, or every stage in `stage_order` (currently "
        "phenology and yield; lai and the joint fine-tune are switched off). Stages "
        "accumulate terms rather than replacing the objective, so running one "
        "alone is a diagnostic, not the plan.",
    )
    calibrate.add_argument(
        "--crop-file", type=Path,
        help="The workspace crop_<crop>.yaml the run starts from — normally "
        "<TC_OUT_DIR>/workspace/crop_wheat.yaml. Without it the bundled "
        "torchcrop preset is calibrated, which is a different crop from the "
        "one the production run uses.",
    )
    calibrate.add_argument(
        "--sowing-file", type=Path,
        help="sowing_from_simplace.csv from a finished SIMPLACE run — the "
        "sowing convention the production run and every CALIBRATION.md number "
        "are on. A static file; nothing here re-runs SIMPLACE. Without it the "
        "export's site calendar is used, which sows a median 22 d later and "
        "gives every cell one fixed date for all seasons.",
    )
    calibrate.add_argument(
        "--pep725", type=Path,
        help="PEP725 export directory. Without it there is no heading stage "
        "and no ground-truth emergence anomaly, so the identifiability rule "
        "binds at two stages everywhere.",
    )
    calibrate.add_argument("--out-dir", type=Path)
    calibrate.add_argument(
        "--fraction", type=float,
        help="Override the per-epoch stratified draw (default 0.20).",
    )
    calibrate.add_argument("--epochs", type=int, help="Override optim.max_epochs.")
    calibrate.add_argument(
        "--batch-size", type=int,
        help="Cell-years per forward pass (default 192). Measured: 768 costs "
        "1.33x the wall time of 192 for 4x the cells, at 1.3 GB — the day loop "
        "is per-day, not per-cell. The floor on batch *count* is the region "
        "count, since a batch carries one region's parameters.",
    )
    calibrate.add_argument(
        "--workers", type=int, default=1,
        help="Processes computing region-batches concurrently (default 1, "
        "exactly sequential). A batch carries one region and the penalties are "
        "scoped to it, so batches of different regions are disjoint and run "
        "together for the same result. The ceiling is the REGION COUNT (~29), "
        "which one node's 80 cores already covers — more nodes have nothing to "
        "do.",
    )
    calibrate.add_argument(
        "--threads", type=int,
        help="torch intra-op threads (default from the config). Measured to "
        "make no difference at all — 1, 4, 10 and 20 threads are within noise "
        "of each other, because the day loop is a Python loop over tiny "
        "[B]-shaped tensors. Ask for cores to run more jobs, not a faster one.",
    )
    calibrate.add_argument(
        "--free-blocked", action="store_true",
        help="Free the parameters the spec files mark blocked. Every current "
        "blocker is a data problem: the N group is unidentifiable at nni 0.985 "
        "and the heat group is unmapped between the two models.",
    )
    calibrate.add_argument(
        "--rebuild-cache", action="store_true",
        help="Re-extract the pooled seasons even if a cache covers them.",
    )
    calibrate.add_argument(
        "--prepare-only", action="store_true",
        help="Build the regions, observations and weather cache, report them "
        "and stop. What to run first on a new export.",
    )
    calibrate.set_defaults(func=_cmd_calibrate, no_management=False)

    export_regions = sub.add_parser(
        "export-regions",
        help="Per-region SIMPLACE crop.xml + cell lists from a finished "
        "'calibrate' run, for a per-region SIMPLACE production build.",
    )
    export_regions.add_argument(
        "--calibration-dir", type=Path, required=True,
        help="A 'cm4eu calibrate --out-dir' directory: holds regions.parquet "
        "and <stage>/crops/crop_region_<NN>.yaml.",
    )
    export_regions.add_argument(
        "--stage", default="yield", choices=["phenology", "yield"],
        help="Which stage's calibrated crop files to use (default: yield, "
        "the final stage -- phenology seeds it and yield fine-tunes on top).",
    )
    export_regions.add_argument(
        "--template", type=Path, required=True,
        help="The SIMPLACE crop.xml whose layout and every non-calibrated "
        "parameter are kept, e.g. the production template's own "
        "data/crop/LINTUL5_crop.xml.",
    )
    export_regions.add_argument(
        "--simplace-crop", required=True,
        help="The <crop> block in --template to overwrite, e.g. WW.",
    )
    export_regions.add_argument("--out-dir", type=Path, required=True)
    export_regions.set_defaults(func=_cmd_export_regions)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the CLI
        # Always with the traceback: a batch job that dies six hours in gets one
        # shot at saying where, and `--verbose` is not set on a submitted run.
        logging.getLogger(__name__).error("%s", exc, exc_info=True)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
