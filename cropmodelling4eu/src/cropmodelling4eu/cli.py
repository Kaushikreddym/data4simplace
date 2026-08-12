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
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the CLI
        logging.getLogger(__name__).error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
