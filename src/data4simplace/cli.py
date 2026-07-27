"""Command-line entry point for ``data4simplace`` / ``simplace-pipeline``.

Usage
-----
    data4simplace --config config.yaml
    data4simplace --config config.yaml --verbose
    data4simplace --config config.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data4simplace import __version__
from data4simplace.config import PipelineConfig, load_config
from data4simplace.pipeline import Pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data4simplace",
        description="Prepare MSWX climate, SoilGrids soil and NPK data as SIMPLACE inputs.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and report enabled stages without running them.",
    )
    parser.add_argument(
        "--tile-deg",
        type=float,
        default=None,
        help="Run tile by tile with ~this edge length in degrees (restartable). "
        "Required for large domains (e.g. continental) to bound memory and keep "
        "SoilGrids WCS requests within limits. Omit for a single in-memory run.",
    )
    parser.add_argument(
        "--tile-index",
        type=int,
        default=None,
        help="Process only this tile (0-based index into the tile list), then "
        "exit. This is the unit of work for an HPC array job: launch one task "
        "per tile, each with its own --tile-index. Requires --tile-deg. Combine "
        "afterwards with --combine-only.",
    )
    parser.add_argument(
        "--count-tiles",
        action="store_true",
        help="Print the number of tiles for the given --tile-deg and exit "
        "(use it to size an array job, e.g. --array=0-$((N-1))).",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip processing; just combine existing tile outputs into final "
        "files (mosaic soil shards into soil.csv). Weather files are already "
        "per-cell global. Pass --tile-deg to also check tile completeness.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"data4simplace {__version__}",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _report_plan(config: PipelineConfig) -> None:
    enabled = [name for name, on in config.flags.model_dump().items() if on]
    logging.getLogger(__name__).info(
        "Dry run — configuration valid. Enabled stages: %s",
        ", ".join(enabled) or "(none)",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Failed to load configuration: %s", exc)
        return 2

    if args.dry_run:
        _report_plan(config)
        return 0

    if args.count_tiles:
        if args.tile_deg is None:
            log.error("--count-tiles requires --tile-deg")
            return 2
        from data4simplace.tiling import count_tiles

        print(count_tiles(config, args.tile_deg))
        return 0

    if args.tile_index is not None and args.tile_deg is None:
        log.error("--tile-index requires --tile-deg")
        return 2

    try:
        if args.combine_only:
            from data4simplace.tiling import combine_tiles

            stats = combine_tiles(config, tile_deg=args.tile_deg)
            log.info(
                "Combined tiles: %d weather files, soil.csv %d rows (%d/%d tiles done) under %s",
                stats["weather_files"], stats["soil_rows"],
                stats["tiles_done"], stats["tiles_expected"], config.paths.output_dir,
            )
        elif args.tile_index is not None:
            from data4simplace.tiling import run_one_tile

            stats = run_one_tile(config, args.tile_index, tile_deg=args.tile_deg)
            log.info(
                "Done (tile %d). %d cells exported%s under %s",
                stats["tile_index"], stats["cells"],
                " (skipped: already done)" if stats["skipped"] else "",
                config.paths.output_dir,
            )
        elif args.tile_deg is not None:
            from data4simplace.tiling import run_tiled

            stats = run_tiled(config, tile_deg=args.tile_deg)
            log.info(
                "Done (tiled). %d/%d tiles, %d cells exported under %s",
                stats["done"], stats["tiles"], stats["cells"], config.paths.output_dir,
            )
        else:
            result = Pipeline(config).run()
            log.info(
                "Done. Wrote %d output file(s) under %s",
                len(result.written), config.paths.output_dir,
            )
    except Exception as exc:  # noqa: BLE001 - surface any stage failure to the CLI
        log.error("Pipeline failed: %s", exc, exc_info=args.verbose)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
