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

    try:
        if args.combine_only:
            from data4simplace.tiling import combine_tiles

            stats = combine_tiles(config, tile_deg=args.tile_deg)
            log.info(
                "Combined tiles: %d weather files, soil.csv %d rows (%d/%d tiles done) under %s",
                stats["weather_files"], stats["soil_rows"],
                stats["tiles_done"], stats["tiles_expected"], config.paths.output_dir,
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
