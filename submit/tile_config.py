"""Write a derived pipeline config for one job of a tiled run.

Two overrides are applied to a copy of the base ``config.yaml``:

``paths.output_dir``
    Points every job of a run at the same run directory, so tile markers, soil
    shards and weather files land together without editing the tracked config.

``flags.*`` (only with ``--flags-only``)
    Turns every execution flag off except the named ones, so a stage the tiled
    run does not cover (NPK / management) can be run once over the whole grid.

``soil.wcs_cache_dir`` (only with ``--tile-index``)
    Gives the array task its own SoilGrids WCS cache. This is **required** for a
    parallel tiled run: :meth:`SoilGridsHandler.fetch_wcs` caches a download as
    ``<cache_dir>/<coverage_id>.tif`` — the key carries no bounding box, while
    the request is subset to the *tile's* bbox. Tiles sharing one cache would
    therefore read back whichever tile downloaded the coverage first, silently
    exporting one tile's soil for the whole domain. A per-tile directory keeps
    the cache correct and still lets a re-run of the *same* tile reuse it.

The YAML is edited as a plain mapping rather than through ``PipelineConfig`` so
the emitted file stays a faithful copy of the original, with no pydantic
defaults or normalisations baked in.

Usage
-----
    python submit/tile_config.py --config config.yaml \\
        --output-dir /beegfs/.../europe --out /beegfs/.../config_run.yaml
    python submit/tile_config.py --config config_run.yaml --tile-index 7 \\
        --cache-root /beegfs/.../wcs_cache --out /beegfs/.../config_tile_7.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


def build(
    base: Path,
    output_dir: str | None,
    tile_index: int | None,
    cache_root: str | None,
    flags_only: str | None = None,
) -> dict[str, Any]:
    """Return the base config as a dict with the run/tile overrides applied."""
    with base.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{base} does not contain a YAML mapping")

    if output_dir is not None:
        cfg.setdefault("paths", {})["output_dir"] = str(output_dir)

    if flags_only is not None:
        keep = {name.strip() for name in flags_only.split(",") if name.strip()}
        flags = cfg.get("flags") or {}
        unknown = keep - set(flags)
        if unknown:
            raise ValueError(f"unknown flag(s): {', '.join(sorted(unknown))}")
        cfg["flags"] = {name: (name in keep) for name in flags}

    if tile_index is not None:
        if cache_root is None:
            raise ValueError("--tile-index requires --cache-root")
        cache_dir = Path(cache_root) / f"tile_{tile_index:04d}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.setdefault("soil", {})["wcs_cache_dir"] = str(cache_dir)

    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="Base YAML config.")
    parser.add_argument("--out", type=Path, required=True, help="Path to write.")
    parser.add_argument("--output-dir", default=None, help="Override paths.output_dir.")
    parser.add_argument(
        "--tile-index",
        type=int,
        default=None,
        help="Array task id; gives the task a private soil.wcs_cache_dir.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Parent directory for the per-tile WCS caches.",
    )
    parser.add_argument(
        "--flags-only",
        default=None,
        help="Comma-separated flag names to keep true; all others are set false.",
    )
    args = parser.parse_args(argv)

    try:
        cfg = build(
            args.config,
            args.output_dir,
            args.tile_index,
            args.cache_root,
            args.flags_only,
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
