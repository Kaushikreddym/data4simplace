"""Report which tiles of a tiled run are finished, and which array ids are not.

A tile is finished when :func:`data4simplace.tiling.run_one_tile` has written its
``<output_dir>/.tiles/tile_<r0>_<c0>.done`` marker. The array task id of a tile is
its position in :func:`data4simplace.tiling.list_windows`, so mapping markers back
to ids gives exactly the ``--array`` list a retry job needs.

Usage
-----
    python submit/tile_status.py --config config_run.yaml --tile-deg 5.0
    python submit/tile_status.py --config config_run.yaml --tile-deg 5.0 --missing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data4simplace.config import load_config
from data4simplace.tiling import list_windows


def _compress(ids: list[int]) -> str:
    """Compress sorted ids into SLURM ``--array`` notation (``0-3,7,10-12``)."""
    if not ids:
        return ""
    parts: list[str] = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tile-deg", type=float, required=True)
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Print only the unfinished array ids, in --array notation.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    windows = list_windows(config, args.tile_deg)
    markers = Path(config.paths.output_dir) / ".tiles"
    done = {p.stem for p in markers.glob("*.done")} if markers.is_dir() else set()

    missing = [i for i, w in enumerate(windows) if w.name not in done]

    if args.missing:
        print(_compress(missing))
        return 0

    n_done = len(windows) - len(missing)
    pct = 100.0 * n_done / len(windows) if windows else 0.0
    print(f"run          : {config.paths.output_dir}")
    print(f"tile size    : {args.tile_deg} deg")
    print(f"tiles        : {n_done}/{len(windows)} done ({pct:.1f}%)")
    if missing:
        head = ", ".join(str(i) for i in missing[:20])
        tail = ", ..." if len(missing) > 20 else ""
        print(f"unfinished   : {len(missing)} -> {head}{tail}")
        print(f"retry --array: {_compress(missing)}")
    else:
        print("unfinished   : none")

    out = Path(config.paths.output_dir)
    weather = len(list((out / "weather").glob("*.csv.gz")))
    shards = len(list((out / "soil" / "_shards").glob("tile_*.csv")))
    print(f"weather files: {weather}")
    print(f"soil shards  : {shards}")
    soil_csv = out / "soil" / "soil.csv"
    print(f"soil.csv     : {'present' if soil_csv.is_file() else 'not written yet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
