"""Build, run and collect a SIMPLACE run over an explicit cell list.

The counterpart of ``run_cells_torchcrop.py``: same cells, same seasons, so
the two models' results differ only in the model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import load_config
from cropmodelling4eu.export import resolve_export
from cropmodelling4eu.simplace import (
    build_workspace,
    collect_run,
    line_ranges,
    run_lines,
)

logger = logging.getLogger("run_cells_simplace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config)
    wanted = pd.read_csv(args.cells)["SimplaceID"].to_numpy(np.int64)

    bundle = resolve_export(config, require_management=config.simplace.iopt != 1)
    workspace = build_workspace(config, bundle, cells=wanted)

    for first, last in line_ranges(workspace.n_lines, config.simplace.lines_per_task):
        code = run_lines(workspace, config, first, last, debug=args.debug)
        if code != 0:
            raise SystemExit(f"SIMPLACE failed on lines {first}-{last}")

    parquet = collect_run(workspace.out_dir, args.out.parent, config.grid)
    frame = pd.read_parquet(parquet)
    frame["yield_t_ha"] = frame["yield_g_m2"] / 100.0
    frame.to_parquet(args.out, index=False, compression="zstd")

    logger.info(
        "wrote %s: %d rows, %d cells, yield %.2f-%.2f t/ha (mean %.2f)",
        args.out, len(frame), frame["SimplaceID"].nunique(),
        frame["yield_t_ha"].min(), frame["yield_t_ha"].max(),
        frame["yield_t_ha"].mean(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
