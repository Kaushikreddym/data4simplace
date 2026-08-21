"""Run torchcrop over an explicit cell list, for a like-for-like comparison.

`torchcrop.run.run_shard` deals cells round-robin out of the whole runnable
set, which is right for a production array job and wrong for a comparison: the
point here is to run *these* cells, the same ones SIMPLACE gets. This drives
the same batching and the same model, over a list read from a CSV.

With ``--sowing-file`` it also sows on the dates SIMPLACE *simulated* rather
than the ones the export proposed — the handoff
``submit/submit_cropmodelling.sh`` uses for its smoke test. With
``--crop-file`` it runs the crop parameters written to a run's workspace
(``crop_<crop>.yaml``, see ``cropmodelling4eu.torchcrop.workspace``) instead of
torchcrop's bundled preset.

``--daily-out`` writes the day-by-day trajectory *alongside* ``--out`` from one
pass (``run_cells(..., mode="both")``), sharing the model's forward simulation
between the two rather than running the cells twice — the smoke test's own
counterpart of ``torchcrop.run.run_shard``'s ``--daily``. ``--daily`` alone
still writes the trajectory *to* ``--out`` instead of a summary, for a
daily-only file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.config import load_config
from cropmodelling4eu.torchcrop.run import DAILY_VARIABLES, run_cells
from cropmodelling4eu.torchcrop.workspace import load_crop_parameters

logger = logging.getLogger("run_cells_torchcrop")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True, help="CSV with a SimplaceID column")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+",
                        help="harvest years to run; default is the config's own "
                             "season.start_year..end_year")
    parser.add_argument("--sowing-file", type=Path,
                        help="CSV of (SimplaceID, year, sowing_doy) to sow on "
                             "instead of the export's calendar")
    parser.add_argument("--crop-file", type=Path,
                        help="crop parameter YAML from a run's workspace "
                             "(crop_<crop>.yaml); without it torchcrop's "
                             "bundled preset is used")
    parser.add_argument("--daily", action="store_true",
                        help="write per-day trajectories to --out instead of "
                             "summaries; ignored if --daily-out is also given")
    parser.add_argument("--daily-out", type=Path,
                        help="also write per-day trajectories here, from the "
                             "same simulation --out's summary comes from "
                             "(one pass, not two)")
    parser.add_argument("--daily-variables", nargs="+", choices=list(DAILY_VARIABLES),
                        help="which series --daily/--daily-out writes; default "
                             "is all of " + ", ".join(DAILY_VARIABLES))
    parser.add_argument("--iopt", type=int, choices=[1, 2, 3, 4],
                        help="override the config's torchcrop.iopt, e.g. to "
                             "sweep it over several calls against one config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config)
    if args.iopt is not None:
        config = config.model_copy(
            update={"torchcrop": config.torchcrop.model_copy(update={"iopt": args.iopt})}
        )
    ids = pd.read_csv(args.cells)["SimplaceID"].to_numpy(np.int64)
    years = args.years if args.years else config.season.years
    sowing = pd.read_csv(args.sowing_file) if args.sowing_file else None
    if sowing is not None:
        logger.info(
            "sowing on SIMPLACE's simulated dates from %s (%d cell-seasons)",
            args.sowing_file, len(sowing),
        )
    crop_params = load_crop_parameters(args.crop_file, config.season.crop)

    daily_kwargs = (
        {"variables": tuple(args.daily_variables)} if args.daily_variables else {}
    )
    mode = "both" if args.daily_out else "daily" if args.daily else "summary"
    result = run_cells(
        config, ids, years, mode=mode,
        sowing=sowing, crop_params=crop_params, **daily_kwargs,
    )
    summary, daily = (
        result if mode == "both" else
        (result, None) if mode == "summary" else
        (None, result)
    )

    # Written on every parquet, daily or summary: --iopt never appears in the
    # output otherwise, so a scenario run (potential/limited) is otherwise only
    # distinguishable by its file path.
    for frame in (summary, daily):
        if frame is not None:
            frame["iopt"] = config.torchcrop.iopt
    if summary is not None:
        summary["model"] = "torchcrop"
        # t/ha alongside g/m2, so a yield reads the same way on both sides.
        summary["yield_t_ha"] = summary["yield_g_m2"] / 100.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (summary if summary is not None else daily).to_parquet(
        args.out, index=False, compression="zstd"
    )
    if summary is not None:
        logger.info(
            "wrote %s: %d rows, %d cells, yield %.2f-%.2f t/ha (mean %.2f)",
            args.out, len(summary), summary["SimplaceID"].nunique(),
            summary["yield_t_ha"].min(), summary["yield_t_ha"].max(),
            summary["yield_t_ha"].mean(),
        )
    else:
        logger.info(
            "wrote %s: %d rows, %d cells, %d seasons",
            args.out, len(daily), daily["SimplaceID"].nunique(), daily["year"].nunique(),
        )

    if args.daily_out:
        args.daily_out.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(args.daily_out, index=False, compression="zstd")
        logger.info(
            "wrote %s: %d rows, %d cells, %d seasons",
            args.daily_out, len(daily), daily["SimplaceID"].nunique(),
            daily["year"].nunique(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
