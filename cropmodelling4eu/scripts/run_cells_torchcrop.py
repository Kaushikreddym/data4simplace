"""Run torchcrop over an explicit cell list, for a like-for-like comparison.

`torchcrop.run.run_shard` deals cells round-robin out of the whole runnable
set, which is right for a production array job and wrong for a comparison: the
point here is to run *these* cells, the same ones SIMPLACE gets. This drives
the same batching and the same model, over a list read from a CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cropmodelling4eu.config import load_config
from cropmodelling4eu.export import resolve_export
from cropmodelling4eu.export.weather import load_season_block
from cropmodelling4eu.torchcrop.run import group_by_sowing, run_batch

logger = logging.getLogger("run_cells_torchcrop")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True, help="CSV with a SimplaceID column")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_config(args.config)
    settings = config.torchcrop
    if settings.torch_threads > 0:
        torch.set_num_threads(settings.torch_threads)

    wanted = pd.read_csv(args.cells)["SimplaceID"].to_numpy(np.int64)
    bundle = resolve_export(config, require_management=settings.iopt != 1)

    ids = np.intersect1d(wanted, bundle.ids)
    if ids.size != wanted.size:
        logger.warning(
            "%d of %d requested cells are not runnable and were dropped",
            wanted.size - ids.size, wanted.size,
        )
    logger.info("running %d cells, %d seasons", ids.size, len(config.season.years))

    device = torch.device(settings.device)
    results: list[pd.DataFrame] = []
    for doy, group in group_by_sowing(ids, bundle.site.sowing_doy(ids)).items():
        for start in range(0, group.size, settings.batch_size):
            batch = group[start : start + settings.batch_size]
            blocks = load_season_block(
                bundle.export_dir, batch, config.season.years, doy, config.grid,
                settings.io_workers, settings.spinup_months,
                settings.min_days_after_sowing,
            )
            for year, (weather, start_doy) in blocks.items():
                results.append(
                    run_batch(batch, weather, year, start_doy, doy, bundle, device)
                )

    if not results:
        raise SystemExit("no (cell, season) produced a result")
    frame = pd.concat(results, ignore_index=True)
    frame["model"] = "torchcrop"
    # t/ha alongside g/m2, so a yield reads the same way on both sides.
    frame["yield_t_ha"] = frame["yield_g_m2"] / 100.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
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
