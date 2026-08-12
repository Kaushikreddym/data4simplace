"""Score a German smoke run against CyBench yields and PEP725 phenology.

A terminal front end for :mod:`cropmodelling4eu.evaluation.germany`, which holds
the logic and its caveats — the same module
``evaluation/germany_smoke_evaluation.ipynb`` calls, so the notebook and the
command line cannot drift apart.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from cropmodelling4eu.evaluation import germany


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--simplace", type=Path, required=True)
    parser.add_argument("--torchcrop", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(args.cells)
    paths = {"simplace": args.simplace}
    if args.torchcrop is not None:
        paths["torchcrop"] = args.torchcrop
    runs = germany.load_runs(**paths)

    lo, hi = germany.common_window(runs)
    runs = {name: f[f["year"].between(lo, hi)] for name, f in runs.items()}
    logging.info("common window: %d-%d", lo, hi)

    print("\n" + "=" * 78)
    print(f"YIELD vs CyBench (NUTS-3, t/ha)   {lo}-{hi}, {len(cells)} cells")
    print("=" * 78)
    yields = germany.validate_yield(runs, cells)
    if not yields.empty:
        columns = ["model", "n", "regions", "years", "sim_mean", "obs_mean",
                   "bias", "rmse", "mae", "r"]
        print(yields[columns].round(2).to_string(index=False))
        yields.to_csv(args.out_dir / "yield_vs_cybench.csv", index=False)

    print("\n" + "=" * 78)
    print(f"PHENOLOGY vs PEP725 (day of year), stations within {germany.MATCH_KM:.0f} km")
    print("=" * 78)
    phenology = germany.validate_phenology(runs, germany.load_pep725(cells, (lo, hi)))
    if phenology.empty:
        print("no stage could be paired")
        return 0

    columns = ["stage", "model", "n", "sim_mean", "obs_mean", "bias", "rmse", "mae", "r"]
    print(phenology[columns].round(1).to_string(index=False))
    phenology.to_csv(args.out_dir / "phenology_vs_pep725.csv", index=False)
    print("\nCaveats:")
    for stage, caveat in phenology[["stage", "caveat"]].drop_duplicates().values:
        print(f"  {stage:24s} {caveat}")
    print(f"\nTables written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
