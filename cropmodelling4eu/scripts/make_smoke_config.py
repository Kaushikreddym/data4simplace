"""Derive the German smoke-test config from the main run config.

Both `submit/submit_simplace.sh --smoke` and `submit/submit_torchcrop.sh --smoke`
call this, and both write the *same* file: the smoke test only means anything if
the two models are given the same cells and the same seasons, so the derivation
has to be one implementation rather than one per submit script. Deriving it from
the main config rather than shipping a second config keeps the smoke test from
drifting away from the run it is a smoke test of.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the main run config")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--run-name", default="de_smoke")
    parser.add_argument("--iopt", type=int, choices=[1, 2, 3, 4],
                        help="override both models' vIOPT/iopt in the derived "
                             "config -- SIMPLACE has no CLI override for it "
                             "the way torchcrop has --iopt, so an IOPT sweep "
                             "needs its own config per value")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    config["run_name"] = args.run_name
    config["season"] |= {"start_year": args.start_year, "end_year": args.end_year}
    if "simplace" in config:
        # A season needs the year before it, and SIMPLACE's own window must
        # cover both. lines_per_task collapses the array to a single task.
        config["simplace"] |= {
            "start_date": f"{args.start_year - 2}-01-01",
            "end_date": f"{args.end_year + 1}-12-31",
            "lines_per_task": 10_000,
        }
    if args.iopt is not None:
        if "simplace" in config:
            config["simplace"]["iopt"] = args.iopt
        if "torchcrop" in config:
            config["torchcrop"]["iopt"] = args.iopt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print(f"Smoke config: {args.out}  ({args.start_year}-{args.end_year})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
