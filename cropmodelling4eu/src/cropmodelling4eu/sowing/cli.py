"""``cm4eu sowing <step>``: the subcommands of the learned-sowing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["add_parser"]

PIPELINE = ("split", "build-cache", "fit-tsumem", "train", "predict")
STEPS = (*PIPELINE, "override-crops", "shift", "oracle", "all")


def _run(args: argparse.Namespace) -> int:
    from .config import load_sowing_config

    if args.sowing_command == "shift":
        from .oracle import shift_table

        shift_table(args.source, args.offset, args.target, tuple(args.bounds))
        return 0
    if args.sowing_command == "oracle":
        from .oracle import assemble_oracle

        runs = {int(o): Path(p) for o, p in (item.split("=", 1) for item in args.run)}
        assemble_oracle(runs, args.target, metric=args.metric, years=args.years)
        return 0

    overrides = {k: v for k, v in {"out_dir": args.out_dir, "workers": args.workers}.items() if v is not None}
    config = load_sowing_config(args.sowing_config, **overrides)
    step = args.sowing_command
    if step == "override-crops":
        from .crops import override_crops

        values = dict(item.split("=", 1) for item in args.set)
        if not values:
            from .train import emergence_params

            p = emergence_params(config)
            values = {"tsumem": p.tsumem, "tbasem": p.tbasem}
        override_crops(args.crops_dir, args.target, {k: float(v) for k, v in values.items()})
        return 0

    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / f"config.{step}.json").write_text(config.model_dump_json(indent=1))
    todo = PIPELINE if step == "all" else (step,)
    for name in todo:
        match name:
            case "split":
                from .targets import pep725_split

                pep725_split(config, overwrite=args.overwrite)
            case "build-cache":
                from .cache import build_cache

                ids = None
                if args.cells:
                    import pandas as pd

                    ids = pd.read_csv(args.cells)["SimplaceID"].to_numpy()
                build_cache(config, ids=ids)
            case "fit-tsumem":
                from .tsumem import fit_tsumem_pep725

                path = fit_tsumem_pep725(config)
                result = json.loads(path.read_text())
                print(f"tsumem {result['tsumem']:.0f} C d  (n {result['n']}, MAE {result['mae_days']:.1f} d)")
            case "train":
                from .train import train

                train(config)
            case "predict":
                from .predict import predict

                print(predict(config))
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``sowing`` on the main parser."""
    parser = sub.add_parser("sowing", help="Learned sowing dates: cache, tsumem, network, tables.")
    steps = parser.add_subparsers(dest="sowing_command", required=True)
    for name in STEPS:
        step = steps.add_parser(name)
        step.add_argument("--sowing-config", type=Path, help="YAML overriding SowingConfig defaults.")
        step.add_argument("--out-dir", type=Path, help="Override sowing out_dir.")
        step.add_argument("--workers", type=int)
        if name == "split":
            step.add_argument("--overwrite", action="store_true",
                              help="Redraw the PEP725 blocks (invalidates every later step).")
        if name in ("build-cache", "all"):
            step.add_argument("--cells", type=Path, help="CSV with SimplaceID; default every cell.")
        if name == "all":
            step.set_defaults(overwrite=False)
        if name == "override-crops":
            step.add_argument("--crops-dir", type=Path, required=True)
            step.add_argument("--target", type=Path, required=True)
            step.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                              help="Scalars to set; default tsumem/tbasem from tsumem.json.")
        if name == "shift":
            step.add_argument("--source", type=Path, required=True, help="Sowing table to shift.")
            step.add_argument("--offset", type=int, required=True, help="Days added to every sowing_doy.")
            step.add_argument("--target", type=Path, required=True)
            step.add_argument("--bounds", type=int, nargs=2, default=(200, 365), metavar=("MIN", "MAX"))
        if name == "oracle":
            step.add_argument("--run", action="append", required=True, metavar="OFFSET=PARQUET",
                              help="torchcrop_europe.parquet of one sowing offset; repeat per offset, "
                                   "written --run=-14=path so a negative offset is not read as an option.")
            step.add_argument("--target", type=Path, required=True, help="Arm directory to write.")
            step.add_argument("--metric", default="yield_t_ha")
            step.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"))
    parser.set_defaults(func=_run)
