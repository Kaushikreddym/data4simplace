"""Yield-maximising sowing: an upper bound, not a sowing method.

Arm D of the sowing comparison runs torchcrop on the network's sowing table
shifted by a fixed grid of offsets, then keeps, per (cell, season), the run with
the highest simulated yield. The choice reads the whole season's weather and
the model's own yield, so it is neither causal nor observable: it bounds how
much yield sowing alone can move in torchcrop, and shows where that optimum
sits against PEP725 and the network.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["offset_label", "shift_table", "assemble_oracle"]

log = logging.getLogger(__name__)

KEY = ["SimplaceID", "year"]


def offset_label(offset: int) -> str:
    """Directory-safe offset name: ``-14`` -> ``m14``, ``14`` -> ``p14``, ``0`` -> ``p0``."""
    return f"{'m' if offset < 0 else 'p'}{abs(offset)}"


def shift_table(source: Path, offset: int, target: Path, bounds: tuple[int, int] = (200, 365)) -> Path:
    """Write ``source`` with every ``sowing_doy`` moved by ``offset`` days, clipped to ``bounds``.

    The clip keeps the sowing inside the season torchcrop simulates (the
    cache opens on DOY 193 and the tables never cross New Year); clipped rows
    duplicate a neighbouring offset and are resolved towards the smaller shift
    by :func:`assemble_oracle`.
    """
    table = pd.read_csv(source)
    shifted = (table["sowing_doy"] + offset).clip(*bounds)
    clipped = int((shifted != table["sowing_doy"] + offset).sum())
    target.parent.mkdir(parents=True, exist_ok=True)
    table.assign(sowing_doy=shifted.astype(int)).to_csv(target, index=False)
    log.info("offset %+d d -> %s (%d of %d rows clipped to %s)", offset, target, clipped, len(table), bounds)
    return target


def assemble_oracle(
    runs: Mapping[int, Path],
    out_dir: Path,
    metric: str = "yield_t_ha",
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per (cell, season), the torchcrop row of the offset with the highest ``metric``.

    ``runs`` maps offset (days) -> ``torchcrop_europe.parquet``. Ties, and the
    duplicates :func:`shift_table`'s clip produces, go to the smallest
    ``|offset|``. Writes ``torchcrop/torchcrop_europe.parquet`` (the chosen rows,
    readable as any other arm), ``sowing_yield_max.csv`` (the sowing table) and
    ``oracle_summary.json`` under ``out_dir``.
    """
    missing = [str(p) for p in runs.values() if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"oracle grid incomplete: {missing}")
    frames = []
    for offset, path in sorted(runs.items()):
        frame = pd.read_parquet(path)
        if years is not None:
            frame = frame[frame["year"].between(*years)]
        frames.append(frame.assign(sowing_offset=np.int16(offset)))
        log.info("offset %+d d: %s rows from %s", offset, f"{len(frame):,}", path)
    grid = pd.concat(frames, ignore_index=True)
    grid["_abs"] = grid["sowing_offset"].abs()
    best = (grid.sort_values([metric, "_abs"], ascending=[False, True], na_position="last")
            .drop_duplicates(KEY).drop(columns="_abs").sort_values(KEY).reset_index(drop=True))

    reference = grid[grid["sowing_offset"] == 0].set_index(KEY)[metric] if 0 in runs else None
    torchcrop = out_dir / "torchcrop"
    torchcrop.mkdir(parents=True, exist_ok=True)
    best.to_parquet(torchcrop / "torchcrop_europe.parquet", index=False)
    best[[*KEY, "sowing_doy"]].to_csv(out_dir / "sowing_yield_max.csv", index=False)

    share = best["sowing_offset"].value_counts(normalize=True).sort_index()
    summary = {
        "metric": metric,
        "offsets": sorted(int(o) for o in runs),
        "runs": {str(int(o)): str(p) for o, p in sorted(runs.items())},
        "cell_seasons": int(len(best)),
        "offset_share": {str(int(o)): round(float(s), 4) for o, s in share.items()},
        "at_grid_edge": round(float(best["sowing_offset"].isin([min(runs), max(runs)]).mean()), 4),
        "median_metric": float(best[metric].median()),
    }
    if reference is not None:
        gain = best.set_index(KEY)[metric] - reference.reindex(best.set_index(KEY).index)
        summary["median_gain_vs_offset0"] = float(gain.median())
        summary["mean_gain_vs_offset0"] = float(gain.mean())
    (out_dir / "oracle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("oracle: %s", summary)
    return best
