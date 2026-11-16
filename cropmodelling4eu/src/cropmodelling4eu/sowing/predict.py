"""Sowing tables from a trained network, in the format torchcrop already reads.

Two tables, both ``(SimplaceID, year, sowing_doy)``:

* ``sowing_network.csv`` — the network's date for each season (arms B0, B1);
* ``sowing_network_climatology.csv`` — each cell's median of that, repeated
  every season (arm C). Comparing the two isolates the year-to-year weather
  response from the better level.

The date is the **median** of the first-event distribution: the first day its
CDF reaches 0.5. ``sowing_network_detail.parquet`` keeps the 10/90 % days and
the mean for diagnostics. A DOY of 366 is written as 365: torchcrop's engine
day wraps at 365, so 366 would never latch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import SeasonCache, index_to_doy
from .config import SowingConfig
from .model import first_event_log_probs
from .train import Rows, load_model

logger = logging.getLogger(__name__)

__all__ = ["NETWORK_TABLE", "CLIMATOLOGY_TABLE", "predict"]

NETWORK_TABLE = "sowing_network.csv"
CLIMATOLOGY_TABLE = "sowing_network_climatology.csv"


def _quantile(cdf: torch.Tensor, q: float) -> np.ndarray:
    return (cdf < q).sum(dim=1).clamp(max=cdf.shape[1] - 1).numpy()


def predict(config: SowingConfig, chunk: int = 4096, out_dir: Path | None = None) -> Path:
    """Predict every valid cached season and write both tables."""
    cache = SeasonCache.open(config.cache_dir)
    model, norm, meta = load_model(config.model_dir)
    torch.set_num_threads(config.train.threads)
    keys = cache.row_keys()
    keys = keys[cache.valid[keys["row"].to_numpy()]].reset_index(drop=True)

    parts = []
    for start in range(0, len(keys), chunk):
        block = keys.iloc[start:start + chunk]
        data = Rows.load(cache, block["row"].to_numpy())
        idx = np.arange(len(block))
        with torch.no_grad():
            logits = model(norm.apply(data.short, "short"), norm.apply(data.medium, "medium"),
                           norm.apply(data.long, "long"), torch.from_numpy(data.window_start))
            prob = first_event_log_probs(logits).exp()
        cdf = prob.cumsum(dim=1)
        years = block["year"].to_numpy()
        to_doy = lambda i: index_to_doy(i, years, meta["cache_start_doy"])
        parts.append(pd.DataFrame({
            "SimplaceID": block["SimplaceID"].to_numpy(), "year": years,
            "sowing_doy": to_doy(_quantile(cdf, 0.5)),
            "sowing_doy_p10": to_doy(_quantile(cdf, 0.1)),
            "sowing_doy_p90": to_doy(_quantile(cdf, 0.9)),
            "sowing_index_mean": (prob * torch.arange(prob.shape[1])).sum(dim=1).numpy(),
            "p_deadline": prob[:, -1].numpy(),
        }))
        if (start // chunk) % 50 == 0:
            logger.info("predicted %d / %d seasons", start + len(block), len(keys))
        del data, idx

    detail = pd.concat(parts, ignore_index=True)
    for col in ("sowing_doy", "sowing_doy_p10", "sowing_doy_p90"):
        detail[col] = detail[col].clip(upper=365).astype(np.int16)
    out = Path(out_dir or config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(out / "sowing_network_detail.parquet", index=False)
    detail[["SimplaceID", "year", "sowing_doy"]].to_csv(out / NETWORK_TABLE, index=False)

    median = detail.groupby("SimplaceID")["sowing_doy"].median().round().astype(np.int16)
    climatology = detail[["SimplaceID", "year"]].assign(sowing_doy=lambda f: f["SimplaceID"].map(median))
    climatology.to_csv(out / CLIMATOLOGY_TABLE, index=False)
    logger.info("sowing tables -> %s (%d seasons, %d cells, DOY %d-%d, deadline share %.3f)",
                out, len(detail), detail["SimplaceID"].nunique(), detail["sowing_doy"].min(),
                detail["sowing_doy"].max(), float((detail["p_deadline"] > 0.5).mean()))
    return out
