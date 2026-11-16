"""Fit the global emergence thermal sum on PEP725 sowing -> emergence lags.

Only station-seasons that observed **both** dates, and only on the
``train`` blocks of :func:`~cropmodelling4eu.sowing.targets.pep725_split`: the
val/test blocks later score the sowing network and must not have shaped the
clock it was trained through. No CLMS and no network enter, so the value is
independent of both — the condition for freezing it in the calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .cache import SeasonCache
from .config import SowingConfig
from .emergence import fit_tsumem, read_teffmx
from .features import SHORT_CHANNELS
from .targets import load_targets

logger = logging.getLogger(__name__)

__all__ = ["fit_tsumem_pep725"]


def fit_tsumem_pep725(config: SowingConfig, splits: tuple[str, ...] = ("train",)) -> Path:
    """Write ``tsumem.json`` and return its path."""
    cache = SeasonCache.open(config.cache_dir)
    targets = load_targets(config)
    pairs = targets[(targets["source"] == "pep725") & targets["split"].isin(splits)
                    & targets["sow_idx"].notna() & targets["em_idx"].notna()].copy()
    lag = pairs["em_idx"] - pairs["sow_idx"]
    # Same-day or >120-day lags are filing errors (an emergence entered for
    # the wrong season), not a thermal clock.
    pairs = pairs[(lag >= 1) & (lag <= 120) & pairs["sow_idx"].between(0, config.cache_days - 2)]
    pairs["row"] = cache.rows(pairs["SimplaceID"].to_numpy(), pairs["year"].to_numpy())
    pairs = pairs[(pairs["row"] >= 0)]
    pairs = pairs[cache.valid[pairs["row"].to_numpy()]]
    if pairs.empty:
        raise ValueError("no PEP725 station-season with both sowing and emergence in the cache")

    rows = pairs["row"].to_numpy()
    order = np.argsort(rows)
    davtmp = np.asarray(cache.short[rows[order]][..., SHORT_CHANNELS.index("davtmp")], np.float64)
    davtmp = davtmp[np.argsort(order)]
    result = fit_tsumem(davtmp, pairs["sow_idx"].to_numpy().astype(int), pairs["em_idx"].to_numpy(),
                        tbasem=config.tbasem, teffmx=read_teffmx(config.crop_path))
    result |= {"splits": list(splits), "stations": int(pairs["s_id"].nunique()),
               "source": "PEP725 BBCH 0 -> BBCH 10, train blocks"}
    config.tsumem_file.parent.mkdir(parents=True, exist_ok=True)
    config.tsumem_file.write_text(json.dumps(result, indent=1))
    logger.info("tsumem %.0f C d (n %d, MAE %.1f d, bias %+.1f d, lag obs %.1f / sim %.1f d) -> %s",
                result["tsumem"], result["n"], result["mae_days"], result["bias_days"],
                result["lag_obs_median_days"], result["lag_sim_median_days"], config.tsumem_file)
    return config.tsumem_file
