"""The fixed emergence step: sowing day -> emergence day on the thermal clock.

It is torchcrop's own clock, reproduced so the sowing network is trained
against the lag the crop model will apply:

* ``rate_t = clamp(davtmp_t - tbasem, 0, teffmx - tbasem)`` from the sowing day
  on (``processes/phenology.py``), with ``davtmp = (tmin + tmax) / 2``;
* the state after day ``i`` is the sum of the rates of days ``s .. i``;
* emergence is ``crossing_day(tsump, tsumem)``: the fractional state index at
  which that sum reaches ``tsumem``, interpolated linearly between the last day
  below and the first at or above it (``torchcrop.run.crossing_day``).

The result is a **cache index** — days since the cache opened — so
``emergence_index - sowing_index`` is exactly torchcrop's
``days_to_emergence``. Nothing here is learned, which is what keeps weather
after the sowing day out of the network: it reaches the loss only through this
table.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["EmergenceParams", "emergence_index", "fit_tsumem", "read_teffmx"]


@dataclass(slots=True, frozen=True)
class EmergenceParams:
    """The three crop constants the thermal clock reads."""

    tsumem: float
    tbasem: float = 0.0
    teffmx: float = 30.0


def emergence_index(
    davtmp: torch.Tensor, sow: torch.Tensor, params: EmergenceParams
) -> torch.Tensor:
    """Fractional emergence index for each candidate sowing index.

    Args:
        davtmp: ``[B, T]`` daily mean temperature, ``(tmin + tmax) / 2``.
        sow: ``[B, K]`` integer sowing indices, ``0 <= sow < T``.
        params: The thermal constants.

    Returns:
        ``[B, K]`` float cache index; NaN where the sum is never reached
        inside the cache.
    """
    rate = (davtmp.double() - params.tbasem).clamp(0.0, params.teffmx - params.tbasem)
    cum = F.pad(rate.cumsum(dim=1), (1, 0))
    target = cum.gather(1, sow.long()) + params.tsumem
    k = torch.searchsorted(cum.contiguous(), target.contiguous())
    never = k > davtmp.shape[1]
    k = k.clamp(1, davtmp.shape[1])
    lo, hi = cum.gather(1, k - 1), cum.gather(1, k)
    frac = ((target - lo) / (hi - lo).clamp(min=1e-9)).clamp(0.0, 1.0)
    index = (k - 2).to(cum.dtype) + frac
    return index.masked_fill(never, float("nan")).to(davtmp.dtype)


def fit_tsumem(
    davtmp: np.ndarray,
    sow_index: np.ndarray,
    emergence_obs: np.ndarray,
    tbasem: float = 0.0,
    teffmx: float = 30.0,
    grid: np.ndarray | None = None,
) -> dict:
    """One ``tsumem`` minimising the absolute lag error on observed pairs.

    The absolute error, not the squared one: PEP725 dates are typed in by
    volunteers and a few are a month off. A season that never emerges inside
    the cache is scored at the cache end, so a too-large ``tsumem`` is
    penalised rather than silently ignored.

    Args:
        davtmp: ``[N, T]`` daily mean temperature of each station-season.
        sow_index: ``[N]`` observed sowing, cache index.
        emergence_obs: ``[N]`` observed emergence, cache index.
        tbasem, teffmx: Held fixed.
        grid: Candidate values [C d]; default 40..320 in steps of 1.

    Returns:
        ``tsumem``, the lag statistics at the optimum, and the error curve.
    """
    grid = np.arange(40.0, 321.0, 1.0) if grid is None else np.asarray(grid, float)
    temp = torch.as_tensor(davtmp, dtype=torch.float64)
    sow = torch.as_tensor(sow_index, dtype=torch.long).unsqueeze(1)
    obs = np.asarray(emergence_obs, float)
    end = float(davtmp.shape[1] - 1)

    curve = []
    for value in grid:
        pred = emergence_index(temp, sow, EmergenceParams(float(value), tbasem, teffmx))
        pred = np.nan_to_num(pred.squeeze(1).numpy(), nan=end)
        error = pred - obs
        curve.append((float(value), float(np.mean(np.abs(error))), float(np.mean(error))))
    table = np.array(curve)
    best = int(np.argmin(table[:, 1]))
    tsumem = float(table[best, 0])

    pred = emergence_index(temp, sow, EmergenceParams(tsumem, tbasem, teffmx)).squeeze(1).numpy()
    lag_sim = pred - np.asarray(sow_index, float)
    lag_obs = obs - np.asarray(sow_index, float)
    result = {
        "tsumem": tsumem,
        "tbasem": tbasem,
        "teffmx": teffmx,
        "n": int(len(obs)),
        "mae_days": float(table[best, 1]),
        "bias_days": float(table[best, 2]),
        "lag_obs_median_days": float(np.median(lag_obs)),
        "lag_sim_median_days": float(np.nanmedian(lag_sim)),
        "never_emerged": int(np.isnan(pred).sum()),
        "at_grid_edge": best in (0, len(grid) - 1),
        "curve": {"tsumem": table[:, 0].tolist(), "mae": table[:, 1].tolist(),
                  "bias": table[:, 2].tolist()},
    }
    if result["at_grid_edge"]:
        logger.warning("tsumem %.0f sits on the search grid's edge", tsumem)
    return result


def read_teffmx(crop_yaml) -> float:
    """``teffmx`` from a calibrated crop YAML (first numeric occurrence)."""
    import re
    from pathlib import Path

    match = re.search(r"^\s+teffmx:\s*([-0-9.eE]+)\s*$", Path(crop_yaml).read_text(), re.M)
    if match is None:
        raise ValueError(f"no numeric teffmx in {crop_yaml}")
    return float(match.group(1))
