"""Per-region calibration loss terms for both models, in one long table.

torchcrop's are written during training (``<stage>/batch_history.parquet``,
one row per epoch, batch, region and term). SIMPLACE is not calibrated — it
re-runs on the crop files torchcrop fitted — so its terms are scored here, post
hoc, on **the same pool, observations and split** the calibration persisted
(``pool.parquet``, ``observations.parquet``, ``<stage>_split.parquet``), with
the same reduction: cells area-weighted (equally, as the loss does) to the
unit-year mean, *then* compared.

The term names match torchcrop's report, so the two overlay directly:

=================================  =========================================
Term                               Definition
=================================  =========================================
``phenology/harvest_date_rmse_d``  Weighted RMSE, simulated maturity vs
                                   observed harvest date [d]
``phenology/harvest_date_bias_d``  Mean simulated - observed [d]
``yield/yield_rmse_t_ha``          Weighted RMSE of the unit-mean dry-matter
                                   yield against ``yield_level`` [t/ha]
``yield/yield_bias_t_ha``          Mean simulated - observed [t/ha]
=================================  =========================================

Maturity stands in for harvest exactly as torchcrop's DVS-2 crossing does;
neither model simulates the drydown between them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["load_batch_history", "load_simplace_runs", "score_simplace"]

#: g DM m-2 to t/ha, as ``calibration.losses.YieldLoss`` converts ``wso``.
G_M2_TO_T_HA: float = 0.01


def load_batch_history(calibration_dir: Path, stages: Iterable[str] | None = None) -> pd.DataFrame:
    """Every stage's ``batch_history.parquet`` under one calibration run."""
    calibration_dir = Path(calibration_dir)
    paths = sorted(calibration_dir.glob("*/batch_history.parquet"))
    if stages is not None:
        paths = [p for p in paths if p.parent.name in set(stages)]
    if not paths:
        raise FileNotFoundError(f"no batch_history.parquet under {calibration_dir}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_simplace_runs(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate ``simplace_europe.parquet`` files, e.g. one per region run.

    A cell-season present in more than one file keeps its first occurrence, so
    pass the per-region runs before a domain-wide fallback.
    """
    frames = [
        pd.read_parquet(p, columns=["SimplaceID", "year", "maturity_doy", "yield_g_m2"])
        for p in map(Path, paths)
    ]
    if not frames:
        raise ValueError("no SIMPLACE output given")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["SimplaceID", "year"], keep="first")
        .reset_index(drop=True)
    )


def _days(dates: pd.Series) -> pd.Series:
    """Days since the epoch, whatever resolution pandas stored the dates in."""
    return (pd.to_datetime(dates) - pd.Timestamp(0)) / pd.Timedelta(days=1)


def _unit_means(pool: pd.DataFrame, simulated: pd.DataFrame) -> pd.DataFrame:
    """Unit-year means of maturity date and yield over the pooled cells."""
    cells = pool.merge(simulated, on=["SimplaceID", "year"], how="left")
    # SIMPLACE's `year` is the harvest year; maturity lands in it.
    cells["maturity"] = pd.to_datetime(
        cells["year"].astype(str) + "-01-01"
    ) + pd.to_timedelta(cells["maturity_doy"] - 1, unit="D")
    cells["yield_t_ha"] = cells["yield_g_m2"] * G_M2_TO_T_HA
    cells["maturity_d"] = _days(cells["maturity"])
    means = (
        cells.groupby(["unit", "year", "region_id"], observed=True)
        .agg(maturity_d=("maturity_d", "mean"), yield_t_ha=("yield_t_ha", "mean"),
             cells=("SimplaceID", "size"), simulated=("maturity_doy", "count"))
        .reset_index()
    )
    return means


#: ``(simulated, observed, weight) -> (rmse, bias, n)`` term names.
_TERMS: dict[tuple[str, str, str], tuple[str, str, str]] = {
    ("maturity_d", "harvest_d", "harvest_date_weight"): (
        "phenology/harvest_date_rmse_d", "phenology/harvest_date_bias_d",
        "phenology/harvest_date_n",
    ),
    ("yield_t_ha", "yield_level", "yield_level_weight"): (
        "yield/yield_rmse_t_ha", "yield/yield_bias_t_ha", "yield/yield_n",
    ),
}


def _terms(frame: pd.DataFrame) -> dict[str, float]:
    """Weighted RMSE, plain bias and count of each term ``frame`` can score."""
    out: dict[str, float] = {}
    for (simulated, observed, weight), names in _TERMS.items():
        if not {simulated, observed} <= set(frame.columns):
            continue
        block = frame[frame[[simulated, observed]].notna().all(axis=1)]
        if block.empty:
            continue
        error = block[simulated] - block[observed]
        w = block[weight].fillna(1.0).to_numpy(float)
        rmse = float(np.sqrt(np.sum(w * error**2) / max(w.sum(), 1e-9)))
        out |= dict(zip(names, (rmse, float(error.mean()), float(len(block)))))
    return out


def score_simplace(
    calibration_dir: Path,
    simulated: pd.DataFrame,
    stage: str,
    label: str = "simplace",
) -> pd.DataFrame:
    """SIMPLACE's harvest-date and yield terms per region and split.

    Args:
        calibration_dir: A calibration run's output directory, holding
            ``pool.parquet``, ``observations.parquet`` and
            ``<stage>_split.parquet``.
        simulated: ``SimplaceID, year, maturity_doy, yield_g_m2`` — see
            :func:`load_simplace_runs`.
        stage: Whose split to score on.
        label: Carried into the output, e.g. ``default`` / ``calibrated``.

    Returns:
        ``model, label, stage, split, region_id, term, value`` — the long
        format of ``batch_history.parquet``.
    """
    calibration_dir = Path(calibration_dir)
    pool = pd.read_parquet(calibration_dir / "pool.parquet")
    observations = pd.read_parquet(calibration_dir / "observations.parquet")
    split = pd.read_parquet(calibration_dir / f"{stage}_split.parquet")

    means = _unit_means(pool[["unit", "year", "region_id", "SimplaceID"]], simulated)
    # One variable at a time: `value` mixes timestamps and floats, and a joint
    # pivot coerces the whole frame to whichever dtype it meets first.
    def observed(variable: str, convert) -> pd.DataFrame:
        block = observations[observations["variable"] == variable]
        return (
            block.assign(**{variable: convert(block["value"]),
                            f"{variable}_weight": block["weight"].astype(float)})
            [["unit", "year", variable, f"{variable}_weight"]]
            .drop_duplicates(["unit", "year"])
        )

    harvest = observed(
        "harvest_date",
        _days,
    ).rename(columns={"harvest_date": "harvest_d"})
    yields = observed("yield_level", lambda v: pd.to_numeric(v))
    table = (
        means.merge(harvest, on=["unit", "year"], how="left")
        .merge(yields, on=["unit", "year"], how="left")
        .merge(split[["unit", "year", "split"]], on=["unit", "year"], how="inner")
    )

    out = pd.DataFrame(
        [
            {"model": "simplace", "label": label, "stage": stage, "split": split_name,
             "region_id": int(region), "term": term, "value": value}
            for (split_name, region), block in table.groupby(["split", "region_id"])
            for term, value in _terms(block).items()
        ]
    )
    logger.info(
        "SIMPLACE %s scored on %d unit-years over %d regions",
        label, len(table), table["region_id"].nunique(),
    )
    return out
