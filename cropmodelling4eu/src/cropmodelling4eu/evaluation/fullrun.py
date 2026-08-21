"""Load both models' full continental runs for a side-by-side comparison.

TorchCrop and SIMPLACE write different schemas for the same quantities.
:mod:`.torchcrop` derives ``maturity_doy``/``harvest_doy``/``season_length_days``
from ``days_to_maturity`` and a sowing latch; SIMPLACE's own collector
(``cm4eu simplace collect``) already writes the dates it simulated, plus
``sowing_forced`` and the sowing window it chose within. :func:`load_runs`
reads each with its own model's rules and returns both tagged by ``model``, so
the rest of a notebook does not need to know which pipeline produced a row.

Used by ``evaluation/full_run_evaluation.ipynb``, which compares the two
against CyBench and against each other. Unlike
:mod:`~cropmodelling4eu.evaluation.germany` (a matched 30-cell smoke test) or
:mod:`~cropmodelling4eu.evaluation.stresstest` (every non-model input forced
equal), this reads each pipeline's *delivered* production output as
``submit/submit_cropmodelling.sh`` writes it -- whatever inputs actually
differ between the two runs (sowing date, year coverage, cell set) stay in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import torchcrop as torchcrop_mod
from .config import SIM_PARQUET, SIMPLACE_PARQUET

logger = logging.getLogger(__name__)

__all__ = ["load_runs", "pair_models"]

#: Quantities read (as available) from both models' Parquets.
_COLUMNS = (
    "SimplaceID", "year", "lon", "lat",
    "yield_t_ha", "yield_g_m2", "biomass_g_m2", "max_lai",
    "days_to_maturity", "tranrf_mean", "nni_mean",
    "sowing_doy", "maturity_doy",
)


def _load_simplace(path: Path, years: tuple[int, int] | None) -> pd.DataFrame:
    """SIMPLACE's own collected Parquet: no phenology derivation needed.

    Its ``maturity_doy``/``sowing_doy`` are the model's simulated dates, not a
    latch plus an offset, so nothing here mirrors
    :func:`torchcrop.add_phenology_columns` -- doing so would silently
    recompute a date the run already measured.
    """
    frame = pd.read_parquet(path)
    frame = frame[[c for c in _COLUMNS if c in frame.columns]].copy()
    frame["year"] = frame["year"].astype("int16")
    if years is not None:
        first, last = years
        frame = frame[frame["year"].between(first, last)].reset_index(drop=True)
    frame["model"] = "simplace"
    # LINTUL-5's harvest = maturity while HARVEST_LAG_DAYS = 0 (see
    # torchcrop.add_phenology_columns); SIMPLACE's rule-based solution models
    # no drydown either, so the same convention is applied for a like-for-like
    # column on both sides.
    frame["harvest_doy"] = frame["maturity_doy"]
    return frame


def load_runs(
    simplace_path: Path | str = SIMPLACE_PARQUET,
    torchcrop_path: Path | str = SIM_PARQUET,
    years: tuple[int, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Each model's full run, normalised onto a shared column set.

    A run not yet on disk is skipped with a warning rather than raised on: the
    notebook stays usable with one model's array finished before the other's,
    the same tolerance :func:`~cropmodelling4eu.evaluation.germany.load_runs`
    gives the smoke test.

    Args:
        simplace_path: ``simplace_europe.parquet`` from ``cm4eu simplace
            collect``.
        torchcrop_path: ``torchcrop_europe.parquet`` from the shard combine.
        years: Inclusive ``(first, last)`` harvest-year filter, applied to
            both sides before anything downstream sees them.

    Returns:
        ``{"simplace": frame, "torchcrop": frame}`` for whichever exist, each
        carrying ``model`` and, where the source has it, ``sowing_doy`` /
        ``maturity_doy`` / ``harvest_doy`` / ``days_to_maturity``.

    Raises:
        FileNotFoundError: If neither path exists.
    """
    runs: dict[str, pd.DataFrame] = {}

    torchcrop_path = Path(torchcrop_path)
    if torchcrop_path.is_file():
        frame = torchcrop_mod.load_simulation(
            torchcrop_path, years=years, with_phenology=True,
        )
        frame["model"] = "torchcrop"
        runs["torchcrop"] = frame
    else:
        logger.warning("torchcrop: no run at %s, skipping it", torchcrop_path)

    simplace_path = Path(simplace_path)
    if simplace_path.is_file():
        runs["simplace"] = _load_simplace(simplace_path, years)
    else:
        logger.warning("simplace: no run at %s, skipping it", simplace_path)

    if not runs:
        raise FileNotFoundError(
            f"neither run exists: simplace={simplace_path}, "
            f"torchcrop={torchcrop_path}"
        )
    for name, frame in runs.items():
        logger.info(
            "%s: %d cell-seasons, %d cells, %d-%d",
            name, len(frame), frame["SimplaceID"].nunique(),
            int(frame["year"].min()), int(frame["year"].max()),
        )
    return runs


def pair_models(
    runs: dict[str, pd.DataFrame],
    columns: tuple[str, ...] = (
        "yield_t_ha", "biomass_g_m2", "max_lai", "sowing_doy",
        "maturity_doy", "days_to_maturity",
    ),
) -> pd.DataFrame:
    """One row per ``(SimplaceID, year)`` both models cover, side by side.

    Sign convention throughout is ``torchcrop - simplace``, matching
    :mod:`~cropmodelling4eu.evaluation.stresstest`: SIMPLACE sits in the
    "observed" slot for the arithmetic only -- **neither model is a
    reference** here, unlike every CyBench comparison in this notebook.

    Args:
        runs: Output of :func:`load_runs`; both keys must be present.
        columns: Quantities to pair, where both sides carry them.

    Returns:
        Paired rows with ``<col>_simplace``, ``<col>_torchcrop`` and
        ``<col>_delta`` per shared column, plus ``yield_ratio``.

    Raises:
        KeyError: If ``runs`` holds only one model.
    """
    missing = {"simplace", "torchcrop"} - set(runs)
    if missing:
        raise KeyError(f"pair_models needs both models; missing {sorted(missing)}")

    keys = ["SimplaceID", "year"]
    sides = {
        model: runs[model].drop(columns=["model"]).set_index(keys)
        for model in ("simplace", "torchcrop")
    }
    shared = [c for c in columns if all(c in f.columns for f in sides.values())]
    dropped = set(columns) - set(shared)
    if dropped:
        logger.info("not on both sides, dropped from the pairing: %s", sorted(dropped))

    paired = sides["simplace"][["lon", "lat"]].join(
        sides["simplace"][shared].add_suffix("_simplace"), how="inner"
    ).join(sides["torchcrop"][shared].add_suffix("_torchcrop"), how="inner")

    for col in shared:
        paired[f"{col}_delta"] = paired[f"{col}_torchcrop"] - paired[f"{col}_simplace"]
    if "yield_t_ha" in shared:
        paired["yield_ratio"] = (
            paired["yield_t_ha_torchcrop"]
            / paired["yield_t_ha_simplace"].replace(0, np.nan)
        )
    logger.info(
        "%d (cell, year) pairs common to both runs (of %d simplace, %d torchcrop)",
        len(paired), len(sides["simplace"]), len(sides["torchcrop"]),
    )
    return paired.reset_index()
