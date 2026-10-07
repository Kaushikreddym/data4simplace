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
from .config import MIN_PLAUSIBLE_SEASON_DAYS, SIM_PARQUET, SIMPLACE_PARQUET

logger = logging.getLogger(__name__)

__all__ = ["load_runs", "pair_models"]

#: Quantities read (as available) from both models' Parquets.
_COLUMNS = (
    "SimplaceID", "year", "lon", "lat",
    "yield_t_ha", "yield_g_m2", "biomass_g_m2", "max_lai",
    "days_to_maturity", "days_to_emergence", "tranrf_mean", "nni_mean",
    "sowing_doy", "emergence_doy", "anthesis_doy", "maturity_doy",
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
    # column on both sides. Conditional, not assumed: a solution whose output
    # carries no maturity date at all (collect.to_run_schema warns when this
    # happens) simply has no harvest_doy either, rather than a KeyError.
    if "maturity_doy" in frame.columns:
        frame["harvest_doy"] = frame["maturity_doy"]

    # The run's own season (sowing -> maturity) and the endpoint-matched one
    # (emergence -> maturity, what CLMS observes). Both are durations, so
    # neither is wrapped; the second is what config.CLMS_STAGES scores, since
    # pairing a sowing-based duration against an emergence-based observation
    # would carry the whole sowing-to-emergence lag as an apparent bias.
    if "days_to_maturity" in frame.columns:
        frame["season_length_days"] = frame["days_to_maturity"].astype(float)
        if "days_to_emergence" in frame.columns:
            frame["emergence_to_maturity_days"] = (
                frame["season_length_days"] - frame["days_to_emergence"].astype(float)
            )
        # collect.to_run_schema wraps a duration forward into (0, 365], so a
        # season running past its own sowing anniversary is written as the
        # remainder. Dropping the *durations* is the honest repair: the dates
        # are still what SIMPLACE wrote, so such a row still carries a harvest
        # bias -- it just cannot report how long the season was. See
        # config.MIN_PLAUSIBLE_SEASON_DAYS for why the cut is where it is.
        wrapped = frame["season_length_days"] < MIN_PLAUSIBLE_SEASON_DAYS
        if wrapped.any():
            logger.warning(
                "simplace: %d of %d rows (%.2f%%) carry a season shorter than "
                "%.0f days -- the (0, 365] wrap in collect.to_run_schema, not "
                "a %.0f-day crop. Their durations are dropped; their dates are "
                "kept", int(wrapped.sum()), len(frame),
                100.0 * float(wrapped.mean()), MIN_PLAUSIBLE_SEASON_DAYS,
                float(frame.loc[wrapped, "season_length_days"].median()),
            )
            duration_cols = [c for c in ("season_length_days",
                                         "emergence_to_maturity_days",
                                         "days_to_maturity", "days_to_emergence")
                             if c in frame.columns]
            frame.loc[wrapped, duration_cols] = np.nan
        frame["season_wrapped"] = wrapped
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
        "emergence_doy", "maturity_doy", "days_to_maturity",
        "days_to_emergence",
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
    if "sowing_doy" in shared:
        # simplace_europe.parquet's own sowing_doy is the raw value SIMPLACE
        # recorded, one day earlier than the date the crop actually starts
        # growing (see collect.sowing_table's docstring) -- sowing_from_
        # simplace.csv corrects it by +1 before torchcrop ever sees it. A
        # chained run therefore shows sowing_doy_delta == +1 everywhere, not
        # 0; comparing torchcrop against the *corrected* date here (what it
        # was actually handed) is what "sown the same day" means.
        paired["sowing_doy_delta"] = (
            paired["sowing_doy_torchcrop"] - (paired["sowing_doy_simplace"] + 1)
        )
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
