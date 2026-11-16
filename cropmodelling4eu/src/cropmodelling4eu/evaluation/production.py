"""Score a calibrated production run per administrative unit.

The references are the ones the calibration was fitted against, on the same
basis it used:

* **CLMS** phenology on the CyBench units, with the PEP725-measured level
  offsets applied — harvest +18.7 d, emergence −6.0 d
  (:data:`~cropmodelling4eu.calibration.config.CLMS_HARVEST_OFFSET_DAYS`), so
  the season length gains their sum. Scoring against raw ``CPMCH`` would report
  the product's own bias as a model error.
* **CyBench** yield converted to dry matter at
  :data:`~cropmodelling4eu.evaluation.config.YIELD_MOISTURE_CONTENT`, since both
  models write grain dry matter.

Simulated cells go to the unit their centre falls in (``cell_to_adm.parquet``,
written by the calibration), averaged with equal weight as
:func:`~cropmodelling4eu.evaluation.aggregate.aggregate_simulated` does, dates
on the circle. Every (unit, year) row is then tagged with the calibration split
it fell in, so held-out skill can be read apart from fitted skill.

Only a run driven by **observed** weather can be scored year by year. A CMIP6
historical year is a draw from that model's climate, not the observed season
of the same number, so :func:`check_year_matched` refuses it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cropmodelling4eu.calibration.config import (
    CLMS_EMERGENCE_OFFSET_DAYS,
    CLMS_HARVEST_OFFSET_DAYS,
)

from .aggregate import to_dry_matter
from .clms import load_phenology
from .config import DAYS_IN_YEAR, YIELD_MOISTURE_CONTENT, _env_path
from .cybench import load_yield
from .doy import circular_mean_doy, doy_difference, unwrap_doy, wrap_doy
from .metrics import AGREEMENT_METRIC_ORDER, agreement_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "CALIBRATION_DIR",
    "MIN_YEARS",
    "PRODUCTION_RUN_DIR",
    "TARGETS",
    "Target",
    "check_year_matched",
    "load_reference",
    "pair_units",
    "pooled_scores",
    "unit_climatology",
    "unit_scores",
    "units_from_cells",
    "yearly_scores",
]

PRODUCTION_RUN_DIR: Path = _env_path(
    "PRODUCTION_RUN_DIR",
    "/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated/mswx",
)
CALIBRATION_DIR: Path = _env_path(
    "CALIBRATION_DIR",
    "/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/winter_wheat_2000_2024_euval/calibration",
)

#: Seasons a unit needs before its own RMSE/R²/IA/MNB is mapped. CLMS has eight
#: at most, and an IA over three points is mostly the one outlier among them.
MIN_YEARS: int = 5


@dataclass(slots=True, frozen=True)
class Target:
    """One simulated quantity paired with its reference."""

    key: str
    label: str
    unit: str
    sim_col: str
    obs_col: str
    circular: bool
    split_file: str


TARGETS: tuple[Target, ...] = (
    Target("yield", "Grain yield", "t DM/ha", "yield_t_ha", "obs_yield_t_ha",
           False, "yield_split.parquet"),
    Target("emergence", "Emergence", "days", "emergence_doy", "obs_emergence_doy",
           True, "phenology_split.parquet"),
    Target("harvest", "Harvest", "days", "maturity_doy", "obs_harvest_doy",
           True, "phenology_split.parquet"),
    Target("season_length", "Emergence to harvest", "days",
           "emergence_to_maturity_days", "obs_season_length_days",
           False, "phenology_split.parquet"),
)


def check_year_matched(run_dir: Path) -> None:
    """Refuse a run whose years are not the observed seasons."""
    if "cmip6" in Path(run_dir).parts:
        raise ValueError(
            f"{run_dir} is a CMIP6 run: its year N is a sample of that model's "
            "climate, not the observed season N, so per-year RMSE/IA/MNB "
            "against CyBench or CLMS measure nothing. Compare climatologies instead."
        )


def load_reference(countries: list[str], years: tuple[int, int]) -> pd.DataFrame:
    """Corrected CLMS phenology and dry-matter CyBench yield per (unit, year)."""
    clms = load_phenology(countries, zone="adm", years=years)
    phenology = pd.DataFrame({
        "adm_id": clms["adm_id"].astype("string"),
        "year": clms["year"].astype(int),
        "obs_emergence_doy": wrap_doy(clms["obs_emergence_doy"] - CLMS_EMERGENCE_OFFSET_DAYS),
        "obs_harvest_doy": wrap_doy(clms["obs_harvest_doy"] + CLMS_HARVEST_OFFSET_DAYS),
        "obs_season_length_days": clms["obs_season_length_days"]
        + CLMS_HARVEST_OFFSET_DAYS + CLMS_EMERGENCE_OFFSET_DAYS,
    })
    cybench = load_yield(countries, years=years)
    yields = pd.DataFrame({
        "adm_id": cybench["adm_id"].astype("string"),
        "year": cybench["harvest_year"].astype(int),
        "obs_yield_t_ha": to_dry_matter(cybench["yield"], YIELD_MOISTURE_CONTENT),
    })
    return phenology.merge(yields, on=["adm_id", "year"], how="outer")


def units_from_cells(frame: pd.DataFrame, cell_to_adm: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight unit means per (adm_id, year); dates averaged on the circle."""
    cells = frame.merge(cell_to_adm[["SimplaceID", "adm_id"]], on="SimplaceID", how="inner")
    cells = cells[cells["adm_id"].notna()]
    keys = ["adm_id", "year"]
    linear = [t.sim_col for t in TARGETS if not t.circular and t.sim_col in cells]
    circular = [t.sim_col for t in TARGETS if t.circular and t.sim_col in cells]

    grouped = cells.groupby(keys, observed=True)
    out = grouped[linear].mean()
    for col in circular:
        angle = 2.0 * np.pi * (cells[col].astype(float) - 1.0) / DAYS_IN_YEAR
        vectors = pd.DataFrame({"s": np.sin(angle), "c": np.cos(angle)}, index=cells.index)
        means = vectors.groupby([cells[k] for k in keys], observed=True).mean()
        out[col] = wrap_doy(np.arctan2(means["s"], means["c"]) * DAYS_IN_YEAR / (2.0 * np.pi) + 1.0)
    out["n_cells"] = grouped.size()
    return out.reset_index()


def pair_units(
    runs: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    cell_to_adm: pd.DataFrame,
    calibration_dir: Path = CALIBRATION_DIR,
) -> pd.DataFrame:
    """Simulated unit means beside the reference, one row per (model, unit, year).

    Adds ``split_<target>`` for each target: the calibration split the row was
    in, or ``"unused"`` for a unit-year the calibration never saw.
    """
    splits = {
        name: pd.read_parquet(Path(calibration_dir) / name)
        .assign(unit=lambda f: f["unit"].astype("string"))
        .set_index(["unit", "year"])["split"]
        for name in {t.split_file for t in TARGETS}
    }
    frames = []
    for model, frame in runs.items():
        units = units_from_cells(frame, cell_to_adm)
        units["adm_id"] = units["adm_id"].astype("string")
        frames.append(units.merge(reference, on=["adm_id", "year"], how="inner").assign(model=model))
    paired = pd.concat(frames, ignore_index=True)

    index = pd.MultiIndex.from_frame(paired[["adm_id", "year"]].rename(columns={"adm_id": "unit"}))
    for target in TARGETS:
        paired[f"split_{target.key}"] = (
            splits[target.split_file].reindex(index).fillna("unused").to_numpy()
        )
    return paired


def _score(frame: pd.DataFrame, target: Target) -> pd.Series:
    return pd.Series(agreement_metrics(
        frame[target.obs_col].to_numpy(), frame[target.sim_col].to_numpy(),
        circular=target.circular,
    ))[list(AGREEMENT_METRIC_ORDER)]


def unit_scores(paired: pd.DataFrame, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Per-unit Bias, RMSE, R², IA and MNB over its seasons, every model and target.

    Returns long rows ``model, target, adm_id, n, bias, rmse, pearson_r2, ia,
    mnb``; a unit
    with fewer than ``min_years`` paired seasons keeps its row with NaN metrics,
    so a map can tell "too few seasons" from "not covered".
    """
    rows = []
    for target in TARGETS:
        if target.sim_col not in paired:
            logger.warning("%s: no %s column in the runs, skipped", target.key, target.sim_col)
            continue
        scored = (
            paired.dropna(subset=[target.sim_col, target.obs_col])
            .groupby(["model", "adm_id"], observed=True)
            .apply(_score, target=target, include_groups=False)
            .reset_index()
        )
        short = scored["n"] < min_years
        scored.loc[short, list(AGREEMENT_METRIC_ORDER[1:])] = np.nan
        rows.append(scored.assign(target=target.key))
    return pd.concat(rows, ignore_index=True)[
        ["model", "target", "adm_id", *AGREEMENT_METRIC_ORDER]
    ]


def unit_climatology(paired: pd.DataFrame, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """Each unit's mean over its own paired seasons, observed and simulated.

    Both sides average the **same** unit-years, so ``sim - obs`` is a
    climatological bias with no sampling difference in it. Dates are circular
    means, then unwrapped around the domain's observed circular mean so a map
    has no seam at New Year (DOY 370 is 5 January of the harvest year).

    Returns long rows ``model, target, adm_id, n, obs, sim, diff``; a unit with
    fewer than ``min_years`` seasons is dropped.
    """
    rows = []
    for target in TARGETS:
        if target.sim_col not in paired:
            continue
        valid = paired.dropna(subset=[target.sim_col, target.obs_col])
        centre = circular_mean_doy(valid[target.obs_col].to_numpy()) if target.circular else None
        for (model, adm_id), block in valid.groupby(["model", "adm_id"], observed=True):
            if len(block) < min_years:
                continue
            obs, sim = block[target.obs_col].to_numpy(), block[target.sim_col].to_numpy()
            if target.circular:
                obs_mean = float(unwrap_doy(circular_mean_doy(obs), centre))
                sim_mean = float(unwrap_doy(circular_mean_doy(sim), centre))
                diff = float(doy_difference(circular_mean_doy(sim), circular_mean_doy(obs)))
            else:
                obs_mean, sim_mean = float(obs.mean()), float(sim.mean())
                diff = sim_mean - obs_mean
            rows.append({"model": model, "target": target.key, "adm_id": adm_id,
                         "n": len(block), "obs": obs_mean, "sim": sim_mean, "diff": diff})
    return pd.DataFrame(rows)


def yearly_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """Spatial skill per season: every unit of one year scored as one sample.

    Where :func:`unit_scores` asks how well a unit's seasons are tracked, this
    asks how well one season's map is reproduced. Rows are ``model, target,
    year``, the agreement metrics, ``obs_mean`` / ``sim_mean`` (dates as
    circular means) and ``held_out`` -- true when no unit-year of that season
    was in the calibration's ``train`` split, i.e. a temporally held-out year.
    """
    rows = []
    for target in TARGETS:
        if target.sim_col not in paired:
            continue
        split_col = f"split_{target.key}"
        valid = paired.dropna(subset=[target.sim_col, target.obs_col])
        for (model, year), block in valid.groupby(["model", "year"], observed=True):
            obs, sim = block[target.obs_col].to_numpy(), block[target.sim_col].to_numpy()
            rows.append({
                "model": model, "target": target.key, "year": int(year),
                **_score(block, target).to_dict(),
                "obs_mean": circular_mean_doy(obs) if target.circular else float(obs.mean()),
                "sim_mean": circular_mean_doy(sim) if target.circular else float(sim.mean()),
                "held_out": not (block[split_col] == "train").any(),
            })
    return pd.DataFrame(rows)


def pooled_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """All unit-years pooled, per model, target and calibration split.

    ``all`` pools every split. Pooled R², IA and MNB include the between-unit
    spread, so they read higher than the median of the per-unit maps.
    """
    rows = []
    for target in TARGETS:
        if target.sim_col not in paired:
            continue
        split_col = f"split_{target.key}"
        valid = paired.dropna(subset=[target.sim_col, target.obs_col])
        everything = valid.assign(**{split_col: "all"})
        for (model, split), block in pd.concat([valid, everything]).groupby(
            ["model", split_col], observed=True
        ):
            rows.append({"model": model, "target": target.key, "split": split,
                         **_score(block, target).to_dict()})
    return pd.DataFrame(rows)
