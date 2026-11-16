"""Calibrate SIMPLACE's crop parameters with Optuna, on torchcrop's objective.

SIMPLACE has no gradient, so the torchcrop calibrator cannot move it: its loss
was only ever *scored* (``evaluation.calibration_loss.score_simplace``) on crops
torchcrop had fitted. This module closes that loop with a derivative-free
search. One Optuna study per region and stage; each trial writes a crop.xml,
runs SIMPLACE over the region's calibration cells, and scores the run.

**The objective is torchcrop's, term for term** (``calibration.losses``):

==========================  ==============================================
Term                        As in torchcrop
==========================  ==============================================
emergence / heading /       weighted MSE of unit-year means against the
harvest level               observation, ``LossConfig.*_level`` weights
emergence / harvest         MSE of (sim - unit climatology) against
anomaly                     (obs - unit climatology)
maturity hinge              ``maturity_after_harvest * (mean(excess^2) +
                            0.1 * mean(early^2))``, ``drydown_days`` slack
yield level                 weighted MSE of the unit-year mean, t/ha
yield anomaly               Huber(sim - unit climatology, obs anomaly)
stage retention             yield stage adds ``lambda_phenology * L_phen``
prior                       ``prior_l2 * sum(((v - mu) / sigma)^2)`` plus the
                            tsum1/(tsum1+tsum2) ratio prior
==========================  ==============================================

Three differences are structural, not choices:

* **Climatology.** torchcrop's is an EMA over batches, because a batch never
  sees a unit's whole record. A SIMPLACE trial scores the whole split at once,
  so the unit mean is computed exactly — the limit the EMA converges to.
* **Heading is DVS 0.87**, which SIMPLACE's yearly file does not report. With
  ``heading_from_daily`` the solution's daily output is kept and the date is
  the linear crossing of ``DevStage`` through 0.87 (``crossing_day``'s rule);
  without it the heading term is dropped, never approximated by anthesis.
* **No pooling penalty.** It shrinks a region toward the cross-region mean,
  which needs every region in one optimisation; the studies are independent.

Best trial = lowest *validation* loss among completed trials, as torchcrop
keeps its best-validation epoch. The search itself minimises the training
loss plus the prior.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from cropmodelling4eu.calibration.config import LossConfig
from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.crops import CROP_PROFILE
from cropmodelling4eu.simplace.collect import daily_frame, read_yearly, to_run_schema
from cropmodelling4eu.simplace.run import build_command, runtime_env, runtime_image
from cropmodelling4eu.simplace.workspace import Workspace

logger = logging.getLogger(__name__)

__all__ = [
    "OptunaSettings",
    "Parameter",
    "SimplaceRunner",
    "SimplaceScorer",
    "StageSpace",
    "apply_values",
    "calibrate_region",
    "load_space",
]

PARAMS_FILE = Path(__file__).parents[1] / "calibration" / "params_simplace.yaml"
#: DVS at PEP725 BBCH 51, as ``calibration.losses.HEADING_DVS``.
HEADING_DVS = 0.87
#: Stage order and the retention weight each later stage gives an earlier one.
STAGES = ("phenology", "yield")
_DAY = pd.Timedelta(days=1)
#: torchcrop writes its crop tables from float32 tensors, so the 1.3 knot
#: comes back as 1.2999999523.
KNOT_TOL = 1e-4


# --------------------------------------------------------------------------- #
# Parameter space
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Parameter:
    """One free parameter: ``crop.<field>`` or a table knot ``crop.<field>@<x>``."""

    name: str
    low: float
    high: float
    init: float | None = None

    @property
    def field(self) -> str:
        return self.name.removeprefix("crop.").split("@", 1)[0]

    @property
    def knot(self) -> float | None:
        return float(self.name.split("@", 1)[1]) if "@" in self.name else None


@dataclass(frozen=True, slots=True)
class StageSpace:
    """A stage's free parameters, priors and ratio priors."""

    stage: str
    parameters: tuple[Parameter, ...]
    priors: dict[str, tuple[float, float]]
    ratio_priors: tuple[tuple[str, str, float, float], ...] = ()

    def penalty(self, values: dict[str, float], weight: float) -> float:
        """torchcrop's ``prior_penalty`` for one region."""
        total = sum(
            ((values[name] - mu) / sigma) ** 2
            for name, (mu, sigma) in self.priors.items()
            if name in values and sigma > 0
        )
        total += sum(
            ((values[a] / max(values[a] + values[b], 1e-6) - mu) / sigma) ** 2
            for a, b, mu, sigma in self.ratio_priors
            if a in values and b in values and sigma > 0
        )
        return weight * float(total)


def load_space(stage: str, path: Path = PARAMS_FILE) -> StageSpace:
    """The stage's block of ``params_simplace.yaml``."""
    spec = yaml.safe_load(Path(path).read_text())["stages"][stage]
    return StageSpace(
        stage=stage,
        parameters=tuple(
            Parameter(name, float(v["bounds"][0]), float(v["bounds"][1]),
                      None if v.get("init") is None else float(v["init"]))
            for name, v in spec["parameters"].items()
        ),
        priors={k: (float(v["value"]), float(v["spread"]))
                for k, v in (spec.get("priors") or {}).items()},
        ratio_priors=tuple(
            (r["numerator"], r["denominator"], float(r["value"]), float(r["spread"]))
            for r in spec.get("ratio_priors") or ()
        ),
    )


def read_value(crop: dict, parameter: Parameter) -> float:
    """A parameter's current value in a flattened crop (``flatten_crop_yaml``)."""
    if parameter.knot is None:
        # scale_factor_rue is absent from the SUSTAg-derived crop: 1 is its
        # meaning there, and the crop.xml writer treats absence the same way.
        return float(crop.get(parameter.field, 1.0 if parameter.field == "scale_factor_rue" else np.nan))
    row = next((r for r in crop[parameter.field] if abs(r[0] - parameter.knot) < KNOT_TOL), None)
    if row is None:
        raise KeyError(f"{parameter.name}: no knot at {parameter.knot} in {crop[parameter.field]}")
    return float(row[1])


def apply_values(crop: dict, space: StageSpace, values: dict[str, float]) -> dict:
    """A copy of ``crop`` with ``values`` written into its scalars and table knots."""
    out = {k: [list(r) for r in v] if isinstance(v, list) else v for k, v in crop.items()}
    for parameter in space.parameters:
        if parameter.name not in values:
            continue
        value = float(values[parameter.name])
        if parameter.knot is None:
            out[parameter.field] = value
            continue
        rows = out[parameter.field]
        index = next((i for i, r in enumerate(rows) if abs(r[0] - parameter.knot) < KNOT_TOL), None)
        if index is None:
            raise KeyError(f"{parameter.name}: no knot at {parameter.knot} in {rows}")
        rows[index][1] = value
    return out


def starting_values(space: StageSpace, crop: dict) -> dict[str, float]:
    """Where torchcrop starts a stage: ``init``, else the crop, clipped to the bounds.

    torchcrop also pulls an init sitting *on* a bound 2 % inward, because the
    sigmoid bijection has no gradient there. TPE has no such problem, so the
    value is only clipped.
    """
    return {
        p.name: float(np.clip(p.init if p.init is not None else read_value(crop, p), p.low, p.high))
        for p in space.parameters
    }


# --------------------------------------------------------------------------- #
# Scoring: torchcrop's loss on SIMPLACE output
# --------------------------------------------------------------------------- #


def _days(values: pd.Series) -> pd.Series:
    """Days since the epoch, from timestamps or epoch-nanosecond floats."""
    return (pd.to_datetime(values) - pd.Timestamp(0)) / _DAY


def _weighted_mse(sim: pd.Series, obs: pd.Series, weight: pd.Series) -> tuple[float, int]:
    mask = sim.notna() & obs.notna()
    if not mask.any():
        return 0.0, 0
    w = weight[mask].fillna(1.0).to_numpy(float)
    err = (sim[mask] - obs[mask]).to_numpy(float)
    return float(np.sum(w * err**2) / max(w.sum(), 1e-9)), int(mask.sum())


def _huber(x: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(x)
    return np.where(a < delta, 0.5 * x**2, delta * (a - 0.5 * delta))


def heading_dates(daily: pd.DataFrame, level: float = HEADING_DVS) -> pd.DataFrame:
    """Per cell-season, the day ``DVS`` crosses ``level``, linearly interpolated.

    ``daily`` is :func:`collect.daily_frame` output. The fraction between the
    last day below and the first day at or above the level is the same rule
    ``torchcrop.run.crossing_day`` applies to the uncapped DVS.
    """
    if daily.empty:
        return pd.DataFrame(columns=["SimplaceID", "year", "heading_d"])
    frame = daily.sort_values(["SimplaceID", "year", "date"])
    keys = ["SimplaceID", "year"]
    previous = frame.groupby(keys)["DVS"].shift()
    previous_date = frame.groupby(keys)["date"].shift()
    hit = (frame["DVS"] >= level) & (previous < level)
    first = frame[hit].groupby(keys).head(1)
    prev, prev_date = previous[first.index], previous_date[first.index]
    fraction = (level - prev) / (first["DVS"] - prev).where(first["DVS"] > prev)
    days = _days(prev_date) + fraction.clip(0, 1)
    return pd.DataFrame({"SimplaceID": first["SimplaceID"].to_numpy(),
                         "year": first["year"].to_numpy(), "heading_d": days.to_numpy()})


def season_dates(sim: pd.DataFrame) -> pd.DataFrame:
    """Absolute emergence and maturity days from SIMPLACE's harvest-year DOYs.

    ``year`` is the harvest year. Emergence normally falls in the autumn before
    it: a DOY later than maturity's belongs to the previous calendar year.
    """
    year_start = pd.to_datetime(sim["year"].astype(int).astype(str) + "-01-01")
    base = (year_start - pd.Timestamp(0)) / _DAY - 1.0
    maturity = base + sim["maturity_doy"]
    emergence_year_start = pd.to_datetime(
        (sim["year"].astype(int) - (sim["emergence_doy"] > sim["maturity_doy"]).astype(int))
        .astype(str) + "-01-01"
    )
    emergence = (emergence_year_start - pd.Timestamp(0)) / _DAY - 1.0 + sim["emergence_doy"]
    return sim.assign(maturity_d=maturity, emergence_d=emergence)


@dataclass(slots=True)
class SimplaceScorer:
    """torchcrop's per-region stage objective, evaluated on a SIMPLACE run.

    Args:
        pool: The region's ``(unit, year, SimplaceID)`` rows.
        observations: ``observations.parquet`` of the calibration run.
        split: ``<stage>_split.parquet`` — which unit-years are train/val/test.
        loss: torchcrop's :class:`LossConfig`.
        stage: ``phenology`` or ``yield``; the yield stage carries the
            phenology terms forward at ``lambda_phenology``.
    """

    pool: pd.DataFrame
    observations: pd.DataFrame
    split: pd.DataFrame
    loss: LossConfig
    stage: str
    targets: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        frames = []
        for variable in ("emergence_date", "heading_date", "harvest_date",
                         "yield_level", "yield_anomaly"):
            block = self.observations[self.observations["variable"] == variable]
            if block.empty:
                continue
            value = _days(block["value"]) if variable.endswith("_date") else pd.to_numeric(block["value"])
            frames.append(
                block.assign(**{variable: value, f"{variable}_w": block["weight"].astype(float)})
                [["unit", "year", variable, f"{variable}_w"]].drop_duplicates(["unit", "year"])
                .set_index(["unit", "year"])
            )
        self.targets = pd.concat(frames, axis=1).reset_index()

    @property
    def weights(self) -> dict[str, float]:
        return {"phenology": 1.0} if self.stage == "phenology" else {
            "yield": 1.0, "phenology": self.loss.lambda_phenology}

    def unit_means(self, sim: pd.DataFrame) -> pd.DataFrame:
        """Unit-year means over the pooled cells; a cell that never matured drops out."""
        cells = self.pool[["unit", "year", "SimplaceID"]].merge(
            sim, on=["SimplaceID", "year"], how="left")
        cells["yield_t_ha"] = cells["yield_g_m2"] * 0.01
        columns = [c for c in ("emergence_d", "heading_d", "maturity_d", "yield_t_ha") if c in cells]
        means = cells.groupby(["unit", "year"])[columns].mean()
        means["cells"] = cells.groupby(["unit", "year"]).size()
        means["matured"] = cells.groupby(["unit", "year"])["maturity_d"].count()
        return means.reset_index()

    def score(self, sim: pd.DataFrame, split: str) -> tuple[float, dict[str, float]]:
        """``(stage loss, report)`` on one split, without the prior."""
        table = (
            self.unit_means(sim)
            .merge(self.targets, on=["unit", "year"], how="left")
            .merge(self.split[["unit", "year", "split"]], on=["unit", "year"], how="inner")
        )
        table = table[table["split"] == split]
        report: dict[str, float] = {
            "unit_years": float(len(table)),
            "never_matured_frac": float(1 - table["matured"].sum() / max(table["cells"].sum(), 1)),
        }
        total = 0.0
        for stage, weight in self.weights.items():
            part = self._phenology(table, report) if stage == "phenology" else self._yield(table, report)
            report[f"{stage}/loss"] = part
            total += weight * part
        report["loss"] = total
        return total, report

    def _anomaly(self, table: pd.DataFrame, sim: str, obs: str) -> tuple[pd.Series, pd.Series]:
        """Within-season anomalies against each unit's own climatology.

        Dates are re-based to 1 January of the harvest year first. torchcrop
        works in days since the season's window start, so a unit's seasons
        share an origin; on absolute days the simulated and observed means span
        different year sets, and the 365 d per year between them would not
        cancel.
        """
        origin = (pd.to_datetime(table["year"].astype(int).astype(str) + "-01-01")
                  - pd.Timestamp(0)) / _DAY
        rel_sim, rel_obs = table[sim] - origin, table[obs] - origin
        return (rel_sim - rel_sim.groupby(table["unit"]).transform("mean"),
                rel_obs - rel_obs.groupby(table["unit"]).transform("mean"))

    def _phenology(self, table: pd.DataFrame, report: dict[str, float]) -> float:
        cfg, parts = self.loss, {}
        pairs = {"emergence_date": ("emergence_d", cfg.emergence_level),
                 "heading_date": ("heading_d", cfg.anthesis_level),
                 "harvest_date": ("maturity_d", cfg.harvest_level)}
        for name, (sim, weight) in pairs.items():
            if weight <= 0 or name not in table or sim not in table:
                continue
            mse, n = _weighted_mse(table[sim], table[name], table[f"{name}_w"])
            if n:
                parts[f"{name}_level"] = weight * mse
                report[f"phenology/{name}_rmse_d"] = float(np.sqrt(mse))
                report[f"phenology/{name}_bias_d"] = float((table[sim] - table[name]).mean())
                report[f"phenology/{name}_n"] = float(n)
        for name, sim, weight in (("emergence_date", "emergence_d", cfg.emergence_anomaly_clms),
                                  ("harvest_date", "maturity_d", cfg.harvest_anomaly)):
            if weight <= 0 or name not in table or sim not in table:
                continue
            sim_anom, obs_anom = self._anomaly(table, sim, name)
            mse, n = _weighted_mse(sim_anom, obs_anom, table[f"{name}_w"])
            if n:
                parts[f"{name}_anomaly"] = weight * mse
                report[f"phenology/{name}_anomaly_rmse_d"] = float(np.sqrt(mse))
        if "harvest_date" in table and cfg.maturity_after_harvest > 0:
            mask = table["maturity_d"].notna() & table["harvest_date"].notna()
            if mask.any():
                gap = (table["maturity_d"] - table["harvest_date"])[mask]
                excess, early = gap.clip(lower=0), (-gap - cfg.drydown_days).clip(lower=0)
                parts["maturity_hinge"] = cfg.maturity_after_harvest * float(
                    (excess**2).mean() + 0.1 * (early**2).mean())
                report["phenology/maturity_after_harvest_d"] = float(excess.mean())
        report["phenology/n_terms"] = float(len(parts))
        return float(sum(parts.values()))

    def _yield(self, table: pd.DataFrame, report: dict[str, float]) -> float:
        cfg, parts = self.loss, {}
        if "yield_level" in table and cfg.yield_level > 0:
            mse, n = _weighted_mse(table["yield_t_ha"], table["yield_level"], table["yield_level_w"])
            if n:
                parts["yield_level"] = cfg.yield_level * mse
                report["yield/yield_rmse_t_ha"] = float(np.sqrt(mse))
                report["yield/yield_bias_t_ha"] = float((table["yield_t_ha"] - table["yield_level"]).mean())
                report["yield/yield_n"] = float(n)
        if "yield_anomaly" in table and cfg.yield_anomaly > 0:
            sim_anom = table["yield_t_ha"] - table.groupby("unit")["yield_t_ha"].transform("mean")
            mask = sim_anom.notna() & table["yield_anomaly"].notna()
            if mask.any():
                w = table.loc[mask, "yield_anomaly_w"].fillna(1.0).to_numpy(float)
                h = _huber((sim_anom - table["yield_anomaly"])[mask].to_numpy(float), cfg.yield_huber_delta)
                parts["yield_anomaly"] = cfg.yield_anomaly * float(np.sum(w * h) / max(w.sum(), 1e-9))
                report["yield/yield_anomaly_huber"] = parts["yield_anomaly"] / cfg.yield_anomaly
        return float(sum(parts.values()))


# --------------------------------------------------------------------------- #
# Running SIMPLACE for one trial
# --------------------------------------------------------------------------- #


class TrialFailed(RuntimeError):
    """A trial's SIMPLACE run did not produce a usable output."""


@dataclass(slots=True)
class SimplaceRunner:
    """Runs one crop over a prebuilt workspace, in a private copy of it.

    The base workspace (cells, weather links, soil, management, solution) is
    built once per region. A trial copies only ``workspace/`` — 0.4 MB, its
    links preserved — and replaces ``data/crop/<file>.xml`` with its own crop,
    *a copy rather than a link*, because a symlink out of the bound tree
    dangles inside the container. Trials therefore never share a crop file and
    can run concurrently.
    """

    base: Workspace
    config: RunConfig
    template_crop: Path
    crop_block: str = CROP_PROFILE.simplace_block
    workers: int = 24
    java_heap: str | None = "4g"
    java_opts: str = "-XX:ActiveProcessorCount=2 -XX:+UseSerialGC"
    heading_from_daily: bool = True
    retries: int = 1

    def run(self, trial_dir: Path, crop: dict) -> pd.DataFrame:
        """Simulate ``crop`` over every base line; return the season table."""
        from cropmodelling4eu.torchcrop.params import write_simplace_crop_xml

        trial_dir = Path(trial_dir)
        work_dir = trial_dir / "workspace"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        shutil.copytree(self.base.work_dir, work_dir, symlinks=True)
        crop_file = next((work_dir / "data" / "crop").glob("*.xml"))
        crop_file.unlink()
        write_simplace_crop_xml(crop_file, crop, self.template_crop, self.crop_block)

        out_dir = trial_dir / "out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        for directory in (out_dir, trial_dir / "projects", trial_dir / "log"):
            directory.mkdir(parents=True, exist_ok=True)
        workspace = replace(
            self.base, root=trial_dir, work_dir=work_dir, out_dir=out_dir,
            solution=work_dir / self.base.solution.relative_to(self.base.work_dir),
            project=work_dir / self.base.project.relative_to(self.base.work_dir),
        )
        self._run_lines(workspace)
        return self._collect(workspace, crop)

    def _chunks(self) -> list[tuple[int, int]]:
        n = self.base.n_lines
        size = -(-n // max(1, min(self.workers, n)))
        return [(a, min(a + size - 1, n)) for a in range(1, n + 1, size)]

    def _run_lines(self, workspace: Workspace) -> None:
        image = runtime_image(self.config.simplace)
        if not image.is_file():
            # Not a TrialFailed: every trial would fail the same way, so the
            # study stops here instead of burning its budget on it.
            raise FileNotFoundError(f"SIMPLACE image not found: {image}")
        env = {**os.environ, **runtime_env(self.config.simplace)}
        vm = " ".join(filter(None, (f"-Xmx{self.java_heap}" if self.java_heap else "", self.java_opts)))
        if vm:
            env["INSTALL4J_ADD_VM_PARAMS"] = vm
        pending = self._chunks()
        for attempt in range(1 + self.retries):
            running = []
            for first, last in pending:
                log = (workspace.root / "log" / f"lines_{first}-{last}.out").open("w")
                command = self._command(workspace, first, last)
                running.append(((first, last), subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
            failed = []
            for chunk, process, log in running:
                if process.wait() != 0:
                    failed.append(chunk)
                log.close()
            if not failed:
                return
            # A JVM abort (the musl/glibc failure mode) loses only its own
            # range; rerunning those lines is cheaper than failing the trial.
            logger.warning("attempt %d: line ranges %s failed", attempt + 1, failed)
            pending = failed
        raise TrialFailed(f"SIMPLACE line ranges {pending} failed after {1 + self.retries} attempts")

    def _command(self, workspace: Workspace, first: int, last: int) -> list[str]:
        """``build_command`` plus the directories the export's links resolve into.

        A derived export (EU_val_nfix links most of EU_val) is unreadable in the
        container unless those targets are bound as well; ``Workspace.binds``
        binds the export alone, and the production jobs add the rest through
        ``run.env`` (``submit._export_bind_roots``).
        """
        from cropmodelling4eu.simplace.submit import _export_bind_roots

        command = build_command(workspace, self.config, first, last)
        index = command.index("-B") + 1
        extra = [f"{r}:{r}:ro" for r in _export_bind_roots(workspace.export_dir)
                 if r != Path(workspace.export_dir)]
        command[index] = ",".join([command[index], *extra])
        return command

    def _collect(self, workspace: Workspace, crop: dict) -> pd.DataFrame:
        raw = read_yearly(workspace.out_dir)
        # The yearly row echoes crop.TSUM1; a mismatch means the solution did
        # not read this trial's file and every score below would be fiction.
        if "vTSUM1" in raw and not np.allclose(raw["vTSUM1"].astype(float), float(crop["tsum1"]), rtol=1e-4, atol=1e-2):
            raise TrialFailed(
                f"SIMPLACE ran TSUM1 {raw['vTSUM1'].unique()[:3]} but the trial wrote {crop['tsum1']}")
        sim = season_dates(to_run_schema(raw, self.config.grid))
        columns = ["SimplaceID", "year", "emergence_d", "maturity_d", "yield_g_m2",
                   "emergence_doy", "anthesis_doy", "maturity_doy", "sowing_doy"]
        sim = sim[[c for c in columns if c in sim]]
        if self.heading_from_daily:
            daily = [
                daily_frame(pd.read_csv(p, sep=";", usecols=["DATE", "SowingDOY", "DevStage"]),
                            int(p.name.split("_", 1)[0]))
                for p in sorted(workspace.out_dir.rglob("*_daily.csv"))
            ]
            if not daily:
                raise TrialFailed("heading_from_daily is on but the run wrote no *_daily.csv")
            sim = sim.merge(heading_dates(pd.concat(daily, ignore_index=True)),
                            on=["SimplaceID", "year"], how="left")
        return sim


# --------------------------------------------------------------------------- #
# One region's study
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class OptunaSettings:
    """Search settings for one region and stage."""

    n_trials: int = 40
    n_startup_trials: int = 10
    seed: int = 0
    workers: int = 24
    java_heap: str | None = "4g"
    java_opts: str = "-XX:ActiveProcessorCount=2 -XX:+UseSerialGC"
    heading_from_daily: bool = True
    #: Cells per unit kept from the pool, an even subsample; 0 keeps all. The
    #: pool holds up to 12, and a trial's cost is linear in cells.
    max_cells_per_unit: int = 4
    #: Seasons simulated before the first scored harvest, so the soil water
    #: and nitrogen state is not the solution's cold start.
    spinup_years: int = 2
    #: A trial whose cell-seasons mature less often than this is pruned: an
    #: unmatured season drops out of the means, so without the guard a crop
    #: that never matures would score as if it fitted.
    min_matured_frac: float = 0.8
    keep_trial_dirs: bool = False
    #: Also try torchcrop's calibrated values for the region. Off for a race
    #: against torchcrop, where that would hand SIMPLACE torchcrop's answer.
    warm_start: bool = True


def _sample_cells(pool: pd.DataFrame, per_unit: int) -> np.ndarray:
    cells = (pool[["unit", "SimplaceID"]].drop_duplicates()
             .sort_values(["unit", "SimplaceID"]).reset_index(drop=True))
    if per_unit > 0:
        position = cells.groupby("unit").cumcount().to_numpy()
        size = cells.groupby("unit")["SimplaceID"].transform("size").to_numpy()
        # An even spread over each unit's sorted cells, as `max_cells` subsamples
        # a whole run: the first, the last and evenly between.
        keep = np.zeros(len(cells), bool)
        for n in np.unique(size):
            picks = np.unique(np.linspace(0, n - 1, min(per_unit, n)).round().astype(int))
            keep |= (size == n) & np.isin(position, picks)
        cells = cells[keep]
    return np.unique(cells["SimplaceID"].to_numpy(np.int64))


def _region_config(config: RunConfig, years: tuple[int, int], settings: OptunaSettings,
                   overrides: dict[str, Any]) -> RunConfig:
    """The run config narrowed to the scored harvests plus the spin-up."""
    first, last = years
    simplace = config.simplace.model_copy(update={
        "start_date": f"{first - 1 - settings.spinup_years}-01-01",
        "end_date": f"{last}-12-31",
        "daily_output": settings.heading_from_daily,
        "workers_per_task": settings.workers,
        **overrides,
    })
    return config.model_copy(update={"simplace": simplace})


def _build_base(config: RunConfig, cells: np.ndarray, root: Path) -> Workspace:
    """Build (or reload) the region's shared workspace."""
    from cropmodelling4eu.export import resolve_export
    from cropmodelling4eu.simplace.workspace import build_workspace

    marker = root / "base.json"
    if marker.is_file():
        meta = json.loads(marker.read_text())
        if meta["cells"] == cells.tolist() and meta["window"] == [config.simplace.start_date,
                                                                  config.simplace.end_date]:
            return Workspace(**{k: Path(v) if k not in ("n_lines",) and v is not None else v
                                for k, v in meta["workspace"].items()})
        shutil.rmtree(root)
    bundle = resolve_export(config, require_management=config.simplace.iopt != 1)
    workspace = build_workspace(config, bundle, cells=cells, root=root)
    marker.write_text(json.dumps({
        "cells": cells.tolist(),
        "window": [config.simplace.start_date, config.simplace.end_date],
        "workspace": {k: (str(getattr(workspace, k)) if getattr(workspace, k) is not None else None)
                      if k != "n_lines" else workspace.n_lines
                      for k in Workspace.__slots__},
    }))
    return workspace


def _torchcrop_values(crops_dir: Path, region_id: int, space: StageSpace) -> dict[str, float] | None:
    from cropmodelling4eu.torchcrop.params import flatten_crop_yaml

    path = crops_dir / f"crop_region_{region_id:02d}.yaml"
    if not path.is_file():
        return None
    crop = flatten_crop_yaml(path)
    values = {}
    for p in space.parameters:
        value = read_value(crop, p)
        if np.isfinite(value):
            values[p.name] = float(np.clip(value, p.low, p.high))
    return values


def calibrate_region(
    region_id: int,
    stage: str,
    calibration_dir: Path,
    config: RunConfig,
    crop_file: Path,
    out_dir: Path,
    settings: OptunaSettings | None = None,
    loss: LossConfig | None = None,
    simplace_overrides: dict[str, Any] | None = None,
    params_file: Path = PARAMS_FILE,
) -> dict:
    """Run (or resume) one region's study for one stage; write its best crop.

    Layout under ``out_dir/region_<NN>/``: ``base_<stage>/`` (the stage's shared workspace),
    ``<stage>/study.log`` (Optuna journal — rerunning resumes it),
    ``<stage>/trials/<n>/`` (per-trial ``season.parquet`` and ``crop.xml``),
    ``<stage>/trials.csv``, ``<stage>/best.json`` and
    ``<stage>/crop_region_<NN>.xml``.

    The yield stage starts from this region's SIMPLACE phenology best when it
    exists, else from torchcrop's phenology crop, so the phenology it carries
    is a SIMPLACE fit whenever one is available — and says which it used.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    from cropmodelling4eu.torchcrop.params import flatten_crop_yaml, write_simplace_crop_xml

    settings = settings or OptunaSettings()
    loss = loss or LossConfig()
    calibration_dir = Path(calibration_dir)
    region_dir = Path(out_dir) / f"region_{region_id:02d}"
    stage_dir = region_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(calibration_dir / "pool.parquet")
    pool = pool[pool["region_id"] == region_id]
    split = pd.read_parquet(calibration_dir / f"{stage}_split.parquet")
    split = split[split["region_id"] == region_id]
    pool = pool.merge(split[["unit", "year"]], on=["unit", "year"], how="inner")
    if pool.empty:
        raise ValueError(f"region {region_id}: no pooled unit-year in the {stage} split")
    observations = pd.read_parquet(calibration_dir / "observations.parquet")

    cells = _sample_cells(pool, settings.max_cells_per_unit)
    pool = pool[pool["SimplaceID"].isin(cells)]
    years = (int(pool["year"].min()), int(pool["year"].max()))
    run_config = _region_config(config, years, settings, simplace_overrides or {})
    # Per stage: the two splits score different unit-years, so different cells.
    base = _build_base(run_config, cells, region_dir / f"base_{stage}")
    template = next(
        p for p in sorted((Path(run_config.simplace.template_dir) / "data" / "crop").glob("*.xml"))
    )
    runner = SimplaceRunner(
        base, run_config, template, workers=settings.workers, java_heap=settings.java_heap,
        java_opts=settings.java_opts, heading_from_daily=settings.heading_from_daily,
    )
    scorer = SimplaceScorer(pool, observations, split, loss, stage)
    space = load_space(stage, params_file)

    crop = flatten_crop_yaml(crop_file)
    carried = "crop file"
    if stage == "yield":
        phenology = load_space("phenology", params_file)
        own = region_dir / "phenology" / "best.json"
        if own.is_file():
            crop, carried = apply_values(crop, phenology, json.loads(own.read_text())["values"]), "simplace phenology best"
        elif (tc := _torchcrop_values(calibration_dir / "phenology" / "crops", region_id, phenology)):
            crop, carried = apply_values(crop, phenology, tc), "torchcrop phenology crop"
    logger.info("region %02d %s: %d cells, %d unit-years, harvests %d-%d, phenology from %s",
                region_id, stage, len(cells), pool[["unit", "year"]].drop_duplicates().shape[0],
                *years, carried)

    storage = JournalStorage(JournalFileBackend(str(stage_dir / "study.log")))
    study = optuna.create_study(
        study_name=f"region_{region_id:02d}_{stage}", storage=storage, load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=settings.seed + region_id, multivariate=True,
            n_startup_trials=settings.n_startup_trials),
    )
    if not study.trials:
        study.enqueue_trial(starting_values(space, crop), user_attrs={"source": "torchcrop start"})
        if settings.warm_start and (tc := _torchcrop_values(calibration_dir / stage / "crops", region_id, space)):
            study.enqueue_trial(tc, user_attrs={"source": "torchcrop calibrated"})
        study.set_user_attr("phenology_from", carried)
        study.set_user_attr("cells", int(len(cells)))

    # A warm start that failed (a node problem, a bad image) is queued again:
    # Optuna never retries a failed trial, and these two are the comparison
    # the whole study is read against.
    finished = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    sources = {t.user_attrs.get("source") for t in study.trials if t.state in finished}
    waiting = {t.user_attrs.get("source") for t in study.trials
               if t.state == optuna.trial.TrialState.WAITING}
    for t in study.trials:
        source = t.user_attrs.get("source")
        if (t.state == optuna.trial.TrialState.FAIL and source
                and source not in sources | waiting and t.params):
            study.enqueue_trial(t.params, user_attrs={"source": source})
            waiting.add(source)

    objective = _objective(runner, scorer, space, crop, stage_dir, settings, loss)
    done = sum(t.state in finished for t in study.trials)
    if (remaining := settings.n_trials - done) > 0:
        study.optimize(objective, n_trials=remaining, catch=(TrialFailed,))

    return _write_best(study, space, crop, stage_dir, region_id, template, runner.crop_block)


def _objective(runner: SimplaceRunner, scorer: SimplaceScorer, space: StageSpace, crop: dict,
               stage_dir: Path, settings: OptunaSettings, loss: LossConfig) -> Callable:
    import optuna

    def objective(trial: optuna.Trial) -> float:
        values = {p.name: trial.suggest_float(p.name, p.low, p.high) for p in space.parameters}
        trial_dir = stage_dir / "trials" / f"{trial.number:04d}"
        started = time.time()
        sim = runner.run(trial_dir, apply_values(crop, space, values))
        sim.to_parquet(trial_dir / "season.parquet", index=False)
        penalty = space.penalty(values, loss.prior_l2)
        reports = {name: scorer.score(sim, name)[1] for name in ("train", "val", "test")}
        trial.set_user_attr("seconds", time.time() - started)
        trial.set_user_attr("prior_penalty", penalty)
        for name, report in reports.items():
            for key, value in report.items():
                trial.set_user_attr(f"{name}/{key}", value)
        if not settings.keep_trial_dirs:
            for sub in ("workspace", "out", "projects", "log"):
                shutil.rmtree(trial_dir / sub, ignore_errors=True)
        matured = 1.0 - reports["train"]["never_matured_frac"]
        if matured < settings.min_matured_frac:
            raise optuna.TrialPruned(f"only {matured:.0%} of cell-seasons matured")
        return reports["train"]["loss"] + penalty

    return objective


def _write_best(study, space: StageSpace, crop: dict, stage_dir: Path, region_id: int,
                template: Path, crop_block: str) -> dict:
    """Pick the best-validation trial and write its crop and summary."""
    import optuna

    from cropmodelling4eu.torchcrop.params import write_simplace_crop_xml

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(stage_dir / "trials.csv", index=False)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"region {region_id}: no completed trial in {stage_dir}")
    best = min(complete, key=lambda t: (t.user_attrs.get("val/loss", np.inf), t.value))
    values = {p.name: best.params[p.name] for p in space.parameters}
    write_simplace_crop_xml(stage_dir / f"crop_region_{region_id:02d}.xml",
                            apply_values(crop, space, values), template, crop_block)
    summary = {
        "region_id": region_id, "stage": space.stage, "trial": best.number,
        "source": best.user_attrs.get("source", "tpe"), "values": values,
        "objective": best.value, "completed": len(complete), "trials": len(study.trials),
        "phenology_from": study.user_attrs.get("phenology_from"),
        **{k: v for k, v in best.user_attrs.items() if k.split("/", 1)[0] in ("train", "val", "test")},
    }
    (stage_dir / "best.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("region %02d %s: best trial %d (%s), val loss %.4g", region_id, space.stage,
                best.number, summary["source"], best.user_attrs.get("val/loss", float("nan")))
    return summary
