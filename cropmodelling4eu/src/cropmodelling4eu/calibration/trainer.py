"""The stage loop: Adam, early stopping, diagnostics, resumable state.

**Stages accumulate terms rather than replacing the objective.** The requested
order is phenology, then yield, then LAI, and it has one structural problem:
LAI is causally *upstream* of yield, so moving the canopy in stage 3 de-tunes
what stage 2 fitted. Keeping the earlier stages' losses as retention terms
costs nothing — one forward pass produces all three — and it is what stops
stage 3 silently undoing stage 2:

=====  ==============================  ==========================================
Stage  Freed                           Loss
=====  ==============================  ==========================================
1      phenology                       ``L_phen``
2      yield, phenology frozen         ``L_yield + l1 L_phen``
3      canopy, yield anchored          ``L_lai + l2 L_yield + l1 L_phen``
4      all                             joint fine-tune at a small learning rate
=====  ==============================  ==========================================

**Stage 4 is not optional on this run.** The harmonised run puts the canopy
20-60 % *above* target rather than below it, so stage 3 removes leaf area that
stage 2 was fitting yield through — a strictly larger perturbation than the one
the retention weight was sized for, and running the wrong way.

**Stage 1 moves more than its position suggests.** ``tsumem`` going from 60 to
~200 C.d delays emergence by about a month, which removes a month of early
growth from every season before stage 2 ever runs. Do not size stage 2's
starting point on the pre-stage-1 numbers.

Two diagnostics run **before** any stage, not after, because both catch a
problem that would otherwise be paid for in wall time: a finite-difference
gradient check per free parameter (a zero or non-finite gradient almost always
means a non-differentiable target slipped into the spec), and the Jacobian
condition number and parameter correlation matrix per region (a correlation
above ~0.95 between two free parameters means one should be frozen).
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cropmodelling4eu.calibration.checkpointed import run_trajectories
from cropmodelling4eu.calibration.config import CalibrationConfig, STAGE_PARAMETERS
from cropmodelling4eu.calibration.dataset import (
    CellYearDataset,
    RegionBatch,
    StratifiedFractionSampler,
)
from cropmodelling4eu.calibration.losses import (
    CanopyLoss,
    PhenologyLoss,
    RunningClimatology,
    YieldLoss,
)
from cropmodelling4eu.calibration.observations import ObservationSet
from cropmodelling4eu.calibration.parallel import BatchOutcome, RegionPool, waves
from cropmodelling4eu.calibration.problem import CalibrationProblem, load_stage_specs
from cropmodelling4eu.calibration.regions import CalibrationRegions
from cropmodelling4eu.calibration.splits import Split, make_split
from cropmodelling4eu.config import RunConfig
from cropmodelling4eu.export import ExportBundle
from cropmodelling4eu.torchcrop.run import (
    build_site_params,
    build_soil_params,
    fertilizer_from_dvs,
)

logger = logging.getLogger(__name__)

__all__ = ["Calibrator", "StageResult"]

#: Trajectories each stage's loss reads. Asking for more costs memory in the
#: checkpointed forward and buys nothing.
STAGE_TRAJECTORIES: dict[str, tuple[str, ...]] = {
    "phenology": ("tsump", "dvs_uncapped"),
    "yield": ("tsump", "dvs_uncapped", "wso"),
    "lai": ("tsump", "dvs_uncapped", "wso", "lai", "frac_intercepted"),
}

#: Which observed variables each stage's loss needs present in a batch.
STAGE_VARIABLES: dict[str, tuple[str, ...]] = {
    "phenology": ("emergence_date", "harvest_date", "anthesis_date", "heading_date"),
    "yield": ("yield_level", "yield_anomaly"),
    "lai": ("fapar_peak", "fapar_half_fall_date"),
}

_DATE_VARIABLES = frozenset(
    {"emergence_date", "harvest_date", "anthesis_date", "heading_date",
     "fapar_half_fall_date"}
)


@dataclass(slots=True)
class StageResult:
    """What one stage produced."""

    stage: str
    epochs: int
    best_val: float
    history: pd.DataFrame
    parameters: pd.DataFrame
    blocked: dict[str, str]
    test: dict[str, float] = field(default_factory=dict)


class _Containers:
    """A stand-in exposing the three parameter containers a manager writes into.

    The manager holds the containers, not the model: it writes into
    ``crop_params`` by ``setattr``, and every stage's specs address ``crop.*``.
    So one persistent ``CropParameters`` per region is built here and handed to
    a fresh ``Lintul5Model`` on every batch — the model changes with the batch's
    soil and site, the calibrated crop does not.
    """

    def __init__(self, crop_params: Any, soil_params: Any, site_params: Any) -> None:
        self.crop_params = crop_params
        self.soil_params = soil_params
        self.site_params = site_params


class Calibrator:
    """Drive the staged calibration over a prepared dataset.

    Args:
        config: The run configuration (grid, export, seasons).
        settings: The calibration configuration.
        bundle: The resolved export.
        dataset: The pooled cell-years and their cached weather.
        observations: The pooled observations.
        regions: The cell-to-region map.
        crop_params: The crop the run starts from — the workspace's
            ``crop_<crop>.yaml``, i.e. the SUSTAg ``WW`` block. Deep-copied per
            region, so the regions diverge from one common starting point.
        out_dir: Where checkpoints, logs and the calibrated crop files land.
        crop_file: The workspace crop YAML ``crop_params`` was loaded from. It
            supplies the layout the calibrated files are written on, so each
            one diffs cleanly against the run's own starting point.
        workers: Processes computing region-batches concurrently. ``1`` is the
            exactly-sequential path. The useful ceiling is the region count,
            since a batch carries one region — see
            :mod:`cropmodelling4eu.calibration.parallel`.
    """

    def __init__(
        self,
        config: RunConfig,
        settings: CalibrationConfig,
        bundle: ExportBundle,
        dataset: CellYearDataset,
        observations: ObservationSet,
        regions: CalibrationRegions,
        crop_params: Any,
        out_dir: Path,
        crop_file: Path | None = None,
        workers: int = 1,
    ) -> None:
        self.config = config
        self.settings = settings
        self.bundle = bundle
        self.dataset = dataset
        self.observations = observations
        self.regions = regions
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.crop_file = Path(crop_file) if crop_file else None
        self.workers = max(1, int(workers))
        self.device = torch.device(config.torchcrop.device)
        if config.torchcrop.torch_threads > 0:
            # Measured to change nothing (1 / 4 / 10 / 20 threads are within
            # noise), but it is still set rather than left to torch's default of
            # every core: a run that grabs 80 threads to no benefit is a run
            # that stops nine others from starting.
            torch.set_num_threads(config.torchcrop.torch_threads)

        crop_params.iopt = torch.tensor(float(config.torchcrop.iopt))
        self.region_crops = {
            int(region): copy.deepcopy(crop_params)
            for region in dataset.pool["region_id"].unique()
        }
        self.climatology = RunningClimatology()
        self._observed = self._index_observations()
        self._third_stage = self._regions_with_a_third_stage()
        logger.info(
            "%d of %d regions have a third dated stage (heading or anthesis)",
            len(self._third_stage), len(self.region_crops),
        )
        # A region no observation reaches gets no manager and no calibrated crop
        # file: there is nothing to fit it on. It is named rather than left to
        # be discovered, because its cells will still run — on the *uncalibrated*
        # crop — and a map of the result would show no discontinuity.
        self.unobserved_regions = sorted(
            set(regions.summary["region_id"].astype(int)) - set(self.region_crops)
        )
        if self.unobserved_regions:
            names = regions.summary.set_index("region_id")["region"]
            logger.warning(
                "%d region(s) carry no observation and will not be calibrated: %s",
                len(self.unobserved_regions),
                ", ".join(f"{r} ({names.get(r, r)})" for r in self.unobserved_regions),
            )

    # ------------------------------------------------------------------ #
    # Observations, indexed for the loss
    # ------------------------------------------------------------------ #

    def _index_observations(self) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
        """``(unit, year) -> {variable: (value, weight)}``.

        Dates stay absolute here — the conversion to "days since the window's
        first day" needs the batch's own start date, which is not known until
        the batch is drawn.
        """
        out: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
        for row in self.observations.table.itertuples():
            value = (
                pd.Timestamp(row.value).value
                if row.variable in _DATE_VARIABLES
                else float(row.value)
            )
            out.setdefault((str(row.unit), int(row.year)), {})[row.variable] = (
                value, float(row.weight)
            )
        return out

    def _regions_with_a_third_stage(self) -> set[int]:
        """Regions where heading or anthesis is observed.

        The identifiability rule is *free parameters per region <= independent
        observed stages per region*. With CLMS alone every region has two —
        emergence and harvest — which is what tier 1 is sized for. A region
        that also reaches PEP725 heading gets tier 2 as well, and ``tsum1`` and
        ``tsum2`` are genuinely separated there rather than pinned by the ratio
        prior.
        """
        third = self.observations.table[
            self.observations.table["variable"].isin(("heading_date", "anthesis_date"))
        ]
        if third.empty:
            return set()
        units = set(third["unit"].astype(str))
        pool = self.dataset.pool
        return set(pool.loc[pool["unit"].isin(units), "region_id"].unique().tolist())

    def _targets(self, batch: RegionBatch, variables: tuple[str, ...]) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        """Per-unit-year observed values and weights, aligned to each own window.

        An observed **date** becomes a day index on the batch's shared relative
        axis, using that unit-year's own window start — the batch spans several
        harvest years, so a single origin would misdate every season but one.
        """
        day = 86_400_000_000_000  # nanoseconds, so a date lands on a day index
        starts = batch.start_dates.astype("datetime64[ns]").astype(np.int64)
        values: dict[str, list[float]] = {v: [] for v in variables}
        weights: dict[str, list[float]] = {v: [] for v in variables}
        for i, (unit, year) in enumerate(zip(batch.units, batch.unit_years)):
            observed = self._observed.get((str(unit), int(year)), {})
            for variable in variables:
                value, weight = observed.get(variable, (float("nan"), 0.0))
                if variable in _DATE_VARIABLES and np.isfinite(value):
                    value = (value - starts[i]) / day
                values[variable].append(value)
                weights[variable].append(weight)

        as_tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device)
        return (
            {k: as_tensor(v) for k, v in values.items() if np.isfinite(v).any()},
            {k: as_tensor(v) for k, v in weights.items()},
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _model(self, batch: RegionBatch) -> Any:
        """A model for one batch, on this region's persistent crop parameters."""
        from torchcrop import Lintul5Model

        settings = self.config.torchcrop
        soil = build_soil_params(
            self.bundle.soil.select(batch.ids), settings.rootzone_m,
            settings.profile_bottom_m,
        )
        soil.irri = torch.as_tensor(
            self.bundle.irrigated.reindex(batch.ids).fillna(0).to_numpy(float),
            dtype=torch.float32,
        )
        if settings.iopt == 1:
            soil.irri = torch.ones_like(soil.irri)
        site = build_site_params(
            batch.ids, batch.years, batch.sowing_doy, self.bundle
        )
        return Lintul5Model(
            self.region_crops[batch.region_id],
            soil.to(device=self.device),
            site.to(device=self.device),
        ).to(self.device)

    def _sow_index(self, batch: RegionBatch) -> torch.Tensor:
        """``[B]`` day index at which each cell's sowing latch fires.

        Rebuilt from the engine's own wrapped day-of-year, which is what the
        latch compares against — not from the weather's ``doy`` channel, which
        counts leap days the engine's 365-day wrap does not.
        """
        n_days = batch.weather.shape[1]
        engine_doy = (batch.start_doy - 1 + np.arange(n_days)) % 365 + 1
        sown = engine_doy[None, :] >= batch.sowing_doy[:, None]
        index = np.where(sown.any(axis=1), sown.argmax(axis=1), 0)
        return torch.as_tensor(index, dtype=torch.float32, device=self.device)

    def _forward(self, batch: RegionBatch, stage: str) -> tuple[dict, torch.Tensor]:
        """Run one batch and return its trajectories plus the live ``tsumem``.

        The two-pass fertilizer schedule stays under ``no_grad``: pass 1 runs
        unfertilised only to date the DVS-keyed schedule, and keeping it
        detached means the gradient ignores the derivative of those dates,
        which is defensible because they move discretely anyway.
        """
        from torchcrop import WeatherDriver

        model = self._model(batch)
        weather = WeatherDriver(
            torch.as_tensor(batch.weather, dtype=torch.float32, device=self.device)
        )
        keep = STAGE_TRAJECTORIES[stage]
        segment = self.settings.data.checkpoint_segment

        fertilizer = None
        if self.bundle.plans and self.config.torchcrop.iopt != 1:
            with torch.no_grad():
                pass1 = run_trajectories(
                    model, weather, batch.start_doy, keep=("dvs",), segment=0
                )
            # `fertilizer_from_dvs` expects the trajectory with its leading
            # pre-sowing entry, which `run_trajectories` has already dropped —
            # so one is prepended rather than the schedule landing a day early
            # on every event.
            dvs = torch.cat(
                [torch.zeros_like(pass1["dvs"][:, :1]), pass1["dvs"]], dim=1
            )
            fertilizer = fertilizer_from_dvs(
                batch.ids, dvs, self.bundle.plans
            ).to(self.device)

        trajectories = run_trajectories(
            model, weather, batch.start_doy, fertilizer=fertilizer,
            keep=keep, segment=segment,
        )
        return trajectories, model.crop_params.tsumem

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #

    def _batch_loss(
        self, batch: RegionBatch, stages: tuple[str, ...], weights: dict[str, float]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one batch under an accumulated objective.

        ``batch.units`` is passed to every loss as the climatology key, and it
        carries the **unit alone**, not ``(unit, year)``: the climatology is a
        per-unit reference level *across* seasons, and a batch spanning several
        years is exactly what lets it see them.
        """
        keep_stage = stages[-1]
        trajectories, tsumem = self._forward(batch, keep_stage)
        unit_index = torch.as_tensor(batch.unit_index, device=self.device)
        total = torch.zeros((), device=self.device)
        report: dict[str, float] = {}
        scored = 0

        for stage in stages:
            variables = STAGE_VARIABLES[stage]
            targets, obs_weights = self._targets(batch, variables)
            if not targets:
                continue
            weight = weights.get(stage, 1.0)
            if stage == "phenology":
                loss, info = PhenologyLoss(self.settings.loss, self.climatology)(
                    trajectories, tsumem, targets, unit_index, batch.units,
                    self._sow_index(batch), obs_weights,
                )
            elif stage == "yield":
                loss, info = YieldLoss(self.settings.loss, self.climatology)(
                    trajectories["wso"][:, -1], targets, unit_index, batch.units,
                    obs_weights,
                )
            else:
                loss, info = CanopyLoss(self.settings.loss)(
                    trajectories["frac_intercepted"], targets, unit_index,
                    batch.units, obs_weights,
                )
            total = total + weight * loss
            scored += 1
            report |= {f"{stage}/{k}": v for k, v in info.items()}
        report["scored_stages"] = scored
        return total, report

    def batch_gradients(
        self,
        problem: CalibrationProblem,
        drawn: pd.DataFrame,
        stages: tuple[str, ...],
        weights: dict[str, float],
    ) -> BatchOutcome | None:
        """One batch's gradients, loss and diagnostics — the unit of parallelism.

        Runs in a worker process or in this one, identically. It returns
        **gradients, not a graph**: one float per latent of one region, so what
        crosses a process boundary is a few dozen numbers rather than a
        465-day autograd tape.

        Both penalties are scoped to this batch's region. Summed over every
        region they would be applied once per *batch* rather than once per
        *step*, and every batch would touch every region's latents — which is
        both a bias toward the prior and the reason the batches could not run
        independently.

        Returns:
            The :class:`BatchOutcome`, or ``None`` if the loss was not finite.
        """
        batch = self.dataset.batch(drawn)
        problem.zero_grad(set_to_none=True)
        # Materialized **once** per batch, and the same values are handed to the
        # prior: the forward pass reads the parameter containers, so a second
        # materialize would replace them underneath it and build the penalty on
        # a different graph from the loss.
        values = problem.materialize(batch.region_id)
        loss, report = self._batch_loss(batch, stages, weights)
        penalty = problem.prior_penalty(
            self.settings.loss.prior_l2, batch.region_id, values
        ) + problem.pooling_penalty(
            self.settings.loss.pooling_l2
            if self.settings.optim.pooling == "hierarchical"
            else 0.0,
            batch.region_id,
        )
        if not report.get("scored_stages"):
            # No observation in this batch scores any active stage, so the only
            # gradient available is the prior's. Stepping on it would move the
            # region toward its prior for a batch that carries no evidence at
            # all — the same failure as applying the penalties once per batch,
            # arriving by a different route. `scorable_unit_years` keeps these
            # out of the draw; this is the guard for when one slips through.
            logger.debug(
                "region %d: no active-stage observation in this batch — skipped",
                batch.region_id,
            )
            return None

        total = loss + penalty.to(loss.device)
        if not torch.isfinite(total):
            logger.warning(
                "non-finite loss on region %d (%d unit-years) — skipped",
                batch.region_id, batch.n_units,
            )
            return None
        if total.grad_fn is None:
            # Scored, finite, and yet nothing in it reaches a latent: every
            # segment of the checkpointed day loop came out independent of the
            # parameters (a batch whose seasons never leave the pre-emergence
            # branch does this). `backward` on it raises "element 0 of tensors
            # does not require grad" and kills the run, so it is skipped the
            # same way a non-finite one is. `gradient_check` already guards
            # this; the training path did not.
            logger.warning(
                "region %d: loss carries no gradient path (%d unit-years) — skipped",
                batch.region_id, batch.n_units,
            )
            return None
        total.backward()

        manager = problem.manager(batch.region_id)
        return BatchOutcome(
            region_id=batch.region_id,
            grads={
                name: float(p.grad)
                for name, p in manager._latents.items()
                if p.grad is not None
            },
            loss=float(loss.detach()),
            report=report,
        )

    @torch.no_grad()
    def batch_score(
        self,
        problem: CalibrationProblem,
        drawn: pd.DataFrame,
        stages: tuple[str, ...],
        weights: dict[str, float],
    ) -> BatchOutcome | None:
        """One batch's loss and diagnostics, with no gradient — see :meth:`evaluate`."""
        batch = self.dataset.batch(drawn)
        problem.materialize(batch.region_id)
        loss, report = self._batch_loss(batch, stages, weights)
        if not torch.isfinite(loss):
            return None
        return BatchOutcome(
            region_id=batch.region_id, grads={},
            loss=float(loss), report=report,
        )

    def _apply(
        self,
        problem: CalibrationProblem,
        optimizer: torch.optim.Optimizer,
        outcome: BatchOutcome,
        grad_clip: float,
    ) -> None:
        """Write one region's gradients back and take its Adam step.

        Adam skips parameters whose ``grad`` is ``None``, so setting only this
        region's gradients steps only this region — a region no longer takes
        fifty steps an epoch on prior pull alone, carrying momentum with it.
        """
        problem.zero_grad(set_to_none=True)
        manager = problem.manager(outcome["region_id"])
        touched = []
        for name, value in outcome["grads"].items():
            latent = manager._latents[name]
            latent.grad = torch.full_like(latent, value)
            touched.append(latent)
        if not touched:
            return
        if grad_clip > 0:
            # This region's parameters only: over the whole set the norm is
            # dominated by the 28 regions the batch never touched, so a global
            # clip would scale the one real gradient by the wrong factor.
            torch.nn.utils.clip_grad_norm_(touched, grad_clip)
        optimizer.step()

    def scorable_unit_years(self, stages: tuple[str, ...]) -> pd.DataFrame:
        """The unit-years an objective over ``stages`` can actually score.

        A stage's objective covers its own terms *and* the retention terms of
        the stages before it, but nothing else — so a CyBench-yield-only unit
        contributes no phenology target and a batch drawn from it computes a
        loss with no terms in it. That is a whole 465-day forward pass spent on
        a number that is discarded.

        Filtering also makes the split honest: a "training set" counting
        unit-years the stage cannot score reports a size it does not have.
        """
        variables = {v for stage in stages for v in STAGE_VARIABLES[stage]}
        observed = self.observations.table
        keys = set(
            map(
                tuple,
                observed.loc[
                    observed["variable"].isin(variables), ["unit", "year"]
                ].drop_duplicates().to_numpy(),
            )
        )
        unit_years = self.dataset.unit_years()
        keep = [
            (str(u), int(y)) in keys
            for u, y in zip(unit_years["unit"], unit_years["year"])
        ]
        kept = unit_years[keep].reset_index(drop=True)
        logger.info(
            "%d of %d unit-years carry an observation these stages score (%s)",
            len(kept), len(unit_years), ", ".join(stages),
        )
        if kept.empty:
            raise ValueError(
                f"no unit-year carries any of {sorted(variables)} — the stage "
                "has nothing to fit"
            )
        return kept

    def _retention_weights(self, stage: str) -> dict[str, float]:
        """The stage's own weight of 1 plus the earlier stages' retention terms."""
        order = list(self.settings.stage_order)
        loss = self.settings.loss
        retention = {"phenology": loss.lambda_phenology, "yield": loss.lambda_yield}
        if stage not in order:
            # The joint fine-tune, which is not a stage but every stage: no
            # term is a retention term there, so all of them weigh 1.
            return {name: 1.0 for name in order}
        weights = {stage: 1.0}
        for earlier in order[: order.index(stage)]:
            weights[earlier] = retention.get(earlier, 1.0)
        return weights

    # ------------------------------------------------------------------ #
    # Diagnostics that run before any stage
    # ------------------------------------------------------------------ #

    def report_split_coverage(self, split: Split) -> pd.DataFrame:
        """Which references the training years actually reach.

        A year-block split holds out the *last* block, which is the right shape
        for testing temporal extrapolation — but the references do not share a
        window. CLMS covers 2017-2024 only, and a 70/15/15 split of 25 years
        holds out eight, so on the yield stage the entire CLMS record can land
        in validation and test with none of it left to train on. The retention
        phenology term then trains on PEP725 alone. That is survivable and it
        is not obvious, so it is stated.
        """
        table = self.observations.table
        train_years = set(split.train["year"].astype(int))
        rows = []
        for (variable, source), block in table.groupby(["variable", "source"]):
            years = set(block["year"].astype(int))
            shared = years & train_years
            rows.append(
                {
                    "variable": variable, "source": source,
                    "years": f"{min(years)}-{max(years)}",
                    "train_years": len(shared),
                    "train_rows": int(block["year"].isin(train_years).sum()),
                }
            )
        frame = pd.DataFrame(rows).sort_values(["variable", "source"])
        for row in frame[frame["train_years"] == 0].itertuples():
            logger.warning(
                "%s/%s (%s) has NO training year in this split — it contributes "
                "to validation and test only",
                row.variable, row.source, row.years,
            )
        return frame

    def _diagnostic_draw(self, stage: str, split: Split) -> pd.DataFrame:
        """One training unit-year that actually scores **this** stage.

        The diagnostics score the stage alone, not the accumulated objective,
        so the batch has to carry that stage's own observations. Taking the
        first training row instead picks whatever the sort produced — for the
        yield stage that is a PEP725 station-year, which has no yield at all,
        and the check then dies on a loss with no ``grad_fn``.

        Raises:
            ValueError: If no training unit-year scores the stage — which
                means the stage cannot be fitted, and is worth saying loudly
                here rather than as an autograd error later.
        """
        variables = set(STAGE_VARIABLES[stage])
        observed = self.observations.table
        keys = set(
            map(
                tuple,
                observed.loc[
                    observed["variable"].isin(variables), ["unit", "year"]
                ].drop_duplicates().to_numpy(),
            )
        )
        for start in range(len(split.train)):
            row = split.train.iloc[start : start + 1]
            if (str(row["unit"].iloc[0]), int(row["year"].iloc[0])) in keys:
                return row
        raise ValueError(
            f"no training unit-year carries any of {sorted(variables)}, so "
            f"stage {stage!r} has nothing of its own to fit"
        )

    def gradient_check(
        self, stage: str, problem: CalibrationProblem, batch: RegionBatch,
        epsilon: float = 1e-3,
    ) -> pd.DataFrame:
        """Finite-difference check of every free parameter on one batch.

        A zero or non-finite gradient almost always means a non-differentiable
        target slipped into the spec, or that ``smooth=False`` cut the path the
        parameter needed. Cheap to run and expensive to skip: without it the
        first sign is a stage that trains for an hour and moves nothing.
        """
        manager = problem.manager(batch.region_id)
        stages = (stage,)
        weights = {stage: 1.0}

        problem.zero_grad(set_to_none=True)
        problem.materialize(batch.region_id)
        loss, report = self._batch_loss(batch, stages, weights)
        if not report.get("scored_stages") or loss.grad_fn is None:
            # Nothing to differentiate. Reported rather than raised: the check
            # is a diagnostic, and killing a multi-day run over it would be the
            # wrong trade.
            logger.warning(
                "gradient check skipped for %s: the drawn batch scores no term",
                stage,
            )
            return pd.DataFrame(
                columns=["parameter", "analytic", "numeric", "live", "relative_error"]
            )
        loss.backward()
        analytic = {
            name: (None if p.grad is None else float(p.grad))
            for name, p in manager._latents.items()
        }

        rows = []
        for name, latent in manager._latents.items():
            original = latent.detach().clone()
            with torch.no_grad():
                latent.add_(epsilon)
            problem.materialize(batch.region_id)
            with torch.no_grad():
                up, _ = self._batch_loss(batch, stages, weights)
            with torch.no_grad():
                latent.copy_(original - epsilon)
            problem.materialize(batch.region_id)
            with torch.no_grad():
                down, _ = self._batch_loss(batch, stages, weights)
            with torch.no_grad():
                latent.copy_(original)
            numeric = float((up - down) / (2 * epsilon))
            exact = analytic.get(name)
            # A parameter is judged on the **analytic** gradient. The finite
            # difference is the cross-check, and it has a resolution floor: the
            # loss is in days, so a parameter whose analytic gradient is a
            # fraction of a day per unit latent reads as exactly 0 at this
            # epsilon without being dead.
            live = exact is not None and np.isfinite(exact) and exact != 0.0
            scale = max(abs(exact or 0.0), abs(numeric), 1e-6)
            rows.append(
                {
                    "parameter": name,
                    "analytic": exact,
                    "numeric": numeric,
                    "live": bool(live),
                    "relative_error": abs((exact or 0.0) - numeric) / scale,
                }
            )
        problem.materialize(batch.region_id)

        frame = pd.DataFrame(rows)
        dead = frame.loc[~frame["live"], "parameter"].tolist()
        if dead:
            logger.warning(
                "%d parameter(s) have no gradient at all on this batch: %s — "
                "check the spec for a non-differentiable target",
                len(dead), ", ".join(dead),
            )
        # Disagreement only counts where the finite difference can resolve the
        # gradient in the first place.
        resolvable = frame["live"] & (frame["numeric"].abs() > 1e-3)
        wrong = frame[resolvable & (frame["relative_error"] > 0.2)]
        for row in wrong.itertuples():
            logger.warning(
                "%s: analytic %.4g vs finite-difference %.4g — the gradient does "
                "not describe the loss surface",
                row.parameter, row.analytic, row.numeric,
            )
        return frame

    def identifiability(
        self, stage: str, problem: CalibrationProblem, batch: RegionBatch,
        epsilon: float = 1e-2,
    ) -> pd.DataFrame:
        """Per-unit sensitivity Jacobian, its condition number and correlations.

        A correlation above ~0.95 between two free parameters means the
        observations do not separate them and one should be frozen. This is the
        measurement the ``(tsum1, tsum2)`` ratio prior exists because of — with
        anthesis unobserved, only their sum is well identified — and the one
        that put ``vbase``, ``phottb`` and ``vernrt`` at tier 2 rather than
        tier 1: freed alongside ``versat`` on two dated stages they correlate at
        0.997-0.999.

        It runs on **one batch, so on one region**, which the output's
        ``region_id`` column names. Two free parameters that a region's own
        observations cannot separate is the question this answers; whether some
        other region could separate them is a different one.
        """
        manager = problem.manager(batch.region_id)
        names = list(manager._latents)
        columns = []
        for name in names:
            latent = manager._latents[name]
            original = latent.detach().clone()
            responses = []
            for delta in (epsilon, -epsilon):
                with torch.no_grad():
                    latent.copy_(original + delta)
                problem.materialize(batch.region_id)
                with torch.no_grad():
                    trajectories, tsumem = self._forward(batch, stage)
                responses.append(self._signature(trajectories, tsumem))
            with torch.no_grad():
                latent.copy_(original)
            columns.append((responses[0] - responses[1]) / (2 * epsilon))
        problem.materialize(batch.region_id)

        jacobian = np.column_stack(columns)
        finite = np.isfinite(jacobian).all(axis=1)
        jacobian = jacobian[finite]
        if jacobian.size == 0 or jacobian.shape[0] < 2:
            logger.warning("identifiability: too few finite responses to score")
            return pd.DataFrame(columns=["a", "b", "correlation"])

        scale = np.linalg.norm(jacobian, axis=0)
        scaled = jacobian / np.where(scale > 0, scale, 1.0)
        condition = float(np.linalg.cond(scaled))
        correlation = np.corrcoef(scaled, rowvar=False)
        logger.info(
            "region %d: Jacobian condition number %.1f over %d free parameters",
            batch.region_id, condition, len(names),
        )

        rows = [
            {
                "region_id": batch.region_id,
                "a": names[i], "b": names[j],
                "correlation": float(correlation[i, j]),
            }
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        frame = pd.DataFrame(rows).sort_values(
            "correlation", key=abs, ascending=False
        ).reset_index(drop=True)
        frame.attrs["condition_number"] = condition
        near = frame[frame["correlation"].abs() > 0.95]
        for row in near.itertuples():
            logger.warning(
                "%s and %s correlate at %.3f — the observations do not separate "
                "them; freeze one", row.a, row.b, row.correlation,
            )
        return frame

    @staticmethod
    def _signature(trajectories: dict[str, torch.Tensor], tsumem: torch.Tensor) -> np.ndarray:
        """The quantities the loss reads, as one vector, for the Jacobian."""
        from cropmodelling4eu.torchcrop.run import crossing_day

        parts = [
            crossing_day(trajectories["tsump"], tsumem),
            crossing_day(trajectories["dvs_uncapped"], 1.0),
            crossing_day(trajectories["dvs_uncapped"], 2.0),
        ]
        if "wso" in trajectories:
            parts.append(trajectories["wso"][:, -1])
        if "lai" in trajectories:
            parts.append(trajectories["lai"].max(dim=1).values)
        return torch.cat(parts).detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # The stage loop
    # ------------------------------------------------------------------ #

    def build_problem(self, stage: str, free_blocked: bool = False) -> CalibrationProblem:
        """One :class:`CalibrationProblem` for a stage, tiered per region."""
        spec_path = Path(__file__).parent / STAGE_PARAMETERS[stage]
        base = load_stage_specs(spec_path, max_tier=1, free_blocked=free_blocked)
        extended = load_stage_specs(spec_path, max_tier=2, free_blocked=free_blocked)

        models = {}
        specs_by_region = {}
        for region, crop in self.region_crops.items():
            holder = _Containers(
                crop,
                build_soil_params(
                    self.bundle.soil.select(self.bundle.ids[:1]),
                    self.config.torchcrop.rootzone_m,
                    self.config.torchcrop.profile_bottom_m,
                ),
                build_site_params(
                    self.bundle.ids[:1], self.config.season.years[0],
                    np.array([270]), self.bundle,
                ),
            )
            models[region] = holder
            specs_by_region[region] = (
                extended if region in self._third_stage else base
            )

        # **Per region**, which is what the identifiability rule says: a region
        # with a third dated stage can separate what a region with two cannot,
        # so it frees tier 2 and its neighbours do not. `pooling_penalty` means
        # over the managers carrying a given latent, so a parameter free in
        # eleven regions is pooled across those eleven and absent from the rest
        # — nothing needs every region to share one spec set.
        chosen = {
            region: (extended if region in self._third_stage else base)
            for region in models
        }
        with_third = sorted(self._third_stage & set(models))
        logger.info(
            "tier 2 freed in %d of %d regions (%s); the rest keep tier 1 and "
            "the ratio prior",
            len(with_third), len(models),
            ", ".join(str(r) for r in with_third) or "none",
        )

        # Where anthesis *is* observed the near-singular (tsum1, tsum2)
        # direction is data rather than a prior, so the ratio prior is released
        # there. It stays tight everywhere else, which is the whole reason it
        # exists.
        def ratios(region: int) -> dict:
            tight = chosen[region].ratio_priors
            if region not in self._third_stage:
                return dict(tight)
            return {k: (v[0], v[1] * 5.0) for k, v in tight.items()}

        problem = CalibrationProblem(
            models,
            {r: chosen[r].specs for r in models},
            {r: chosen[r].groups for r in models},
            {r: chosen[r].simplex for r in models},
            base.priors,
            {r: ratios(r) for r in models},
        ).to(self.device)
        problem.blocked = base.blocked  # type: ignore[attr-defined]
        return problem

    def run_stage(
        self,
        stage: str,
        problem: CalibrationProblem,
        split: Split,
        stages: tuple[str, ...] | None = None,
        lr_scale: float = 1.0,
        freeze: tuple[str, ...] = (),
    ) -> StageResult:
        """Fit one stage. ``stages`` overrides the accumulated objective."""
        stages = stages or tuple(
            s for s in self.settings.stage_order
            if self.settings.stage_order.index(s)
            <= self.settings.stage_order.index(stage)
        )
        weights = self._retention_weights(stage)
        optim_settings = self.settings.optim
        for name, parameter in problem.named_parameters():
            if any(f in name for f in freeze):
                parameter.requires_grad_(False)
        trainable = [p for p in problem.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            trainable,
            lr=optim_settings.lr * lr_scale,
            betas=optim_settings.betas,
        )
        sampler = StratifiedFractionSampler(
            split.train, self.settings.data.fraction,
            self.settings.data.batch_size, seed=self.settings.data.seed,
        )

        history: list[dict] = []
        best_val, best_state, since_best = float("inf"), None, 0
        # The fork boundary is where the invariant has to hold: a worker
        # inherits the containers as they stand now and keeps them for the
        # stage. `gradient_check` backwards and `identifiability` materializes
        # outside `no_grad`, both after `build_problem`, so both leave residue
        # here — harmless only by accident, because this stage happens to
        # rewrite its own targets every batch.
        problem.detach_containers()
        pool_context = RegionPool(self, problem, stages, weights, self.workers)
        with pool_context as pool:
          for epoch in range(optim_settings.max_epochs):
            started = time.time()
            batches = sampler.epoch(epoch)
            epoch_loss, n_batches = 0.0, 0
            reports: list[dict] = []
            for wave in waves(batches):
                for outcome in pool.run(wave):
                    self._apply(problem, optimizer, outcome, optim_settings.grad_clip)
                    epoch_loss += outcome["loss"]
                    n_batches += 1
                    reports.append(outcome["report"])

            validation = self.evaluate(problem, split.val, stages, weights, pool)
            record = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, n_batches),
                "val_loss": validation["loss"],
                "batches": n_batches,
                "seconds": time.time() - started,
                **_mean_reports(reports),
                **{f"val/{k}": v for k, v in validation.items() if k != "loss"},
            }
            history.append(record)
            logger.info(
                "%s epoch %d/%d: train %.4f, val %.4f (%d batches, %.1f s)%s",
                stage, epoch, optim_settings.max_epochs - 1,
                record["train_loss"], record["val_loss"], n_batches,
                record["seconds"], _headline(record),
            )

            if validation["loss"] < best_val - 1e-6:
                best_val, since_best = validation["loss"], 0
                best_state = copy.deepcopy(problem.state_dict())
            else:
                since_best += 1
                if since_best >= optim_settings.patience:
                    logger.info("%s: no improvement for %d epochs, stopping",
                                stage, since_best)
                    break

        if best_state is not None:
            problem.load_state_dict(best_state)

        result = StageResult(
            stage=stage,
            epochs=len(history),
            best_val=best_val,
            history=pd.DataFrame(history),
            parameters=problem.parameter_table(),
            blocked=getattr(problem, "blocked", {}),
            test=self.evaluate(problem, split.test, stages, weights),
        )
        self._write_stage(result, problem)
        return result

    def evaluate(
        self,
        problem: CalibrationProblem,
        unit_years: pd.DataFrame,
        stages: tuple[str, ...],
        weights: dict[str, float],
        pool: RegionPool | None = None,
    ) -> dict[str, float]:
        """Mean loss and diagnostics over a held-out set, all of it.

        Scored in **full**, not sampled, so it is more batches than a 20 %
        training draw — which makes it the dominant cost once training is one
        parallel wave. ``pool`` scores them concurrently; without one it runs
        in this process.
        """
        if unit_years.empty:
            return {"loss": float("nan")}
        batches = StratifiedFractionSampler(
            unit_years, 1.0, self.settings.data.batch_size, seed=0
        ).epoch(0)
        outcomes = (
            pool.score(batches) if pool is not None
            else [
                o for drawn in batches
                if (o := self.batch_score(problem, drawn, stages, weights)) is not None
            ]
        )
        if not outcomes:
            return {"loss": float("nan")}
        return {
            "loss": sum(o["loss"] for o in outcomes) / len(outcomes),
            **_mean_reports([o["report"] for o in outcomes]),
        }

    def _write_stage(self, result: StageResult, problem: CalibrationProblem) -> None:
        """Persist a stage: history, parameters, the blocked reasons, the crops."""
        stage_dir = self.out_dir / result.stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        result.history.to_csv(stage_dir / "history.csv", index=False)
        result.parameters.to_csv(stage_dir / "parameters.csv", index=False)
        torch.save(problem.state_dict(), stage_dir / "latents.pt")
        (stage_dir / "summary.json").write_text(
            json.dumps(
                {
                    "stage": result.stage,
                    "epochs": result.epochs,
                    "best_val": result.best_val,
                    "test": result.test,
                    # Carried into the output, not left in the source file: a
                    # stage that could not identify something has to say why.
                    "blocked_parameters": result.blocked,
                    "regions_with_a_third_stage": sorted(self._third_stage),
                    "regions_without_observations": self.unobserved_regions,
                },
                indent=2, default=float,
            )
        )
        self.write_crops(stage_dir / "crops")
        logger.info("%s written to %s", result.stage, stage_dir)

    def write_crops(self, directory: Path) -> None:
        """One ``crop_<region>.yaml`` per region, in torchcrop's own preset layout.

        The output of a calibration is a *runnable crop file*, not a table of
        numbers: the production runner already loads one with
        ``CropParameters(config_file=...)``, so a calibrated region is used by
        pointing at its file rather than by writing new code.
        """
        from cropmodelling4eu.torchcrop.params import write_crop_yaml_from_parameters

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        names = self.regions.summary.set_index("region_id")["region"]
        for region, crop in self.region_crops.items():
            label = str(names.get(region, region))
            write_crop_yaml_from_parameters(
                directory / f"crop_region_{region:02d}.yaml",
                crop,
                template=self.crop_file,
                crop_name=f"{self.config.season.crop}_{label}",
                description=(
                    f"Calibrated for region {region} ({label}) by "
                    f"cropmodelling4eu.calibration"
                ),
            )

    def run(self, free_blocked: bool = False) -> list[StageResult]:
        """Every stage in order, then the joint fine-tune."""
        results: list[StageResult] = []
        for stage in self.settings.stage_order:
            variables = STAGE_VARIABLES[stage]
            available = set(self.observations.variables())
            if not available & set(variables):
                logger.warning(
                    "stage %s skipped: none of %s is in the observation pool",
                    stage, list(variables),
                )
                continue
            problem = self.build_problem(stage, free_blocked)
            # Drawn on the stage's **own** variables, not the accumulated set.
            # Only CyBench carries yield, so a stage-2 draw over the union
            # would spend forward passes on PEP725 station-cells and CLMS-only
            # unit-years that can never score the term being fitted. The
            # retention terms still score whatever the drawn units happen to
            # carry — and they do carry it, because CLMS and CyBench are both
            # reported on `adm_id`, so a CyBench unit-year in 2017-2024 brings
            # its own CLMS phenology with it.
            split = make_split(
                self.scorable_unit_years((stage,)),
                mode="loyo" if stage == "phenology" else "block",
                seed=self.settings.data.seed,
                protect=self.settings.phenology_years,
            )
            self.report_split_coverage(split).to_csv(
                self.out_dir / f"{stage}_split_coverage.csv", index=False
            )
            first = self.dataset.batch(self._diagnostic_draw(stage, split))
            self.gradient_check(stage, problem, first).to_csv(
                self.out_dir / f"{stage}_gradient_check.csv", index=False
            )
            self.identifiability(stage, problem, first).to_csv(
                self.out_dir / f"{stage}_identifiability.csv", index=False
            )
            results.append(self.run_stage(stage, problem, split))

        if self.settings.joint_finetune and results:
            stage = self.settings.stage_order[-1]
            problem = self.build_problem(stage, free_blocked)
            split = make_split(
                self.scorable_unit_years(self.settings.stage_order),
                mode="block", seed=self.settings.data.seed,
                protect=self.settings.phenology_years,
            )
            results.append(
                self.run_stage(
                    "joint", problem, split,
                    stages=self.settings.stage_order,
                    lr_scale=self.settings.joint_lr_scale,
                )
            )
        return results


#: Diagnostics worth a line in the log rather than only a column in the CSV:
#: these are the quantities CALIBRATION.md's target table is written in, so a
#: run stays readable against it while it is still going.
_HEADLINE: tuple[tuple[str, str, str], ...] = (
    ("val/phenology/emergence_date_bias_d", "emergence bias", "{:+.1f} d"),
    ("val/phenology/harvest_date_bias_d", "harvest bias", "{:+.1f} d"),
    ("val/phenology/sowing_to_emergence_d", "sow->emergence", "{:.1f} d"),
    ("val/phenology/never_matured_frac", "never matured", "{:.2%}"),
    ("val/yield/yield_bias_t_ha", "yield bias", "{:+.2f} t/ha"),
    ("val/lai/fapar_peak_rmse", "fAPAR RMSE", "{:.3f}"),
)


def _headline(record: dict) -> str:
    """The target-table diagnostics of one epoch, as a short trailing clause."""
    parts = [
        f"{label} {fmt.format(record[key])}"
        for key, label, fmt in _HEADLINE
        if key in record and np.isfinite(record[key])
    ]
    return f" — {', '.join(parts)}" if parts else ""


def _mean_reports(reports: list[dict]) -> dict[str, float]:
    """Mean of each diagnostic over an epoch's batches, ignoring absent keys."""
    if not reports:
        return {}
    frame = pd.DataFrame(reports)
    return {k: float(v) for k, v in frame.mean(numeric_only=True).items()}
