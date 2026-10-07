"""Loss terms, in the units the answer is reported in.

**Dates come off the accumulator by linear interpolation**, through
``cropmodelling4eu.torchcrop.run.crossing_day`` — the same function the
production run dates emergence with, so the loss and the reported column cannot
drift apart. Not an argmax, which quantises to whole days and carries no
gradient; and not a sigmoid-sum date, which is exact only in the sharp limit
and biased at any finite sharpness. The quantity is reported in days, so a
smoothing bias is a bias in the answer.

**Which accumulator, per stage, is not a free choice:**

===================  =================  ==========  ===============================
Observation          Accumulator        Threshold   Why not DVS
===================  =================  ==========  ===============================
emergence            ``tsump``          ``tsumem``  DVS is pinned at 0 until the
                                                    crop emerges, so it carries no
                                                    emergence signal, and its first
                                                    non-zero step is a day late
heading (BBCH 51)    ``dvs_uncapped``   ~0.87       anthesis is BBCH 61 = DVS 1.0
anthesis             ``dvs_uncapped``   1.0
harvest              ``dvs_uncapped``   2.0,        harvest >= maturity
                                        hinged
===================  =================  ==========  ===============================

**``dvs_uncapped``, not ``dvs``.** ``euler_update`` clamps ``dvs`` to
``[0, 2]``, so on the day the crop matures the stored value is exactly 2.0 and
``crossing_day``'s interpolation fraction sits exactly on its own
``clamp(0, 1)`` boundary — where the derivative is zero. The maturity date then
comes back as a whole number and ``d(maturity)/d(tsum1)`` is 0 analytically
while a finite difference measures 0.27 d per 100 C.d. Nothing raises and the
stage simply does not train. See
:mod:`cropmodelling4eu.calibration.checkpointed`.

**Everything is in days since the window's first day**, never day-of-year: that
removes circular arithmetic from the loss entirely — no wrap, no shortest-path
difference. The evaluation notebook still needs the circular machinery because
it pairs two products that share no origin; the loss does not.

**Aggregate-then-compare.** CLMS, CyBench yield and CyBench fAPAR are reported
on an ``adm_id``, so the unit's cells are simulated, area-weighted to the unit
mean, and *then* compared. The weighted mean is linear, so gradients pass
cleanly through it. PEP725 is point-matched to the cell that contains the
station and needs no reduction.

**The anomaly terms are taken against a running per-unit climatology.** A batch
is one year by construction, so a within-unit anomaly cannot be formed inside
it. The climatology is an exponential moving average over batches, held
**detached**: the gradient then flows only through the current year's simulated
date, which is exactly what an anomaly term should move — the level is the
other term's job.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from cropmodelling4eu.calibration.config import LossConfig
from cropmodelling4eu.torchcrop.run import crossing_day

logger = logging.getLogger(__name__)

__all__ = [
    "CanopyLoss",
    "PhenologyLoss",
    "RunningClimatology",
    "YieldLoss",
    "segment_mean",
    "sigmoid_fall_day",
]

#: DVS at PEP725 BBCH 51 (heading). Anthesis is BBCH 61 = DVS 1.0; heading runs
#: a few days ahead of it, which on LINTUL-5's linear generative clock is
#: 0.85-0.90.
HEADING_DVS: float = 0.87


def segment_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    n_segments: int,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean of ``values`` within each segment of ``index``.

    The aggregate-then-compare reduction. NaN entries — a cell whose crop never
    reached the threshold — are dropped from their segment rather than
    poisoning it, and the returned mask says which segments kept anything.

    Args:
        values: ``[B]`` per-cell quantity, possibly carrying NaN.
        index: ``[B]`` segment (unit) index in ``[0, n_segments)``.
        n_segments: Number of units in the batch.
        weights: Optional ``[B]`` per-cell weights; ``None`` is equal weight.

    Returns:
        ``(mean [n_segments], valid [n_segments])``. Empty segments hold 0 and
        are ``False`` in the mask.
    """
    finite = torch.isfinite(values)
    w = torch.ones_like(values) if weights is None else weights
    w = torch.where(finite, w, torch.zeros_like(w))
    clean = torch.where(finite, values, torch.zeros_like(values))

    total = torch.zeros(n_segments, dtype=values.dtype, device=values.device)
    total = total.index_add(0, index, clean * w)
    norm = torch.zeros_like(total).index_add(0, index, w)
    valid = norm > 0
    return total / norm.clamp(min=1e-9), valid


def sigmoid_fall_day(
    series: torch.Tensor, level: torch.Tensor, sharpness: float = 1.0
) -> torch.Tensor:
    """Differentiable day at which a **non-monotone** series falls below ``level``.

    ``crossing_day``'s linear interpolation assumes the series is monotone, and
    fAPAR is not: it rises, plateaus and falls, so "the first day below half the
    peak" would be day 0. The sigmoid sum ``t0 + sum_t sigmoid(k (x_t - level))``
    counts the days spent above the level instead, which is well defined on a
    hump and differentiable everywhere.

    It carries the smoothing bias ``crossing_day`` exists to avoid, so it is
    used **only** here — the one target that is not monotone.

    Args:
        series: ``[B, T]`` trajectory.
        level: ``[B]`` threshold, typically half the peak.
        sharpness: Sigmoid steepness ``k``; larger is sharper and stiffer.

    Returns:
        ``[B]`` fractional day index.
    """
    above = torch.sigmoid(sharpness * (series - level.unsqueeze(1)))
    return above.sum(dim=1)


@dataclass(slots=True)
class RunningClimatology:
    """Per-unit means of a simulated quantity, as a detached EMA over batches.

    Seeded on first sight and updated with ``momentum``, so a unit's
    climatology is available from its second appearance. Detached by
    construction: it is a *reference level*, and letting the gradient reach it
    would turn the anomaly term back into a second level term.
    """

    momentum: float = 0.1
    values: dict[tuple[str, str], torch.Tensor] = field(default_factory=dict)

    def update(
        self, variable: str, units: np.ndarray, values: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Fold this batch in and return the climatology for its units.

        Returns:
            ``[n_units]`` reference level, NaN where a unit has never been seen
            with a valid value.
        """
        detached = values.detach()
        out = torch.full_like(detached, float("nan"))

        # Group by unit first. A batch spans several years, so one unit appears
        # in it more than once, and every season of that unit must be scored
        # against the *same* level: updating as we went would compare each
        # season to a reference that had already absorbed the earlier ones, and
        # the anomaly would depend on position in the batch.
        entries: dict[tuple[str, str], list[int]] = {}
        for i, unit in enumerate(units):
            if bool(valid[i]):
                entries.setdefault((variable, str(unit)), []).append(i)

        for i, unit in enumerate(units):
            key = (variable, str(unit))
            if key in self.values:
                out[i] = self.values[key]
            elif key in entries:
                # First sighting: the mean of the unit's seasons *in this batch*
                # is the natural first estimate of its climatology, and it is
                # the same for every one of them.
                out[i] = detached[entries[key]].mean()

        for key, rows in entries.items():
            batch_mean = detached[rows].mean()
            current = self.values.get(key)
            self.values[key] = (
                batch_mean
                if current is None
                else (1.0 - self.momentum) * current + self.momentum * batch_mean
            )
        return out

    def state_dict(self) -> dict[str, float]:
        return {f"{v}|{u}": float(t) for (v, u), t in self.values.items()}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.values = {
            tuple(key.split("|", 1)): torch.tensor(value)  # type: ignore[misc]
            for key, value in state.items()
        }


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Weighted mean squared error over the valid entries, and their count."""
    mask = valid & torch.isfinite(target) & torch.isfinite(prediction)
    n = int(mask.sum())
    if n == 0:
        return prediction.sum() * 0.0, 0
    w = torch.ones_like(target) if weights is None else weights
    w = w[mask]
    error = (prediction[mask] - target[mask]) ** 2
    return (error * w).sum() / w.sum().clamp(min=1e-9), n


def _masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    delta: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    mask = valid & torch.isfinite(target) & torch.isfinite(prediction)
    n = int(mask.sum())
    if n == 0:
        return prediction.sum() * 0.0, 0
    w = torch.ones_like(target) if weights is None else weights
    w = w[mask]
    loss = torch.nn.functional.huber_loss(
        prediction[mask], target[mask], reduction="none", delta=delta
    )
    return (loss * w).sum() / w.sum().clamp(min=1e-9), n


class PhenologyLoss:
    """Stage 1: emergence, heading/anthesis and harvest, in days.

    The three terms are **not** equally weighted, and that is measured rather
    than stylistic. CLMS emergence is only +6 d from ground truth and its
    spatial gradient is sound, so its *level* carries full weight — but its
    interannual scatter is 1.7x the ground observation's and correlates with it
    at r 0.11, so its *anomaly* is weighted to near zero and the anomaly signal
    is taken from PEP725 where stations exist. CLMS harvest needs no such
    discount: anomaly r 0.54-0.58 against PEP725.

    Season length is deliberately absent: it is the difference of the emergence
    and harvest terms and adds no independent information.
    """

    def __init__(self, settings: LossConfig, climatology: RunningClimatology) -> None:
        self.settings = settings
        self.climatology = climatology

    def __call__(
        self,
        trajectories: dict[str, torch.Tensor],
        tsumem: torch.Tensor,
        targets: dict[str, torch.Tensor],
        unit_index: torch.Tensor,
        units: np.ndarray,
        sow_index: torch.Tensor,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one batch.

        Args:
            trajectories: ``tsump`` and ``dvs_uncapped``, ``[B, T]`` each. The
                **uncapped** DVS, not the stored one: ``euler_update`` clamps
                ``dvs`` at 2, which puts ``crossing_day``'s interpolation
                fraction exactly on its own clamp boundary and silently zeroes
                the maturity gradient. See
                :mod:`cropmodelling4eu.calibration.checkpointed`.
            tsumem: The batch's emergence threshold — the **materialized**
                value, so the gradient reaches ``tsumem`` through the
                threshold, which is the only path it has at ``smooth=False``.
            targets: ``{variable: [n_units]}`` observed day index within the
                window, NaN where unobserved.
            unit_index: ``[B]`` cell -> unit map.
            units: ``[n_units]`` unit identifiers, for the climatology.
            sow_index: ``[B]`` day index of each cell's own sowing latch, used
                only for the reported sowing-to-emergence lag.
            weights: Optional ``{variable: [n_units]}`` observation weights.

        Returns:
            ``(loss, diagnostics)``.
        """
        n_units = len(units)
        weights = weights or {}
        parts: dict[str, torch.Tensor] = {}
        report: dict[str, float] = {}

        dvs = trajectories["dvs_uncapped"]
        simulated = {
            "emergence_date": crossing_day(trajectories["tsump"], tsumem),
            "anthesis_date": crossing_day(dvs, 1.0),
            "heading_date": crossing_day(dvs, HEADING_DVS),
            "harvest_date": crossing_day(dvs, 2.0),
        }
        means = {
            name: segment_mean(value, unit_index, n_units)
            for name, value in simulated.items()
        }

        term_weights = {
            "emergence_date": self.settings.emergence_level,
            "anthesis_date": self.settings.anthesis_level,
            "heading_date": self.settings.anthesis_level,
            "harvest_date": self.settings.harvest_level,
        }
        for name, weight in term_weights.items():
            if weight <= 0 or name not in targets:
                continue
            value, valid = means[name]
            loss, n = _masked_mse(value, targets[name], valid, weights.get(name))
            if n:
                parts[f"{name}_level"] = weight * loss
                report[f"{name}_rmse_d"] = float(loss.detach().sqrt())
                report[f"{name}_bias_d"] = float(
                    (value - targets[name])[valid & torch.isfinite(targets[name])]
                    .detach().mean()
                )
                report[f"{name}_n"] = n

        for name, weight in (
            ("emergence_date", self.settings.emergence_anomaly_clms),
            ("harvest_date", self.settings.harvest_anomaly),
        ):
            if weight <= 0 or name not in targets:
                continue
            value, valid = means[name]
            reference = self.climatology.update(name, units, value, valid)
            observed_reference = self.climatology.update(
                f"obs_{name}", units, targets[name], torch.isfinite(targets[name])
            )
            loss, n = _masked_mse(
                value - reference,
                targets[name] - observed_reference,
                valid & torch.isfinite(reference) & torch.isfinite(observed_reference),
                weights.get(name),
            )
            if n:
                parts[f"{name}_anomaly"] = weight * loss
                report[f"{name}_anomaly_rmse_d"] = float(loss.detach().sqrt())

        # Harvest >= maturity is physics, not a fit: penalise maturity landing
        # *after* the observed harvest hard, and maturing more than
        # `drydown_days` before it only mildly -- an early maturity followed by
        # drydown is an ordinary season.
        if "harvest_date" in targets and self.settings.maturity_after_harvest > 0:
            maturity, valid = means["harvest_date"]
            mask = valid & torch.isfinite(targets["harvest_date"])
            if bool(mask.any()):
                excess = (maturity - targets["harvest_date"])[mask].clamp(min=0.0)
                early = (
                    targets["harvest_date"] - maturity - self.settings.drydown_days
                )[mask].clamp(min=0.0)
                parts["maturity_hinge"] = self.settings.maturity_after_harvest * (
                    (excess**2).mean() + 0.1 * (early**2).mean()
                )
                report["maturity_after_harvest_d"] = float(excess.detach().mean())

        emergence_lag = (simulated["emergence_date"] - sow_index)
        report["sowing_to_emergence_d"] = float(
            emergence_lag[torch.isfinite(emergence_lag)].detach().mean()
            if bool(torch.isfinite(emergence_lag).any())
            else float("nan")
        )
        report["never_matured_frac"] = float(
            (~torch.isfinite(simulated["harvest_date"])).float().mean()
        )

        total = sum(parts.values()) if parts else dvs.sum() * 0.0
        report["n_terms"] = len(parts)
        return total, report


class YieldLoss:
    """Stage 2: the level per region, and a Huber anomaly.

    ``L = w_lvl (mean_sim - mean_obs)^2 + w_ano Huber(anom_sim, anom_obs)``.

    The level term is **per region, never pooled**: the bias is three regimes,
    not one — Mediterranean near zero, north-west uniformly low, Baltic
    uniformly high — and a pooled term averages them into a number that moves
    nothing where it matters. One region per batch makes that automatic.

    The observation is detrended per unit before it arrives (see
    :mod:`~cropmodelling4eu.calibration.observations`): CyBench carries a
    technology trend LINTUL-5 structurally cannot produce, and fitting raw
    levels lets the parameters absorb it.
    """

    #: g DM m-2 to t/ha. `wso` is dry matter and CyBench is fresh weight at
    #: standard moisture, so the observation is routed through
    #: `evaluation.aggregate.to_dry_matter` before it reaches here.
    G_M2_TO_T_HA: float = 0.01

    def __init__(self, settings: LossConfig, climatology: RunningClimatology) -> None:
        self.settings = settings
        self.climatology = climatology

    def __call__(
        self,
        wso: torch.Tensor,
        targets: dict[str, torch.Tensor],
        unit_index: torch.Tensor,
        units: np.ndarray,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one batch. ``wso`` is ``[B]`` final storage-organ dry weight."""
        weights = weights or {}
        simulated = wso * self.G_M2_TO_T_HA
        value, valid = segment_mean(simulated, unit_index, len(units))

        parts: dict[str, torch.Tensor] = {}
        report: dict[str, float] = {}

        if "yield_level" in targets and self.settings.yield_level > 0:
            loss, n = _masked_mse(
                value, targets["yield_level"], valid, weights.get("yield_level")
            )
            if n:
                parts["yield_level"] = self.settings.yield_level * loss
                report["yield_bias_t_ha"] = float(
                    (value - targets["yield_level"])[
                        valid & torch.isfinite(targets["yield_level"])
                    ].detach().mean()
                )
                report["yield_n"] = n

        if "yield_anomaly" in targets and self.settings.yield_anomaly > 0:
            reference = self.climatology.update("yield", units, value, valid)
            loss, n = _masked_huber(
                value - reference,
                targets["yield_anomaly"],
                valid & torch.isfinite(reference),
                self.settings.yield_huber_delta,
                weights.get("yield_anomaly"),
            )
            if n:
                parts["yield_anomaly"] = self.settings.yield_anomaly * loss
                report["yield_anomaly_huber"] = float(loss.detach())

        total = sum(parts.values()) if parts else wso.sum() * 0.0
        return total, report


class CanopyLoss:
    """Stage 3: fAPAR, compared **in fAPAR space**.

    ``DiagnosticState.frac_intercepted`` *is* the model's own
    ``1 - exp(-k.LAI)``, so the observation is never inverted into LAI and
    ``k`` never has to be guessed. Two consequences the parameter spec carries:
    ``kdiftb`` is frozen — against fAPAR, ``k`` and ``LAI`` are exactly
    degenerate, and calibrating both fits the observation while getting the
    canopy wrong — and the loss is weighted toward **shape**, because fAPAR
    saturates above LAI ~ 3 and says far more about timing than about the
    plateau.
    """

    def __init__(self, settings: LossConfig) -> None:
        self.settings = settings

    def __call__(
        self,
        fapar: torch.Tensor,
        targets: dict[str, torch.Tensor],
        unit_index: torch.Tensor,
        units: np.ndarray,
        weights: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Score one batch. ``fapar`` is the ``[B, T]`` interception trajectory."""
        weights = weights or {}
        parts: dict[str, torch.Tensor] = {}
        report: dict[str, float] = {}

        peak = fapar.max(dim=1).values
        peak_unit, valid = segment_mean(peak, unit_index, len(units))
        if "fapar_peak" in targets and self.settings.canopy_level > 0:
            loss, n = _masked_mse(
                peak_unit, targets["fapar_peak"], valid, weights.get("fapar_peak")
            )
            if n:
                parts["fapar_peak"] = self.settings.canopy_level * loss
                report["fapar_peak_rmse"] = float(loss.detach().sqrt())
                report["fapar_n"] = n

        if (
            "fapar_half_fall_date" in targets
            and self.settings.canopy_senescence > 0
        ):
            fall = sigmoid_fall_day(fapar, 0.5 * peak.detach())
            fall_unit, fall_valid = segment_mean(fall, unit_index, len(units))
            loss, n = _masked_mse(
                fall_unit,
                targets["fapar_half_fall_date"],
                fall_valid,
                weights.get("fapar_half_fall_date"),
            )
            if n:
                parts["fapar_senescence"] = self.settings.canopy_senescence * loss
                report["fapar_senescence_rmse_d"] = float(loss.detach().sqrt())

        total = sum(parts.values()) if parts else fapar.sum() * 0.0
        return total, report
