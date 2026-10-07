"""A segmented, gradient-checkpointed forward pass over the day loop.

``Lintul5Model.forward`` retains every per-day ``ModelState``, rate dict and
``DiagnosticState``. At batch 2048 that is 1.7 GB *without* a graph; with
autograd, expect 10-20x. The three mitigations, in order of cheapness, are: cut
the window (done in :mod:`~cropmodelling4eu.calibration.dataset`), shrink the
batch (done in the sampler), and checkpoint the day loop — this module, which
is the only one that is real work.

The construction is the standard one: split ``T`` days into segments of length
``s``, keep only the segment boundaries in memory, and recompute each segment's
interior during the backward pass. Peak activation memory falls from ``O(T)``
to ``O(T/s + s)``, minimised at ``s = sqrt(T)``, for about 1.3x compute. At
``T = 345`` that is ``s = 18`` and a ~19x reduction.

**Only the trajectories the loss reads are kept.** The full ``ModelOutput`` is
what makes the stock forward expensive, and a calibration loss reads four or
five series out of it — the dates come off ``tsump`` and ``dvs``, the canopy
term off ``lai`` and ``frac_intercepted``, the yield term off ``wso``. Asking
for the rest costs memory and buys nothing.

``torch.utils.checkpoint`` passes tensors, not dataclasses, so ``ModelState``
is flattened to a tuple in field order and rebuilt on the other side. The order
comes from ``dataclasses.fields`` rather than being written out, so a state
field added upstream is carried automatically instead of being silently
dropped.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any, Iterable

import torch
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_TRAJECTORIES", "run_trajectories"]

#: What each requested trajectory is read off. ``state`` fields are taken
#: *after* the day's Euler step, so entry ``t`` describes the day weather row
#: ``t`` produced — the same convention ``ModelOutput``'s ``[:, 1:]`` slices
#: give, with the leading initial condition already dropped.
DEFAULT_TRAJECTORIES: tuple[str, ...] = ("tsump", "dvs_uncapped", "lai", "wso")

_DIAGNOSTIC_FIELDS = frozenset({"frac_intercepted", "tranrf", "nni"})
_DERIVED = {"biomass": ("wlv", "wst", "wso")}

#: Trajectories rebuilt from the day's rate and the state *before* the Euler
#: step, as ``{name: (state field, rate key)}``.
#:
#: ``dvs_uncapped`` exists because ``euler_update`` clamps ``dvs`` to
#: ``[0, 2]``, and that clamp **silently destroys the maturity gradient**. On
#: the day the crop crosses DVS 2 the stored value is exactly 2.0, so
#: ``crossing_day``'s interpolation fraction ``(2 - x0) / (x1 - x0)`` is exactly
#: 1.0 — sitting on its own ``clamp(0, 1)`` boundary, where the derivative is
#: zero. The date then comes back as a whole number and ``d(maturity)/d(tsum1)``
#: is 0 analytically while a finite difference measures 0.27 d per 100 C.d.
#: Nothing raises; the stage simply does not train.
#:
#: ``dvs_prev + dvs_rate`` is the value *before* the clamp — identical to the
#: stored trajectory on every other day, since the rate is gated on
#: ``dvs < 2`` — so the interpolation lands strictly inside the interval and
#: the gradient survives. It dips back to 2.0 after the crossing, which
#: ``crossing_day`` is indifferent to: it counts days strictly below the
#: threshold, and 2.0 is not one of them.
_DERIVED_RATE = {"dvs_uncapped": ("dvs", "dvs_rate")}


def _flatten(state: Any) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(state, f.name) for f in fields(state))


def _rebuild(template: Any, values: Iterable[torch.Tensor]) -> Any:
    return type(template)(**{f.name: v for f, v in zip(fields(template), values)})


def _read(
    name: str, previous: Any, state: Any, rates: dict[str, torch.Tensor], diagnostic: Any
) -> torch.Tensor:
    if name in _DIAGNOSTIC_FIELDS:
        return getattr(diagnostic, name)
    if name in _DERIVED:
        return sum(getattr(state, f) for f in _DERIVED[name])
    if name in _DERIVED_RATE:
        field_name, rate_key = _DERIVED_RATE[name]
        return getattr(previous, field_name) + rates[rate_key]
    return getattr(state, name)


def run_trajectories(
    model: Any,
    weather: Any,
    start_doy: int,
    fertilizer: torch.Tensor | None = None,
    irrigation: torch.Tensor | None = None,
    keep: tuple[str, ...] = DEFAULT_TRAJECTORIES,
    segment: int = 18,
    initial_state: Any | None = None,
) -> dict[str, torch.Tensor]:
    """Run the day loop and return only the requested ``[B, T]`` trajectories.

    Args:
        model: A :class:`torchcrop.Lintul5Model`. Its parameter containers must
            already have been written by the calibration manager — this runs
            the model, it does not materialize it.
        weather: A :class:`torchcrop.WeatherDriver`.
        start_doy: Day-of-year of the window's first day, shared by the batch.
        fertilizer: Optional ``[B, T, 3]`` daily N/P/K in g element m-2 d-1.
        irrigation: Optional ``[B, T]`` daily irrigation in mm d-1.
        keep: Trajectory names — ``ModelState`` fields, the ``DiagnosticState``
            fields in :data:`_DIAGNOSTIC_FIELDS`, or ``"biomass"``.
        segment: Days per checkpoint segment; ``0`` disables checkpointing and
            runs the plain loop, which is what a gradient check wants.
        initial_state: Optional pre-built state; ``None`` calls
            ``model.initialize``, which is the SIMPLACE-parity setup.

    Returns:
        ``{name: [B, T]}``, one entry per requested trajectory.

    Raises:
        ValueError: If a requested trajectory is not a readable field.
    """
    from torchcrop.states.model_state import ModelState

    readable = (
        {f.name for f in fields(ModelState)}
        | _DIAGNOSTIC_FIELDS
        | set(_DERIVED)
        | set(_DERIVED_RATE)
    )
    unknown = sorted(set(keep) - readable)
    if unknown:
        raise ValueError(
            f"not a readable trajectory: {unknown}; known: {sorted(readable)}"
        )

    model.crop_params.validate()
    model.soil_params.validate()
    model.site_params.validate()

    n_days = weather.n_days
    state = initial_state or model.initialize(
        batch_size=weather.batch_size,
        dtype=weather.data.dtype,
        device=weather.data.device,
    )
    template = state
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in keep}

    def advance(t0: int, t1: int, *flat: torch.Tensor):
        """Run days ``[t0, t1)`` and return the end state plus the kept series."""
        current = _rebuild(template, flat)
        series: list[torch.Tensor] = []
        for t in range(t0, t1):
            doy = torch.full_like(
                current.dvs, float(((start_doy - 1 + t) % 365) + 1)
            )
            rates, diagnostic = model._compute_rates_dispatch(
                state=current,
                weather_day=weather.day(t),
                doy=doy,
                crop_params=model.crop_params,
                soil_params=model.soil_params,
                site_params=model.site_params,
                irrigation=None if irrigation is None else irrigation[:, t],
                fertilizer=None if fertilizer is None else fertilizer[:, t, :],
            )
            previous, current = current, model.update_state(current, rates, 1.0)
            series += [
                _read(name, previous, current, rates, diagnostic) for name in keep
            ]
        return (*_flatten(current), *series)

    flat = _flatten(state)
    n_state = len(flat)
    step = n_days if segment <= 0 else segment
    for t0 in range(0, n_days, step):
        t1 = min(t0 + step, n_days)
        if segment <= 0 or not torch.is_grad_enabled():
            out = advance(t0, t1, *flat)
        else:
            out = checkpoint(advance, t0, t1, *flat, use_reentrant=False)
        flat, series = out[:n_state], out[n_state:]
        for i, name in enumerate(keep):
            collected[name] += list(series[i :: len(keep)])

    return {name: torch.stack(values, dim=1) for name, values in collected.items()}
