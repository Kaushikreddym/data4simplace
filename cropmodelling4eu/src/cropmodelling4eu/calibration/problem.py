"""One calibration manager per region, plus the constraints the stock one cannot express.

Three things live here that ``torchcrop.calibration`` does not provide.

**A simplex group.** ``fl + fs + fo = 1`` is a constraint the stock
``ConstraintGroup`` cannot express — it does ordering only. Calibrating the
three independently produces an invalid partitioning that still *runs*, which
is the worst kind of failure: nothing raises and the crop quietly allocates
more than the day's assimilate. :class:`SimplexGroup` is applied as a
post-``materialize`` hook, either deriving the third member from the other two
or rescaling the members back onto the simplex.

**Hierarchical pooling.** One global latent per parameter plus a per-region
deviation under L2 shrinkage, so a data-poor region inherits the global fit
instead of running to its bounds. It is implemented as a penalty on each
region's distance from the cross-region mean rather than as a reparameterised
latent: the two are the same estimator, and the penalty form leaves the stock
:class:`~torchcrop.calibration.manager.CalibrationManager` untouched, which
matters because its bijections are what make every iterate feasible.

**The ``(tsum1, tsum2)`` ratio prior.** At ``idsl = 2`` the split does move
maturity, but weakly — 0.003 d per degree-day reallocated, so a 21 % shift of
the thermal budget moves maturity by two thirds of a day against a CLMS harvest
RMSE of 21 d. Both stay free everywhere; what stops the optimiser wandering
that near-singular direction is a prior on the **ratio**, strong wherever
anthesis is unobserved and released where it is. That is the reason the
anthesis output had to exist before stage 1 could free ``phottb`` or
``vernalisation_devstage``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from torchcrop.calibration import (
    CalibrationManager,
    ConstraintGroup,
    ParameterSpec,
    load_calibration_config,
)
from torchcrop.calibration.paths import (
    parse_path,
    read_value,
    rebuild_table,
    write_scalar,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INIT_MARGIN",
    "CalibrationProblem",
    "RegionManager",
    "SimplexGroup",
    "StageSpecs",
    "load_stage_specs",
]


@dataclass(frozen=True, slots=True)
class SimplexGroup:
    """``sum(members) + derived = 1`` over one table abscissa.

    Attributes:
        members: Spec names calibrated freely, e.g. the ``fltb`` and ``fstb``
            ordinates at one DVS.
        derived: Target path receiving ``1 - sum(members)``. ``None`` where the
            crop file carries no matching row — the SUSTAg ``fotb`` has knots
            at 0.99 and 1.0 only, so at DVS 0.646 there is nothing to write and
            the group degrades to enforcing ``sum(members) <= 1``.
    """

    members: tuple[str, ...]
    derived: str | None = None

    def apply(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Project one materialized value set onto the simplex.

        **Pure**: it adjusts the value dict and writes nothing. Writing here was
        a real bug — ``rebuild_table`` detaches every ordinate *not* in its own
        ``updates``, so a second group rebuilding the same table wiped the
        gradient the first had just established on it. With ``fltb`` and
        ``fstb`` each carrying two groups, three of the four partitioning
        ordinates silently stopped training while a finite difference showed a
        clear response. :meth:`RegionManager.materialize` now writes every
        table once, from the original, with all updates at once.
        """
        total = sum(values[m] for m in self.members)
        if self.derived is None:
            # Rescale rather than clamp: clamping one member would silently
            # decide which organ loses the excess, and the ratio between them
            # is the thing being calibrated.
            scale = torch.where(
                total > 1.0, 1.0 / total.clamp(min=1e-6), torch.ones_like(total)
            )
            for member in self.members:
                values[member] = values[member] * scale
            return values

        values[self.derived] = (1.0 - total).clamp(min=0.0)
        return values


def member_targets(names: Iterable[str], containers: dict[str, Any]) -> dict[str, Any]:
    return {name: parse_path(name, containers) for name in names}


def _write(
    targets: dict[str, Any], values: dict[str, torch.Tensor], containers: dict[str, Any]
) -> None:
    """Write a set of values back through their targets, tables batched."""
    tables: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    for name, target in targets.items():
        if target.is_table:
            tables.setdefault(target.table_key, {})[target.row] = values[name]
        else:
            write_scalar(target, values[name], containers)
    for (container, field), updates in tables.items():
        current = getattr(containers[container], field)
        setattr(containers[container], field, rebuild_table(current, updates))


#: Fraction of a parameter's range an initial value is pushed off its bound by.
#: A latent seeded **at** a bound is frozen, not free: the manager's bijection
#: is ``lo + (hi - lo) sigmoid(z)``, and inverting it at the bound clamps ``z``
#: to about +-13.8, where ``sigma'(z)`` is ~1e-6. The gradient is then
#: numerically zero and the parameter never moves — which matters here because
#: the SUSTAg crop starts *on* several bounds by construction: ``versat`` at 70,
#: the ``vernrt`` plateau at 0 and 1, both ``phottb`` knots at 0 and 1.
INIT_MARGIN: float = 0.02


def _off_the_bound(spec: ParameterSpec, current: float, margin: float) -> ParameterSpec:
    """Return ``spec`` with its init pulled ``margin`` of the range off a bound."""
    if spec.bounds is None or margin <= 0:
        return spec
    lo, hi = spec.bounds
    value = spec.init if spec.init is not None else current
    step = margin * (hi - lo)
    moved = min(max(value, lo + step), hi - step)
    if moved == value:
        return spec
    logger.debug(
        "%s: init %.4g moved to %.4g, off the bound %s", spec.name, value, moved,
        spec.bounds,
    )
    return ParameterSpec(
        name=spec.name, bounds=spec.bounds, kind=spec.kind,
        transform=spec.transform, init=moved, categories=spec.categories,
    )


class RegionManager(CalibrationManager):
    """A :class:`CalibrationManager` that also honours :class:`SimplexGroup`.

    ``materialize`` runs the stock reconstruction first — every bijection, every
    ordered group — and then applies the simplex hook, which is the only order
    that keeps both: the simplex reads materialized values, and the ordered
    groups are what guarantees the members are inside their boxes to begin with.

    It also seeds every latent :data:`INIT_MARGIN` off its bounds, without which
    a parameter starting on one is silently frozen.
    """

    def __init__(
        self,
        model: Any,
        specs: Sequence[ParameterSpec],
        groups: Sequence[ConstraintGroup] | None = None,
        simplex: Sequence[SimplexGroup] | None = None,
        init_margin: float = INIT_MARGIN,
    ) -> None:
        containers = {
            "crop": model.crop_params,
            "soil": model.soil_params,
            "site": model.site_params,
        }
        specs = [
            _off_the_bound(
                spec, read_value(parse_path(spec.name, containers), containers),
                init_margin,
            )
            for spec in specs
        ]
        super().__init__(model, specs, groups)
        self._simplex = list(simplex or [])
        self._derived_targets: dict[str, Any] | None = None
        for group in self._simplex:
            missing = [m for m in group.members if m not in self._specs]
            if missing:
                raise ValueError(f"simplex member(s) {missing} have no ParameterSpec")
        # After the parent, which has already taken its detached
        # `_table_originals` snapshot and seeded the latents through
        # `read_value` — a `float()`, so nothing here moves a seeded value.
        self.detach_containers()

    def detach_containers(self) -> None:
        """Drop any autograd graph left in the parameter containers.

        ``materialize`` writes ``bij.forward(latent)`` into the containers, so
        between it and ``backward`` they legitimately hold graph. The stock
        manager launders only the **tables it targets** (``_table_originals``)
        and has no scalar equivalent, which is fine while one manager owns the
        containers for the process lifetime — and wrong here, because
        ``Calibrator`` keeps one ``CropParameters`` per region alive across
        every stage. A scalar written by stage 1 is still in the container when
        stage 2 forwards through it: the first ``backward`` frees that stale
        bijection's saved tensors and the second raises "backward through the
        graph a second time".

        The predicate is ``grad_fn is not None``, never a blanket ``detach``: a
        container field may legitimately be an ``nn.Parameter``, which is
        torchcrop's direct calibration path
        (``Lintul5Model.learnable_parameter_groups``). Detaching those would
        replace the leaf the caller's optimiser holds with a plain tensor and
        unhook it silently. Only ``materialize`` residue is non-leaf.
        """
        for container in self._containers.values():
            for entry in fields(container):
                value = getattr(container, entry.name)
                if isinstance(value, torch.Tensor) and value.grad_fn is not None:
                    setattr(container, entry.name, value.detach())

    def materialize(self) -> dict[str, torch.Tensor]:
        """Constrained values written into the containers, simplex included.

        The parent's ``materialize`` is *not* called: it writes the tables, and
        writing them twice is what broke the gradients. Instead the parent's
        value computation is reused and the write is done once here, from
        ``_table_originals``, with the simplex adjustments already folded in —
        so every free ordinate of a table survives in one graph.
        """
        values = self._materialize_values()
        for group in self._simplex:
            values = group.apply(values)
        if self._derived_targets is None:
            self._derived_targets = {
                group.derived: parse_path(group.derived, self._containers)
                for group in self._simplex
                if group.derived is not None
            }

        targets = {**self._targets, **self._derived_targets}
        table_updates: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        for name, value in values.items():
            target = targets[name]
            if target.is_table:
                table_updates.setdefault(target.table_key, {})[target.row] = value
            else:
                write_scalar(target, value, self._containers)

        for table_key, updates in table_updates.items():
            container, field = table_key
            original = self._table_originals.get(table_key)
            if original is None:
                original = getattr(self._containers[container], field).detach().clone()
                self._table_originals[table_key] = original
            setattr(self._containers[container], field,
                    rebuild_table(original, updates))
        return values


class CalibrationProblem(nn.Module):
    """The free parameters of every region, and the priors tying them together.

    One :class:`RegionManager` per region — forced, not chosen: a table
    ordinate cannot be ``[B]``-shaped, so a forward pass carries exactly one
    region's parameters (see
    :mod:`~cropmodelling4eu.calibration.regions`).

    Args:
        models: ``{region_id: Lintul5Model}``. Each region needs its own model
            instance because the manager writes into the model's parameter
            containers.
        specs: The stage's free parameters.
        groups: Ordering constraints.
        simplex: Simplex constraints.
        priors: ``{spec name: (value, spread)}`` — the physical prior and the
            scale the L2 pull is measured in. JRC per region for phenology, the
            SIMPLACE ``crop.xml`` value elsewhere.
        ratio_priors: ``{(a, b): (ratio, spread)}`` — a prior on
            ``a / (a + b)``, which is how the near-singular ``(tsum1, tsum2)``
            direction is pinned without freezing either.
    """

    def __init__(
        self,
        models: dict[int, Any],
        specs: Sequence[ParameterSpec] | dict[int, Sequence[ParameterSpec]],
        groups: Sequence[ConstraintGroup] | dict[int, Sequence[ConstraintGroup]] | None = None,
        simplex: Sequence[SimplexGroup] | dict[int, Sequence[SimplexGroup]] | None = None,
        priors: dict[str, tuple[float, float]] | None = None,
        ratio_priors: dict[tuple[str, str], tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()

        def per_region(value: Any, region: int) -> Any:
            """One shared set, or a different one per region."""
            return value.get(region, ()) if isinstance(value, dict) else value

        self.managers = nn.ModuleDict(
            {
                str(region): RegionManager(
                    model,
                    per_region(specs, region),
                    per_region(groups, region),
                    per_region(simplex, region),
                )
                for region, model in sorted(models.items())
            }
        )
        # The union, because regions may free different parameters — the
        # identifiability rule is *per region*, and a region with a third dated
        # stage can separate what a region with two cannot. `pooling_penalty`
        # already means over the managers that carry a given latent, so a
        # parameter free in eleven regions is pooled across those eleven and
        # simply absent from the rest.
        self.spec_names = tuple(
            dict.fromkeys(
                spec.name
                for manager in self.managers.values()
                for spec in manager._specs.values()
            )
        )
        self.priors = priors or {}
        # Per region, because the ratio prior is *released* exactly where
        # anthesis is observed: there the (tsum1, tsum2) split is data, and
        # elsewhere it is the only thing holding a near-singular direction.
        self.ratio_priors = {
            int(region): dict(per_region(ratio_priors or {}, int(region)) or {})
            for region in models
        }
        counts = {
            len(m._specs) for m in self.managers.values()
        }
        logger.info(
            "%d region managers, %d distinct free parameters, %d latents "
            "(%s per region)",
            len(self.managers), len(self.spec_names),
            sum(len(m._latents) for m in self.managers.values()),
            "/".join(str(c) for c in sorted(counts)),
        )

    def manager(self, region_id: int) -> RegionManager:
        key = str(int(region_id))
        if key not in self.managers:
            raise KeyError(
                f"region {region_id} has no manager; it was not in the region set "
                f"the problem was built from ({sorted(self.managers)})"
            )
        return self.managers[key]  # type: ignore[return-value]

    def materialize(self, region_id: int) -> dict[str, torch.Tensor]:
        """Write one region's constrained values into its model. Call per batch."""
        return self.manager(region_id).materialize()

    @torch.no_grad()
    def values(self) -> dict[int, dict[str, float]]:
        """Every region's current physical values, for logging and the output.

        ``no_grad`` is load-bearing, not an optimisation. This wants floats, but
        ``materialize`` **writes the containers** on the way to producing them —
        so with grad enabled the end-of-stage ``parameter_table`` leaves a live
        graph in every region's persistent ``CropParameters``, which the next
        stage then forwards through. See :meth:`RegionManager.detach_containers`.
        """
        return {
            int(region): {k: float(v) for k, v in manager.materialize().items()}
            for region, manager in self.managers.items()
        }

    def detach_containers(self) -> None:
        """Launder every region's containers — see :meth:`RegionManager.detach_containers`."""
        for manager in self.managers.values():
            manager.detach_containers()

    def pooling_penalty(self, weight: float, region_id: int | None = None) -> torch.Tensor:
        """L2 of a region's latent from the cross-region mean.

        The hierarchical model, written as a penalty: the "global latent" is
        the mean and the deviation is what is shrunk.

        ``region_id`` scopes it to **one** region, and a training loop must pass
        it. A batch carries one region, so a penalty summed over all of them
        would be added once per *batch* rather than once per *step* — with 51
        batches over 29 regions that applies each region's shrinkage ~28 times
        for every time its data is seen, and the prior wins by arithmetic
        rather than by evidence. It also makes every batch touch every
        region's latents, which is what stops the batches being independent.

        The mean is detached, so a region is pulled toward where the others
        currently are without dragging them back.
        """
        if weight <= 0 or len(self.managers) < 2:
            return torch.zeros(())
        total = torch.zeros(())
        targets = (
            [self.manager(region_id)] if region_id is not None
            else list(self.managers.values())
        )
        for name in self.spec_names:
            key = _latent_key(name)
            latents = [m._latents[key] for m in self.managers.values() if key in m._latents]
            if len(latents) < 2:
                continue
            mean = torch.stack(latents).mean().detach()
            for manager in targets:
                if key in manager._latents:
                    total = total + (manager._latents[key] - mean) ** 2
        return weight * total / max(1, len(self.spec_names))

    def prior_penalty(
        self,
        weight: float,
        region_id: int | None = None,
        values: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """L2 toward the physical prior, scaled by the prior's own spread.

        Scaling by the spread is what makes one weight serve parameters in
        degree-days and parameters in ``m2 g-1``: a deviation is counted in
        units of how much the prior itself varies across zones, not in the
        parameter's own units.

        ``region_id`` scopes it to one region, for the reason
        :meth:`pooling_penalty` gives: summed over every region it is applied
        once per batch instead of once per step, which multiplies the prior's
        weight by the batches-per-region ratio and quietly freezes the weakly
        identified parameters. It also re-materializes every region on every
        batch, which is pure cost.
        """
        if weight <= 0 or not (self.priors or any(self.ratio_priors.values())):
            return torch.zeros(())
        regions = (
            [int(region_id)] if region_id is not None
            else [int(r) for r in self.managers]
        )
        managers = [self.manager(r) for r in regions]
        total = torch.zeros(())
        for region, manager in zip(regions, managers):
            # `values` is passed in by a training step that has already
            # materialized. Re-materializing here would rewrite the parameter
            # containers *after* the forward pass had read them, so the loss and
            # the penalty would reference two different sets of tensors built
            # from the same latents — one graph for each, per batch.
            current = values if values is not None else manager.materialize()
            for name, (prior, spread) in self.priors.items():
                if name in current and spread > 0:
                    total = total + ((current[name] - prior) / spread) ** 2
            for (a, b), (prior, spread) in self.ratio_priors[region].items():
                if a in current and b in current and spread > 0:
                    ratio = current[a] / (current[a] + current[b]).clamp(min=1e-6)
                    total = total + ((ratio - prior) / spread) ** 2
        return weight * total / max(1, len(managers))

    def parameter_table(self) -> "Any":
        """Every region's values as a tidy frame, which is what gets written out."""
        import pandas as pd

        return pd.DataFrame(
            [
                {"region_id": region, "parameter": name, "value": value}
                for region, values in self.values().items()
                for name, value in values.items()
            ]
        )


def _latent_key(name: str) -> str:
    for ch in ".@[]":
        name = name.replace(ch, "__")
    return name


@dataclass(frozen=True, slots=True)
class StageSpecs:
    """A stage's free parameters and everything that constrains them.

    Attributes:
        specs: The parameters actually freed, after the tier and blocked
            filters.
        groups: Ordering constraints over the survivors.
        simplex: Simplex constraints over the survivors.
        priors: ``{name: (value, spread)}``.
        ratio_priors: ``{(a, b): (ratio, spread)}``.
        tiers: ``{name: tier}`` for every parameter in the file, kept so the
            trainer can report what a region did *not* get to free.
        blocked: ``{name: reason}`` for the parameters excluded, so the reason
            reaches the run's output instead of only its source file.
    """

    specs: list[ParameterSpec]
    groups: list[ConstraintGroup]
    simplex: list[SimplexGroup]
    priors: dict[str, tuple[float, float]]
    ratio_priors: dict[tuple[str, str], tuple[float, float]]
    tiers: dict[str, int]
    blocked: dict[str, str]


def load_stage_specs(
    path: Path,
    max_tier: int = 1,
    free_blocked: bool = False,
) -> StageSpecs:
    """Read a stage's parameter file and apply the tier and blocked filters.

    The ``parameters`` and ``constraints`` blocks are torchcrop's **existing**
    ``load_calibration_config`` schema, unmodified, so a spec file is portable
    to any torchcrop calibration. Three keys are additions this package reads
    itself, and each records a decision rather than a preference:

    ``tier``
        2 means the parameter needs a **third dated stage** in the region —
        PEP725 heading, or torchcrop's anthesis date. The identifiability rule
        is *free parameters per region <= independent observed stages*, and
        with CLMS alone a region has two.
    ``blocked``
        A reason string. The parameter is excluded and the reason is carried
        into the run's output, so a stage that could not identify something
        says why instead of quietly omitting it.
    ``simplex``
        Constraints ``ConstraintGroup`` cannot express; see
        :class:`SimplexGroup`.

    Args:
        path: The stage's YAML.
        max_tier: Highest tier to free. 1 is the default everywhere; the
            trainer raises it per region where a third stage is observed.
        free_blocked: Free the blocked parameters anyway. Deliberately awkward:
            every current blocker is a data problem, not a code one.

    Returns:
        The :class:`StageSpecs`.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    entries = raw.get("parameters") or {}
    tiers = {name: int((opts or {}).get("tier", 1)) for name, opts in entries.items()}
    blocked = {
        name: str((opts or {})["blocked"])
        for name, opts in entries.items()
        if (opts or {}).get("blocked")
    }

    kept = {
        name: opts
        for name, opts in entries.items()
        if tiers[name] <= max_tier and (free_blocked or name not in blocked)
    }
    dropped = sorted(set(entries) - set(kept))
    if dropped:
        logger.info(
            "%d of %d parameters not freed: %s",
            len(dropped), len(entries),
            ", ".join(f"{n} ({blocked.get(n, f'tier {tiers[n]}')})" for n in dropped),
        )

    specs, groups = load_calibration_config({**raw, "parameters": kept})
    names = {s.name for s in specs}
    # A constraint over a parameter that was filtered out is dropped rather
    # than raising: the filters are the identifiability rule doing its job, and
    # an ordering over one surviving member constrains nothing anyway.
    groups = [
        g for g in groups if set(g.members) <= names and len(set(g.members)) >= 2
    ]

    simplex = [
        SimplexGroup(members=tuple(entry["members"]), derived=entry.get("derived"))
        for entry in raw.get("simplex") or []
        if set(entry["members"]) <= names
    ]
    priors = {
        name: (float(entry["value"]), float(entry["spread"]))
        for name, entry in (raw.get("priors") or {}).items()
        if name in names
    }
    ratio_priors = {
        (entry["numerator"], entry["denominator"]): (
            float(entry["value"]), float(entry["spread"])
        )
        for entry in raw.get("ratio_priors") or []
        if {entry["numerator"], entry["denominator"]} <= names
    }
    return StageSpecs(specs, groups, simplex, priors, ratio_priors, tiers, blocked)
