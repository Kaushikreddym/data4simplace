"""Run a wave of region-batches concurrently, on one node's cores.

**The parallel axis is regions, and its ceiling is the region count.** A batch
carries exactly one region's parameters (a table ordinate cannot be
``[B]``-shaped), and since the penalties were scoped to the batch's region
nothing in a batch touches another region's latents. So the gradients of two
batches from *different* regions are disjoint, and computing them concurrently
gives the same numbers as computing them in sequence.

At batch 768 each region contributes about one batch an epoch, so ~29 batches
can run at once on this domain. A ``compute`` node has 80 cores — **one node
already covers the whole available parallelism**, and a second would idle. This
is deliberately not a multi-node design: there is nothing for the extra nodes
to do.

Why threads would not have worked: the day loop is a Python loop, so it holds
the GIL, and torch's intra-op threading measured flat at 1/4/10/20 threads
because each day's tensors are ``[B]``-shaped. Only separate *processes* help.

**Fork, not spawn, and no pickling of the model.** Workers are forked once per
stage and inherit the calibrator, the problem and the memory-mapped season
cache as they stood then; per wave they receive only the current latent vector
(29 regions x <=18 floats) and return only their region's gradients. Nothing
large crosses a process boundary, and the memmap is file-backed so fork does
not copy it.

**The one approximation.** Within a wave every worker sees the latents as they
were at the wave's start, so the pooling penalty's cross-region mean is that
of the wave start rather than being updated between the steps inside it.
Sequential execution updates it after each batch. The difference is one step of
staleness in a *detached* mean under a penalty weighted 0.1 — and it is the
same synchronous-versus-sequential distinction any data-parallel optimiser
makes. Set ``workers=1`` for the exactly-sequential path.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Any

import pandas as pd
import torch

logger = logging.getLogger(__name__)

__all__ = ["BatchOutcome", "RegionPool", "waves"]

#: Set in each forked worker; never pickled, never sent.
_STATE: dict[str, Any] = {}


class BatchOutcome(dict):
    """``{grads, loss, report, region_id}`` — what one worker sends back.

    A plain dict so it pickles cheaply: the gradients are one float per latent
    of one region, which is the whole point of returning them rather than the
    graph.
    """


def waves(batches: list[pd.DataFrame]) -> list[list[pd.DataFrame]]:
    """Group an epoch's batches so no wave repeats a region.

    Two batches of the *same* region must stay sequential — the second reads
    the parameters the first wrote. Two batches of different regions never
    touch the same latents, so they may run together. Ordering inside a wave is
    irrelevant, which is what makes the result independent of how many workers
    are available.
    """
    remaining = list(batches)
    grouped: list[list[pd.DataFrame]] = []
    while remaining:
        seen: set[int] = set()
        wave, deferred = [], []
        for drawn in remaining:
            region = int(drawn["region_id"].iloc[0])
            (deferred if region in seen else wave).append(drawn)
            seen.add(region)
        grouped.append(wave)
        remaining = deferred
    return grouped


def _initialise(calibrator: Any, problem: Any, stages: tuple, weights: dict) -> None:
    """Runs in the forked child: adopt the inherited objects as this worker's."""
    _STATE.update(
        calibrator=calibrator, problem=problem, stages=stages, weights=weights
    )
    # One thread per worker. The measurement says threading buys nothing, and
    # N workers each spawning N threads is how a node ends up oversubscribed.
    torch.set_num_threads(1)


def _score_batch(payload: tuple) -> BatchOutcome | None:
    """Forward-only scoring in a worker — the validation and test passes.

    Cheaper to parallelise than training and, once training is a single wave,
    the thing that dominates an epoch: a held-out set is scored in full, so it
    is *more* batches than a 20 % training draw, not fewer. No gradients, no
    steps, and no waves — nothing here writes a parameter, so every batch is
    independent of every other regardless of region.
    """
    drawn, latents = payload
    problem = _STATE["problem"]
    problem.load_state_dict(latents)
    return _STATE["calibrator"].batch_score(
        problem, drawn, _STATE["stages"], _STATE["weights"]
    )


def _run_batch(payload: tuple) -> BatchOutcome | None:
    """Compute one batch's gradients in a worker; ``None`` if it was not finite."""
    drawn, latents = payload
    calibrator = _STATE["calibrator"]
    problem = _STATE["problem"]
    problem.load_state_dict(latents)
    return calibrator.batch_gradients(
        problem, drawn, _STATE["stages"], _STATE["weights"]
    )


class RegionPool:
    """A forked worker pool for one stage, or a passthrough when ``workers <= 1``.

    Used as a context manager so the workers die with the stage rather than
    outliving it holding a copy of the season cache.
    """

    def __init__(
        self,
        calibrator: Any,
        problem: Any,
        stages: tuple,
        weights: dict,
        workers: int,
    ) -> None:
        self.calibrator = calibrator
        self.problem = problem
        self.stages = stages
        self.weights = weights
        self.workers = max(1, int(workers))
        self._pool: Any = None

    def __enter__(self) -> "RegionPool":
        if self.workers > 1:
            # Fork so the children inherit the calibrator, the problem and the
            # memmap as they stand now. Spawn would re-import and re-open
            # everything, and would need all of it picklable.
            context = mp.get_context("fork")
            self._pool = context.Pool(
                processes=self.workers,
                initializer=_initialise,
                initargs=(self.calibrator, self.problem, self.stages, self.weights),
            )
            # `sched_getaffinity`, not `cpu_count`: under SLURM the latter
            # reports the node's 80 cores rather than the job's allocation, so
            # a log built on it would tell a 32-core job it had 80 and hide a
            # genuine oversubscription.
            allowed = len(os.sched_getaffinity(0))
            logger.info(
                "region pool: %d forked workers on %d allocated cores",
                self.workers, allowed,
            )
            if self.workers > allowed:
                logger.warning(
                    "%d workers exceed the %d cores this job was given; they "
                    "will contend rather than run in parallel. Submit with "
                    "--cpus-per-task >= %d (CAL_CPUS)",
                    self.workers, allowed, self.workers,
                )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def run(self, wave: list[pd.DataFrame]) -> list[BatchOutcome]:
        """Gradients for one wave of region-disjoint batches."""
        if self._pool is None:
            return [
                outcome
                for drawn in wave
                if (outcome := self.calibrator.batch_gradients(
                    self.problem, drawn, self.stages, self.weights
                )) is not None
            ]
        return self._map(_run_batch, wave)

    def score(self, batches: list[pd.DataFrame]) -> list[BatchOutcome]:
        """Forward-only scores for a held-out set, in any order.

        No waves: scoring writes no parameter, so two batches of the *same*
        region are independent here even though training's are not.
        """
        if self._pool is None:
            return [
                outcome
                for drawn in batches
                if (outcome := self.calibrator.batch_score(
                    self.problem, drawn, self.stages, self.weights
                )) is not None
            ]
        return self._map(_score_batch, batches)

    def _map(self, function: Any, batches: list[pd.DataFrame]) -> list[BatchOutcome]:
        latents = {k: v.detach().clone() for k, v in self.problem.state_dict().items()}
        payloads = [(drawn, latents) for drawn in batches]
        return [o for o in self._pool.map(function, payloads) if o is not None]
