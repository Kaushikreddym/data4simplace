"""Throughput of torchcrop (CPU and GPU) against SIMPLACE (CPU), per cell-year.

Two sources, one table. **torchcrop** is timed here, on the calibration pool's
real cached seasons and real soil/site, through the same
:meth:`Calibrator.batch_score` / :meth:`Calibrator.batch_gradients` a stage
runs — so a row is the cost of a calibration step, not of a toy forward pass.
**SIMPLACE** cannot run on a GPU and is not re-run: its SLURM logs already
record ``OK lines A-B in XmYs`` per task, and a project line is one cell over
the run's whole year window, so each task is a measured cell-years-per-second.

GPU time is only meaningful after ``torch.cuda.synchronize``: kernels launch
asynchronously, and an unsynchronised timer measures the Python loop that
queued them. :meth:`Calibrator._clock` synchronises.

Output columns: ``model, label, device, mode, batch_size, regions, T,
threads, workers, cell_years, seconds, cell_years_per_s, peak_mem_gb``.
"""

from __future__ import annotations

import json
import logging
import re
import resource
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

__all__ = ["BenchRow", "bench_torchcrop", "simplace_bench_rows", "simplace_log_timings"]

_OK_LINE = re.compile(r"^OK\s+lines\s+(\d+)-(\d+)\s+in\s+(?:(\d+)m)?(\d+)s", re.M)
_WORKERS = re.compile(r"\((\d+) worker\(s\)\)")


@dataclass(slots=True)
class BenchRow:
    """One timed configuration."""

    model: str
    label: str
    device: str
    mode: str
    batch_size: int
    regions: int
    T: int
    threads: int
    workers: int
    cell_years: int
    seconds: float
    cell_years_per_s: float
    peak_mem_gb: float


def _peak_memory_gb(device: torch.device) -> float:
    """Peak allocator memory on CUDA; the process's peak RSS on CPU.

    RSS is a high-water mark for the whole process, so on CPU it only ever
    rises across a sweep — read it as "at most", and run the sweep ascending.
    """
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e9
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _draw(calibrator: Any, unit_years: pd.DataFrame, rows: int, mix: bool) -> pd.DataFrame:
    """The first batch of about ``rows`` cell-years a stage would draw."""
    from cropmodelling4eu.calibration.dataset import StratifiedFractionSampler

    batches = StratifiedFractionSampler(
        unit_years, 1.0, rows, seed=calibrator.settings.data.seed, mix_regions=mix
    ).epoch(0)
    return max(batches, key=lambda b: int(b["n_cells"].sum()))


def bench_torchcrop(
    calibrator: Any,
    stage: str,
    batch_sizes: Iterable[int],
    devices: Iterable[str] = ("cpu",),
    modes: Iterable[str] = ("loader", "forward", "fwd+bwd"),
    mix: bool = True,
    repeats: int = 2,
) -> pd.DataFrame:
    """Time one calibration batch per size, device and mode.

    Args:
        calibrator: A prepared :class:`Calibrator` (``calibration.prepare``).
        stage: Whose objective and trajectories are timed.
        batch_sizes: Target cell-years per batch, ascending. The pool caps it,
            so a row records the size actually run.
        devices: ``cpu`` and/or ``cuda``.
        modes: ``loader`` (cache gather + collate only), ``forward``
            (``batch_score``, no graph) and ``fwd+bwd`` (``batch_gradients``,
            checkpointed).
        mix: Mixed-region batches; needs ``data.start_doy``. ``False`` times
            the region-packed path, whose batch the largest region caps.
        repeats: Timed repetitions after one untimed warm-up; the minimum is
            kept, the standard estimator for a noisy wall clock.
    """
    from cropmodelling4eu.calibration.losses import RunningClimatology

    unit_years = calibrator.scorable_unit_years((stage,))
    stages = tuple(
        s for s in calibrator.settings.stage_order
        if calibrator.settings.stage_order.index(s)
        <= calibrator.settings.stage_order.index(stage)
    ) if stage in calibrator.settings.stage_order else (stage,)
    weights = calibrator._retention_weights(stage)
    threads = torch.get_num_threads()
    rows: list[BenchRow] = []

    for device_name in devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda requested but unavailable — skipped")
            continue
        calibrator.device = device
        problem = calibrator.build_problem(stage).to(device)
        problem.detach_containers()
        for size in batch_sizes:
            drawn = _draw(calibrator, unit_years, int(size), mix)
            batch = calibrator.dataset.batch(drawn)
            n_rows, n_days = batch.weather.shape[:2]
            for mode in modes:
                # A fresh climatology per timing: its EMA would otherwise carry
                # tensors from the previous device into this one.
                calibrator.climatology = RunningClimatology()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

                def once() -> None:
                    match mode:
                        case "loader":
                            calibrator.dataset.batch(drawn)
                        case "forward":
                            calibrator.batch_score(problem, drawn, stages, weights)
                        case "fwd+bwd":
                            calibrator.batch_gradients(problem, drawn, stages, weights)
                        case _:
                            raise ValueError(f"unknown mode {mode!r}")

                once()
                timings = []
                for _ in range(max(1, repeats)):
                    started = calibrator._clock()
                    once()
                    timings.append(calibrator._clock() - started)
                seconds = min(timings)
                rows.append(
                    BenchRow(
                        model="torchcrop", label="mixed" if mix else "region-packed",
                        device=device.type, mode=mode,
                        batch_size=n_rows, regions=len(batch.regions), T=n_days,
                        threads=threads, workers=1, cell_years=n_rows,
                        seconds=seconds, cell_years_per_s=n_rows / seconds,
                        peak_mem_gb=_peak_memory_gb(device),
                    )
                )
                logger.info(
                    "torchcrop %s %-8s B=%5d (%2d regions, T=%d): %.2f s, "
                    "%.0f cell-years/s",
                    device.type, mode, n_rows, len(batch.regions), n_days,
                    seconds, n_rows / seconds,
                )
    return pd.DataFrame([asdict(r) for r in rows])


def simplace_log_timings(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Every completed SIMPLACE task's measured throughput.

    A project line is one cell simulated over ``build.json``'s
    ``start_date``..``end_date``, and each JVM is one thread, so
    ``cells x seasons / seconds / workers`` is cell-years per second per core.

    Args:
        run_dirs: ``<run>/simplace`` directories (holding ``build.json`` and
            ``log/``), or per-region ones holding ``log/`` under a parent run.
    """
    rows: list[dict] = []
    for run_dir in map(Path, run_dirs):
        build = next(
            (p / "build.json" for p in (run_dir, *run_dir.parents)
             if (p / "build.json").is_file()),
            None,
        )
        if build is None:
            logger.warning("%s: no build.json above it — skipped", run_dir)
            continue
        meta = json.loads(build.read_text())
        seasons = (
            pd.Timestamp(meta["end_date"]).year - pd.Timestamp(meta["start_date"]).year
        )
        for log in sorted((run_dir / "log").glob("sp_*.out")):
            text = log.read_text(errors="replace")
            workers = int(m.group(1)) if (m := _WORKERS.search(text)) else 1
            for match in _OK_LINE.finditer(text):
                first, last = int(match.group(1)), int(match.group(2))
                seconds = int(match.group(3) or 0) * 60 + int(match.group(4))
                cells = last - first + 1
                rows.append(
                    {
                        "run": meta.get("run_name", run_dir.name),
                        "source": str(run_dir), "log": log.name,
                        "first_line": first, "last_line": last,
                        "cells": cells, "seasons": seasons, "workers": workers,
                        "seconds": seconds,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["cell_years"] = frame["cells"] * frame["seasons"]
    frame["cell_years_per_s"] = frame["cell_years"] / frame["seconds"]
    frame["cell_years_per_core_s"] = frame["cell_years_per_s"] / frame["workers"]
    return frame


def simplace_bench_rows(timings: pd.DataFrame) -> pd.DataFrame:
    """The SIMPLACE logs summarised into :class:`BenchRow` form, per run."""
    if timings.empty:
        return pd.DataFrame(columns=list(BenchRow.__slots__))
    out = []
    for source, block in timings.groupby("source"):
        out.append(
            BenchRow(
                model="simplace", label=Path(source).name if Path(source).name != "simplace"
                else Path(source).parent.name,
                device="cpu", mode="forward",
                batch_size=int(block["cells"].median()), regions=0, T=0,
                threads=1, workers=int(block["workers"].median()),
                cell_years=int(block["cell_years"].sum()),
                seconds=float(block["seconds"].sum()),
                cell_years_per_s=float(np.median(block["cell_years_per_core_s"])),
                peak_mem_gb=float("nan"),
            )
        )
    return pd.DataFrame([asdict(r) for r in out])
