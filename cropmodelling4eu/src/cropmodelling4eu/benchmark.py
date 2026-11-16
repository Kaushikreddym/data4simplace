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
threads, workers, cell_years, seconds, cell_years_per_s, peak_mem_gb``; the
per-pass losses go to a second table in ``batch_history`` form.
"""

from __future__ import annotations

import json
import logging
import re
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

__all__ = ["BenchRow", "bench_production", "bench_torchcrop", "simplace_bench_rows", "simplace_log_timings"]

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


def _children_peak_rss_gb() -> float:
    """Summed peak RSS (``VmHWM``) of this process's live child processes [GB].

    The weather readers of ``torchcrop.io_processes`` are children, and
    ``ru_maxrss`` of RUSAGE_SELF never sees them. Each worker's own high-water
    mark is summed, so it is an upper bound on their joint peak. Linux only; 0
    elsewhere.
    """
    proc = Path("/proc")
    children = [int(pid) for task in (proc / "self" / "task").glob("*")
                for pid in (task / "children").read_text().split()] if proc.is_dir() else []
    total_kb = 0
    for pid in children:
        try:
            status = (proc / str(pid) / "status").read_text()
        except OSError:
            continue
        total_kb += next((int(line.split()[1]) for line in status.splitlines()
                          if line.startswith("VmHWM:")), 0)
    return total_kb / 1e6


def _draw(
    calibrator: Any, unit_years: pd.DataFrame, rows: int, mix: bool, n: int = 1
) -> list[pd.DataFrame]:
    """The ``n`` largest batches of about ``rows`` cell-years a stage would draw."""
    from cropmodelling4eu.calibration.dataset import StratifiedFractionSampler

    batches = StratifiedFractionSampler(
        unit_years, 1.0, rows, seed=calibrator.settings.data.seed, mix_regions=mix
    ).epoch(0)
    return sorted(batches, key=lambda b: -int(b["n_cells"].sum()))[: max(1, n)]


def bench_torchcrop(
    calibrator: Any,
    stage: str,
    batch_sizes: Iterable[int],
    devices: Iterable[str] = ("cpu",),
    modes: Iterable[str] = ("loader", "forward", "fwd+bwd"),
    mix: bool = True,
    repeats: int = 2,
    n_batches: int = 1,
    steps: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time calibration batches per size, device and mode, and log every pass's loss.

    Args:
        calibrator: A prepared :class:`Calibrator` (``calibration.prepare``).
        stage: Whose objective and trajectories are timed.
        batch_sizes: Target cell-years per batch, ascending. The pool caps it,
            so a row records the size actually run.
        devices: ``cpu`` and/or ``cuda``.
        modes: ``loader`` (cache gather + collate only), ``forward``
            (``batch_score``, no graph), ``fwd+bwd`` (``batch_gradients``,
            checkpointed) and ``step`` (``fwd+bwd`` plus the stage's own clipped
            Adam step, so the loss per pass is the optimiser's progress).
        mix: Mixed-region batches; needs ``data.start_doy``. ``False`` times
            the region-packed path, whose batch the largest region caps.
        repeats: Timed passes over each batch after one untimed warm-up. The
            summary keeps the minimum, the standard estimator for a noisy wall
            clock. Without a step every pass must give the same loss, so the
            pass log doubles as a determinism check.
        n_batches: Batches per size, largest first. The summary times the first;
            the rest only add to the pass log.
        steps: Passes over the size's batches in ``step`` mode, each pass one
            Adam step per batch, from a freshly built problem per size so one
            size's steps never leak into the next.

    Returns:
        ``(summary, passes)``: one :class:`BenchRow` per size, device and mode,
        and the ``batch_history`` long format of every pass (``epoch`` is the
        pass, ``split`` is ``bench:<mode>``) plus ``label, mode, size, timed``.
    """
    from cropmodelling4eu.calibration.losses import RunningClimatology

    unit_years = calibrator.scorable_unit_years((stage,))
    stages = tuple(
        s for s in calibrator.settings.stage_order
        if calibrator.settings.stage_order.index(s)
        <= calibrator.settings.stage_order.index(stage)
    ) if stage in calibrator.settings.stage_order else (stage,)
    weights = calibrator._retention_weights(stage)
    optim = calibrator.settings.optim
    threads = torch.get_num_threads()
    label = "mixed" if mix else "region-packed"
    rows: list[BenchRow] = []
    passes: list[dict] = []

    for device_name in devices:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda requested but unavailable — skipped")
            continue
        calibrator.device = device
        problem = calibrator.build_problem(stage).to(device)
        problem.detach_containers()
        for size in batch_sizes:
            drawn = _draw(calibrator, unit_years, int(size), mix, n_batches)
            first = calibrator.dataset.batch(drawn[0])
            n_rows, n_days = first.weather.shape[:2]
            for mode in modes:
                # A fresh climatology per timing: its EMA would otherwise carry
                # tensors from the previous device into this one.
                calibrator.climatology = RunningClimatology()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                target, optimizer = problem, None
                if mode == "step":
                    target = calibrator.build_problem(stage).to(device)
                    target.detach_containers()
                    optimizer = torch.optim.Adam(
                        [p for p in target.parameters() if p.requires_grad],
                        lr=optim.lr, betas=optim.betas,
                    )

                def once(batch: pd.DataFrame) -> Any:
                    match mode:
                        case "loader":
                            calibrator.dataset.batch(batch)
                            return None
                        case "forward":
                            return calibrator.batch_score(target, batch, stages, weights)
                        case "fwd+bwd":
                            return calibrator.batch_gradients(target, batch, stages, weights)
                        case "step":
                            outcome = calibrator.batch_gradients(target, batch, stages, weights)
                            if outcome is not None:
                                calibrator._apply(target, optimizer, outcome, optim.grad_clip)
                            return outcome
                        case _:
                            raise ValueError(f"unknown mode {mode!r}")

                n_pass = max(1, steps) if mode == "step" else 1 + max(1, repeats)
                timings: list[float] = []
                for pass_ in range(n_pass):
                    for index, batch in enumerate(drawn):
                        if mode != "step":
                            # Without a step every pass must score the same, which
                            # only holds if the anomaly climatology starts afresh.
                            calibrator.climatology = RunningClimatology()
                        started = calibrator._clock()
                        outcome = once(batch)
                        seconds = calibrator._clock() - started
                        timed = mode == "step" or pass_ > 0
                        if timed and index == 0:
                            timings.append(seconds)
                        if outcome is None:
                            continue
                        outcome["seconds"] = seconds
                        passes += [
                            r | {"label": label, "mode": mode, "size": int(size), "timed": timed}
                            for r in calibrator._batch_records(
                                outcome, stage, pass_, f"bench:{mode}", index
                            )
                        ]
                        logger.info(
                            "torchcrop %s %-8s B=%5d batch %d pass %d: loss %.4g, %.2f s",
                            device.type, mode, outcome["rows"], index, pass_,
                            outcome["loss"], seconds,
                        )
                seconds = float(np.median(timings) if mode == "step" else min(timings))
                rows.append(
                    BenchRow(
                        model="torchcrop", label=label,
                        device=device.type, mode=mode,
                        batch_size=n_rows, regions=len(first.regions), T=n_days,
                        threads=threads, workers=1, cell_years=n_rows,
                        seconds=seconds, cell_years_per_s=n_rows / seconds,
                        peak_mem_gb=_peak_memory_gb(device),
                    )
                )
                logger.info(
                    "torchcrop %s %-8s B=%5d (%2d regions, T=%d): %.2f s, "
                    "%.0f cell-years/s",
                    device.type, mode, n_rows, len(first.regions), n_days,
                    seconds, n_rows / seconds,
                )
    return pd.DataFrame([asdict(r) for r in rows]), pd.DataFrame(passes)


def bench_production(
    config: Any,
    batch_sizes: Iterable[int],
    device: str = "cpu",
    crop_file: Path | None = None,
    sowing_file: Path | None = None,
    years: tuple[int, int] = (2000, 2024),
    n_batches: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Time the **production** runner (``torchcrop.run``), not the calibrator.

    Same batching (``sowing_plan`` + ``pack_seasons``), weather reader
    (``load_row_block``) and two-pass ``run_batch`` a production shard runs,
    on the default crop unless ``crop_file`` says otherwise. Each batch is
    timed in two parts: ``io`` (reading and windowing the weather) and
    ``model`` (``run_batch``, both passes, no gradient). The first batch of a
    size is kept but flagged ``warmup``.

    Cells are drawn at random from the export, enough to fill ``n_batches``
    batches of the largest size, and the same cells serve every size.

    Returns:
        One row per (size, batch): ``device, batch_size, rows, T, io_processes,
        io_s`` (time the model waited for weather), ``reader_peak_gb`` (the
        readers' summed peak RSS), ``model_s, cell_years_per_s, peak_mem_gb, warmup, error``. A batch that
        runs out of memory is recorded with ``error`` set, not raised.
    """
    from cropmodelling4eu.export import resolve_export
    from cropmodelling4eu.torchcrop.run import pack_seasons, row_blocks, run_batch, sowing_plan
    from cropmodelling4eu.torchcrop.workspace import load_crop_parameters

    settings = config.torchcrop
    if settings.torch_threads > 0:
        torch.set_num_threads(settings.torch_threads)
    bundle = resolve_export(config, require_management=settings.iopt != 1)
    crop = load_crop_parameters(crop_file, config.season.crop) if crop_file else None
    sowing = pd.read_csv(sowing_file) if sowing_file else None
    seasons = list(range(years[0], years[1] + 1))
    sizes = sorted(int(b) for b in batch_sizes)
    wanted = int(np.ceil(n_batches * sizes[-1] / len(seasons) * 1.2))
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(bundle.ids, size=min(wanted, bundle.ids.size), replace=False))
    plan = sowing_plan(ids, seasons, bundle.site.sowing_doy(ids), sowing)
    target = torch.device(device)
    rows: list[dict] = []
    for size in sizes:
        # A fresh cache (and reader pool) per size, so no size reads warm.
        records: dict = {}
        batches = pack_seasons(plan, size)[:n_batches]
        blocks_of = row_blocks(batches, bundle, config, records)
        for index in range(len(batches)):
            if target.type == "cuda":
                torch.cuda.synchronize(target)
                torch.cuda.reset_peak_memory_stats(target)
            model_s = 0.0
            n_rows, days, error = 0, 0, None
            started = time.perf_counter()
            # With io_processes the read overlaps the previous batch's model,
            # so io_s is the time the model *waited* for its weather.
            (batch_ids, batch_years, batch_doy), blocks = next(blocks_of)
            io_s = time.perf_counter() - started
            try:
                for block, weather, start_doy in blocks:
                    started = time.perf_counter()
                    run_batch(batch_ids[block], weather, batch_years[block], start_doy,
                              batch_doy[block], bundle, target, crop)
                    if target.type == "cuda":
                        torch.cuda.synchronize(target)
                    model_s += time.perf_counter() - started
                    n_rows, days = n_rows + len(block), max(days, weather.shape[1])
            except torch.cuda.OutOfMemoryError as exc:
                error = f"OOM: {str(exc).splitlines()[0][:120]}"
                torch.cuda.empty_cache()
            rows.append({
                "model": "torchcrop", "label": "production", "device": target.type,
                "batch_size": int(batch_ids.size), "target": size, "batch": index, "rows": n_rows,
                "T": days, "threads": torch.get_num_threads(), "io_processes": settings.io_processes,
                "io_s": io_s, "model_s": model_s,
                "cell_years_per_s": n_rows / model_s if model_s and not error else np.nan,
                "peak_mem_gb": _peak_memory_gb(target),
                "reader_peak_gb": _children_peak_rss_gb() if settings.io_processes else 0.0,
                "warmup": index == 0, "error": error,
            })
            logger.info("production %s B=%6d batch %d: %d rows, T=%d, io wait %.1f s, model %.1f s%s",
                        target.type, size, index, n_rows, days, io_s, model_s,
                        f" ({error})" if error else "")
            if error:
                break
        blocks_of.close()
    return pd.DataFrame(rows)


def _find_build(run_dir: Path) -> Path | None:
    """The ``build.json`` describing ``run_dir``'s year window.

    A plain SIMPLACE run holds it beside ``log/``; a production run
    (``runs/<crop>/<variant>/<forcing>``) keeps its task logs in ``logs/`` and
    one ``build.json`` per region under ``simplace_regions/``, all sharing the
    run's dates.
    """
    candidates = (
        *(p / "build.json" for p in (run_dir, *run_dir.parents)),
        run_dir / "simplace" / "build.json",
        *sorted(run_dir.glob("simplace_regions/region_*/build.json")),
    )
    return next((p for p in candidates if p.is_file()), None)


def simplace_log_timings(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Every completed SIMPLACE task's measured throughput.

    A project line is one cell simulated over ``build.json``'s
    ``start_date``..``end_date``, and each JVM is one thread, so
    ``cells x seasons / seconds / workers`` is cell-years per second per core.

    Args:
        run_dirs: ``<run>/simplace`` directories (``build.json`` + ``log/``),
            per-region ones under a parent run, or production run directories
            (``logs/`` + ``simplace_regions/region_*/build.json``).
    """
    rows: list[dict] = []
    for run_dir in map(Path, run_dirs):
        build = _find_build(run_dir)
        if build is None:
            logger.warning("%s: no build.json above or below it — skipped", run_dir)
            continue
        meta = json.loads(build.read_text())
        seasons = (
            pd.Timestamp(meta["end_date"]).year - pd.Timestamp(meta["start_date"]).year
        )
        logs = sorted({*run_dir.glob("log/sp_*.out"), *run_dir.glob("logs/sp_*.out")})
        if not logs:
            logger.warning("%s: no sp_*.out task logs in log/ or logs/", run_dir)
        for log in logs:
            text = log.read_text(errors="replace")
            workers = int(m.group(1)) if (m := _WORKERS.search(text)) else 1
            for match in _OK_LINE.finditer(text):
                first, last = int(match.group(1)), int(match.group(2))
                seconds = int(match.group(3) or 0) * 60 + int(match.group(4))
                rows.append(
                    {
                        "run": meta.get("run_name", run_dir.name),
                        "source": str(run_dir), "log": log.name,
                        "first_line": first, "last_line": last,
                        "cells": last - first + 1, "seasons": seasons,
                        "workers": workers, "seconds": seconds,
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
