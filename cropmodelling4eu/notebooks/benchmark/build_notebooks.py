"""Generate ``benchmark.ipynb`` and ``calibration_loss.ipynb``.

Written from source rather than by hand so the cells stay diffable and
re-generatable; run this once and keep the ``.ipynb`` files beside it.

Inputs:

* ``benchmark.parquet`` from ``cm4eu benchmark`` — a CPU job and a GPU job
  appended into one file (``--append``), plus SIMPLACE's task logs
  (``--simplace-runs``).
* A calibration run's ``<stage>/batch_history.parquet`` (written by every
  ``cm4eu calibrate`` since mixed-region batching), its ``pool.parquet``,
  ``observations.parquet``, ``<stage>_split.parquet`` and ``regions.parquet``,
  plus SIMPLACE outputs on the default and the calibrated crops.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

MD = lambda text: nbf.v4.new_markdown_cell(text.strip("\n"))
CODE = lambda text: nbf.v4.new_code_cell(text.strip("\n"))

SETUP = """
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cropmodelling4eu.evaluation.style import (
    DIVERGING_CMAP, SEQUENTIAL_CMAP, diverging_norm, use_style,
)

logging.basicConfig(level=logging.WARNING, force=True)
P = use_style("light")
RUN = Path("/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/winter_wheat_2000_2024_euval")
"""

# --------------------------------------------------------------------------- #
# benchmark.ipynb
# --------------------------------------------------------------------------- #

benchmark = [
MD("""
# Throughput: torchcrop on CPU and GPU, SIMPLACE on CPU

**Unit: cell-years per second.** One cell simulated through one season. It is
the only unit both models share: SIMPLACE runs a project line (one cell, every
season from 1979 to 2024) per JVM thread, and torchcrop runs a `[B, T]` batch of
cell-seasons per forward pass.

| Source | How it is measured | Device |
|---|---|---|
| torchcrop | `cm4eu benchmark` times real calibration batches: cached pool seasons, real soil and site, through `Calibrator.batch_score` (*forward*) and `batch_gradients` (*fwd+bwd*, checkpointed). Minimum of the repeats after one warm-up; GPU timers synchronise first. | CPU, GPU |
| SIMPLACE | Not re-run. Each SLURM task logs `OK lines A-B in XmYs`, and lines × seasons ÷ seconds ÷ JVMs is the per-core rate. | CPU only (Java) |

SIMPLACE cannot run on a GPU, so it appears as a **per-core band**, not a
curve. torchcrop's CPU throughput is also per process: its day loop is a Python
loop, so threads add nothing (measured flat at 1, 4, 10 and 20) and only
processes scale. That makes "per core" the fair comparison on CPU, and a whole
GPU against one core the honest framing on the GPU side.
"""),
MD("## 0. Setup"),
CODE(SETUP + """
#: `cm4eu benchmark --output ... [--append]` from the CPU and the GPU job.
BENCH = RUN / "benchmark" / "benchmark.parquet"
#: Per-task SIMPLACE timings the same command wrote beside it.
TASKS = BENCH.with_name(BENCH.stem + "_simplace_tasks.parquet")
#: Cell-years one calibration epoch simulates: pool rows x data.fraction.
#: Read off the run's log ("pool: N (unit, year, cell) rows"); EU_val phenology.
EPOCH_CELL_YEARS = int(73_562 * 0.20)
#: Forked region workers on the CPU path (CAL_WORKERS).
CPU_WORKERS = 29

bench = pd.read_parquet(BENCH)
tasks = pd.read_parquet(TASKS) if TASKS.is_file() else pd.DataFrame()
torchcrop = bench[bench["model"] == "torchcrop"].copy()
simplace = bench[bench["model"] == "simplace"].copy()

#: Colour follows the entity, never its rank: fixed slots, never cycled.
COLOUR = {"torchcrop cpu": P["series_1"], "torchcrop cuda": P["series_2"],
          "simplace": P["series_3"]}
bench.groupby(["model", "label", "device", "mode"]).size().rename("rows").to_frame()
"""),

MD("""
## 1. Throughput against batch size

Solid lines are mixed-region batches; dashed lines, where present, are the
region-packed path, whose batch the largest region caps. The grey band is
SIMPLACE's per-core rate across every logged task (5th to 95th percentile),
with the median line.
"""),
CODE("""
modes = [m for m in ("forward", "fwd+bwd") if m in set(torchcrop["mode"])]
fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 3.8), sharey=True,
                         squeeze=False)
for ax, mode in zip(axes[0], modes):
    block = torchcrop[torchcrop["mode"] == mode]
    for (device, label), line in block.groupby(["device", "label"]):
        line = line.sort_values("batch_size")
        key = f"torchcrop {device}"
        ax.plot(line["batch_size"], line["cell_years_per_s"], marker="o", ms=4, lw=2,
                ls="-" if label == "mixed" else "--", color=COLOUR[key],
                label=f"{key} ({label})")
        last = line.iloc[-1]
        ax.annotate(f"{last['cell_years_per_s']:,.0f}", (last["batch_size"], last["cell_years_per_s"]),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    color=P["secondary"], fontsize=8)
    if not tasks.empty:
        lo, mid, hi = tasks["cell_years_per_core_s"].quantile([0.05, 0.5, 0.95])
        ax.axhspan(lo, hi, color=P["grid"], zorder=0, label="SIMPLACE per core (p5-p95)")
        ax.axhline(mid, color=COLOUR["simplace"], lw=2, label=f"SIMPLACE median {mid:.1f}")
    ax.set(xscale="log", yscale="log", xlabel="batch size [cell-years]", title=mode)
axes[0][0].set_ylabel("cell-years per second")
axes[0][-1].legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout()
"""),

MD("""
## 2. Peak memory

On the GPU this is `torch.cuda.max_memory_allocated`, reset per timing. On the
CPU it is the process's peak RSS, a high-water mark that only rises across a
sweep. So the CPU line is an upper bound, and the sweep runs in ascending order
for that reason.
"""),
CODE("""
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for (device, mode), line in torchcrop[torchcrop["label"] == "mixed"].groupby(["device", "mode"]):
    if mode == "loader":
        continue
    line = line.sort_values("batch_size")
    ax.plot(line["batch_size"], line["peak_mem_gb"], marker="o", ms=4, lw=2,
            ls="-" if mode == "fwd+bwd" else ":", color=COLOUR[f"torchcrop {device}"],
            label=f"{device} {mode}")
ax.set(xscale="log", xlabel="batch size [cell-years]", ylabel="peak memory [GB]")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
"""),

MD("""
## 3. What one calibration epoch costs

This is an estimate, not a measurement: `EPOCH_CELL_YEARS` divided by each
path's best fwd+bwd throughput. The CPU path is the forked pool: one process per
region, `CPU_WORKERS` at once, each at the region-packed per-process rate. The
GPU path is one process running mixed batches. Validation is excluded on both
sides, as it scales the same way. The forked bar needs the region-packed CPU
sweep (`--region-packed`), which `submit_benchmark.sh` runs.
"""),
CODE("""
step = torchcrop[torchcrop["mode"] == "fwd+bwd"]
best = step.groupby(["device", "label"])["cell_years_per_s"].max()
paths = {}
# The fork pool runs region-packed batches, so it is scaled from that rate only;
# multiplying the mixed rate by the worker count would credit it batches it
# cannot form.
if ("cpu", "region-packed") in best:
    paths[f"CPU, {CPU_WORKERS} forked workers, region-packed"] = (
        EPOCH_CELL_YEARS / (best["cpu", "region-packed"] * CPU_WORKERS))
if ("cpu", "mixed") in best:
    paths["CPU, 1 process, mixed"] = EPOCH_CELL_YEARS / best["cpu", "mixed"]
if ("cuda", "mixed") in best:
    paths["GPU, 1 process, mixed"] = EPOCH_CELL_YEARS / best["cuda", "mixed"]

epoch = pd.Series(paths, name="minutes").div(60).sort_values()
fig, ax = plt.subplots(figsize=(5.6, 0.5 + 0.45 * len(epoch)))
colours = [COLOUR["torchcrop cuda"] if "GPU" in k else COLOUR["torchcrop cpu"] for k in epoch.index]
ax.barh(epoch.index, epoch.values, color=colours, height=0.6)
for y, v in enumerate(epoch.values):
    ax.text(v, y, f"  {v:.1f} min", va="center", color=P["secondary"], fontsize=8)
ax.set(xlabel=f"minutes per epoch ({EPOCH_CELL_YEARS:,} cell-years, fwd+bwd)")
ax.invert_yaxis()
fig.tight_layout()
epoch.to_frame()
"""),

MD("""
## 4. How much of a step is the data loader

A cache gather plus collate (`loader`) as a share of a full fwd+bwd step. A
share that grows with B means the memmap read, not the model, is becoming the
bottleneck. The fix there is pinned host memory or a prefetch thread, not a
bigger GPU.
"""),
CODE("""
pivot = torchcrop.pivot_table(index=["device", "label", "batch_size"], columns="mode",
                              values="seconds", aggfunc="min")
if {"loader", "fwd+bwd"} <= set(pivot.columns):
    share = (pivot["loader"] / pivot["fwd+bwd"]).rename("loader share").reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for (device, label), line in share.groupby(["device", "label"]):
        ax.plot(line["batch_size"], 100 * line["loader share"], marker="o", ms=4, lw=2,
                ls="-" if label == "mixed" else "--", color=COLOUR[f"torchcrop {device}"],
                label=f"{device} ({label})")
    ax.set(xscale="log", xlabel="batch size [cell-years]", ylabel="loader / step [%]")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
"""),

MD("""
## 5. SIMPLACE's spread across tasks

Task ranges are contiguous in SimplaceID order, which runs north to south. So
the per-task rate carries a latitude signal, which a single median hides.
"""),
CODE("""
if not tasks.empty:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for i, (source, block) in enumerate(tasks.groupby("source")):
        ax.hist(block["cell_years_per_core_s"], bins=40, histtype="step", lw=2,
                color=P[f"series_{3 + i % 6}"], label=Path(source).parent.name
                if Path(source).name == "simplace" else Path(source).name)
    ax.set(xlabel="cell-years per second per core", ylabel="tasks")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
"""),

MD("## 6. Table view"),
CODE("""
bench.sort_values(["model", "device", "label", "mode", "batch_size"]).round(3)
"""),
]

# --------------------------------------------------------------------------- #
# calibration_loss.ipynb
# --------------------------------------------------------------------------- #

loss = [
MD("""
# Calibration loss by region, batch and epoch: harvest date and yield

The objective is a **sum of per-region means**: one forward pass, then each
region's loss on its own rows. So a mixed-region batch gives every region
exactly the gradient its own batch would. It also means the per-region terms
plotted here are the quantities the optimiser actually sees.
`<stage>/batch_history.parquet` keeps them, one row per epoch, split, batch,
region and term. `history.csv` averages them away.

| Term | Meaning |
|---|---|
| `phenology/harvest_date_rmse_d` | Unit-mean simulated maturity (DVS 2 crossing) against the observed harvest date [d] |
| `phenology/harvest_date_bias_d` | Simulated minus observed [d]; positive is late |
| `yield/yield_rmse_t_ha` | Unit-mean dry-matter yield against the detrended `yield_level` [t/ha] |
| `loss` | The region's weighted objective, all active terms, retention included |

**SIMPLACE is not calibrated.** It re-runs on the crop files torchcrop fits, so
it appears as horizontal references: once on the default crop and once on the
per-region calibrated crops. Both are scored by
`evaluation.calibration_loss.score_simplace` on the **same pool, observations
and validation split** as torchcrop. Maturity stands in for harvest in both
models.
"""),
MD("## 0. Setup"),
CODE(SETUP + """
from cropmodelling4eu.evaluation.calibration_loss import (
    load_batch_history, load_simplace_runs, score_simplace,
)
from cropmodelling4eu.evaluation.plots import cell_map

#: Calibration runs to compare, e.g. the CPU fork-pool run and the GPU mixed run.
RUNS = {
    "cpu mixed": RUN / "calibration_mixcheck_cpu",
}
STAGE = "phenology"
TERMS = {
    "phenology/harvest_date_rmse_d": "harvest-date RMSE [d]",
    "yield/yield_rmse_t_ha": "yield RMSE [t/ha]",
}
#: SIMPLACE on the default crop, and on the per-region calibrated crops.
SIMPLACE = {
    # `simplace/`, not the top-level RUN/simplace_europe.parquet: that one is
    # the region-merged *calibrated* run, so it would score calibrated twice.
    "SIMPLACE default": [RUN / "simplace" / "simplace_europe.parquet"],
    "SIMPLACE calibrated": sorted((RUN / "simplace_regions").glob("region_*/simplace_europe.parquet")),
}

history = pd.concat(
    [load_batch_history(path, [STAGE]).assign(run=name) for name, path in RUNS.items()],
    ignore_index=True,
)
TERMS = {k: v for k, v in TERMS.items() if k in set(history["term"])}
primary = next(iter(RUNS))
regions = pd.read_parquet(RUNS[primary] / "regions.parquet")
history.groupby(["run", "split"])[["epoch", "batch"]].max()
"""),

MD("""
## 1. SIMPLACE references on the same split

This cell is skipped with a note if the run predates `pool.parquet` and
`<stage>_split.parquet`. Those files are written by every calibration since
mixed-region batching.
"""),
CODE("""
reference = pd.DataFrame()
needed = [RUNS[primary] / n for n in ("pool.parquet", "observations.parquet", f"{STAGE}_split.parquet")]
if all(p.is_file() for p in needed):
    reference = pd.concat(
        [score_simplace(RUNS[primary], load_simplace_runs(paths), STAGE, label)
         for label, paths in SIMPLACE.items() if paths],
        ignore_index=True,
    )
else:
    print("no persisted pool/split under", RUNS[primary], "- SIMPLACE references skipped")

# A calibrated run that reproduces the default one exactly did not apply its
# crops; say so here rather than draw one line on top of the other.
if reference["label"].nunique() == 2:
    pair = reference.pivot_table(index=["split", "region_id", "term"], columns="label", values="value")
    same = (pair.iloc[:, 0] - pair.iloc[:, 1]).abs().lt(1e-9).mean()
    if same > 0.99:
        print(f"WARNING: SIMPLACE default and calibrated agree on {same:.0%} of terms - "
              "check the per-region runs took their crops, and that 'default' is RUN/simplace/")
reference.head()
"""),

MD("""
## 2. Region × epoch

Each cell is the row-weighted mean over that epoch's training batches for one
region. Magnitude uses a sequential ramp; the colour scale is shared within each
term, not across terms.
"""),
CODE("""
train = history[(history["split"] == "train") & (history["run"] == primary)]
rows = train[train["term"] == "rows"][["epoch", "batch", "region_id", "value"]].rename(
    columns={"value": "w"})
for term, label in TERMS.items():
    block = train[train["term"] == term].merge(rows, on=["epoch", "batch", "region_id"])
    grid = (block.assign(wv=block["value"] * block["w"])
            .groupby(["region_id", "epoch"])[["wv", "w"]].sum()
            .pipe(lambda g: g["wv"] / g["w"]).unstack("epoch"))
    fig, ax = plt.subplots(figsize=(1.2 + 0.45 * grid.shape[1], 0.6 + 0.22 * grid.shape[0]))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=SEQUENTIAL_CMAP)
    ax.set(xticks=range(grid.shape[1]), xticklabels=grid.columns,
           yticks=range(grid.shape[0]), yticklabels=grid.index,
           xlabel="epoch", ylabel="region", title=f"{label}, training batches")
    fig.colorbar(image, ax=ax, label=label, shrink=0.8)
    fig.tight_layout()
"""),

MD("""
## 3. Per region: every batch, the epoch mean, validation, SIMPLACE

The dots are individual training batches, each placed at its epoch plus its
position within that epoch. The line is the epoch mean and the diamonds are
validation. Horizontal lines are SIMPLACE on the validation split, solid for the
calibrated crops and dotted for the default crop.
"""),
CODE("""
for term, label in TERMS.items():
    block = history[(history["term"] == term) & (history["run"] == primary)]
    ids = sorted(block["region_id"].unique())
    ncol = 6
    fig, axes = plt.subplots(-(-len(ids) // ncol), ncol, figsize=(13, 1.9 * -(-len(ids) // ncol)),
                             sharex=True, squeeze=False)
    for ax, region in zip(axes.flat, ids):
        own = block[block["region_id"] == region]
        tr = own[own["split"] == "train"].copy()
        span = tr.groupby("epoch")["batch"].transform("max").clip(lower=1)
        ax.scatter(tr["epoch"] + 0.8 * tr["batch"] / span, tr["value"], s=8,
                   color=P["muted"], linewidths=0)
        ax.plot(*zip(*tr.groupby("epoch")["value"].mean().items()), color=P["series_1"], lw=2)
        va = own[own["split"] == "val"].groupby("epoch")["value"].mean()
        ax.plot(va.index, va.values, marker="D", ms=4, lw=0, color=P["series_2"])
        if not reference.empty:
            ref = reference[(reference["split"] == "val") & (reference["term"] == term)
                            & (reference["region_id"] == region)]
            for _, r in ref.iterrows():
                ax.axhline(r["value"], color=P["series_3"], lw=1.5,
                           ls="-" if "calibrated" in r["label"] else ":")
        ax.set_title(f"region {region}", fontsize=8)
    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=P["muted"], marker="o", lw=0, label="train batch"),
               plt.Line2D([], [], color=P["series_1"], lw=2, label="train epoch mean"),
               plt.Line2D([], [], color=P["series_2"], marker="D", lw=0, label="validation"),
               plt.Line2D([], [], color=P["series_3"], lw=1.5, label="SIMPLACE calibrated"),
               plt.Line2D([], [], color=P["series_3"], lw=1.5, ls=":", label="SIMPLACE default")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8)
    fig.supylabel(label)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
"""),

MD("""
## 4. Where the loss sits on the map

These are the last validation epoch's per-region terms, painted onto every cell
of the region: torchcrop, then SIMPLACE on the calibrated crops, then the change
in torchcrop since epoch 0. A sequential ramp is used for levels and a diverging
one centred on zero for the change, where blue is improvement.
"""),
CODE("""
val = history[(history["split"] == "val") & (history["run"] == primary)]
cells = regions[["SimplaceID", "lon", "lat", "region_id"]]
for term, label in TERMS.items():
    by_epoch = val[val["term"] == term].groupby(["epoch", "region_id"])["value"].mean().unstack("epoch")
    if by_epoch.empty:
        continue
    last, first = by_epoch.iloc[:, -1], by_epoch.iloc[:, 0]
    frame = cells.assign(
        torchcrop=cells["region_id"].map(last),
        change=cells["region_id"].map(last - first),
    )
    if not reference.empty:
        cal = reference[(reference["label"] == "SIMPLACE calibrated") & (reference["split"] == "val")
                        & (reference["term"] == term)].set_index("region_id")["value"]
        frame["simplace"] = cells["region_id"].map(cal)
    vmax = np.nanpercentile(frame[[c for c in ("torchcrop", "simplace") if c in frame]], 98)
    cell_map(frame.dropna(subset=["torchcrop"]), "torchcrop", cmap=SEQUENTIAL_CMAP, vmin=0,
             vmax=vmax, title=f"torchcrop, last epoch", cbar_label=label)
    if "simplace" in frame:
        cell_map(frame.dropna(subset=["simplace"]), "simplace", cmap=SEQUENTIAL_CMAP, vmin=0,
                 vmax=vmax, title="SIMPLACE, calibrated crops", cbar_label=label)
    cell_map(frame.dropna(subset=["change"]), "change", cmap=DIVERGING_CMAP,
             norm=diverging_norm(frame["change"].dropna()),
             title="torchcrop, change since epoch 0", cbar_label=f"Δ {label}")
"""),

MD("""
## 5. Loss against wall-clock time

This compares the same stage run two ways, for example the CPU fork pool and
the GPU with mixed batches. Each point is one epoch: x is cumulative training
time, from the per-batch `seconds`, and y is the row-weighted validation term.
"""),
CODE("""
for term, label in TERMS.items():
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for i, (run, block) in enumerate(history.groupby("run")):
        tr = block[(block["split"] == "train") & (block["term"] == term)]
        seconds = tr.drop_duplicates(["epoch", "batch"]).groupby("epoch")["seconds"].sum().cumsum()
        va = block[(block["split"] == "val") & (block["term"] == term)].groupby("epoch")["value"].mean()
        joined = pd.concat([seconds.rename("s"), va.rename("v")], axis=1).dropna()
        ax.plot(joined["s"] / 60, joined["v"], marker="o", ms=4, lw=2,
                color=P[f"series_{1 + i}"], label=run)
    ax.set(xlabel="training wall time [min]", ylabel=f"validation {label}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
"""),

MD("## 6. Table view"),
CODE("""
summary = (history[history["term"].isin(TERMS)]
           .groupby(["run", "split", "term", "region_id", "epoch"])["value"].mean()
           .unstack("epoch"))
pd.concat([summary, reference.pivot_table(index=["label", "split", "term", "region_id"],
                                          values="value")
           if not reference.empty else None]).round(2)
"""),
]

for name, cells in (("benchmark.ipynb", benchmark), ("calibration_loss.ipynb", loss)):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"]["kernelspec"] = {
        "name": "python3", "display_name": "Python 3", "language": "python",
    }
    nbf.write(notebook, HERE / name)
    print("wrote", HERE / name)
