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
#: `submit_benchmark.sh` writes one file per device, each with a per-pass loss
#: log (`*_passes.parquet`, batch_history format) beside it.
BENCH_DIR = RUN / "benchmark"
BENCH_FILES = [BENCH_DIR / f"benchmark_{d}.parquet" for d in ("cpu", "gpu")]
#: Per-task SIMPLACE timings the CPU job wrote beside its file.
TASKS = BENCH_DIR / "benchmark_cpu_simplace_tasks.parquet"
#: The production calibration whose epochs section 3 prices.
CALIBRATION = RUN / "calibration"
#: Cell-years one training epoch actually simulated, from its own batch log.
_epochs = (pd.read_parquet(CALIBRATION / "phenology" / "batch_history.parquet")
           .query("split == 'train'").drop_duplicates(["epoch", "batch"]))
EPOCH_CELL_YEARS = int(_epochs.groupby("epoch")["batch_rows"].sum().median())
#: Forked region workers on the CPU path (CAL_WORKERS).
CPU_WORKERS = 29

bench = pd.concat([pd.read_parquet(p) for p in BENCH_FILES if p.is_file()], ignore_index=True)
passes = pd.concat(
    [pd.read_parquet(q) for p in BENCH_FILES
     if (q := p.with_name(p.stem + "_passes.parquet")).is_file()],
    ignore_index=True,
)
tasks = pd.read_parquet(TASKS) if TASKS.is_file() else pd.DataFrame()
torchcrop = bench[bench["model"] == "torchcrop"].copy()
simplace = bench[bench["model"] == "simplace"].copy()

#: Colour follows the entity, never its rank: fixed slots, never cycled.
COLOUR = {"torchcrop cpu": P["torchcrop_cpu"], "torchcrop cuda": P["torchcrop_gpu"],
          "simplace": P["simplace"]}
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
        ax.plot(line["batch_size"], line["cell_years_per_s"], ms=4, lw=2,
                marker="s" if device == "cpu" else "o",
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

MD("""
## 7. Loss per pass and batch

Each timed pass also logs its loss: one row per region and term, in the
`batch_history` format (`epoch` is the pass). The loss plotted here is the
batch's **row-weighted mean of the per-region losses**, the same number
`history.csv` averages. It leaves out the prior and pooling penalties, which
are added only to the gradient.

* `forward` / `fwd+bwd` take no optimiser step. Their anomaly terms read a
  running per-unit climatology, which runs from passes earlier in the same mode.
  Runs before 2026-11-09 evening let it accumulate across passes, so their
  repeated passes differ by a few units; later runs reset it per pass, and any
  spread is then non-determinism.
* At a size both devices ran, CPU and GPU draw the **same** batch (same sampler
  seed), so their losses should agree to float tolerance.
* `step` takes the stage's own clipped Adam step after every batch, from a
  fresh problem per size. Its curve is the optimiser's progress per pass, and
  x = wall seconds puts CPU and GPU on one axis.
"""),
CODE("""
if passes.empty:
    print("no *_passes.parquet yet - rerun submit_benchmark.sh")
else:
    loss = passes[passes["term"] == "loss"].merge(
        passes[passes["term"] == "rows"][["device", "label", "mode", "size", "epoch", "batch", "region_id", "value"]]
        .rename(columns={"value": "w"}),
        on=["device", "label", "mode", "size", "epoch", "batch", "region_id"],
    )
    per_batch = (loss.assign(wv=loss["value"] * loss["w"])
                 .groupby(["device", "label", "mode", "size", "batch", "epoch", "batch_rows"])
                 .agg(wv=("wv", "sum"), w=("w", "sum"), seconds=("seconds", "first"))
                 .reset_index())
    per_batch["loss"] = per_batch["wv"] / per_batch["w"]

    fixed = per_batch[per_batch["mode"].isin(["forward", "fwd+bwd"])]
    spread = fixed.groupby(["device", "label", "mode", "size", "batch"])["loss"].agg(["min", "max"])
    print("max loss spread across repeated passes:", float((spread["max"] - spread["min"]).abs().max()))
    first = fixed[fixed["epoch"] == 0].pivot_table(
        index=["label", "mode", "size", "batch"], columns="device", values="loss")
    if {"cpu", "cuda"} <= set(first.columns):
        both = first.dropna()
        print("CPU vs GPU, same batch: max relative difference",
              float(((both["cuda"] - both["cpu"]).abs() / both["cpu"].abs()).max()))

    # Mixed batches only: the CPU job also times one-region batches, whose
    # losses are of a different set of rows and would pull its curves down.
    steps = per_batch[(per_batch["mode"] == "step") & (per_batch["label"] == "mixed")]
    if not steps.empty:
        sizes = sorted(steps["size"].unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for i, ((device, size), block) in enumerate(steps.groupby(["device", "size"])):
            colour = P[f"series_{1 + sizes.index(size)}"]
            ls = "-" if device == "cuda" else "--"
            epoch_loss = block.groupby("epoch").apply(
                lambda g: np.average(g["loss"], weights=g["batch_rows"]), include_groups=False)
            wall = block.groupby("epoch")["seconds"].sum().cumsum()
            name = f"{device} B={size}"
            axes[0].plot(epoch_loss.index, epoch_loss.values, ls=ls, marker="o", ms=3, color=colour, label=name)
            axes[1].plot(wall.values, epoch_loss.values, ls=ls, marker="o", ms=3, color=colour, label=name)
        axes[0].set(xlabel="pass (one Adam step per batch)", ylabel="loss, row-weighted over batches")
        axes[1].set(xlabel="cumulative wall time [s]", xscale="log")
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle("step mode: solid = GPU, dashed = CPU")
        fig.tight_layout()
    per_batch.pivot_table(index=["device", "label", "mode", "size", "batch"], columns="epoch",
                          values="loss").round(2)
"""),

MD("""
## 8. Wall time for the EU historical production run, on one node

What the CMIP6 *historical* production run (MIROC6, 1851-2014 harvests, every
exported cell) would take on one node, from the measured rates. It is a
**forward** run, so torchcrop's `forward` mode is the rate that applies. Every
configuration is packed onto a whole node as tightly as its memory allows:

| | Per process | Processes per node |
|---|---|---|
| SIMPLACE, CPU | measured: median rate of the CMIP6 hist-aer tasks, each one node of 24 JVMs | 24 JVMs (~72 GB); 39 ran out of memory |
| torchcrop, CPU | `forward`, one thread | `min(cores, RAM / peak RSS)` at that batch size |
| torchcrop, GPU | `forward`, one V100 | the node's 2 GPUs |

SIMPLACE has no batch, so its bar is the same in every group. It is **21.5 h**,
not the 7 h of earlier drafts: that figure multiplied one JVM running alone
(~11 cell-years/s) by 39 JVMs per node, but 39 ran out of memory and packed JVMs
each run slower, so the measured 24-JVM node rate replaces it. The y axis is
logarithmic. Each bar is the total wall time, weather reading included; with
reader processes the model's wait for weather is close to zero, so it is not
drawn separately (the table below keeps `io_h`).

**Why the GPU takes longer than the CPU.** A GPU pass has a floor of
~21 s whatever the batch: the 465-day loop launches many small kernels per day,
and that latency, not arithmetic, is the time (21 s at 64 cell-years, 29 s at
65,536). One CPU process does a 4,096 batch in 12 s. So per pass the GPU only
catches up at ~16k, and a GPU node holds 2 GPUs where a CPU node holds 19-61
processes (by memory). At 65,536 one V100 does 2,265 cell-years/s, about 6.7 CPU
processes' worth, and a CPU node runs 36 of them. The GPU uses 2.2 of its 16 GB
there, so it can take ~7x more per pass; with the time almost flat in batch that
is where it would overtake the CPU node. The GPU rows below 4096 come from the earlier GPU-only run (872622,
same node type).
"""),
CODE("""
# Segoe UI is a licensed Microsoft font and not installed here. Selawik is
# Microsoft's open (OFL) metric-compatible fallback for it, so the PNG and PDF
# embed Selawik; the SVG keeps its text as text, naming Segoe UI first, so it
# renders in Segoe UI wherever that is installed (PowerPoint, Illustrator).
import logging
from matplotlib import font_manager

SELAWIK = Path("/data01/FDS/muduchuru/Data/SIMPLACE/fonts/selawik/zip")
for ttf in sorted(SELAWIK.glob("*.ttf")):
    font_manager.fontManager.addfont(str(ttf))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
# A list as font.family (not font.sans-serif) so matplotlib falls back per
# glyph to DejaVu Sans for what Selawik lacks, such as the minus sign.
POSTER_FONT = {"font.family": ["Segoe UI", "Selawik", "DejaVu Sans"],
               "svg.fonttype": "none", "pdf.fonttype": 42}

# The poster's figure style, shared by section 8 and section 10: one column of
# the two-column poster, bold left titles, black text and 1.2 pt black spines
# (1.5x the style's 0.8 pt), and text 1.25x the first poster draft's sizes.
POSTER_RC = {**POSTER_FONT, "font.size": 9.38, "axes.titlesize": 10.62,
             "axes.titleweight": "bold", "axes.titlelocation": "left",
             "axes.labelsize": 9.38, "xtick.labelsize": 8.75, "ytick.labelsize": 8.75,
             "axes.edgecolor": "black", "axes.linewidth": 1.2,
             "xtick.color": "black", "ytick.color": "black",
             "xtick.major.width": 1.2, "ytick.major.width": 1.2,
             "xtick.labelcolor": "black", "ytick.labelcolor": "black",
             "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black"}
POSTER_SIZE = (7.8, 1.72)

#: The run being priced: every (cell, harvest) the historical production simulated.
PRODUCTION = Path("/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/"
                  "calibrated/cmip6/MIROC6/historical/torchcrop/torchcrop_europe.parquet")
PROD_CELL_YEARS = len(pd.read_parquet(PRODUCTION, columns=["year"]))
#: Node shapes (`scontrol show node`): compute 80 cores / 95 GB, gpu 2 x V100 16 GB.
CPU_CORES, NODE_GB, GPUS_PER_NODE = 80, 95.0, 2

# The production path (`cm4eu benchmark --production`): torchcrop.run's own
# batching, weather read from disk, two-pass run_batch. Every run of
# submit/benchmark/submit_baseline_bench.sh, any reader count.
BASELINE = BENCH_DIR / "baseline"
prod = pd.concat([pd.read_parquet(f) for f in sorted(BASELINE.glob("production_*.parquet"))],
                 ignore_index=True)
prod = prod[~prod["warmup"].astype(bool) & prod["error"].isna()].copy()
for column in ("io_processes", "reader_peak_gb"):
    prod[column] = prod[column].fillna(0) if column in prod else 0
prod["io_processes"] = prod["io_processes"].astype(int)
by = (prod.groupby(["device", "target", "io_processes"])
      .agg(rows=("rows", "sum"), io=("io_s", "sum"), model=("model_s", "sum"),
           mem=("peak_mem_gb", "max"), readers_gb=("reader_peak_gb", "max")).reset_index())
# Processes per node. GPU: one per GPU, its readers on the 48 host cores.
# CPU: as many as both cores (a process and each of its readers take one) and
# RAM (the process's peak plus its readers') allow.
by["procs"] = np.where(
    by["device"] == "cuda", GPUS_PER_NODE,
    np.minimum(CPU_CORES // (1 + by["io_processes"]),
               np.floor(NODE_GB / (by["mem"] + by["readers_gb"]))),
)
by["io_h"] = PROD_CELL_YEARS * by["io"] / by["rows"] / by["procs"] / 3600
by["model_h"] = PROD_CELL_YEARS * by["model"] / by["rows"] / by["procs"] / 3600
by["total_h"] = by["io_h"] + by["model_h"]
# Per device and batch, the reader count that finishes first.
pick = by.loc[by.groupby(["device", "target"])["total_h"].idxmin()]
# The sizes both devices ran, plus the GPU's largest (it needs them to fill).
SIZES = [1024, 4096, 8192, 16384, 32768, 40960]
pick = pick[pick["target"].isin(SIZES)]

# SIMPLACE: the measured rate of whole nodes. One JVM alone runs ~11
# cell-years/s, but 39 per node (the old RAM / 2.4 GB packing) ran out of
# memory and thrashed (job 872800), and packed JVMs each run slower. The CMIP6
# hist-aer tasks are one node of 24 JVMs each, so their median task rate is the
# node rate directly. It is the same number section 8b uses.
from cropmodelling4eu.benchmark import simplace_log_timings

HIST_AER = Path("/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/"
                "calibrated/cmip6/MIROC6/hist-aer")
sp = simplace_log_timings([HIST_AER])
sp_node = sp["cell_years_per_s"].median()
sp_hours = PROD_CELL_YEARS / sp_node / 3600

by_device = lambda d, col: pick[pick["device"] == d].set_index("target")[col].to_dict()
series = {
    "SIMPLACE · CPU": (P["simplace"], dict.fromkeys(SIZES, sp_hours), {}),
    "torchcrop · CPU": (P["torchcrop_cpu"], by_device("cpu", "total_h"), by_device("cpu", "io_h")),
    "torchcrop · GPU": (P["torchcrop_gpu"], by_device("cuda", "total_h"), by_device("cuda", "io_h")),
}
def duration(hours: float) -> str:
    # Hours as the unit a reader thinks in: minutes, hours or days.
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.0f} d"

from matplotlib.patches import Patch

FIGURES = BENCH_DIR / "figures"
FIGURES.mkdir(exist_ok=True)
NAMES = {"SIMPLACE · CPU": "SIMPLACE", "torchcrop · CPU": "TorchCrop CPU",
         "torchcrop · GPU": "TorchCrop GPU"}
with plt.rc_context(POSTER_RC):
    fig, ax = plt.subplots(figsize=(POSTER_SIZE[0], POSTER_SIZE[1] * 1.15))
    x, width = np.arange(len(SIZES)), 0.27
    floor = 0.05
    for i, (name, (colour, values, loading)) in enumerate(series.items()):
        heights = np.array([values.get(s, np.nan) for s in SIZES])
        pos = x + (i - 1) * width
        ax.bar(pos, heights - floor, width - 0.03, bottom=floor, color=colour,
               label=NAMES[name], zorder=2)
        best = int(np.nanargmin(heights))
        ax.annotate(duration(heights[best]), (pos[best], heights[best]), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.1,
                    fontweight="bold", color="black")
    ax.set_yscale("log")
    ax.set_ylim(floor, 24 * 7)
    ticks = [5 / 60, 0.25, 1, 4, 24, 24 * 7]
    ax.set_yticks(ticks, ["5 min", "15 min", "1 h", "4 h", "1 d", "1 wk"])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    for i, (_, values, _) in enumerate(series.values()):
        for j, s in enumerate(SIZES):
            if s not in values:
                ax.annotate("not run", (x[j] + (i - 1) * width, floor), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.9, color="black", rotation=90)
    ax.set_xticks(x, [f"{s:,}" for s in SIZES])
    ax.tick_params(axis="x", length=0)
    ax.set(xlabel="batch size [cell-years per batch]", ylabel="wall time, one node",
           title=f"EU historical run, {PROD_CELL_YEARS / 1e6:.1f} M cell-years")
    ax.grid(axis="x", visible=False)
    # Inside the plot, in the band above 1 day that only the small-batch GPU
    # bars on the left reach.
    ax.legend(frameon=False, ncol=3, loc="upper right",
              bbox_to_anchor=(1.0, 1.02), fontsize=8.1, handlelength=1.2, columnspacing=1.0,
              borderaxespad=0.2)
    ax.annotate("Bold: each model's best batch. Production runs, weather read from disk "
                "while the model computes; SIMPLACE 24 JVMs per node.",
                (0, 0), xycoords="axes fraction", xytext=(0, -37), textcoords="offset points",
                ha="left", va="top", fontsize=7.25)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"walltime_eu_historical.{ext}", dpi=300, bbox_inches="tight")

table = pd.DataFrame({name: [v.get(s, np.nan) for s in SIZES] for name, (_, v, _) in series.items()},
                     index=pd.Index(SIZES, name="batch size"))
for name, (_, values, loading) in series.items():
    if loading:
        table[f"{name} loading %"] = [100 * loading.get(s, np.nan) / values.get(s, np.nan) for s in SIZES]
table.rename(columns=lambda c: c if c.endswith("%") else f"{c} [h]").round(2)
"""),

MD("""
## 8b. Best single-node setup, production path

Section 8 times **calibration** batches. This one times what the EU run actually
executes, `torchcrop.run` on the default crop (`cm4eu benchmark --production`,
`submit/benchmark/submit_baseline_bench.sh`): two-pass `run_batch`, the
production batching, and the weather reader. Each setup is shown at the batch
that is best **on one node**:

* **torchcrop · CPU**: the batch whose per-process rate times the processes RAM
  allows (95 GB / peak RSS) is highest, including weather I/O.
* **torchcrop · GPU**: the batch with the highest rate, including weather I/O;
  the largest that fits in 16 GB is 40,960 (49,152 ran out of memory). Two GPUs
  per node.
* **SIMPLACE · CPU**: the measured packed rate of the CMIP6 hist-aer run, 24 JVMs
  per node (39 thrashed and was OOM-killed). That run writes daily output, so
  yearly-only would be somewhat faster.

With `torchcrop.io_processes: 0` weather I/O and the model run **one after the
other**, so the bar is their sum. The reader's threads hold the GIL, so one
process reads at ~4.5 ms per row whatever its thread count: harmless with 22
processes per CPU node, but it left a GPU waiting 6x longer for weather than it
computed. With `io_processes > 0` worker processes read ahead of the model, and
the hatched part is only the time the model still **waited**; each setup's
best reader count is chosen like its batch. The CPU bar assumes
the shared filesystem keeps up with every process reading at once, which is not
verified. SIMPLACE reads its weather inside the JVM, so its bar is not split.
"""),
CODE("""
from cropmodelling4eu.benchmark import simplace_log_timings

# Section 8's production table (`by`), every reader count included.
best = by.loc[by.groupby("device")["total_h"].idxmin()].set_index("device")

# SIMPLACE: section 8's measured node rate.
sp_h = PROD_CELL_YEARS / sp_node / 3600

bars = [
    ("SIMPLACE · CPU", P["simplace"], np.nan, sp_h, f"{int(sp['workers'].median())} JVMs per node"),
    ("torchcrop · CPU", P["torchcrop_cpu"], best.loc["cpu", "io_h"], best.loc["cpu", "model_h"],
     f"B={int(best.loc['cpu', 'target']):,} x {int(best.loc['cpu', 'procs'])} processes"),
    ("torchcrop · GPU", P["torchcrop_gpu"], best.loc["cuda", "io_h"], best.loc["cuda", "model_h"],
     f"B={int(best.loc['cuda', 'target']):,} x {int(best.loc['cuda', 'procs'])} GPUs"
     + (f", {int(best.loc['cuda', 'io_processes'])} readers each"
        if best.loc['cuda', 'io_processes'] else "")),
]
with plt.rc_context(POSTER_RC):
    fig, ax = plt.subplots(figsize=(POSTER_SIZE[0], POSTER_SIZE[1] * 1.15))
    y = np.arange(len(bars))[::-1]
    floor = 0.05
    for yi, (name, colour, io_h, model_h, setup) in zip(y, bars):
        wait = io_h if np.isfinite(io_h) else 0.0
        total = wait + model_h
        # Log axis: the hatched part ends at the waiting time, the bar at the total.
        if wait > floor:
            ax.barh(yi, wait - floor, 0.62, left=floor, color=colour, alpha=0.45,
                    hatch="////", edgecolor=P["surface"], linewidth=0, zorder=2)
        ax.barh(yi, total - max(wait, floor), 0.62, left=max(wait, floor), color=colour, zorder=2)
        speed = "" if name.startswith("SIMPLACE") else f"  ({sp_h / total:.0f}x faster)"
        ax.annotate(f"{duration(total)}  ·  {setup}{speed}", (total, yi), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.1,
                    color="black")
    ax.set_xscale("log")
    ax.set_xlim(floor, 24 * 30)
    ticks = [5 / 60, 0.25, 1, 4, 24, 24 * 7]
    ax.set_xticks(ticks, ["5 min", "15 min", "1 h", "4 h", "1 d", "1 wk"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, [NAMES[b[0]] for b in bars])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="wall time, one node",
           title=f"EU historical run, {PROD_CELL_YEARS / 1e6:.1f} M cell-years: best setup per node")
    if any(np.isfinite(b[2]) and b[2] > floor for b in bars):
        ax.legend(handles=[Patch(facecolor=P["surface"], hatch="////", edgecolor="black",
                                 label="waiting for weather")],
                  frameon=False, fontsize=8.1, loc="lower right")
    ax.annotate("Production path, weather read ahead by worker processes; "
                "SIMPLACE is the measured CMIP6 hist-aer node rate.",
                (0, 0), xycoords="axes fraction", xytext=(0, -40), textcoords="offset points",
                ha="left", va="top", fontsize=7.25)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"best_setup_eu_historical.{ext}", dpi=300, bbox_inches="tight")
by.assign(**{c: by[c].round(2) for c in ("io_h", "model_h", "total_h")}).drop(columns=["io", "model"])
"""),

MD("""
## 9. Calibration loss against wall time

One region's phenology stage, fitted three ways by `submit/benchmark/submit_loss_walltime.sh`,
each on a whole node with the largest batch it can take:

| | Optimiser | One step is |
|---|---|---|
| torchcrop · GPU | Adam on the gradient | the region's **whole** epoch in one batch, one V100 |
| torchcrop · CPU | Adam on the gradient | the same batch, one process |
| SIMPLACE · CPU | Optuna TPE, no gradient | one trial: every cell of the region, as many JVMs as RAM holds |

All three score the **same** unit-years and the same loss terms: torchcrop reuses
the production calibration's split (`--split-from`), and the SIMPLACE scorer is
the torchcrop objective recomputed on SIMPLACE output. SIMPLACE starts at
torchcrop's starting crop. The losses are **without** the prior penalty, which is
what `history.csv` and the trial reports both log.

The x axis is optimisation time only: setup (pool, weather cache, the SIMPLACE
base workspace) is excluded for all three. The **solid** line is the loss of
each step as it happened — one epoch for torchcrop, one trial (dots) for
SIMPLACE, whose TPE trials jump around by design. The **dotted** line is the
best loss reached so far in that panel; the line under each panel gives each run's best and length. The
axes are logarithmic, so equal-length steps look shorter as time grows.
"""),
CODE("""
LW = BENCH_DIR / "loss_walltime"
LW_REGION, LW_STAGE = 16, "phenology"
curves = {}
for name, device in (("torchcrop · GPU", "tc_gpu"), ("torchcrop · CPU", "tc_cpu")):
    path = LW / device / LW_STAGE / "history.csv"
    if path.is_file():
        h = pd.read_csv(path)
        curves[name] = pd.DataFrame({"minutes": h["seconds"].cumsum() / 60,
                                     "train": h["train_loss"], "val": h["val_loss"]})
trials_csv = LW / "sp_cpu" / f"region_{LW_REGION:02d}" / LW_STAGE / "trials.csv"
if trials_csv.is_file():
    t = pd.read_csv(trials_csv)
    t = t[t["state"] == "COMPLETE"].sort_values("number")
    curves["SIMPLACE · CPU"] = pd.DataFrame({
        "minutes": t["user_attrs_seconds"].cumsum().to_numpy() / 60,
        "train": t["user_attrs_train/loss"].to_numpy(),
        "val": t["user_attrs_val/loss"].to_numpy(),
    })

COLOURS = {"SIMPLACE · CPU": P["simplace"], "torchcrop · CPU": P["torchcrop_cpu"],
           "torchcrop · GPU": P["torchcrop_gpu"]}
# The poster's shapes: torchcrop CPU squares, GPU circles; SIMPLACE's trials are
# discrete points, so they get small dots.
MARKERS = {"SIMPLACE · CPU": ".", "torchcrop · CPU": "s", "torchcrop · GPU": "o"}
if not curves:
    print("no runs under", LW, "- submit submit/benchmark/submit_loss_walltime.sh first")
else:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, column, title in zip(axes, ("train", "val"), ("training", "validation")):
        for name, colour in COLOURS.items():
            if name not in curves:
                continue
            c = curves[name]
            best = c[column].cummin()
            # The step's own loss: an epoch for torchcrop, one trial for SIMPLACE
            # (TPE explores, so its trials jump around by design).
            ax.plot(c["minutes"], c[column], color=colour, lw=1.2, alpha=0.9,
                    marker=MARKERS[name], ms=5 if name.startswith("SIMPLACE") else 3.5,
                    markevery=1 if name.startswith("SIMPLACE") else 6)
            ax.plot(c["minutes"], best, color=colour, lw=2, ls=":")
        # The runs end within a few units of each other, so their best values
        # go in one note per panel instead of colliding at the line ends.
        note = "   ".join(f"{n.replace('torchcrop', 'tc')} {curves[n][column].min():,.0f} "
                           f"({curves[n]['minutes'].iloc[-1]:.0f} min)"
                           for n in COLOURS if n in curves)
        ax.text(0.5, -0.2, f"best: {note}", transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=P["secondary"])
        ax.set(xscale="log", yscale="log", xlabel="optimisation wall time [min]",
               title=f"{title} loss, region {LW_REGION} {LW_STAGE}")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    axes[0].set_ylabel("loss (no prior)")
    handles = [Line2D([], [], color=c, lw=2, marker=MARKERS[n], ms=5, label=n)
               for n, c in COLOURS.items() if n in curves]
    handles += [Line2D([], [], color=P["muted"], lw=1.2, label="loss at that step"),
                Line2D([], [], color=P["muted"], lw=2, ls=":", label="best so far")]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=5, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "loss_walltime.png", dpi=200, bbox_inches="tight")
    pd.concat({k: v.iloc[[0, -1]] for k, v in curves.items()}).round(1)
"""),

MD("""
## 10. Poster figure: production wall time and calibration validation loss

Sections 8b and 9 (validation only) as one figure sized for **one column of the
two-column TorchCrop poster** (the aspect of its Fig. 2), in the poster's style:
bold panel titles, direct labels instead of a legend, SIMPLACE grey, TorchCrop
CPU olive (squares), TorchCrop GPU teal (circles). Saved as PNG, PDF and SVG.

* **Left**: the EU historical run on one node, each setup at its best batch
  (section 8b). The note under it gives the node hardware.
* **Right**: validation loss against optimisation wall time for region 16
  phenology (section 9). Solid: best so far; dotted: each TorchCrop epoch;
  dots: each SIMPLACE trial, with trials above 1,600 drawn as open triangles on
  the top edge. Each label gives the wall time at which that run first reached
  its best validation loss.
"""),
CODE("""
POSTER = {"SIMPLACE · CPU": "SIMPLACE", "torchcrop · CPU": "TorchCrop CPU",
          "torchcrop · GPU": "TorchCrop GPU"}
Y_CLIP = (830, 1600)
with plt.rc_context(POSTER_RC):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.8, 1.72), gridspec_kw={"width_ratios": (1.1, 1.0), "wspace": 0.26})

    # Left: wall time on one node, each setup at its best batch (section 8b).
    setups = by.loc[by.groupby("device")["total_h"].idxmin()].set_index("device")
    cpu, gpu = setups.loc["cpu"], setups.loc["cuda"]
    hours = [(0 if not np.isfinite(io_h) else io_h) + model_h for _, _, io_h, model_h, _ in bars]
    rows = [
        ("SIMPLACE", bars[0][1], hours[0], "24 JVMs"),
        ("TorchCrop CPU", bars[1][1], hours[1],
         f"B {int(cpu['target']):,} × {int(cpu['procs'])} processes"),
        ("TorchCrop GPU", bars[2][1], hours[2], f"B {int(gpu['target']):,} × {GPUS_PER_NODE} GPUs"),
    ]
    floor, day = 0.05, 24.0
    for i, (name, colour, h, setup) in enumerate(rows):
        left.barh(i, h - floor, 0.6, left=floor, color=colour, zorder=2)
        # Speed-up over SIMPLACE on the same node, the panel's headline number.
        speedup = "" if i == 0 else f" · {hours[0] / h:.0f}× faster"
        # A bar near the 1-day edge has no room right of it: label inside.
        inside = h > 2.0
        side = dict(xytext=(-3, 0), ha="right", color="black") if inside else \
            dict(xytext=(3, 0), ha="left", color="black")
        left.annotate(duration(h) + speedup, (h, i), xytext=(side["xytext"][0], 1.5),
                      textcoords="offset points", va="bottom", ha=side["ha"], fontsize=9.38,
                      color=side["color"], fontweight="bold")
        left.annotate(setup, (h, i), xytext=(side["xytext"][0], -1.5), textcoords="offset points",
                      va="top", ha=side["ha"], fontsize=7.5, color=side["color"])
    left.set_yticks(range(len(rows)), [r[0] for r in rows])
    left.invert_yaxis()
    left.set_xscale("log")
    left.set_xlim(floor, day)
    left.set_xticks([5 / 60, 0.25, 1, 4, day], ["5 min", "15 min", "1 h", "4 h", "1 d"])
    left.xaxis.set_minor_locator(plt.NullLocator())
    left.set(title=f"EU historical run, {PROD_CELL_YEARS / 1e6:.1f} M cell-years",
             xlabel="wall time on one node")
    left.grid(axis="y", visible=False)
    left.tick_params(axis="y", length=0)
    # Notes sit a fixed distance under the axis, whatever the figure height.
    left.annotate(f"CPU node: {CPU_CORES} cores, {NODE_GB:.0f} GB RAM\\n"
              f"GPU node: {GPUS_PER_NODE} × V100, 16 GB each",
              (0, 0), xycoords="axes fraction", xytext=(0, -37), textcoords="offset points",
              ha="left", va="top", fontsize=7.25,
              color="black")

    # Right: validation loss against wall time.
    label_y = {"SIMPLACE · CPU": 1060, "torchcrop · GPU": 950, "torchcrop · CPU": 868}
    for name, colour in COLOURS.items():
        if name not in curves:
            continue
        c = curves[name]
        simplace_run = name.startswith("SIMPLACE")
        clipped = c["val"] > Y_CLIP[1]
        if simplace_run:
            # Trials are independent samples, not a trajectory: dots, no line.
            right.plot(c["minutes"][~clipped], c["val"][~clipped], lw=0, marker="o", ms=2.6,
                       color=colour, alpha=0.85, zorder=3)
            # Trials above the axis: an open triangle on its top edge.
            right.plot(c["minutes"][clipped], np.full(clipped.sum(), Y_CLIP[1]), lw=0,
                       marker="^", ms=3.4, mfc="none", mew=0.8, color=colour, clip_on=False,
                       zorder=3)
        else:
            right.plot(c["minutes"], c["val"], color=colour, lw=1.0, ls=":", zorder=4)
        best = c["val"].cummin()
        right.plot(c["minutes"], best, color=colour, lw=1.8, zorder=5,
                   marker=None if simplace_run else MARKERS[name], ms=2.8, markevery=6)
        # One label column right of every run's end, each tied to its end by a
        # hairline: CPU and GPU finish at the same loss and would collide.
        end = (c["minutes"].iloc[-1], best.iloc[-1])
        # Time to the optimum: the first step that reached the run's best loss.
        reached = c["minutes"].iloc[int(np.argmin(c["val"].to_numpy()))]
        right.annotate(f"{POSTER[name]} ({duration(reached / 60)})", end,
                       xytext=(95, label_y[name]),
                       textcoords="data", ha="left", va="center", fontsize=8.12,
                       color="black",
                       arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.7,
                                   "shrinkA": 1, "shrinkB": 2})
    right.set(xscale="log", yscale="log", ylim=(830, Y_CLIP[1]), xlim=(0.5, 600),
              xlabel="optimisation wall time [min]", ylabel="validation loss",
              title=f"Calibration, region {LW_REGION} {LW_STAGE}")
    right.set_xticks([1, 3, 10, 30, 60], ["1", "3", "10", "30", "60"])
    right.xaxis.set_minor_locator(plt.NullLocator())
    right.set_yticks([900, 1000, 1200, 1400, 1600], ["900", "1,000", "1,200", "1,400", "1,600"])
    right.yaxis.set_minor_locator(plt.NullLocator())
    right.spines["right"].set_visible(False)
    right.annotate("solid: best so far · dotted: each epoch · dots: each trial\\n"
               "open triangles: trials above 1,600 · (time to best loss)",
               (1, 0), xycoords="axes fraction", xytext=(0, -37), textcoords="offset points",
               ha="right", va="top", fontsize=7.25,
               color="black")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"poster_walltime_loss.{ext}", dpi=300, bbox_inches="tight")
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

#: Calibration runs to compare, e.g. the CPU fork-pool run and a GPU mixed run.
#: `calibration/` is the one the production runs took their crops from.
RUNS = {
    "production (cpu)": RUN / "calibration",
}
#: `yield` carries the harvest-date terms too (retention) and is the stage the
#: production crops came from; `phenology` is the stage before it.
STAGE = "yield"
TERMS = {
    "phenology/harvest_date_rmse_d": "harvest-date RMSE [d]",
    "yield/yield_rmse_t_ha": "yield RMSE [t/ha]",
}
#: SIMPLACE on the default crop, and on the per-region calibrated crops.
PRODUCTION = Path("/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated/mswx")
SIMPLACE = {
    # No default-crop SIMPLACE output survives the reorg (RUN/simplace/ is an
    # empty rebuild); the reference is drawn only if this file exists again.
    "SIMPLACE default": [p for p in [RUN / "simplace" / "simplace_europe.parquet"] if p.is_file()],
    "SIMPLACE calibrated": sorted(PRODUCTION.glob("simplace_regions/region_*/simplace_europe.parquet")),
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
if not reference.empty and reference["label"].nunique() == 2:
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
                   color=P["torchcrop_gpu"], alpha=0.3, linewidths=0)
        ax.plot(*zip(*tr.groupby("epoch")["value"].mean().items()), color=P["torchcrop_gpu"], lw=2)
        va = own[own["split"] == "val"].groupby("epoch")["value"].mean()
        ax.plot(va.index, va.values, marker="D", ms=4, lw=0, color=P["torchcrop_cpu"])
        if not reference.empty:
            ref = reference[(reference["split"] == "val") & (reference["term"] == term)
                            & (reference["region_id"] == region)]
            for _, r in ref.iterrows():
                ax.axhline(r["value"], color=P["simplace"], lw=2,
                           ls="-" if "calibrated" in r["label"] else ":")
        ax.set_title(f"region {region}", fontsize=8)
    for ax in axes.flat[len(ids):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=P["torchcrop_gpu"], alpha=0.3, marker="o", lw=0, label="train batch"),
               plt.Line2D([], [], color=P["torchcrop_gpu"], lw=2, label="train epoch mean"),
               plt.Line2D([], [], color=P["torchcrop_cpu"], marker="D", lw=0, label="validation"),
               plt.Line2D([], [], color=P["simplace"], lw=2, label="SIMPLACE calibrated"),
               plt.Line2D([], [], color=P["simplace"], lw=2, ls=":", label="SIMPLACE default")]
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
