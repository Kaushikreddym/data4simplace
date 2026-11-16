#!/bin/bash
# =============================================================================
# Shared settings for the SIMPLACE Europe run.
# Sourced by every simplace_*.sh script.
#
# Separate from torchcrop_env.sh because the two models share nothing at run
# time: SIMPLACE runs a Singularity container and needs no torch, no GPU and no
# Python beyond this package's CLI.
#
# Override anything from the environment before submitting, e.g.
#   SP_LINES_PER_TASK=5000 ./submit/baseline/submit_simplace.sh
# =============================================================================

# Every SP_* setting is exported, not just assigned: submit_simplace.sh passes
# them to the jobs with `sbatch --export=...`, which reads the *environment* of
# the submitting shell.

# --- Project & environment ---------------------------------------------------
export SP_PROJECT_DIR="${SP_PROJECT_DIR:-/data01/FDS/muduchuru/codes/GITHUB/data4simplace/cropmodelling4eu}"
export SP_CONDA_ENV="${SP_CONDA_ENV:-sdba}"
export SP_CONFIG="${SP_CONFIG:-${SP_PROJECT_DIR}/config.yaml}"

# --- Run layout --------------------------------------------------------------
export SP_RUN_NAME="${SP_RUN_NAME:-winter_wheat_2000_2024}"
export SP_LOG_DIR="${SP_LOG_DIR:-${SP_PROJECT_DIR}/submit/logs}"
# Where `cm4eu simplace build` puts the run directory. Empty -> the config's
# own <paths.output_dir>/<run_name>/simplace. Everything the run needs lands
# there, including its own submit.sh and run_task.sh, so this is the only path
# worth remembering afterwards.
export SP_RUN_DIR="${SP_RUN_DIR:-}"

# --- Smoke test ---------------------------------------------------------------
# `./submit/baseline/submit_simplace.sh --smoke` runs a small, evaluable subset instead of
# the continent: N cells spread over Germany, each inside a distinct CyBench
# NUTS-3 region so the yields can be scored against a statistic, and near enough
# to PEP725 stations that the phenology can be scored too. Germany because it is
# the only domain where both references are dense.
export SP_SMOKE_CELLS="${SP_SMOKE_CELLS:-30}"
export SP_SMOKE_DIR="${SP_SMOKE_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/de_smoke}"
# Harvest years to score: 2003, the European drought, and 2005, an ordinary
# year beside it -- the contrast the daily LAI/AGB/NNI/TRANRF comparison in
# germany_smoke_evaluation.ipynb is built around. SIMPLACE runs the window
# continuously, so 2004 is produced too as a byproduct; it is reported but not
# the point. Both inside CyBench DE's 1999-2023 coverage and past where
# PEP725's wheat records thin out.
export SP_SMOKE_START="${SP_SMOKE_START:-1989}"
export SP_SMOKE_END="${SP_SMOKE_END:-2024}"
# Empty -> the config's own simplace.iopt (1, potential production). Set to
# sweep vIOPT: unlike torchcrop's --iopt there is no CLI override for a
# solution's vIOPT, so a non-empty value here builds and runs a *separate*
# solution under its own simplace_iopt<n>/ directory (see --smoke below)
# rather than patching one run. submit_cropmodelling.sh --smoke sets this
# once per value in TC_SMOKE_IOPTS to sweep both models together; set by hand
# for a single other value, e.g. SP_IOPT=2 ./submit/baseline/submit_simplace.sh --smoke.
export SP_IOPT="${SP_IOPT:-}"

# --- Sharding ----------------------------------------------------------------
# SIMPLACE selects work by project-file line range (`-l=START-END`), so the
# task size is a line count rather than a cell list. Unlike the torchcrop
# shards these ranges are contiguous — a round-robin split is not expressible —
# so tasks over Scandinavia and Iberia have different wall times.
export SP_LINES_PER_TASK="${SP_LINES_PER_TASK:-500}"

# --- SLURM -------------------------------------------------------------------
export SP_PARTITION="${SP_PARTITION:-compute}"
# SIMPLACE's `-l=START-END` is one thread in one JVM, so a task only uses
# more cores through SP_CORES_PER_TASK below, each JVM taking ~2 (JIT and GC
# threads). The task-table worker clears SP_* before sourcing a region's
# run.env, so for production the worker count and heap come from the build
# (SlurmResources), and these values only size the allocation -- keep them in
# step.
export SP_CPUS="${SP_CPUS:-16}"
# The image's launcher hard-codes -Xmx30g and each JVM grows into it (~12 GB
# RSS for one task, 70 GB for two in run 785392, OOM at six under 80G). Capped
# at SP_JAVA_HEAP=4g, eight JVMs peaked at 21.5 GB together.
export SP_MEM="${SP_MEM:-40G}"
export SP_TIME="${SP_TIME:-12:00:00}"
export SP_MAX_CONCURRENT="${SP_MAX_CONCURRENT:-20}"
# Concurrent singularity/JVM invocations per task, each over a contiguous
# sub-range of the task's lines. Piloted on region_15 lines 501-1000
# (2026-10-23): 8 workers at a 4g heap took 5m23s against ~35 min for one
# worker, with byte-identical yearly output.
export SP_CORES_PER_TASK="${SP_CORES_PER_TASK:-8}"
# run_task.sh passes this to the launcher as -Xmx; empty keeps its 30g.
export SP_JAVA_HEAP="${SP_JAVA_HEAP:-4g}"

export HDF5_USE_FILE_LOCKING=FALSE      # BeeGFS + h5netcdf/netCDF4

sp_activate() {
    # /etc/bashrc and conda's shell hook both read unset variables, so `set -u`
    # has to stand down for the duration of the activation.
    local had_u=0
    case "$-" in *u*) had_u=1; set +u ;; esac
    # shellcheck disable=SC1090
    source ~/.bashrc
    conda activate "${SP_CONDA_ENV}" || {
        echo "ERROR: cannot activate conda env '${SP_CONDA_ENV}'" >&2
        exit 1
    }
    [ "${had_u}" -eq 1 ] && set -u
    command -v cm4eu >/dev/null || {
        echo "ERROR: 'cm4eu' not on PATH in env '${SP_CONDA_ENV}'." >&2
        echo "       Install it with: pip install -e '${SP_PROJECT_DIR}'" >&2
        exit 1
    }
    command -v singularity >/dev/null || {
        echo "ERROR: singularity not on PATH; SIMPLACE runs in its container." >&2
        exit 1
    }
}

sp_banner() {
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo "  node        : $(hostname)"
    echo "  job         : ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+(task ${SLURM_ARRAY_TASK_ID})}"
    echo "  env         : ${SP_CONDA_ENV}"
    echo "  config      : ${SP_CONFIG}"
    echo "  run         : ${SP_RUN_NAME}"
    echo "  lines/task  : ${SP_LINES_PER_TASK}"
    echo "  cores/task  : ${SP_CORES_PER_TASK}"
    echo "  started     : $(date -Is)"
    echo "=================================================="
}
