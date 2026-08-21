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
#   SP_LINES_PER_TASK=5000 ./submit/submit_simplace.sh
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
# `./submit/submit_simplace.sh --smoke` runs a small, evaluable subset instead of
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
# for a single other value, e.g. SP_IOPT=2 ./submit/submit_simplace.sh --smoke.
export SP_IOPT="${SP_IOPT:-}"

# --- Sharding ----------------------------------------------------------------
# SIMPLACE selects work by project-file line range (`-l=START-END`), so the
# task size is a line count rather than a cell list. Unlike the torchcrop
# shards these ranges are contiguous — a round-robin split is not expressible —
# so tasks over Scandinavia and Iberia have different wall times.
export SP_LINES_PER_TASK="${SP_LINES_PER_TASK:-500}"

# --- SLURM -------------------------------------------------------------------
export SP_PARTITION="${SP_PARTITION:-compute}"
# SIMPLACE's `-l=START-END` is one thread in one JVM, so SP_CPUS only buys
# concurrency through SP_CORES_PER_TASK below -- sized to it (+1 for the
# run_task.sh wrapper), not to the node, since a lone JVM asking for a whole
# node schedules far behind smaller jobs on a busy fair-share cluster and
# leaves every core past SP_CORES_PER_TASK idle regardless.
export SP_CPUS="${SP_CPUS:-2}"
# Each SIMPLACE JVM reports its own container as "RAM: 32.0 GB" regardless of
# what SP_MEM actually is, and several concurrent JVMs each approaching that
# is what OOM-killed SP_CORES_PER_TASK=6 at this same SP_MEM (sacct: MaxRSS
# pinned at the 80G cap, exit 0:125, after ~18 min -- see run 782877). 80G
# comfortably covers one JVM with a wide margin and is unchanged from what
# already ran successfully at up to 6 concurrent JVMs.
export SP_MEM="${SP_MEM:-80G}"
export SP_TIME="${SP_TIME:-12:00:00}"
export SP_MAX_CONCURRENT="${SP_MAX_CONCURRENT:-20}"
# SIMPLACE's `-l=START-END` runs in one thread of one JVM, so SP_CPUS alone
# leaves every core past the first idle. run_task.sh fans a task's own line
# range across this many concurrent singularity/JVM invocations instead --
# the same pattern SP_MAX_CONCURRENT already uses across array tasks, just
# within one.
#
# 1 (default) is the proven-safe setting: one JVM per task cannot oversubscribe
# against itself. Values above 1 multiply per-JVM memory (each JVM behaves as
# though it has ~32 GB to itself, see the SP_MEM note above) faster than
# SP_MEM scales with it -- 6 OOM-killed most tasks of a retry at this SP_MEM.
# 2 completed cleanly (piloted on line range 2501-3000, job 785392: 48m33s,
# peak 70 GB of the 80G budget) -- a viable speedup for a *future* run, not
# applied here since the array already queued at 1. Raise it only with SP_MEM
# raised to match, and only after piloting a single range with
# `submit.sh --lines START-END` first -- a bad value fails an entire array's
# worth of nodes, not one task.
export SP_CORES_PER_TASK="${SP_CORES_PER_TASK:-1}"

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
