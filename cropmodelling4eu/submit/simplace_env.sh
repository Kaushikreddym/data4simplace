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
# Harvest years to score. The full window is wasted here: CyBench DE covers
# 1999-2023 and PEP725's wheat records thin out sharply before 2000.
export SP_SMOKE_START="${SP_SMOKE_START:-2000}"
export SP_SMOKE_END="${SP_SMOKE_END:-2010}"

# --- Sharding ----------------------------------------------------------------
# SIMPLACE selects work by project-file line range (`-l=START-END`), so the
# task size is a line count rather than a cell list. Unlike the torchcrop
# shards these ranges are contiguous — a round-robin split is not expressible —
# so tasks over Scandinavia and Iberia have different wall times.
export SP_LINES_PER_TASK="${SP_LINES_PER_TASK:-500}"

# --- SLURM -------------------------------------------------------------------
export SP_PARTITION="${SP_PARTITION:-compute}"
export SP_CPUS="${SP_CPUS:-80}"
export SP_MEM="${SP_MEM:-80G}"
export SP_TIME="${SP_TIME:-12:00:00}"
export SP_MAX_CONCURRENT="${SP_MAX_CONCURRENT:-10}"

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
    echo "  started     : $(date -Is)"
    echo "=================================================="
}
