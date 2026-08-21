#!/bin/bash
# =============================================================================
# Shared settings for the torchcrop / LINTUL-5 Europe run.
# Sourced by every torchcrop_*.sh script.
#
# Separate from submit/env.sh because this run consumes the *finished* SIMPLACE
# export rather than producing it: it needs no data4simplace CLI, no SoilGrids
# WCS cache and no tiling, and it must not inherit env.sh's activation check on
# `data4simplace` being on PATH.
#
# Override anything from the environment before submitting, e.g.
#   TC_START_YEAR=1990 TC_N_SHARDS=20 ./submit/submit_torchcrop.sh
# =============================================================================

# Every TC_* setting is exported, not just assigned: submit_torchcrop.sh passes
# them to the jobs with `sbatch --export=...`, which reads the *environment* of
# the submitting shell.

# --- Project & environment ---------------------------------------------------
export TC_PROJECT_DIR="${TC_PROJECT_DIR:-/data01/FDS/muduchuru/codes/GITHUB/data4simplace/cropmodelling4eu}"
export TC_CONDA_ENV="${TC_CONDA_ENV:-sdba}"   # holds torch + the torchcrop install

# --- Input & output ----------------------------------------------------------
export TC_DATA_DIR="${TC_DATA_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/EU}"
export TC_RUN_NAME="${TC_RUN_NAME:-winter_wheat_2000_2024}"
export TC_OUT_DIR="${TC_OUT_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/torchcrop/${TC_RUN_NAME}}"
export TC_SHARD_DIR="${TC_SHARD_DIR:-${TC_OUT_DIR}/shards}"
export TC_LOG_DIR="${TC_LOG_DIR:-${TC_PROJECT_DIR}/submit/logs}"

# The fertilizer-composition table is not part of the europe export; it lives
# next to the SIMPLACE reference the exporter was built from.
export TC_COMPOSITION_XML="${TC_COMPOSITION_XML:-/beegfs/muduchuru/simplace/Brandenburg_1KM_winter_wheat/data/management/fertilizer_composition.xml}"

# --- Sowing dates from a finished SIMPLACE run --------------------------------
# Empty -> torchcrop sows on the export's own calendar. Set to the
# sowing_from_simplace.csv a collected SIMPLACE run writes and every shard sows
# on SIMPLACE's *simulated* dates instead, which is what
# submit/submit_cropmodelling.sh chains. A (cell, season) missing from the file
# is dropped rather than falling back, so the two models never end up on two
# different sowing conventions inside one output.
export TC_SOWING_FILE="${TC_SOWING_FILE:-}"
# A SLURM dependency to hang the shard array off, e.g. "afterok:12345". Used by
# submit_cropmodelling.sh to queue torchcrop behind SIMPLACE in one call.
export TC_DEPENDENCY="${TC_DEPENDENCY:-}"

# --- Smoke test ---------------------------------------------------------------
# `./submit/submit_torchcrop.sh --smoke` runs a small, evaluable subset instead
# of the continent: N cells spread over Germany, each inside a distinct CyBench
# NUTS-3 region so the yields can be scored against a statistic, and near enough
# to PEP725 stations that the phenology can be scored too.
#
# The defaults are deliberately identical to simplace_env.sh's SP_SMOKE_*: the
# two smoke runs share one directory, one derived config and one cell list, so
# the models are compared on the same cells and the same seasons rather than on
# two similar-looking subsets. Override one and you are no longer comparing.
export TC_CONFIG="${TC_CONFIG:-${TC_PROJECT_DIR}/config.yaml}"
export TC_SMOKE_CELLS="${TC_SMOKE_CELLS:-30}"
export TC_SMOKE_DIR="${TC_SMOKE_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/de_smoke}"
# Harvest years to score: 2003 (drought) and 2005 (ordinary) -- see
# simplace_env.sh's SP_SMOKE_START/END, which this mirrors.
export TC_SMOKE_START="${TC_SMOKE_START:-2003}"
export TC_SMOKE_END="${TC_SMOKE_END:-2005}"
# torchcrop-only: the smoke test runs all three over the same cells and
# seasons, since torchcrop is cheap enough to sweep and SIMPLACE (one fixed
# vIOPT per solution run) is not. 4 is dropped -- no European soil P or K
# layer exists, so it would run on built-in defaults rather than on data, the
# same reason TC_IOPT itself defaults to 3.
export TC_SMOKE_IOPTS="${TC_SMOKE_IOPTS:-1 2 3}"

# --- What to simulate --------------------------------------------------------
export TC_CROP="${TC_CROP:-wheat}"
# Harvest years. Winter wheat sown 1 Sep of Y-1, so the earliest possible is
# 1980 (the weather export starts 1979-01-01) and the latest is 2024.
export TC_START_YEAR="${TC_START_YEAR:-2000}"
export TC_END_YEAR="${TC_END_YEAR:-2024}"
# 1 potential | 2 water-limited | 3 water+N | 4 water+NPK.
# 3 is the default because no European soil P or K data exists — iopt=4 would
# run on torchcrop's built-in defaults, not on data.
export TC_IOPT="${TC_IOPT:-3}"
# 0/1: also write torchcrop_daily_shard_<n>.parquet beside each summary shard
# -- the same simulation's day-by-day trajectory (see run_shard's `daily`),
# so it costs no extra model pass, only the extra rows written to disk. Off
# by default: the Europe run is already tiled for memory (see the run's own
# notes), and the daily table is long-format, one row per (cell, day,
# variable), so it is far larger than the summary.
export TC_DAILY="${TC_DAILY:-0}"
export TC_DAILY_VARIABLES="${TC_DAILY_VARIABLES:-LAI AGB NNI TRANRF}"

# --- Working directory --------------------------------------------------------
# <TC_OUT_DIR>/workspace holds what the run will *use*, written before the array
# is submitted: crop_<crop>.yaml (the crop parameters, loadable and editable),
# config_run.yaml, and the audit against SIMPLACE's crop file. A run whose
# parameters live only inside the installed torchcrop package cannot be checked
# by opening anything, which is what this fixes.
export TC_WORK_DIR="${TC_WORK_DIR:-${TC_OUT_DIR}/workspace}"
export TC_CROP_FILE="${TC_CROP_FILE:-${TC_WORK_DIR}/crop_${TC_CROP}.yaml}"
# torchcrop | simplace -- whose crop parameters the run uses. `simplace` (the
# default, matching the smoke test) rebuilds the crop file from the solution's
# own crop.xml, which is what makes a SIMPLACE comparison a comparison of the
# models -- 21 of 72 parameters (TSUM1, TSUM2, DTSMTB, NMAXSO, ...) differ from
# torchcrop's bundled preset, which is what made an earlier smoke run's yield
# come out 3.6x low before this was wired up. `torchcrop` reverts to that
# bundled preset, written out unchanged, for a run that intentionally wants
# torchcrop's own calibration rather than SIMPLACE's.
export TC_CROP_SOURCE="${TC_CROP_SOURCE:-simplace}"
# The smoke test's own crop source, kept separate from TC_CROP_SOURCE above so
# each can be overridden independently -- both default to `simplace` now.
export TC_SMOKE_CROP_SOURCE="${TC_SMOKE_CROP_SOURCE:-simplace}"
export TC_SIMPLACE_TEMPLATE="${TC_SIMPLACE_TEMPLATE:-/data01/FDS/muduchuru/codes/SIMPLACE/Brandenburg_1KM_winter_wheat}"

# --- Sharding & batching -----------------------------------------------------
# One SLURM array task per shard; cells are dealt round-robin so every shard
# spans the whole domain and the tasks finish together.
export TC_N_SHARDS="${TC_N_SHARDS:-20}"
# Cells per model call. LINTUL-5 is a Python loop over days, so the per-cell
# cost falls sharply with batch size: measured 40 ms/cell-year at B=128,
# 9.4 at B=1024, 5.2 at B=2048. Peak RSS at B=2048 is 1.7 GB, so the batch is
# limited by ModelOutput retaining every per-day state, not by the cell count.
export TC_BATCH_SIZE="${TC_BATCH_SIZE:-2048}"
# Weather reads are gzip-bound and release the GIL, so threads scale here.
export TC_IO_WORKERS="${TC_IO_WORKERS:-16}"
# The model works on [B]-shaped tensors — too small to repay many intra-op
# threads. Cores are better spent on TC_IO_WORKERS.
export TC_TORCH_THREADS="${TC_TORCH_THREADS:-4}"

# --- SLURM -------------------------------------------------------------------
export TC_PARTITION="${TC_PARTITION:-compute}"
export TC_CPUS="${TC_CPUS:-16}"
export TC_MEM="${TC_MEM:-16G}"
# Measured ~30 min per shard for 45 seasons x 6 900 cells at B=2048 on one
# compute node; 6 h leaves room for a slow node and a wider year range.
export TC_TIME="${TC_TIME:-06:00:00}"
export TC_MAX_CONCURRENT="${TC_MAX_CONCURRENT:-20}"
export TC_MAPS_MEM="${TC_MAPS_MEM:-48G}"
export TC_MAPS_TIME="${TC_MAPS_TIME:-01:00:00}"

# --- Runtime tuning ----------------------------------------------------------
# One task per shard is already the parallelism; stop the BLAS/OpenMP layers
# from oversubscribing the cores SLURM allocated.
export OMP_NUM_THREADS="${TC_TORCH_THREADS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE      # BeeGFS + h5netcdf/netCDF4

tc_activate() {
    # /etc/bashrc and conda's shell hook both read unset variables, so `set -u`
    # has to stand down for the duration of the activation.
    local had_u=0
    case "$-" in *u*) had_u=1; set +u ;; esac
    # shellcheck disable=SC1090
    source ~/.bashrc
    conda activate "${TC_CONDA_ENV}" || {
        echo "ERROR: cannot activate conda env '${TC_CONDA_ENV}'" >&2
        exit 1
    }
    [ "${had_u}" -eq 1 ] && set -u
    python -c "import torch, torchcrop, cropmodelling4eu" 2>/dev/null || {
        echo "ERROR: torch/torchcrop not importable in env '${TC_CONDA_ENV}'." >&2
        echo "       Install with: pip install -e /beegfs/muduchuru/pkgs_fnl/torchcrop and pip install -e ${TC_PROJECT_DIR}[torchcrop]" >&2
        exit 1
    }
}

tc_banner() {
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo "  node        : $(hostname)"
    echo "  job         : ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+(task ${SLURM_ARRAY_TASK_ID})}"
    echo "  env         : ${TC_CONDA_ENV}"
    echo "  input       : ${TC_DATA_DIR}"
    echo "  output      : ${TC_OUT_DIR}"
    echo "  seasons     : ${TC_START_YEAR}-${TC_END_YEAR}  (iopt=${TC_IOPT}, crop=${TC_CROP})"
    echo "  shards      : ${TC_N_SHARDS}  batch=${TC_BATCH_SIZE}"
    echo "  started     : $(date -Is)"
    echo "=================================================="
}

# Resolve submit/ when SLURM runs a *copy* of a script from the node spool dir
# (/var/spool/slurmd/job<id>/slurm_script), where BASH_SOURCE cannot find
# env.sh next to it.
tc_find_submit_dir() {
    local d
    for d in "${TC_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
             "$(dirname "${BASH_SOURCE[1]}")"; do
        [ -f "${d}/torchcrop_env.sh" ] && { echo "${d}"; return 0; }
    done
    echo "ERROR: cannot locate submit/torchcrop_env.sh. Set TC_PROJECT_DIR or" >&2
    echo "       sbatch from the project root so SLURM_SUBMIT_DIR resolves it." >&2
    return 1
}
