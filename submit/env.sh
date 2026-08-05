#!/bin/bash
# =============================================================================
# Shared settings for the tiled Europe run. Sourced by every submit script.
#
# Every value can be overridden from the environment before submitting, e.g.
#   D4S_TILE_DEG=2.5 D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh
# =============================================================================

# Every D4S_* setting is exported, not just assigned: submit_europe.sh passes
# them to the jobs with `sbatch --export=...`, which reads the *environment* of
# the submitting shell. A plain shell variable is invisible there, so a bare
# `: "${VAR:=default}"` would silently propagate nothing.

# --- Project & environment ---------------------------------------------------
export D4S_PROJECT_DIR="${D4S_PROJECT_DIR:-/data01/FDS/muduchuru/codes/GITHUB/data4simplace}"
export D4S_CONDA_ENV="${D4S_CONDA_ENV:-sdba}"  # env holding the editable install
export D4S_CONFIG="${D4S_CONFIG:-${D4S_PROJECT_DIR}/config.yaml}"

# --- Run layout --------------------------------------------------------------
# Outputs go to /beegfs: the weather export writes one gzipped CSV per cropland
# cell (O(10^5) files for Europe), which is far more inodes than /data01 wants.
export D4S_RUN_NAME="${D4S_RUN_NAME:-europe_torchcrop}"
export D4S_OUT_DIR="${D4S_OUT_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/${D4S_RUN_NAME}}"
export D4S_WORK_DIR="${D4S_WORK_DIR:-${D4S_OUT_DIR}/_work}"   # per-task configs
export D4S_LOG_DIR="${D4S_LOG_DIR:-${D4S_PROJECT_DIR}/submit/logs}"

# The SoilGrids WCS cache is keyed by coverage id only (no bbox), so tiles must
# NOT share one cache directory - see submit/tile_config.py for the details.
export D4S_WCS_CACHE_ROOT="${D4S_WCS_CACHE_ROOT:-${D4S_WORK_DIR}/wcs_cache}"

# The run config: a copy of D4S_CONFIG with paths.output_dir pointed at the run
# directory. Written by submit_europe.sh, read by every job of the run.
export D4S_RUN_CONFIG="${D4S_RUN_CONFIG:-${D4S_WORK_DIR}/config_run.yaml}"

# --- Tiling ------------------------------------------------------------------
# 5.0 deg over the 690x380 Europe grid => 112 tiles of 50x50 cells.
# Halve it to 2.5 deg (448 tiles) if a tile OOMs or overruns its walltime.
export D4S_TILE_DEG="${D4S_TILE_DEG:-5.0}"

# --- SLURM -------------------------------------------------------------------
export D4S_PARTITION="${D4S_PARTITION:-compute}"
# Sized to the MSWX read pool (climate.read_workers, capped at 16): that pool is
# the only real parallelism in a tile. Measured peak RSS of a running tile was
# 4.7 GB, so the previous 40 cpus / 80 GB reserved a whole 80-core, 95 GB node
# per tile to run one ~33%-busy core - which is what capped the run at 8
# concurrent tiles regardless of how many nodes were idle. At 16/24G three tiles
# share a node.
export D4S_CPUS="${D4S_CPUS:-16}"
export D4S_MEM="${D4S_MEM:-24G}"
export D4S_TIME="${D4S_TIME:-2-00:00:00}"
# Concurrency cap. Each task pulls its own SoilGrids subsets from the ISRIC WCS
# and its own CORINE tiles from the CLC WMS; keep this modest so the run stays
# within what those public services will serve without throttling us. This is a
# politeness limit on *external services*, not a compute limit - raise it only
# as far as ISRIC/CLC tolerate, even though the cluster could run all 112 at once.
export D4S_MAX_CONCURRENT="${D4S_MAX_CONCURRENT:-24}"

# --- Runtime tuning ----------------------------------------------------------
# One task per tile is already the parallelism; stop the BLAS/OpenMP layers from
# oversubscribing the cores SLURM allocated. OMP is deliberately *not* D4S_CPUS:
# the MSWX reader forks up to 16 processes, and giving each one 16 OpenMP threads
# would oversubscribe the allocation 16-fold.
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GDAL_CACHEMAX=512
export GDAL_NUM_THREADS=ALL_CPUS
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif
export HDF5_USE_FILE_LOCKING=FALSE      # BeeGFS + h5netcdf/netCDF4

d4s_activate() {
    # /etc/bashrc and conda's shell hook both read unset variables, so `set -u`
    # has to stand down for the duration of the activation.
    local had_u=0
    case "$-" in *u*) had_u=1; set +u ;; esac
    # shellcheck disable=SC1090
    source ~/.bashrc
    conda activate "${D4S_CONDA_ENV}" || {
        echo "ERROR: cannot activate conda env '${D4S_CONDA_ENV}'" >&2
        exit 1
    }
    [ "${had_u}" -eq 1 ] && set -u
    command -v data4simplace >/dev/null || {
        echo "ERROR: 'data4simplace' not on PATH in env '${D4S_CONDA_ENV}'." >&2
        echo "       Install it with: pip install -e '${D4S_PROJECT_DIR}[dev]'" >&2
        exit 1
    }
}

d4s_banner() {
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo "  node        : $(hostname)"
    echo "  job         : ${SLURM_JOB_ID:-none} ${SLURM_ARRAY_TASK_ID:+(task ${SLURM_ARRAY_TASK_ID})}"
    echo "  env         : ${D4S_CONDA_ENV}"
    echo "  run config  : ${D4S_RUN_CONFIG}"
    echo "  output dir  : ${D4S_OUT_DIR}"
    echo "  tile size   : ${D4S_TILE_DEG} deg"
    echo "  started     : $(date -Is)"
    echo "=================================================="
}
