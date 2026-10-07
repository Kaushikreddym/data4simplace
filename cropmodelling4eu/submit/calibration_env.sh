#!/bin/bash
# =============================================================================
# Shared settings for the torchcrop calibration over Europe.
# Sourced by submit/submit_calibration.sh and submit/calibrate.sh.
#
# Separate from torchcrop_env.sh because this run consumes the *finished*
# outputs of both models rather than producing one: it needs a completed
# SIMPLACE run (for its sowing dates), a prepared torchcrop workspace (for the
# crop it starts from) and the CLMS/CyBench references — and it must not
# inherit the shard array's settings, which describe a production run.
#
# Override anything from the environment before submitting, e.g.
#   CAL_STAGE=phenology CAL_EPOCHS=20 ./submit/submit_calibration.sh
# =============================================================================

# Every CAL_* setting is exported, not just assigned: submit_calibration.sh
# passes them to the job with `sbatch --export=...`, which reads the
# *environment* of the submitting shell.

# --- Project & environment ---------------------------------------------------
export CAL_PROJECT_DIR="${CAL_PROJECT_DIR:-/data01/FDS/muduchuru/codes/GITHUB/data4simplace/cropmodelling4eu}"
export CAL_CONDA_ENV="${CAL_CONDA_ENV:-sdba}"   # holds torch + the torchcrop install
export CAL_CONFIG="${CAL_CONFIG:-${CAL_PROJECT_DIR}/config.yaml}"

# --- The run this calibrates against -----------------------------------------
export CAL_RUN_NAME="${CAL_RUN_NAME:-winter_wheat_2000_2024_euval}"
export CAL_RUN_DIR="${CAL_RUN_DIR:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/${CAL_RUN_NAME}}"
export CAL_OUT_DIR="${CAL_OUT_DIR:-${CAL_RUN_DIR}/calibration}"

# --- The two inputs that are not optional in practice ------------------------
#
# THE CROP. Without it torchcrop's *bundled* preset is calibrated, which is a
# different crop from the one the production run uses -- different `idsl`,
# `tsum1`, and no vernalisation block at all. Every number in CALIBRATION.md is
# against the harmonised SUSTAg WW block, which is what this file is.
export CAL_CROP_FILE="${CAL_CROP_FILE:-${CAL_RUN_DIR}/torchcrop/workspace/crop_wheat.yaml}"
#
# THE SOWING DATES. A *static* file written by `cm4eu simplace collect` from
# the finished SIMPLACE run -- nothing here re-runs SIMPLACE. It is the sowing
# convention the production run and every CALIBRATION.md number are on, and it
# is per season. The export's own site calendar (the fallback) sows a median
# 22 days later and gives every cell ONE fixed date for all seasons, which
# would leave simulated emergence with no interannual variation at all -- the
# very thing stage 1 exists to fit.
export CAL_SOWING_FILE="${CAL_SOWING_FILE:-${CAL_RUN_DIR}/simplace/sowing_from_simplace.csv}"

# --- PEP725, and what it buys ------------------------------------------------
# The station observations. Empty -> no heading stage and no ground-truth
# emergence anomaly, so the identifiability rule binds at TWO dated stages
# everywhere and tier 2 (`vbase`, `phottb`, `vernrt`,
# `vernalisation_devstage`, `tbasem`, `dtsmtb`) stays frozen. With it, the
# regions PEP725 reaches gain a third stage.
export CAL_PEP725="${CAL_PEP725:-/data01/FDS/muduchuru/Data/Agri/PEP725}"

# --- What to run -------------------------------------------------------------
# `all` runs phenology -> yield -> LAI and then the joint fine-tune, carrying
# each stage's fitted crop into the next. Stages accumulate terms rather than
# replacing the objective, so running one alone is a diagnostic, not the plan.
export CAL_STAGE="${CAL_STAGE:-all}"
export CAL_EPOCHS="${CAL_EPOCHS:-}"        # empty -> optim.max_epochs (80)
export CAL_FRACTION="${CAL_FRACTION:-}"    # empty -> data.fraction (0.20)

# --- Sizing, all of it measured ----------------------------------------------
#
# BATCH SIZE. LINTUL-5's day loop is Python and costs per *day*, so a batch of
# 768 cell-years costs 1.33x the wall time of 192 for 4x the cells, at 1.3 GB
# against 1.0. Measured over a 465-day window with sqrt(T) checkpointing:
#
#     B= 96  13.7 s  142.8 ms/cell  1.0 GB
#     B=192  12.6 s   65.4 ms/cell  1.0 GB
#     B=384  16.0 s   41.5 ms/cell  1.1 GB
#     B=768  16.8 s   21.9 ms/cell  1.3 GB
#
# The gain is capped by region homogeneity, not by memory: a batch carries one
# region's parameters, so the batch *count* has a floor at the region count
# (~29) however large the batch is. Raising `fraction` is what fills those
# batches, and it is close to free for the same reason.
export CAL_BATCH_SIZE="${CAL_BATCH_SIZE:-768}"
#
# THREADS. Measured to make no difference whatsoever -- 1, 4, 10 and 20 threads
# all land within noise of each other, at both B=192 and B=768, because each
# day's tensor ops are on [B]-shaped arrays far too small to repay torch's
# intra-op synchronisation. The wall time is interpreter overhead x 465 days.
# So ask for cores to run more jobs, not a faster one. The cores requested
# below are for the ONE-OFF weather-cache build, which is gzip-bound and does
# scale with `io_workers`.
export CAL_THREADS="${CAL_THREADS:-1}"
#
# WORKERS -- the parallelism that does work, and why it is ONE node.
#
# A batch carries one region's parameters (a table ordinate cannot be
# [B]-shaped), and the prior/pooling penalties are scoped to that region, so
# batches of *different* regions share no latents and their gradients are
# disjoint. Computing them concurrently gives the same numbers as computing
# them in sequence, so `--workers N` is a speedup and not an approximation
# (bar one step of staleness in the pooling mean within a wave -- see
# calibration/parallel.py).
#
# The ceiling is therefore the REGION COUNT, ~29 on this domain, since at
# batch 768 each region contributes about one batch per epoch. A compute node
# has 80 cores, so **one node already covers the whole available parallelism**
# and a second would idle. This is deliberately not a multi-node job: there is
# nothing for the extra nodes to do. Threads are no substitute -- the day loop
# is a Python loop and holds the GIL, which is why 1/4/10/20 threads measured
# identical.
export CAL_WORKERS="${CAL_WORKERS:-29}"

# --- SLURM -------------------------------------------------------------------
export CAL_PARTITION="${CAL_PARTITION:-compute}"
# One core per worker, plus a few for the parent and the gzip-bound weather
# cache. Not more: N workers on N cores is the point, and asking for the whole
# node would keep others off it for no gain.
export CAL_CPUS="${CAL_CPUS:-32}"
# ~1.3 GB per worker at batch 768, and the workers are forked so the season
# cache is shared (it is a file-backed mmap, which fork does not copy) rather
# than duplicated 29 times. 64G leaves room for the parent and the page cache.
export CAL_MEM="${CAL_MEM:-64G}"
# Every partition on this cluster is `infinite`, so the whole staged run goes
# in one job and each stage inherits the previous stage's fitted crop in
# memory. A conservative ceiling: ~15 min an epoch for phenology (8 CLMS
# seasons) and roughly double that for yield and LAI (CyBench's 25 years make
# the pool ~2.6x larger), so 80 epochs x 4 stages is a few days. Early stopping
# (`patience: 12`) usually ends a stage well before its cap.
export CAL_TIME="${CAL_TIME:-7-00:00:00}"
export CAL_LOG_DIR="${CAL_LOG_DIR:-${CAL_PROJECT_DIR}/submit/logs}"

# --- Helpers -----------------------------------------------------------------

cal_activate() {
    # /etc/bashrc and conda's shell hook both read unset variables, so `set -u`
    # has to stand down for the duration of the activation.
    local had_u=0
    case "$-" in *u*) had_u=1; set +u ;; esac
    # shellcheck disable=SC1090
    source ~/.bashrc
    conda activate "${CAL_CONDA_ENV}" || {
        echo "ERROR: cannot activate conda env '${CAL_CONDA_ENV}'" >&2
        exit 1
    }
    [ "${had_u}" -eq 1 ] && set -u
    python -c "import torch, torchcrop, cropmodelling4eu" 2>/dev/null || {
        echo "ERROR: torch/torchcrop not importable in env '${CAL_CONDA_ENV}'." >&2
        exit 1
    }
}

# The two inputs whose absence changes the science silently rather than loudly:
# a missing crop file calibrates a different crop, a missing sowing file
# calibrates a different sowing convention. Both are checked before the job is
# submitted, not after it has spent an hour building a weather cache.
cal_check_inputs() {
    local fatal=0
    if [ ! -f "${CAL_CROP_FILE}" ]; then
        echo "ERROR: no crop file at ${CAL_CROP_FILE}" >&2
        echo "       Without it torchcrop's bundled preset is calibrated, which is" >&2
        echo "       a different crop from the production run's. Build the" >&2
        echo "       workspace first (cm4eu torchcrop workspace / prepare_torchcrop_workspace.py)." >&2
        fatal=1
    fi
    if [ ! -f "${CAL_SOWING_FILE}" ]; then
        echo "ERROR: no sowing table at ${CAL_SOWING_FILE}" >&2
        echo "       It is written by 'cm4eu simplace collect' from the finished" >&2
        echo "       SIMPLACE run. Without it the export's site calendar is used," >&2
        echo "       which sows a median 22 d later and gives every cell one fixed" >&2
        echo "       date for all seasons -- no interannual emergence signal at all." >&2
        fatal=1
    fi
    if [ -n "${CAL_PEP725}" ] && [ ! -d "${CAL_PEP725}" ]; then
        echo "WARNING: no PEP725 at ${CAL_PEP725}; tier-2 parameters stay frozen" >&2
    fi
    return "${fatal}"
}

cal_banner() {
    echo "=================================================="
    echo "$1"
    echo "=================================================="
    echo "  node        : $(hostname)"
    echo "  job         : ${SLURM_JOB_ID:-none}"
    echo "  env         : ${CAL_CONDA_ENV}"
    echo "  config      : ${CAL_CONFIG}"
    echo "  crop        : ${CAL_CROP_FILE}"
    echo "  sowing      : ${CAL_SOWING_FILE}"
    echo "  pep725      : ${CAL_PEP725:-(none -- tier 2 frozen)}"
    echo "  stage(s)    : ${CAL_STAGE}"
    echo "  batch/threads: ${CAL_BATCH_SIZE} / ${CAL_THREADS}"
    echo "  workers     : ${CAL_WORKERS} (ceiling is the region count, ~29)"
    echo "  output      : ${CAL_OUT_DIR}"
    echo "  started     : $(date)"
    echo "=================================================="
}
