#!/bin/bash
# =============================================================================
# Driver for the torchcrop calibration over Europe.
#
#   ./submit/submit_calibration.sh                 # the full staged run
#   ./submit/submit_calibration.sh --prepare       # regions + observations +
#                                                  # weather cache, then stop
#   ./submit/submit_calibration.sh --stage phenology
#   ./submit/submit_calibration.sh --smoke         # 3 epochs, run here, ~1 h
#   ./submit/submit_calibration.sh --dry-run       # print the plan, submit nothing
#
# Tunables live in submit/calibration_env.sh and can be overridden per call:
#   CAL_EPOCHS=20 CAL_FRACTION=0.4 ./submit/submit_calibration.sh
#
# RUN --prepare FIRST on a new export. It is where a missing reference, an
# empty cell set or a region that no observation reaches shows up -- in minutes
# rather than after the first epoch. It also builds the caches the real run
# then reuses, so it costs nothing twice.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/calibration_env.sh"

DRY_RUN=0
SMOKE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)  DRY_RUN=1 ;;
        --prepare)  export CAL_PREPARE_ONLY=1 ;;
        --smoke)    SMOKE=1 ;;
        --stage)    shift; export CAL_STAGE="${1:?--stage needs a value}" ;;
        --stage=*)  export CAL_STAGE="${1#*=}" ;;
        -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

export CAL_PREPARE_ONLY="${CAL_PREPARE_ONLY:-0}"
mkdir -p "${CAL_LOG_DIR}"

# Checked here as well as in the job: a typo in a path should cost a second at
# the prompt, not a place in the queue and an hour of weather-cache building.
cal_check_inputs || exit 1

echo "=================================================="
echo "  torchcrop calibration"
echo "=================================================="
echo "  config      : ${CAL_CONFIG}"
echo "  crop        : ${CAL_CROP_FILE}"
echo "  sowing      : ${CAL_SOWING_FILE}  (static, from the finished SIMPLACE run)"
echo "  pep725      : ${CAL_PEP725:-(none -- tier 2 stays frozen)}"
echo "  stage(s)    : ${CAL_STAGE}"
echo "  epochs      : ${CAL_EPOCHS:-optim.max_epochs}"
echo "  fraction    : ${CAL_FRACTION:-data.fraction}"
echo "  batch       : ${CAL_BATCH_SIZE} cell-years,  threads ${CAL_THREADS}"
echo "  workers     : ${CAL_WORKERS} processes on 1 node (${CAL_CPUS} cores)"
echo "                the ceiling is the region count; more nodes would idle"
echo "  output      : ${CAL_OUT_DIR}"
echo "=================================================="

if [ "${SMOKE}" = "1" ]; then
    # Three epochs in the foreground. Not a fit -- about five Adam steps per
    # region against the ~146 a full run buys -- but it exercises every part of
    # the pipeline and already moves the harvest bias, so it is the right thing
    # to run before spending days of wall clock.
    export CAL_EPOCHS="${CAL_EPOCHS:-3}"
    export CAL_STAGE="${CAL_STAGE:-phenology}"
    echo "SMOKE: ${CAL_EPOCHS} epochs of ${CAL_STAGE}, here, ~1 h"
    [ "${DRY_RUN}" = "1" ] && { echo "(dry run)"; exit 0; }
    exec bash "${SUBMIT_DIR}/calibrate.sh"
fi

CMD=(sbatch
    --job-name=cm_calib
    --partition="${CAL_PARTITION}"
    --cpus-per-task="${CAL_CPUS}"
    --mem="${CAL_MEM}"
    --time="${CAL_TIME}"
    # One file, not two: everything interesting here is *logging*, which goes
    # to stderr, so a split would leave the .out holding only the banner while
    # the run's actual progress hid in the .err. sbatch merges the streams when
    # --error is omitted.
    --output="${CAL_LOG_DIR}/cm_calib_%j.log"
    --export=ALL
    "${SUBMIT_DIR}/calibrate.sh"
)

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi

OUT=$("${CMD[@]}") || { echo "sbatch failed: ${OUT}" >&2; exit 1; }
JOB="${OUT##* }"
echo "submitted job ${JOB}"
echo
echo "  watch    : tail -f ${CAL_LOG_DIR}/cm_calib_${JOB}.log"
echo "  progress : grep 'epoch ' ${CAL_LOG_DIR}/cm_calib_${JOB}.log"
echo "  results  : ${CAL_OUT_DIR}/<stage>/summary.json"
echo "  cancel   : scancel ${JOB}"
