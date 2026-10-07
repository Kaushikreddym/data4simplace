#!/bin/bash
# =============================================================================
# SLURM job: the whole staged torchcrop calibration, in one process.
#
# Submitted by submit/submit_calibration.sh. Safe to sbatch directly if the
# CAL_* environment is already set (source submit/calibration_env.sh first).
#
# ONE job, not a chain, for a reason: `Calibrator.run()` carries each stage's
# fitted crop into the next *in memory* -- stage 2 starts from stage 1's
# parameters, and stage 3 from stage 2's. Splitting the stages across jobs
# would need the per-region crop files reloaded between them, which is not what
# the code does today. Every partition on this cluster is `infinite`, so there
# is nothing to gain by splitting.
#
# Restartable, but not resumable: a killed job re-reads the cached regions and
# weather (minutes) and then restarts the stage from the crop file. Only the
# finished stages on disk survive.
# =============================================================================
#SBATCH --job-name=cm_calib
#SBATCH --output=submit/logs/cm_calib_%j.log
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -uo pipefail

for _d in "${CAL_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/calibration_env.sh" ] && { CAL_SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${CAL_SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/calibration_env.sh. Set CAL_PROJECT_DIR or" >&2
    echo "       sbatch from the project root so SLURM_SUBMIT_DIR resolves it." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${CAL_SUBMIT_DIR}/calibration_env.sh"
cal_activate

cal_banner "TORCHCROP CALIBRATION -- ${CAL_STAGE}"
cal_check_inputs || exit 1

cd "${CAL_PROJECT_DIR}" || exit 1
mkdir -p "${CAL_OUT_DIR}" "${CAL_LOG_DIR}"

ARGS=(
    --config      "${CAL_CONFIG}"
    --stage       "${CAL_STAGE}"
    --crop-file   "${CAL_CROP_FILE}"
    --sowing-file "${CAL_SOWING_FILE}"
    --out-dir     "${CAL_OUT_DIR}"
    --batch-size  "${CAL_BATCH_SIZE}"
    --threads     "${CAL_THREADS}"
    --workers     "${CAL_WORKERS}"
)
[ -n "${CAL_PEP725:-}" ] && [ -d "${CAL_PEP725}" ] && ARGS+=(--pep725 "${CAL_PEP725}")
[ -n "${CAL_EPOCHS:-}" ]   && ARGS+=(--epochs "${CAL_EPOCHS}")
[ -n "${CAL_FRACTION:-}" ] && ARGS+=(--fraction "${CAL_FRACTION}")
[ "${CAL_PREPARE_ONLY:-0}" = "1" ] && ARGS+=(--prepare-only)

echo "cm4eu calibrate ${ARGS[*]}"
echo

# -u so the log is readable while the job runs; an epoch is ~15 minutes and a
# block-buffered log would show nothing for hours.
python -u -m cropmodelling4eu.cli calibrate "${ARGS[@]}"
STATUS=$?

echo
echo "=================================================="
echo "  finished    : $(date)"
echo "  exit status : ${STATUS}"
echo "  outputs     : ${CAL_OUT_DIR}"
echo "=================================================="
exit "${STATUS}"
