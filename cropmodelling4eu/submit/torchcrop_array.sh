#!/bin/bash
# =============================================================================
# SLURM array job: one task = one shard of the runnable cell set.
#
# Submitted by submit/submit_torchcrop.sh, which supplies --array from
# TC_N_SHARDS. Do not sbatch this directly unless you also pass --array.
#
# Each task is fully independent: it reads only its own round-robin slice of
# the cells and writes only its own Parquet file, so tasks never contend and a
# failed shard is re-runnable on its own (--retry skips the finished ones).
# =============================================================================
#SBATCH --job-name=tc_shard
#SBATCH --output=submit/logs/tc_shard_%A_%a.out
#SBATCH --error=submit/logs/tc_shard_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -uo pipefail

for _d in "${TC_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/torchcrop_env.sh" ] && { TC_SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${TC_SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/torchcrop_env.sh. Set TC_PROJECT_DIR or" >&2
    echo "       sbatch from the project root so SLURM_SUBMIT_DIR resolves it." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${TC_SUBMIT_DIR}/torchcrop_env.sh"
tc_activate

SHARD="${SLURM_ARRAY_TASK_ID:?torchcrop_array.sh must run as a SLURM array job}"
tc_banner "TORCHCROP SHARD ${SHARD} / ${TC_N_SHARDS}"

cd "${TC_PROJECT_DIR}" || exit 1
mkdir -p "${TC_SHARD_DIR}"

CROP_ARG=()
if [ -n "${TC_CROP_FILE:-}" ] && [ -f "${TC_CROP_FILE}" ]; then
    CROP_ARG=(--crop-file "${TC_CROP_FILE}")
    echo "Crop params  : ${TC_CROP_FILE}"
fi

SOWING_ARG=()
if [ -n "${TC_SOWING_FILE:-}" ]; then
    [ -f "${TC_SOWING_FILE}" ] || {
        echo "ERROR: TC_SOWING_FILE=${TC_SOWING_FILE} does not exist. The" >&2
        echo "       SIMPLACE run it comes from has not been collected." >&2
        exit 1
    }
    SOWING_ARG=(--sowing-file "${TC_SOWING_FILE}")
    echo "Sowing dates : ${TC_SOWING_FILE} (SIMPLACE-simulated)"
fi

DAILY_ARG=()
if [ "${TC_DAILY:-0}" -eq 1 ]; then
    # shellcheck disable=SC2206
    DAILY_ARG=(--daily --daily-variables ${TC_DAILY_VARIABLES})
    echo "Daily output : torchcrop_daily_shard_<n>.parquet (${TC_DAILY_VARIABLES})"
fi

srun --cpu-bind=none python -m cropmodelling4eu.torchcrop.run \
    --shard "${SHARD}" \
    --n-shards "${TC_N_SHARDS}" \
    --export-dir "${TC_DATA_DIR}" \
    --out-dir "${TC_SHARD_DIR}" \
    --composition "${TC_COMPOSITION_XML}" \
    --start-year "${TC_START_YEAR}" \
    --end-year "${TC_END_YEAR}" \
    --crop "${TC_CROP}" \
    --iopt "${TC_IOPT}" \
    --batch-size "${TC_BATCH_SIZE}" \
    --io-workers "${TC_IO_WORKERS}" \
    --torch-threads "${TC_TORCH_THREADS}" \
    --device cpu \
    ${CROP_ARG[@]:+"${CROP_ARG[@]}"} \
    ${SOWING_ARG[@]:+"${SOWING_ARG[@]}"} \
    ${DAILY_ARG[@]:+"${DAILY_ARG[@]}"}
STATUS=$?

echo ""
if [ ${STATUS} -eq 0 ]; then
    echo "OK   shard ${SHARD} finished at $(date -Is)"
else
    # No Parquet is written on failure, so the shard stays in the retry set
    # reported by submit/torchcrop_status.sh.
    echo "FAIL shard ${SHARD} exited ${STATUS} at $(date -Is)" >&2
fi
exit ${STATUS}
