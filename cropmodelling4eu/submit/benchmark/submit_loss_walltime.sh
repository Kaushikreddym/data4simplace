#!/bin/bash
# =============================================================================
# Calibration loss against wall time: one region's phenology stage fitted three
# ways, each on one whole node and each with the largest batch it can take.
#
#   tc_gpu  torchcrop, one V100, the region's whole epoch in one batch
#   tc_cpu  torchcrop, one CPU process, the same batch
#   sp_cpu  SIMPLACE + Optuna, one CPU node, as many JVMs as its RAM holds
#
# All three score the same unit-years: torchcrop reuses the production
# calibration's split (--split-from), and SIMPLACE scores that split directly.
# SIMPLACE starts where torchcrop starts (--no-warm-start), so neither is handed
# the other's answer. Plot: notebooks/benchmark/benchmark.ipynb, section 9.
#
#   ./submit/benchmark/submit_loss_walltime.sh --dry-run
#
# Settings (environment, with defaults):
#   LW_REGION   16        PAN-2: 161 cells, 1475 cell-years, PEP725 + CLMS
#   LW_STAGE    phenology
#   LW_EPOCHS   80        torchcrop (patience still applies)
#   LW_TRIALS   60        SIMPLACE
#   LW_JVMS     39        floor(95 GB / 2.4 GB per JVM)
#   LW_OUT      ${CAL_RUN_DIR}/benchmark/loss_walltime
#   LW_AFTER    job id(s) all three wait for (afterany), e.g. to stay under the job cap
# =============================================================================
set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SUBMIT_DIR}/utils/calibration_env.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

LW_REGION="${LW_REGION:-16}"
LW_STAGE="${LW_STAGE:-phenology}"
LW_EPOCHS="${LW_EPOCHS:-80}"
LW_TRIALS="${LW_TRIALS:-60}"
LW_JVMS="${LW_JVMS:-39}"
LW_OUT="${LW_OUT:-${CAL_RUN_DIR}/benchmark/loss_walltime}"
LW_AFTER="${LW_AFTER:-}"
PROD_CAL="${CAL_RUN_DIR}/calibration"
SP_CONFIG="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated/mswx/config.yaml"
CONTAINERS="/data01/FDS/muduchuru/Data/SIMPLACE/containers"
# Larger than any region's epoch, so every epoch is one batch: a full gradient.
MAX_BATCH=1000000

for f in "${PROD_CAL}/${LW_STAGE}_split.parquet" "${CAL_CROP_FILE}" "${SP_CONFIG}"; do
    [ -e "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
mkdir -p "${LW_OUT}/logs"

tc_task() {  # device out-dir
    local out="${LW_OUT}/$2"
    # The region and EnZ tables are the production run's, so the pool is too.
    if [ "${DRY_RUN}" -eq 0 ]; then
        mkdir -p "${out}"
        for f in cell_to_adm.parquet regions.parquet regions.settings.json; do
            cp -n "${PROD_CAL}/${f}" "${out}/" 2>/dev/null || true
        done
    fi
    echo "cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/utils/calibration_env.sh' && cal_activate && \
cm4eu calibrate --config '${CAL_CONFIG}' --stage '${LW_STAGE}' --crop-file '${CAL_CROP_FILE}' \
--sowing-file '${CAL_SOWING_FILE}' --pep725 '${CAL_PEP725}' --out-dir '${out}' \
--regions ${LW_REGION} --split-from '${PROD_CAL}' --fraction 1.0 --batch-size ${MAX_BATCH} \
--epochs ${LW_EPOCHS} --workers 1 --threads 1 --device $1"
}

SP_TASK="cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/utils/calibration_env.sh' && cal_activate && \
cm4eu simplace-calibrate --config '${SP_CONFIG}' --calibration-dir '${PROD_CAL}' \
--crop-file '${CAL_CROP_FILE}' --out-dir '${LW_OUT}/sp_cpu' --region ${LW_REGION} --stage '${LW_STAGE}' \
--trials ${LW_TRIALS} --workers ${LW_JVMS} --cells-per-unit 0 --no-warm-start \
--java-image '${CONTAINERS}/temurin-17-jre-jammy.sif' --simplace-dir '${CONTAINERS}/simplace_5.0-3897'"

# --wrap runs /bin/sh, which aborts on ~/.bashrc's hyphenated function names.
wrap() { echo "bash -c $(printf '%q' "$1")"; }
# --exclusive everywhere: the comparison is per node, and a neighbour on the
# node (or on the GPU, which has no GRES here) would be timed too.
JOBS=(
  "tc_gpu|--partition=gpu --exclusive --mem=0 --time=12:00:00|hostname && nvidia-smi && $(tc_task cuda tc_gpu)"
  "tc_cpu|--partition=${CAL_PARTITION} --exclusive --mem=0 --time=1-00:00:00|$(tc_task cpu tc_cpu)"
  "sp_cpu|--partition=${CAL_PARTITION} --exclusive --mem=0 --time=2-00:00:00|${SP_TASK}"
)
for job in "${JOBS[@]}"; do
    IFS='|' read -r name resources task <<< "${job}"
    # shellcheck disable=SC2206
    cmd=(sbatch --parsable --job-name="lw_${name}" ${resources}
         ${LW_AFTER:+--dependency=afterany:${LW_AFTER}}
         --output="${LW_OUT}/logs/${name}_%j.log" --wrap="$(wrap "${task}")")
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "${name}: ${resources}"; echo "  ${task}" | cut -c1-400
    else
        echo "${name}: $("${cmd[@]}")"
    fi
done
echo "out: ${LW_OUT}"
