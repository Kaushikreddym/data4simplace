#!/bin/bash
# =============================================================================
# Optuna calibration of SIMPLACE, one array task per region: phenology, then
# yield on top of the region's SIMPLACE phenology best. Scored with torchcrop's
# loss terms on the torchcrop calibration's pool, observations and split
# (cropmodelling4eu.simplace.optuna_calibration).
#
#   ./submit/calibration/submit_simplace_optuna.sh --dry-run
#   SO_REGIONS="12 17" SO_TRIALS=6 ./submit/calibration/submit_simplace_optuna.sh   # pilot
#   ./submit/calibration/submit_simplace_optuna.sh                                 # all regions
#
# Rerunning resumes: each study is an Optuna journal under
# ${SO_OUT}/region_<NN>/<stage>/study.log, and only the missing trials run.
#
# Settings (environment, with defaults):
#   SO_CONFIG      production mswx config.yaml (the SIMPLACE block production ran)
#   SO_CAL_DIR     ${CAL_RUN_DIR}/calibration       torchcrop calibration (pool, split, warm start)
#   SO_CROP_FILE   ${CAL_CROP_FILE}                 harmonised SUSTAg crop_wheat.yaml
#   SO_OUT         ${CAL_RUN_DIR}/simplace_optuna
#   SO_REGIONS     all region ids in regions.parquet
#   SO_STAGE       all                              phenology | yield | all
#   SO_TRIALS      40                               per region and stage
#   SO_STARTUP     10                               random trials before TPE
#   SO_WORKERS     24                               JVMs per trial
#   SO_CELLS       4                                pooled cells kept per unit
#   SO_SLOTS       3                                regions at once (the 15-job cap is total)
#   SO_TIME        3-00:00:00
# =============================================================================
set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SUBMIT_DIR}/utils/calibration_env.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

RUNS_ROOT="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated"
CONTAINERS="/data01/FDS/muduchuru/Data/SIMPLACE/containers"
SO_CONFIG="${SO_CONFIG:-${RUNS_ROOT}/mswx/config.yaml}"
SO_CAL_DIR="${SO_CAL_DIR:-${CAL_RUN_DIR}/calibration}"
SO_CROP_FILE="${SO_CROP_FILE:-${CAL_CROP_FILE}}"
SO_OUT="${SO_OUT:-${CAL_RUN_DIR}/simplace_optuna}"
SO_STAGE="${SO_STAGE:-all}"
SO_TRIALS="${SO_TRIALS:-40}"
SO_STARTUP="${SO_STARTUP:-10}"
SO_WORKERS="${SO_WORKERS:-24}"
SO_CELLS="${SO_CELLS:-4}"
SO_SLOTS="${SO_SLOTS:-3}"
SO_TIME="${SO_TIME:-3-00:00:00}"
# The glibc Java image: the shipped musl one aborts 15-30% of JVMs once 24
# share a node.
SO_JAVA_IMAGE="${SO_JAVA_IMAGE:-${CONTAINERS}/temurin-17-jre-jammy.sif}"
SO_SIMPLACE_DIR="${SO_SIMPLACE_DIR:-${CONTAINERS}/simplace_5.0-3897}"

for f in "${SO_CONFIG}" "${SO_CROP_FILE}" "${SO_CAL_DIR}/pool.parquet" "${SO_JAVA_IMAGE}"; do
    [ -e "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
grep -q "^source_file: .*LINTUL5_crop.xml" "${SO_CROP_FILE}" \
    || { echo "ERROR: ${SO_CROP_FILE} is not the SUSTAg LINTUL5 crop" >&2; exit 1; }

# /etc/bashrc trips `set -e`; cal_activate already handles `set -u`.
set +e; cal_activate; set -e
SO_REGIONS="${SO_REGIONS:-$(python -c "import pandas as pd, sys; print(' '.join(map(str, sorted(pd.read_parquet(sys.argv[1])['region_id'].unique()))))" "${SO_CAL_DIR}/regions.parquet")}"
read -r -a REGIONS <<< "${SO_REGIONS}"
mkdir -p "${SO_OUT}/logs"
printf '%s\n' "${REGIONS[@]}" > "${SO_OUT}/regions.txt"

# Size the node to the JVMs: ~1.3 cores and ~3 GB each (24 workers used 27
# cores and 58 GB), plus the scorer.
CPUS=$(( SO_WORKERS * 4 / 3 + 2 ))
MEM="$(( SO_WORKERS * 3 + 16 ))G"

TASK="cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/utils/calibration_env.sh' && cal_activate && \
REGION=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID + 1))p\" '${SO_OUT}/regions.txt') && \
cm4eu simplace-calibrate --config '${SO_CONFIG}' --calibration-dir '${SO_CAL_DIR}' \
--crop-file '${SO_CROP_FILE}' --out-dir '${SO_OUT}' --region \"\${REGION}\" --stage '${SO_STAGE}' \
--trials '${SO_TRIALS}' --startup-trials '${SO_STARTUP}' --workers '${SO_WORKERS}' \
--cells-per-unit '${SO_CELLS}' --java-image '${SO_JAVA_IMAGE}' --simplace-dir '${SO_SIMPLACE_DIR}'"
# --wrap runs /bin/sh, which aborts on ~/.bashrc's hyphenated function names.
WRAP="bash -c $(printf '%q' "${TASK}")"

CMD=(sbatch --parsable --job-name=sp_optuna --partition="${CAL_PARTITION}"
     --array="0-$(( ${#REGIONS[@]} - 1 ))%${SO_SLOTS}"
     --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${SO_TIME}"
     --output="${SO_OUT}/logs/sp_optuna_%A_%a.log" --wrap="${WRAP}")

echo "regions : ${REGIONS[*]} (${#REGIONS[@]}, ${SO_SLOTS} at once)"
echo "stage   : ${SO_STAGE}, ${SO_TRIALS} trials each, ${SO_WORKERS} JVMs, ${SO_CELLS} cells/unit"
echo "node    : ${CPUS} CPUs, ${MEM}, ${SO_TIME}"
echo "out     : ${SO_OUT}"
if [ "${DRY_RUN}" -eq 1 ]; then
    printf '%q ' "${CMD[@]}"; echo
    exit 0
fi
echo "submitted $("${CMD[@]}")"
