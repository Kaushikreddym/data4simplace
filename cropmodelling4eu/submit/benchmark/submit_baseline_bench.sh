#!/bin/bash
# =============================================================================
# Throughput of the PRODUCTION runners on the default (uncalibrated) crops —
# the path submit/baseline runs — rather than calibration batches:
#
#   GPU node   torchcrop.run (two-pass run_batch, its own batching and weather
#              I/O), batch sizes BB_GPU_SIZES
#   CPU node   the same at BB_CPU_SIZES, one thread per process; then SIMPLACE
#              on its template crop, BB_SP_CELLS cells over BB_SP_JVMS JVMs
#
# Each job takes its node exclusively, so the times are the node's own.
#
#   ./submit/benchmark/submit_baseline_bench.sh --dry-run
#
# Settings (environment, with defaults):
#   BB_CPU_SIZES  256,1024,2048,4096,16384
#   BB_GPU_SIZES  1024,4096,16384,65536
#   BB_BATCHES    3          batches per size (the first is flagged warm-up)
#   BB_YEARS      "2000 2024"
#   BB_SP_CELLS   780        20 per JVM
#   BB_SP_JVMS    39         floor(95 GB / 2.4 GB)
#   BB_AFTER      job id(s) to wait for (afterany)
#   BB_TAG        suffix for the output files, so a follow-up sweep keeps the first
#   BB_SKIP_SP    1 = torchcrop only on the CPU node
#   BB_ONLY       gpu | cpu      submit one node only (default both)
#   BB_IO_PROCESSES  "0"         weather-reader processes, one benchmark per value,
#                                appended to one file (torchcrop.io_processes)
#   BB_OUT        ${CAL_RUN_DIR}/benchmark/baseline
# =============================================================================
set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SUBMIT_DIR}/utils/calibration_env.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

DATA=/data01/FDS/muduchuru/Data/SIMPLACE
BB_CPU_SIZES="${BB_CPU_SIZES:-256,1024,2048,4096,16384}"
BB_GPU_SIZES="${BB_GPU_SIZES:-1024,4096,16384,65536}"
BB_BATCHES="${BB_BATCHES:-3}"
BB_YEARS="${BB_YEARS:-2000 2024}"
BB_SP_CELLS="${BB_SP_CELLS:-780}"
BB_SP_JVMS="${BB_SP_JVMS:-39}"
BB_AFTER="${BB_AFTER:-}"
BB_TAG="${BB_TAG:-}"
BB_SKIP_SP="${BB_SKIP_SP:-0}"
BB_ONLY="${BB_ONLY:-}"
BB_IO_PROCESSES="${BB_IO_PROCESSES:-0}"
BB_OUT="${BB_OUT:-${CAL_RUN_DIR}/benchmark/baseline}"
CONFIG="${DATA}/cropmodelling4eu/runs/winter_wheat/calibrated/mswx/config.yaml"
SOWING="${DATA}/cropmodelling4eu_inputs/sowing_from_simplace_euval_nfix.csv"
CONTAINERS="${DATA}/containers"

for f in "${CONFIG}" "${SOWING}" "${CAL_CROP_FILE}" "${CONTAINERS}/temurin-17-jre-jammy.sif"; do
    [ -e "${f}" ] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done
grep -q "^source_file: .*LINTUL5_crop.xml" "${CAL_CROP_FILE}" \
    || { echo "ERROR: ${CAL_CROP_FILE} is not the harmonised SUSTAg crop" >&2; exit 1; }
mkdir -p "${BB_OUT}/logs"

# SIMPLACE runs the production config on the glibc Java image (24+ JVMs on the
# shipped musl image abort 15-30 % of the time).
SP_CONFIG="${BB_OUT}/simplace_config.yaml"
if [ "${DRY_RUN}" -eq 0 ]; then
    python3 - "${CONFIG}" "${SP_CONFIG}" "${CONTAINERS}" <<'EOF'
import sys
from pathlib import Path

import yaml

source, target, containers = sys.argv[1:]
config = yaml.safe_load(Path(source).read_text())
config["simplace"].update(
    java_image=f"{containers}/temurin-17-jre-jammy.sif",
    simplace_dir=f"{containers}/simplace_5.0-3897",
    daily_output=False,
)
config["run_name"] = "benchmark_baseline_simplace"
Path(target).write_text(yaml.safe_dump(config, sort_keys=False))
EOF
fi

ACTIVATE="cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/utils/calibration_env.sh' && cal_activate"
tc() {  # device sizes output
    local cmd="rm -f '${BB_OUT}/$3'" n
    for n in ${BB_IO_PROCESSES}; do
        cmd+=" && cm4eu benchmark --production --config '${CONFIG}' --crop-file '${CAL_CROP_FILE}' \
--sowing-file '${SOWING}' --years ${BB_YEARS} --n-batches ${BB_BATCHES} --threads 1 \
--io-processes ${n} --devices $1 --batch-sizes '$2' --output '${BB_OUT}/$3' --append"
    done
    echo "${cmd}"
}
SP_DIR="${BB_OUT}/simplace"
SP="rm -rf '${SP_DIR}' && cm4eu simplace build --config '${SP_CONFIG}' --cells ${BB_SP_CELLS} \
--lines-per-task ${BB_SP_CELLS} --out-dir '${SP_DIR}' && mkdir -p '${SP_DIR}/log' && \
SP_CORES_PER_TASK=${BB_SP_JVMS} SP_JAVA_HEAP=4g '${SP_DIR}/run_task.sh' 1-${BB_SP_CELLS} \
2>&1 | tee '${SP_DIR}/log/sp_bench.out'"

GPU_TASK="hostname && nvidia-smi && ${ACTIVATE} && $(tc cuda "${BB_GPU_SIZES}" "production_gpu${BB_TAG}.parquet")"
CPU_TASK="${ACTIVATE} && $(tc cpu "${BB_CPU_SIZES}" "production_cpu${BB_TAG}.parquet")"
[ "${BB_SKIP_SP}" = "1" ] || CPU_TASK="${CPU_TASK} && ${SP}"

wrap() { echo "bash -c $(printf '%q' "$1")"; }
DEP=(); [ -n "${BB_AFTER}" ] && DEP=(--dependency="afterany:${BB_AFTER}")
GPU=(sbatch --parsable --job-name=bb_gpu${BB_TAG} --partition=gpu --exclusive --mem=0 --time=6:00:00
     "${DEP[@]+"${DEP[@]}"}" --output="${BB_OUT}/logs/gpu${BB_TAG}_%j.log" --wrap="$(wrap "${GPU_TASK}")")
CPU=(sbatch --parsable --job-name=bb_cpu${BB_TAG} --partition="${CAL_PARTITION}" --exclusive --mem=0
     --time=12:00:00 "${DEP[@]+"${DEP[@]}"}" --output="${BB_OUT}/logs/cpu${BB_TAG}_%j.log"
     --wrap="$(wrap "${CPU_TASK}")")

if [ "${DRY_RUN}" -eq 1 ]; then
    [ "${BB_ONLY}" = "cpu" ] || echo "GPU: ${GPU_TASK}"
    [ "${BB_ONLY}" = "gpu" ] || echo "CPU: ${CPU_TASK}" | cut -c1-600
    exit 0
fi
[ "${BB_ONLY}" = "cpu" ] || echo "GPU $("${GPU[@]}")"
[ "${BB_ONLY}" = "gpu" ] || echo "CPU $("${CPU[@]}")"
echo "-> ${BB_OUT}"
