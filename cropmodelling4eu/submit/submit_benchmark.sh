#!/bin/bash
# =============================================================================
# Throughput benchmark: torchcrop calibration batches on a CPU node, then on a
# GPU node, appended into one parquet; SIMPLACE read from its own task logs.
#
#   ./submit/submit_benchmark.sh              # CPU job, then GPU job after it
#   ./submit/submit_benchmark.sh --dry-run    # print the sbatch commands
#
# The GPU job depends on the CPU one (afterany) because both append to the
# same file. The weather cache must already exist under BENCH_CACHE_FROM — a
# `cm4eu calibrate --prepare-only` (or any finished run) with the same
# --start-doy builds it; the benchmark only reads it.
#
# Settings (environment, with defaults):
#   BENCH_SIZES       64,256,1024,4096,16384  target cell-years per batch
#   BENCH_STAGE       phenology
#   BENCH_START_DOY   200
#   BENCH_CACHE_FROM  ${CAL_RUN_DIR}/calibration_mixcheck  (its out-dir/cache)
#   BENCH_OUT         ${CAL_RUN_DIR}/benchmark/benchmark.parquet
#   BENCH_GPU_PART    gpu
# =============================================================================
set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SUBMIT_DIR}/calibration_env.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

BENCH_SIZES="${BENCH_SIZES:-64,256,1024,4096,16384}"
BENCH_STAGE="${BENCH_STAGE:-phenology}"
BENCH_START_DOY="${BENCH_START_DOY:-200}"
BENCH_CACHE_FROM="${BENCH_CACHE_FROM:-${CAL_RUN_DIR}/calibration_mixcheck}"
BENCH_OUT="${BENCH_OUT:-${CAL_RUN_DIR}/benchmark/benchmark.parquet}"
BENCH_GPU_PART="${BENCH_GPU_PART:-gpu}"
mkdir -p "${CAL_LOG_DIR}" "$(dirname "${BENCH_OUT}")"

# Since 2026-10-01 the euval workspace crop is the Brandenburg crop.xml again
# (idsl 0, no vernrt), which the phenology spec cannot even resolve.
if grep -q "^source_file: .*Brandenburg" "${CAL_CROP_FILE}"; then
    echo "ERROR: ${CAL_CROP_FILE} is the Brandenburg crop, not the SUSTAg WW one;" >&2
    echo "       set CAL_CROP_FILE to a harmonised crop_wheat.yaml" >&2
    exit 1
fi

common="cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/calibration_env.sh' && cal_activate && \
cm4eu benchmark --config '${CAL_CONFIG}' --stage '${BENCH_STAGE}' \
--crop-file '${CAL_CROP_FILE}' --sowing-file '${CAL_SOWING_FILE}' \
--pep725 '${CAL_PEP725}' --out-dir '${BENCH_CACHE_FROM}' \
--start-doy '${BENCH_START_DOY}' --batch-sizes '${BENCH_SIZES}' --threads 1 \
--output '${BENCH_OUT}'"

# The CPU job also times the region-packed path and reads the SIMPLACE logs, so
# the GPU job only has to add its own rows.
cpu_wrap="${common} --devices cpu --simplace-runs '${CAL_RUN_DIR}/simplace' \
${CAL_RUN_DIR}/simplace_regions/region_* && \
cm4eu benchmark --config '${CAL_CONFIG}' --stage '${BENCH_STAGE}' \
--crop-file '${CAL_CROP_FILE}' --sowing-file '${CAL_SOWING_FILE}' \
--pep725 '${CAL_PEP725}' --out-dir '${BENCH_CACHE_FROM}' \
--start-doy '${BENCH_START_DOY}' --batch-sizes '${BENCH_SIZES}' --threads 1 \
--devices cpu --region-packed --append --output '${BENCH_OUT}'"
gpu_wrap="${common} --devices cuda --append"

cpu=(sbatch --parsable --job-name=cm_bench_cpu --partition="${CAL_PARTITION}"
     --cpus-per-task=4 --mem=64G --time=12:00:00
     --output="${CAL_LOG_DIR}/cm_bench_cpu_%j.log" --wrap="${cpu_wrap}")

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cpu[@]}"; echo
    echo "sbatch --dependency=afterany:<cpu> --partition=${BENCH_GPU_PART} ... --wrap=\"${gpu_wrap}\""
    exit 0
fi

CPU_JOB=$("${cpu[@]}")
GPU_JOB=$(sbatch --parsable --job-name=cm_bench_gpu --partition="${BENCH_GPU_PART}" \
    ${CAL_GRES:+--gres="${CAL_GRES}"} --dependency=afterany:"${CPU_JOB}" \
    --cpus-per-task=8 --mem=64G --time=6:00:00 \
    --output="${CAL_LOG_DIR}/cm_bench_gpu_%j.log" --wrap="${gpu_wrap}")
echo "submitted CPU ${CPU_JOB}, GPU ${GPU_JOB} -> ${BENCH_OUT}"
