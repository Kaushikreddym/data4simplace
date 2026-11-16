#!/bin/bash
# =============================================================================
# Throughput benchmark: torchcrop calibration batches on a CPU node and on a
# GPU node at the same time, one parquet each, plus a per-pass loss log beside
# each; SIMPLACE read from its own task logs.
#
#   ./submit/benchmark/submit_benchmark.sh              # CPU and GPU jobs
#   BENCH_ONLY=gpu ./submit/benchmark/submit_benchmark.sh
#   ./submit/benchmark/submit_benchmark.sh --dry-run    # print the sbatch commands
#
# The weather cache must already exist under BENCH_CACHE_FROM — a
# `cm4eu calibrate --prepare-only` (or any finished run) with the same
# --start-doy builds it; the benchmark only reads it.
#
# Settings (environment, with defaults):
#   BENCH_SIZES       64,256,1024,4096,16384  CPU target cell-years per batch
#   BENCH_GPU_SIZES   4096,16384,32768,65536  GPU ones: its time is flat in B,
#                     so only large batches say anything (73,562 = whole pool)
#   BENCH_MODES       loader,forward,fwd+bwd,step
#   BENCH_STEPS       5   Adam steps per batch in `step` mode (loss per pass)
#   BENCH_BATCHES     2   batches per size, all logged to *_passes.parquet
#   BENCH_STAGE       phenology
#   BENCH_START_DOY   200
#   BENCH_CACHE_FROM  ${CAL_RUN_DIR}/calibration  (read only: its weather_cache
#                     and region tables; never written to)
#   BENCH_WORK        ${CAL_RUN_DIR}/benchmark/prepare  (prepare()'s own out-dir)
#   BENCH_SIMPLACE    production run dirs whose logs/sp_*.out are read
#                     (default: mswx and the MIROC6 hist-aer daily pilot)
#   BENCH_DIR         ${CAL_RUN_DIR}/benchmark  (benchmark_{cpu,gpu}.parquet,
#                     each with a *_passes.parquet loss log beside it)
#   BENCH_GPU_PART    gpu
#   BENCH_ONLY        cpu | gpu  submit just that job
# =============================================================================
set -euo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SUBMIT_DIR}/utils/calibration_env.sh"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

BENCH_SIZES="${BENCH_SIZES:-64,256,1024,4096,16384}"
BENCH_GPU_SIZES="${BENCH_GPU_SIZES:-4096,16384,32768,65536}"
BENCH_MODES="${BENCH_MODES:-loader,forward,fwd+bwd,step}"
BENCH_STEPS="${BENCH_STEPS:-5}"
BENCH_BATCHES="${BENCH_BATCHES:-2}"
BENCH_STAGE="${BENCH_STAGE:-phenology}"
BENCH_START_DOY="${BENCH_START_DOY:-200}"
BENCH_CACHE_FROM="${BENCH_CACHE_FROM:-${CAL_RUN_DIR}/calibration}"
RUNS_ROOT="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/runs/winter_wheat/calibrated"
BENCH_WORK="${BENCH_WORK:-${CAL_RUN_DIR}/benchmark/prepare}"
BENCH_SIMPLACE="${BENCH_SIMPLACE:-${RUNS_ROOT}/mswx ${RUNS_ROOT}/cmip6/MIROC6/hist-aer}"
# One file per device, so the two jobs can run at once without racing on a
# read-modify-write of one parquet; benchmark.ipynb reads both.
BENCH_DIR="${BENCH_DIR:-${CAL_RUN_DIR}/benchmark}"
CPU_OUT="${BENCH_DIR}/benchmark_cpu.parquet"
GPU_OUT="${BENCH_DIR}/benchmark_gpu.parquet"
BENCH_GPU_PART="${BENCH_GPU_PART:-gpu}"
BENCH_ONLY="${BENCH_ONLY:-}"
mkdir -p "${CAL_LOG_DIR}" "${BENCH_DIR}"
[ -f "${BENCH_CACHE_FROM}/weather_cache/pool_index.parquet" ] || {
    echo "ERROR: no weather cache under ${BENCH_CACHE_FROM}" >&2; exit 1; }
# prepare() rewrites pool/observations in its out-dir, so it gets its own; the
# region tables are seeded from the calibration so the pool is the same one.
if [ "${DRY_RUN}" = "0" ]; then
    mkdir -p "${BENCH_WORK}"
    for f in cell_to_adm.parquet regions.parquet regions.settings.json; do
        cp -n "${BENCH_CACHE_FROM}/${f}" "${BENCH_WORK}/" 2>/dev/null || true
    done
fi

# Since 2026-10-01 the euval workspace crop is the Brandenburg crop.xml again
# (idsl 0, no vernrt), which the phenology spec cannot even resolve.
if grep -q "^source_file: .*Brandenburg" "${CAL_CROP_FILE}"; then
    echo "ERROR: ${CAL_CROP_FILE} is the Brandenburg crop, not the SUSTAg WW one;" >&2
    echo "       set CAL_CROP_FILE to a harmonised crop_wheat.yaml" >&2
    exit 1
fi

common="cd '${CAL_PROJECT_DIR}' && source '${SUBMIT_DIR}/utils/calibration_env.sh' && cal_activate && \
cm4eu benchmark --config '${CAL_CONFIG}' --stage '${BENCH_STAGE}' \
--crop-file '${CAL_CROP_FILE}' --sowing-file '${CAL_SOWING_FILE}' \
--pep725 '${CAL_PEP725}' --out-dir '${BENCH_WORK}' --cache-dir '${BENCH_CACHE_FROM}/weather_cache' \
--start-doy '${BENCH_START_DOY}' --batch-sizes '${BENCH_SIZES}' --threads 1 --modes '${BENCH_MODES}' \
--steps '${BENCH_STEPS}' --n-batches '${BENCH_BATCHES}'"

# The CPU job also times the region-packed path and reads the SIMPLACE logs, so
# the GPU job only has to add its own rows.
cpu_wrap="${common} --devices cpu --append --output '${CPU_OUT}' --simplace-runs ${BENCH_SIMPLACE} && \
cm4eu benchmark --config '${CAL_CONFIG}' --stage '${BENCH_STAGE}' \
--crop-file '${CAL_CROP_FILE}' --sowing-file '${CAL_SOWING_FILE}' \
--pep725 '${CAL_PEP725}' --out-dir '${BENCH_WORK}' --cache-dir '${BENCH_CACHE_FROM}/weather_cache' \
--start-doy '${BENCH_START_DOY}' --batch-sizes '${BENCH_SIZES}' --threads 1 --modes '${BENCH_MODES}' \
--steps '${BENCH_STEPS}' --n-batches '${BENCH_BATCHES}' \
--devices cpu --region-packed --append --output '${CPU_OUT}'"
gpu_wrap="hostname && nvidia-smi && ${common//${BENCH_SIZES}/${BENCH_GPU_SIZES}} --devices cuda --append --output '${GPU_OUT}'"
# --wrap runs under /bin/sh, whose POSIX mode aborts on ~/.bashrc's
# hyphenated function names, so cal_activate needs a real bash.
cpu_wrap="bash -c $(printf '%q' "${cpu_wrap}")"
gpu_wrap="bash -c $(printf '%q' "${gpu_wrap}")"

cpu=(sbatch --parsable --job-name=cm_bench_cpu --partition="${CAL_PARTITION}"
     --exclusive --cpus-per-task=4 --mem=64G --time=12:00:00
     --output="${CAL_LOG_DIR}/cm_bench_cpu_%j.log" --wrap="${cpu_wrap}")

if [ "${DRY_RUN}" = "1" ]; then
    printf '%q ' "${cpu[@]}"; echo
    echo "sbatch --partition=${BENCH_GPU_PART} --exclusive ... --wrap=\"${gpu_wrap}\""
    exit 0
fi

# The gpu partition has no GRES, so nothing stops a second job sharing the
# card; a timing run takes the whole node (as the CasCorrDiff jobs do).
gpu=(sbatch --parsable --job-name=cm_bench_gpu --partition="${BENCH_GPU_PART}"
     ${CAL_GRES:+--gres="${CAL_GRES}"} --exclusive --mem=0 --cpus-per-task=8 --time=6:00:00
     --output="${CAL_LOG_DIR}/cm_bench_gpu_%j.log")

case "${BENCH_ONLY}" in
    gpu) echo "submitted GPU $("${gpu[@]}" --wrap="${gpu_wrap}") -> ${GPU_OUT}"; exit 0 ;;
    cpu) echo "submitted CPU $("${cpu[@]}") -> ${CPU_OUT}"; exit 0 ;;
    "") ;;
    *) echo "ERROR: BENCH_ONLY must be cpu or gpu" >&2; exit 1 ;;
esac

CPU_JOB=$("${cpu[@]}")
GPU_JOB=$("${gpu[@]}" --wrap="${gpu_wrap}")
echo "submitted CPU ${CPU_JOB} -> ${CPU_OUT}, GPU ${GPU_JOB} -> ${GPU_OUT}"
