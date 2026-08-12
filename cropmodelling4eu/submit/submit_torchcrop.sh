#!/bin/bash
# =============================================================================
# Master driver for the torchcrop / LINTUL-5 Europe run.
#
#   ./submit/submit_torchcrop.sh              # submit all shards, then maps
#   ./submit/submit_torchcrop.sh --retry      # submit only the missing shards
#   ./submit/submit_torchcrop.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_torchcrop.sh --no-maps    # shard array only
#   ./submit/submit_torchcrop.sh --maps-only  # re-render maps from the shards
#
# Tunables live in submit/torchcrop_env.sh and can be overridden per call:
#   TC_START_YEAR=1980 TC_TIME=12:00:00 ./submit/submit_torchcrop.sh
#   TC_RUN_NAME=test TC_END_YEAR=2001 ./submit/submit_torchcrop.sh
#
# Submits
#   1. torchcrop_array.sh - array job, one task per shard (throttled)
#   2. torchcrop_maps.sh  - dependent combine + map render
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/torchcrop_env.sh"

DRY_RUN=0
RETRY=0
DO_MAPS=1
MAPS_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)   DRY_RUN=1 ;;
        --retry)     RETRY=1 ;;
        --no-maps)   DO_MAPS=0 ;;
        --maps-only) MAPS_ONLY=1 ;;
        -h|--help)   sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

tc_activate
cd "${TC_PROJECT_DIR}" || exit 1

EXPORT_VARS="ALL,TC_PROJECT_DIR,TC_CONDA_ENV,TC_DATA_DIR,TC_OUT_DIR,TC_SHARD_DIR"
EXPORT_VARS="${EXPORT_VARS},TC_LOG_DIR,TC_COMPOSITION_XML,TC_CROP,TC_START_YEAR"
EXPORT_VARS="${EXPORT_VARS},TC_END_YEAR,TC_IOPT,TC_N_SHARDS,TC_BATCH_SIZE"
EXPORT_VARS="${EXPORT_VARS},TC_IO_WORKERS,TC_TORCH_THREADS"

# --- Which shards still need running -----------------------------------------
missing_shards() {
    local s out=()
    for ((s = 0; s < TC_N_SHARDS; s++)); do
        [ -f "$(printf '%s/torchcrop_shard_%03d.parquet' "${TC_SHARD_DIR}" "${s}")" ] \
            || out+=("${s}")
    done
    # SLURM takes a comma list; a contiguous range would be wrong after a
    # scattered set of failures.
    local IFS=,; echo "${out[*]}"
}

if [ "${MAPS_ONLY}" -eq 1 ]; then
    ARRAY_SPEC=""
elif [ "${RETRY}" -eq 1 ]; then
    ARRAY_SPEC="$(missing_shards)"
    if [ -z "${ARRAY_SPEC}" ]; then
        echo "All ${TC_N_SHARDS} shards already have a Parquet file - nothing to retry."
        [ "${DO_MAPS}" -eq 1 ] && echo "Run 'sbatch submit/torchcrop_maps.sh' to re-render maps."
        exit 0
    fi
else
    ARRAY_SPEC="0-$((TC_N_SHARDS - 1))"
fi

NYEARS=$((TC_END_YEAR - TC_START_YEAR + 1))

echo "=================================================="
echo "TORCHCROP EUROPE RUN"
echo "=================================================="
echo "  run name     : ${TC_RUN_NAME}"
echo "  input        : ${TC_DATA_DIR}"
echo "  output       : ${TC_OUT_DIR}"
echo "  seasons      : ${TC_START_YEAR}-${TC_END_YEAR}  (${NYEARS} harvest years)"
echo "  run mode     : iopt=${TC_IOPT}  crop=${TC_CROP}"
if [ -n "${ARRAY_SPEC}" ]; then
echo "  array        : ${ARRAY_SPEC}%${TC_MAX_CONCURRENT}$([ "${RETRY}" -eq 1 ] && echo '  (retry of missing shards)')"
echo "  batching     : ${TC_BATCH_SIZE} cells/call, ${TC_IO_WORKERS} io threads"
echo "  resources    : ${TC_PARTITION}, ${TC_CPUS} cpus, ${TC_MEM}, ${TC_TIME}"
else
echo "  array        : skipped (--maps-only)"
fi
echo "  logs         : ${TC_LOG_DIR}"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# After the dry-run gate, so a dry run never leaves an empty run directory
# behind that --retry would then read as "shard dir exists, nothing missing".
mkdir -p "${TC_LOG_DIR}" "${TC_SHARD_DIR}" "${TC_OUT_DIR}"

# --- 1. Shard array -----------------------------------------------------------
ARRAY_JOB=""
if [ -n "${ARRAY_SPEC}" ]; then
    ARRAY_JOB=$(sbatch --parsable \
        --partition="${TC_PARTITION}" \
        --cpus-per-task="${TC_CPUS}" \
        --mem="${TC_MEM}" \
        --time="${TC_TIME}" \
        --array="${ARRAY_SPEC}%${TC_MAX_CONCURRENT}" \
        --export="${EXPORT_VARS}" \
        submit/torchcrop_array.sh) \
        || { echo "ERROR: shard array submission failed" >&2; exit 1; }
    echo "Shard array submitted : ${ARRAY_JOB}"
fi

if [ "${DO_MAPS}" -eq 0 ]; then
    echo "Maps skipped (--no-maps). Run them later with:"
    echo "  sbatch --export=${EXPORT_VARS} submit/torchcrop_maps.sh"
    exit 0
fi

# --- 2. Combine + maps --------------------------------------------------------
# afterany, not afterok: the map job checks the shard set itself and prints a
# precise list of what is missing, which is more useful than a silently
# cancelled dependency.
DEP=()
[ -n "${ARRAY_JOB}" ] && DEP=(--dependency=afterany:"${ARRAY_JOB}")

MAPS_JOB=$(sbatch --parsable \
    --partition="${TC_PARTITION}" \
    --mem="${TC_MAPS_MEM}" \
    --time="${TC_MAPS_TIME}" \
    "${DEP[@]}" \
    --export="${EXPORT_VARS}" \
    submit/torchcrop_maps.sh) \
    || { echo "ERROR: maps submission failed" >&2; exit 1; }
echo "Maps submitted        : ${MAPS_JOB}${ARRAY_JOB:+  (afterany:${ARRAY_JOB})}"

cat <<EOF

==================================================
MONITOR
==================================================
  squeue -u \$USER
  ./submit/torchcrop_status.sh
  tail -f ${TC_LOG_DIR}/tc_shard_${ARRAY_JOB:-JOBID}_0.out

RETRY missing shards once the array drains:
  ./submit/submit_torchcrop.sh --retry

RESULTS:
  ${TC_OUT_DIR}/torchcrop_europe.parquet
  ${TC_OUT_DIR}/torchcrop_europe_grid.nc
  ${TC_OUT_DIR}/maps/

CANCEL:
  scancel ${ARRAY_JOB} ${MAPS_JOB}
==================================================
EOF
