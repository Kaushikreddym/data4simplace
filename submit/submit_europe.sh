#!/bin/bash
# =============================================================================
# Master driver for the tiled Europe run.
#
#   ./submit/submit_europe.sh              # submit every tile, then combine
#   ./submit/submit_europe.sh --retry      # submit only the unfinished tiles
#   ./submit/submit_europe.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_europe.sh --no-combine # tile array only
#
# Tunables live in submit/env.sh and can be overridden per invocation:
#   D4S_TILE_DEG=2.5 D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh
#   D4S_RUN_NAME=test D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh
# Submits
#   1. tile_array.sh  - array job, one task per tile (throttled)
#   2. combine.sh     - dependent (afterany) mosaic of the finished tiles
#   3. management.sh  - dependent NPK/fertilizer export, if paths.npk_root is set
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/env.sh"

DRY_RUN=0
RETRY=0
DO_COMBINE=1
for arg in "$@"; do
    case "${arg}" in
        --dry-run)    DRY_RUN=1 ;;
        --retry)      RETRY=1 ;;
        --no-combine) DO_COMBINE=0 ;;
        -h|--help)    sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

d4s_activate
cd "${D4S_PROJECT_DIR}" || exit 1
mkdir -p "${D4S_LOG_DIR}" "${D4S_WORK_DIR}/configs" "${D4S_WCS_CACHE_ROOT}" "${D4S_OUT_DIR}"

# --- 1. Freeze the run config ------------------------------------------------
# One copy of config.yaml with paths.output_dir pointed at the run directory.
# Every job of the run reads this file, so a later edit to the tracked
# config.yaml cannot change the meaning of tiles submitted earlier.
if [ "${RETRY}" -eq 1 ] && [ -f "${D4S_RUN_CONFIG}" ]; then
    echo "Reusing existing run config: ${D4S_RUN_CONFIG}"
else
    python submit/tile_config.py \
        --config "${D4S_CONFIG}" \
        --output-dir "${D4S_OUT_DIR}" \
        --out "${D4S_RUN_CONFIG}" >/dev/null || exit 1
    echo "Wrote run config: ${D4S_RUN_CONFIG}"
fi

# --- 2. Validate it before burning an array job -------------------------------
data4simplace --config "${D4S_RUN_CONFIG}" --dry-run || {
    echo "ERROR: config validation failed; nothing submitted." >&2
    exit 1
}

# --- 3. Size the array --------------------------------------------------------
NTILES=$(data4simplace --config "${D4S_RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}" --count-tiles) || exit 1
if ! [[ "${NTILES}" =~ ^[0-9]+$ ]] || [ "${NTILES}" -eq 0 ]; then
    echo "ERROR: --count-tiles returned '${NTILES}'" >&2
    exit 1
fi

MAX_ARRAY=$(scontrol show config 2>/dev/null | awk '/^MaxArraySize/ {print $3}')
if [ -n "${MAX_ARRAY}" ] && [ "${NTILES}" -gt "${MAX_ARRAY}" ]; then
    echo "ERROR: ${NTILES} tiles exceeds MaxArraySize=${MAX_ARRAY}." >&2
    echo "       Increase D4S_TILE_DEG (fewer, bigger tiles)." >&2
    exit 1
fi

if [ "${RETRY}" -eq 1 ]; then
    ARRAY_SPEC=$(python submit/tile_status.py \
        --config "${D4S_RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}" --missing) || exit 1
    if [ -z "${ARRAY_SPEC}" ]; then
        echo "All ${NTILES} tiles already carry a .done marker - nothing to retry."
        [ "${DO_COMBINE}" -eq 1 ] && echo "Run 'sbatch submit/combine.sh' to (re-)mosaic."
        exit 0
    fi
else
    ARRAY_SPEC="0-$((NTILES - 1))"
fi

echo "=================================================="
echo "TILED EUROPE RUN"
echo "=================================================="
echo "  base config  : ${D4S_CONFIG}"
echo "  run config   : ${D4S_RUN_CONFIG}"
echo "  output dir   : ${D4S_OUT_DIR}"
echo "  tile size    : ${D4S_TILE_DEG} deg  (${NTILES} tiles total)"
echo "  array        : ${ARRAY_SPEC}%${D4S_MAX_CONCURRENT}$([ "${RETRY}" -eq 1 ] && echo '  (retry of unfinished tiles)')"
echo "  resources    : ${D4S_PARTITION}, ${D4S_CPUS} cpus, ${D4S_MEM}, ${D4S_TIME}"
echo "  logs         : ${D4S_LOG_DIR}"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# Pass the run's settings to every job, so a later edit of env.sh does not
# retarget jobs that are already queued.
EXPORT_VARS="ALL,D4S_PROJECT_DIR,D4S_CONDA_ENV,D4S_RUN_CONFIG,D4S_OUT_DIR,D4S_WORK_DIR"
EXPORT_VARS="${EXPORT_VARS},D4S_WCS_CACHE_ROOT,D4S_TILE_DEG,D4S_LOG_DIR,D4S_CPUS"

# --- 4. Tile array ------------------------------------------------------------
ARRAY_JOB=$(sbatch --parsable \
    --partition="${D4S_PARTITION}" \
    --cpus-per-task="${D4S_CPUS}" \
    --mem="${D4S_MEM}" \
    --time="${D4S_TIME}" \
    --array="${ARRAY_SPEC}%${D4S_MAX_CONCURRENT}" \
    --export="${EXPORT_VARS}" \
    submit/tile_array.sh) || { echo "ERROR: tile array submission failed" >&2; exit 1; }
echo "Tile array submitted : ${ARRAY_JOB}"

if [ "${DO_COMBINE}" -eq 0 ]; then
    echo "Combine skipped (--no-combine). Run it later with:"
    echo "  sbatch --export=${EXPORT_VARS} submit/combine.sh"
    exit 0
fi

# --- 5. Combine (afterany: a single failed tile must not block the mosaic) -----
COMBINE_JOB=$(sbatch --parsable \
    --partition="${D4S_PARTITION}" \
    --dependency=afterany:"${ARRAY_JOB}" \
    --export="${EXPORT_VARS}" \
    submit/combine.sh) || { echo "ERROR: combine submission failed" >&2; exit 1; }
echo "Combine submitted    : ${COMBINE_JOB}  (afterany:${ARRAY_JOB})"

# --- 6. NPK / management (independent of the tiles; self-skips if unconfigured)-
MGMT_JOB=$(sbatch --parsable \
    --partition="${D4S_PARTITION}" \
    --dependency=afterany:"${COMBINE_JOB}" \
    --export="${EXPORT_VARS}" \
    submit/management.sh) || { echo "WARNING: management submission failed" >&2; MGMT_JOB=""; }
[ -n "${MGMT_JOB}" ] && echo "Management submitted : ${MGMT_JOB}  (afterany:${COMBINE_JOB})"

cat <<EOF

==================================================
MONITOR
==================================================
  squeue -u \$USER
  ./submit/status.sh
  tail -f ${D4S_LOG_DIR}/tile_${ARRAY_JOB}_0.out

RETRY unfinished tiles once the array drains:
  ./submit/submit_europe.sh --retry

CANCEL:
  scancel ${ARRAY_JOB} ${COMBINE_JOB} ${MGMT_JOB}
==================================================
EOF
