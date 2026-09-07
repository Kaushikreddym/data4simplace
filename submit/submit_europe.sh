#!/bin/bash
# =============================================================================
# Master driver for the tiled Europe run.
#
#   ./submit/submit_europe.sh              # submit every tile, then combine
#   ./submit/submit_europe.sh --retry      # submit only the unfinished tiles
#   ./submit/submit_europe.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_europe.sh --no-combine # tile array only
#   ./submit/submit_europe.sh --no-validation # inputs only, no validation data
#   ./submit/submit_europe.sh --no-clms-fetch # validation, but tiles already local
#
# Tunables live in submit/env.sh and can be overridden per invocation:
#   D4S_TILE_DEG=2.5 D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh
#   D4S_RUN_NAME=test D4S_MAX_CONCURRENT=6 ./submit/submit_europe.sh
# Submits, in two groups.
#
# MODEL INPUTS -- what a crop model consumes:
#   1. tile_array.sh  - array job, one task per tile (throttled)
#   2. combine.sh     - dependent (afterany) mosaic of the finished tiles
#   3. management.sh  - dependent NPK/fertilizer export, if paths.npk_root is set
#
# VALIDATION DATASETS -- what a run is scored and calibrated against. Currently
# CLMS phenology; LAI and the rest join here. Skipped entirely unless
# flags.run_phenology_processing and paths.clms_root are both set.
#   4. clms_fetch.sh                 - per-tile download of the CLMS rasters
#   5. validation_array.sh  (cuts)   - per-tile, finds each country's season cut
#   6. validation_reduce.sh (cuts)   - merges them into season_cuts.csv
#   7. validation_array.sh  (stats)  - per-tile histograms, using those cuts
#   8. validation_reduce.sh (stats)  - merges to validation/phenology/*.parquet
#
# The fetch needs CDSE S3 keys in a named boto3 profile; check them first with
#   python -m data4simplace.phenology.fetch
# Pass --no-clms-fetch when the tiles are already on disk.
#
# 4-7 run after the inputs because they read the same run config and target the
# same run directory, not because they depend on the export.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/env.sh"

DRY_RUN=0
RETRY=0
DO_COMBINE=1
DO_VALIDATION=1
DO_CLMS_FETCH=1
for arg in "$@"; do
    case "${arg}" in
        --dry-run)    DRY_RUN=1 ;;
        --retry)      RETRY=1 ;;
        --no-combine) DO_COMBINE=0 ;;
        --no-validation) DO_VALIDATION=0 ;;
        --no-clms-fetch) DO_CLMS_FETCH=0 ;;
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

# Validation tiles are resolved here, before the banner, so --dry-run prints the
# whole plan and not just the input half. Building the inventory is also the
# right place for it: every array task needs it and none of them should fetch it.
NTILES_VAL=""
if [ "${DO_VALIDATION}" -eq 1 ]; then
    python -m data4simplace.phenology.handler \
        --config "${D4S_RUN_CONFIG}" --stage inventory >/dev/null 2>&1
    NTILES_VAL=$(python -m data4simplace.phenology.handler \
        --config "${D4S_RUN_CONFIG}" --stage count-tiles 2>/dev/null) || NTILES_VAL=""
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
if [[ "${NTILES_VAL}" =~ ^[0-9]+$ ]] && [ "${NTILES_VAL}" -gt 0 ]; then
    echo "  validation   : CLMS phenology, ${NTILES_VAL} tiles$([ "${DO_CLMS_FETCH}" -eq 1 ] && echo ' (fetch + process)' || echo ' (process only)')"
else
    echo "  validation   : none (run_phenology_processing off, or paths.clms_root unset)"
fi
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# Pass the run's settings to every job, so a later edit of env.sh does not
# retarget jobs that are already queued.
EXPORT_VARS="ALL,D4S_PROJECT_DIR,D4S_CONDA_ENV,D4S_RUN_CONFIG,D4S_OUT_DIR,D4S_WORK_DIR"
EXPORT_VARS="${EXPORT_VARS},D4S_WCS_CACHE_ROOT,D4S_TILE_DEG,D4S_LOG_DIR,D4S_CPUS"
EXPORT_VARS="${EXPORT_VARS},D4S_VAL_STAGE,D4S_BIGMEM_PARTITION"

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
    --partition="${D4S_BIGMEM_PARTITION}" \
    --dependency=afterany:"${COMBINE_JOB}" \
    --export="${EXPORT_VARS}" \
    submit/management.sh) || { echo "WARNING: management submission failed" >&2; MGMT_JOB=""; }
[ -n "${MGMT_JOB}" ] && echo "Management submitted : ${MGMT_JOB}  (afterany:${COMBINE_JOB})"

# --- 7. Validation datasets (CLMS phenology) ----------------------------------
# Self-skipping: if the flag or the path is unset there is nothing to submit, and
# burning four jobs to discover that is worse than asking here.
VAL_JOBS=""
if [ "${DO_VALIDATION}" -eq 1 ]; then
    if [[ "${NTILES_VAL}" =~ ^[0-9]+$ ]] && [ "${NTILES_VAL}" -gt 0 ]; then
        VAL_SPEC="0-$((NTILES_VAL - 1))%${D4S_MAX_CONCURRENT}"
        echo "Validation tiles     : ${NTILES_VAL}"

        # CLMS fetch. Resumable and idempotent -- an object already on disk at the
        # right size and ETag is skipped -- so a partial array is fixed by
        # resubmitting, and --no-clms-fetch skips it once the tiles are local.
        FETCH_JOB=""
        if [ "${DO_CLMS_FETCH}" -eq 1 ]; then
            FETCH_JOB=$(sbatch --parsable \
                --partition="${D4S_PARTITION}" \
                --dependency=afterany:"${COMBINE_JOB}" \
                --array="${VAL_SPEC}" \
                --export="${EXPORT_VARS}" \
                submit/clms_fetch.sh) || FETCH_JOB=""
            [ -n "${FETCH_JOB}" ] && echo "CLMS fetch array     : ${FETCH_JOB}"
        fi
        # Cuts wait on the fetch when there is one, else on the mosaic.
        CUTS_AFTER="${FETCH_JOB:-${COMBINE_JOB}}"

        CUTS_JOB=$(D4S_VAL_STAGE=cuts sbatch --parsable \
            --partition="${D4S_PARTITION}" \
            --dependency=afterany:"${CUTS_AFTER}" \
            --array="${VAL_SPEC}" \
            --export="${EXPORT_VARS},D4S_VAL_STAGE=cuts" \
            submit/validation_array.sh) || CUTS_JOB=""
        [ -n "${CUTS_JOB}" ] && echo "Val cuts array       : ${CUTS_JOB}"

        CUTSRED_JOB=$(sbatch --parsable \
            --partition="${D4S_BIGMEM_PARTITION}" \
            --dependency=afterany:"${CUTS_JOB}" \
            --export="${EXPORT_VARS},D4S_VAL_STAGE=reduce-cuts" \
            submit/validation_reduce.sh) || CUTSRED_JOB=""
        [ -n "${CUTSRED_JOB}" ] && echo "Val cuts reduce      : ${CUTSRED_JOB}"

        # afterany, not afterok: a country whose histogram was not bimodal is a
        # fallback to 1 January, not a failure, and the stats pass degrades to
        # that cut on its own if the table is missing entirely.
        STATS_JOB=$(sbatch --parsable \
            --partition="${D4S_PARTITION}" \
            --dependency=afterany:"${CUTSRED_JOB}" \
            --array="${VAL_SPEC}" \
            --export="${EXPORT_VARS},D4S_VAL_STAGE=stats" \
            submit/validation_array.sh) || STATS_JOB=""
        [ -n "${STATS_JOB}" ] && echo "Val stats array      : ${STATS_JOB}"

        STATSRED_JOB=$(sbatch --parsable \
            --partition="${D4S_BIGMEM_PARTITION}" \
            --dependency=afterany:"${STATS_JOB}" \
            --export="${EXPORT_VARS},D4S_VAL_STAGE=reduce-stats" \
            submit/validation_reduce.sh) || STATSRED_JOB=""
        [ -n "${STATSRED_JOB}" ] && echo "Val stats reduce     : ${STATSRED_JOB}"

        VAL_JOBS="${FETCH_JOB} ${CUTS_JOB} ${CUTSRED_JOB} ${STATS_JOB} ${STATSRED_JOB}"
    else
        echo "Validation skipped   : run_phenology_processing off, or paths.clms_root unset"
    fi
fi

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
  scancel ${ARRAY_JOB} ${COMBINE_JOB} ${MGMT_JOB} ${VAL_JOBS}
==================================================
EOF
