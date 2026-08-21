#!/bin/bash
# =============================================================================
# Master driver for the torchcrop / LINTUL-5 Europe run.
#
#   ./submit/submit_torchcrop.sh              # submit all shards, then maps
#   ./submit/submit_torchcrop.sh --retry      # submit only the missing shards
#   ./submit/submit_torchcrop.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_torchcrop.sh --no-maps    # shard array only
#   ./submit/submit_torchcrop.sh --maps-only  # re-render maps from the shards
#   ./submit/submit_torchcrop.sh --smoke      # 30 German cells, run here, then evaluate
#
# Tunables live in submit/torchcrop_env.sh and can be overridden per call:
#   TC_START_YEAR=1980 TC_TIME=12:00:00 ./submit/submit_torchcrop.sh
#   TC_RUN_NAME=test TC_END_YEAR=2001 ./submit/submit_torchcrop.sh
#
# Submits
#   1. torchcrop_array.sh - array job, one task per shard (throttled)
#   2. torchcrop_maps.sh  - dependent combine + map render
#
# --smoke submits nothing. It is the counterpart of submit_simplace.sh --smoke
# and shares its directory, its derived config and its cell list, so the two
# models are scored on the same cells over the same seasons.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/torchcrop_env.sh"

DRY_RUN=0
RETRY=0
DO_MAPS=1
MAPS_ONLY=0
SMOKE=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)   DRY_RUN=1 ;;
        --retry)     RETRY=1 ;;
        --no-maps)   DO_MAPS=0 ;;
        --maps-only) MAPS_ONLY=1 ;;
        --smoke)     SMOKE=1 ;;
        -h|--help)   sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

# The shard flags describe an array job the smoke test does not submit, so a
# combination is a misunderstanding rather than something to resolve silently.
if [ "${SMOKE}" -eq 1 ] && { [ "${RETRY}" -eq 1 ] || [ "${MAPS_ONLY}" -eq 1 ] \
    || [ "${DO_MAPS}" -eq 0 ]; }; then
    echo "--smoke runs no array and no maps; drop --retry/--maps-only/--no-maps." >&2
    exit 2
fi

tc_activate
cd "${TC_PROJECT_DIR}" || exit 1

# --- Smoke test ---------------------------------------------------------------
# Runs in the foreground: 30 cells over 3 seasons is seconds of model time, and
# the point is to look at the answer rather than to queue it. The cells go
# through scripts/run_cells_torchcrop.py, not run_shard, because a shard deals
# cells round-robin out of the whole runnable set — right for the array job and
# wrong here, where the whole point is to run *these* cells, the ones SIMPLACE
# gets.
#
# Outputs land under ${TC_SMOKE_DIR}/torchcrop/, the same layout the production
# run uses for its own ${TC_OUT_DIR}: a workspace/ holding the crop file and
# config this run actually used, and the run's own results beside it. The crop
# file is what makes it checkable -- open it next to
# ${TC_SIMPLACE_TEMPLATE}/data/crop/crop.xml and data/management/management.xml
# and see exactly what did or did not carry across.
if [ "${SMOKE}" -eq 1 ]; then
    SMOKE_CONFIG="${TC_SMOKE_DIR}/smoke.yaml"
    CELLS="${TC_SMOKE_DIR}/de_cells.csv"
    SMOKE_TC_DIR="${TC_SMOKE_DIR}/torchcrop"
    SMOKE_CROP_FILE="${SMOKE_TC_DIR}/workspace/crop_${TC_CROP}.yaml"
    # submit_cropmodelling.sh --smoke builds one SIMPLACE run per swept IOPT,
    # under simplace_iopt<n>/ (see submit_simplace.sh's own --smoke block); an
    # older single-run smoke test left it at the flat simplace/ path instead.
    SIMPLACE_OUT="${TC_SMOKE_DIR}/simplace_iopt3/simplace_europe.parquet"
    [ -f "${SIMPLACE_OUT}" ] || SIMPLACE_OUT="${TC_SMOKE_DIR}/simplace/simplace_europe.parquet"

    echo "=================================================="
    echo "TORCHCROP GERMAN SMOKE TEST"
    echo "=================================================="
    echo "  run mode     : iopt=${TC_SMOKE_IOPTS}  crop=${TC_CROP}"
    echo "  cells        : ${TC_SMOKE_CELLS}, one per CyBench NUTS-3 region"
    echo "  seasons      : ${TC_SMOKE_START}-${TC_SMOKE_END}"
    echo "  crop         : ${SMOKE_CROP_FILE} (${TC_SMOKE_CROP_SOURCE})"
    echo "  directory    : ${TC_SMOKE_DIR}"
    echo "=================================================="

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "--dry-run: nothing run."
        exit 0
    fi

    mkdir -p "${TC_SMOKE_DIR}"
    python scripts/make_smoke_config.py --config "${TC_CONFIG}" \
        --out "${SMOKE_CONFIG}" \
        --start-year "${TC_SMOKE_START}" --end-year "${TC_SMOKE_END}" || exit 1

    if [ ! -f "${CELLS}" ]; then
        echo "Selecting ${TC_SMOKE_CELLS} German cells (one per NUTS-3 region)..."
        python scripts/select_german_cells.py --config "${SMOKE_CONFIG}" \
            --n-cells "${TC_SMOKE_CELLS}" --out "${CELLS}" || exit 1
    else
        echo "Reusing the cell list at ${CELLS} ($(( $(wc -l < "${CELLS}") - 1 )) cells)."
    fi

    TC_TEMPLATE_DATA="${TC_SIMPLACE_TEMPLATE}/data"
    python scripts/prepare_torchcrop_workspace.py --config "${SMOKE_CONFIG}" \
        --out-dir "${SMOKE_TC_DIR}" --crop-source "${TC_SMOKE_CROP_SOURCE}" \
        --crop-xml "${TC_TEMPLATE_DATA}/crop/crop.xml" \
        --seeds-xml "${TC_TEMPLATE_DATA}/crop/seeds.xml" \
        --management-xml "${TC_TEMPLATE_DATA}/management/management.xml" \
        > /dev/null || exit 1

    # One pair of parquets per IOPT (TC_SMOKE_IOPTS, default "1 2 3"), same
    # cells, same seasons, same crop file -- only vIOPT's nutrient-limitation
    # setting differs, so a bias between them is that setting and nothing else.
    # --iopt overrides the smoke config's own torchcrop.iopt per call rather
    # than writing three configs, which would also each need its own
    # resolve_export.
    #
    # --daily-out reads the summary and the daily trajectory off *one*
    # simulation (run_cells(..., mode="both")) rather than running the cells
    # twice -- see run_batch/daily_batch's shared _simulate. The daily table is
    # a build artifact of this run, written once here rather than re-derived on
    # demand by every notebook that wants it, matching SIMPLACE's own
    # out/daily/<id>_daily.csv.
    SMOKE_OUTS=() SMOKE_DAILY_OUTS=()
    for IOPT in ${TC_SMOKE_IOPTS}; do
        OUT="${SMOKE_TC_DIR}/de_torchcrop_iopt${IOPT}.parquet"
        DAILY_OUT="${SMOKE_TC_DIR}/de_torchcrop_daily_iopt${IOPT}.parquet"
        SMOKE_OUTS+=("${OUT}")
        SMOKE_DAILY_OUTS+=("${DAILY_OUT}")

        echo "--- iopt=${IOPT} ---"
        python scripts/run_cells_torchcrop.py --config "${SMOKE_CONFIG}" \
            --cells "${CELLS}" --crop-file "${SMOKE_CROP_FILE}" --iopt "${IOPT}" \
            --out "${OUT}" --daily-out "${DAILY_OUT}" \
            --daily-variables ${TC_DAILY_VARIABLES} || exit 1
    done

    echo ""
    echo "=================================================="
    echo "SMOKE TEST READY TO EVALUATE"
    echo "=================================================="
    echo "  cells     : ${CELLS}"
    for OUT in "${SMOKE_OUTS[@]}"; do
        echo "  torchcrop : ${OUT}"
    done
    for OUT in "${SMOKE_DAILY_OUTS[@]}"; do
        echo "  daily     : ${OUT}"
    done
    echo "  crop file : ${SMOKE_CROP_FILE}"
    echo "  audit     : ${SMOKE_TC_DIR}/workspace/crop_parameter_audit.csv"
    echo "  seasons   : ${TC_SMOKE_START}-${TC_SMOKE_END}"
    echo ""
    if [ -f "${SIMPLACE_OUT}" ]; then
        echo "Score both models against CyBench yields and PEP725 phenology"
        echo "(the CLI takes one torchcrop file; pick the same IOPT SIMPLACE"
        echo "was built with -- 3 below, matching this file's own directory):"
        echo ""
        echo "  python scripts/validate_germany.py --cells ${CELLS} \\"
        echo "      --torchcrop ${SMOKE_TC_DIR}/de_torchcrop_iopt3.parquet \\"
        echo "      --simplace ${SIMPLACE_OUT} \\"
        echo "      --out-dir ${TC_SMOKE_DIR}/validation"
        echo ""
        echo "Or open evaluation/germany_smoke_evaluation.ipynb, which loads all"
        echo "three IOPT runs beside SIMPLACE and plots the sweep."
    else
        # validate_germany.py takes --simplace as required: the comparison is
        # the deliverable, and a torchcrop-only score answers half the question.
        echo "No SIMPLACE run beside it yet. Produce one on the same cells with:"
        echo ""
        echo "  ./submit/submit_simplace.sh --smoke"
    fi
    echo "=================================================="
    exit 0
fi

EXPORT_VARS="ALL,TC_PROJECT_DIR,TC_CONDA_ENV,TC_DATA_DIR,TC_OUT_DIR,TC_SHARD_DIR"
EXPORT_VARS="${EXPORT_VARS},TC_LOG_DIR,TC_COMPOSITION_XML,TC_CROP,TC_START_YEAR"
EXPORT_VARS="${EXPORT_VARS},TC_END_YEAR,TC_IOPT,TC_N_SHARDS,TC_BATCH_SIZE"
EXPORT_VARS="${EXPORT_VARS},TC_IO_WORKERS,TC_TORCH_THREADS,TC_SOWING_FILE"
EXPORT_VARS="${EXPORT_VARS},TC_WORK_DIR,TC_CROP_FILE,TC_DAILY,TC_DAILY_VARIABLES"

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
echo "  crop         : ${TC_CROP_FILE} (${TC_CROP_SOURCE})"
[ -n "${TC_SOWING_FILE:-}" ] && \
echo "  sowing       : ${TC_SOWING_FILE} (SIMPLACE-simulated)"
[ -n "${TC_DEPENDENCY:-}" ] && \
echo "  waits for    : ${TC_DEPENDENCY}"
echo "  logs         : ${TC_LOG_DIR}"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# After the dry-run gate, so a dry run never leaves an empty run directory
# behind that --retry would then read as "shard dir exists, nothing missing".
mkdir -p "${TC_LOG_DIR}" "${TC_SHARD_DIR}" "${TC_OUT_DIR}"

# --- The working directory ----------------------------------------------------
# Prepared here, on the login node, before anything is queued: the crop file the
# tasks read has to exist first, and a mistake in it then fails in seconds
# rather than in every array task.
TC_TEMPLATE_DATA="${TC_SIMPLACE_TEMPLATE}/data"
python scripts/prepare_torchcrop_workspace.py --config "${TC_CONFIG}" \
    --out-dir "${TC_OUT_DIR}" --crop-source "${TC_CROP_SOURCE}" \
    --crop-xml "${TC_TEMPLATE_DATA}/crop/crop.xml" \
    --seeds-xml "${TC_TEMPLATE_DATA}/crop/seeds.xml" \
    --management-xml "${TC_TEMPLATE_DATA}/management/management.xml" \
    > /dev/null || { echo "ERROR: could not prepare the workspace" >&2; exit 1; }

# --- 1. Shard array -----------------------------------------------------------
ARRAY_JOB=""
if [ -n "${ARRAY_SPEC}" ]; then
    # An upstream dependency, when this array is chained behind a SIMPLACE run
    # by submit_cropmodelling.sh. Empty in a standalone run.
    ARRAY_DEP=()
    [ -n "${TC_DEPENDENCY:-}" ] && ARRAY_DEP=(--dependency="${TC_DEPENDENCY}")

    ARRAY_JOB=$(sbatch --parsable \
        ${ARRAY_DEP[@]:+"${ARRAY_DEP[@]}"} \
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
