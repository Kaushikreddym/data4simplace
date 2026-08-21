#!/bin/bash
# =============================================================================
# Both models, in the order that lets the second use the first's answer.
#
#   ./submit/submit_cropmodelling.sh              # build, then submit the chain
#   ./submit/submit_cropmodelling.sh --build-only # build and validate, submit nothing
#   ./submit/submit_cropmodelling.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_cropmodelling.sh --status     # what each stage has finished
#   ./submit/submit_cropmodelling.sh --retry      # rebuild, then re-run what failed
#   ./submit/submit_cropmodelling.sh --smoke      # 30 German cells, run here, then evaluate
#
# Tunables come from BOTH environment files, unchanged:
#   submit/simplace_env.sh   (SP_*)   and   submit/torchcrop_env.sh   (TC_*)
#
# SIMPLACE first, because its sowing date is a *result*: the rule-based
# solution sows on the first day inside the planting window on which a weather
# rule fires, per cell and per season. torchcrop has no such rule -- it latches
# one day-of-year -- so run alone it takes the export's proposed date and the
# two models grow different seasons. Chaining them removes the single largest
# difference between the runs, and it is the only difference that cannot be
# removed by configuration.
#
# Three jobs, dependency-chained, so the whole thing is one submission:
#
#   1. <run>/submit.sh          SIMPLACE array, one task per line range
#   2. cropmodelling_handoff.sh afterany -- collect, and write the sowing table
#   3. submit_torchcrop.sh      afterok  -- shard array on those dates, + maps
#
# `afterany` on the handoff and `afterok` on torchcrop is deliberate: a
# part-failed SIMPLACE array should still be collected and reported (its
# retryable tasks are listed by --status), but torchcrop must not start on a
# sowing table that was never written.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/torchcrop_env.sh"

BUILD_ONLY=0
DRY_RUN=0
STATUS=0
RETRY=0
SMOKE=0
for arg in "$@"; do
    case "${arg}" in
        --build-only) BUILD_ONLY=1 ;;
        --dry-run)    DRY_RUN=1 ;;
        --status)     STATUS=1 ;;
        --retry)      RETRY=1 ;;
        --smoke)      SMOKE=1 ;;
        -h|--help)    sed -n '2,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

# Where the SIMPLACE run lands, and therefore where its sowing table lands.
# Resolved the same way `cm4eu simplace build` resolves it, so --status can
# find a run this script did not submit.
resolve_run_dir() {
    if [ -n "${SP_RUN_DIR}" ]; then
        echo "${SP_RUN_DIR}"
        return
    fi
    python - "${SP_CONFIG}" <<'PY'
import sys
from cropmodelling4eu.config import load_config
print(load_config(sys.argv[1]).run_dir / "simplace")
PY
}

# --- Smoke test ---------------------------------------------------------------
# The same 30 German cells through both models, in the foreground, with the
# handoff done inline: there is no array to wait on, so there is no job to
# depend on either.
#
# torchcrop's outputs and workspace land under ${SP_SMOKE_DIR}/torchcrop/, the
# same layout the production chain uses: workspace/crop_<crop>.yaml is the
# crop this run actually used, checkable directly against
# ${TC_SIMPLACE_TEMPLATE}/data/crop/crop.xml, data/crop/seeds.xml and
# data/management/management.xml in the SIMPLACE template.
if [ "${SMOKE}" -eq 1 ]; then
    CELLS="${SP_SMOKE_DIR}/de_cells.csv"
    SMOKE_CONFIG="${SP_SMOKE_DIR}/smoke.yaml"
    SMOKE_TC_DIR="${SP_SMOKE_DIR}/torchcrop"
    SMOKE_CROP_FILE="${SMOKE_TC_DIR}/workspace/crop_${TC_CROP}.yaml"

    echo "=================================================="
    echo "CROPMODELLING SMOKE TEST — SIMPLACE, then torchcrop"
    echo "=================================================="
    echo "  cells     : ${SP_SMOKE_CELLS}, one per CyBench NUTS-3 region"
    echo "  seasons   : ${SP_SMOKE_START}-${SP_SMOKE_END}"
    echo "  crop      : ${SMOKE_CROP_FILE} (${TC_SMOKE_CROP_SOURCE})"
    echo "  directory : ${SP_SMOKE_DIR}"
    echo "  sowing    : SIMPLACE-simulated, handed to torchcrop (per IOPT)"
    echo "  iopt sweep: ${TC_SMOKE_IOPTS} -- both models, one SIMPLACE build per"
    echo "              value (no CLI override for vIOPT, unlike torchcrop --iopt)"
    echo "=================================================="

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "--dry-run: nothing run."
        exit 0
    fi

    # The base config: cells and the torchcrop workspace (crop file) are
    # iopt-independent, so both are derived once here rather than inside the
    # per-IOPT loop below. Each IOPT's SIMPLACE run derives its own
    # smoke_iopt<n>.yaml from the same main config (see submit_simplace.sh
    # --smoke), so this file drives only the cell selection and torchcrop.
    python scripts/make_smoke_config.py --config "${SP_PROJECT_DIR}/config.yaml" \
        --out "${SMOKE_CONFIG}" \
        --start-year "${SP_SMOKE_START}" --end-year "${SP_SMOKE_END}" || exit 1

    TC_TEMPLATE_DATA="${TC_SIMPLACE_TEMPLATE}/data"
    python scripts/prepare_torchcrop_workspace.py --config "${SMOKE_CONFIG}" \
        --out-dir "${SMOKE_TC_DIR}" --crop-source "${TC_SMOKE_CROP_SOURCE}" \
        --crop-xml "${TC_TEMPLATE_DATA}/crop/crop.xml" \
        --seeds-xml "${TC_TEMPLATE_DATA}/crop/seeds.xml" \
        --management-xml "${TC_TEMPLATE_DATA}/management/management.xml" \
        > /dev/null || exit 1

    # One SIMPLACE build and one torchcrop pass per IOPT (TC_SMOKE_IOPTS,
    # default "1 2 3"), same cells and seasons throughout -- only vIOPT's
    # nutrient-limitation setting differs, so a bias between the three is that
    # setting and nothing else. SP_IOPT drives submit_simplace.sh --smoke into
    # its own simplace_iopt<n>/ run directory (see its own --smoke block);
    # torchcrop's --iopt overrides the (iopt-independent) base config instead
    # of needing a build of its own. Filenames match the naming
    # germany_smoke_evaluation.ipynb reads.
    #
    # --daily-out reads the summary and the daily trajectory off *one*
    # simulation (run_cells(..., mode="both")) instead of running the cells
    # twice -- a build artifact of this run rather than something an
    # evaluation notebook re-derives on demand.
    SIMPLACE_OUTS=() SMOKE_OUTS=() SMOKE_DAILY_OUTS=()
    for IOPT in ${TC_SMOKE_IOPTS}; do
        echo ""
        echo "--- iopt=${IOPT}: SIMPLACE ---"
        SP_IOPT="${IOPT}" "${SUBMIT_DIR}/submit_simplace.sh" --smoke || exit 1

        SP_OUT_DIR="${SP_SMOKE_DIR}/simplace_iopt${IOPT}"
        SOWING="${SP_OUT_DIR}/sowing_from_simplace.csv"
        SIMPLACE_OUT="${SP_OUT_DIR}/simplace_europe.parquet"
        [ -s "${SOWING}" ] || {
            echo "ERROR: ${SOWING} not written by the SIMPLACE smoke run." >&2
            exit 1
        }
        SIMPLACE_OUTS+=("${SIMPLACE_OUT}")

        TC_OUT="${SMOKE_TC_DIR}/de_torchcrop_iopt${IOPT}.parquet"
        TC_DAILY_OUT="${SMOKE_TC_DIR}/de_torchcrop_daily_iopt${IOPT}.parquet"
        SMOKE_OUTS+=("${TC_OUT}")
        SMOKE_DAILY_OUTS+=("${TC_DAILY_OUT}")

        echo "--- iopt=${IOPT}: torchcrop, on SIMPLACE's simulated sowing ---"
        echo "Handing $(( $(wc -l < "${SOWING}") - 1 )) simulated sowing dates to torchcrop..."
        python scripts/run_cells_torchcrop.py --config "${SMOKE_CONFIG}" \
            --cells "${CELLS}" --sowing-file "${SOWING}" \
            --crop-file "${SMOKE_CROP_FILE}" --iopt "${IOPT}" \
            --out "${TC_OUT}" --daily-out "${TC_DAILY_OUT}" \
            --daily-variables ${TC_DAILY_VARIABLES} || exit 1
    done

    echo ""
    echo "=================================================="
    echo "SMOKE TEST READY TO EVALUATE"
    echo "=================================================="
    echo "  cells     : ${CELLS}"
    for OUT in "${SIMPLACE_OUTS[@]}"; do
        echo "  simplace  : ${OUT}"
    done
    for OUT in "${SMOKE_OUTS[@]}"; do
        echo "  torchcrop : ${OUT}"
    done
    for OUT in "${SMOKE_DAILY_OUTS[@]}"; do
        echo "  daily     : ${OUT}"
    done
    echo "  crop file : ${SMOKE_CROP_FILE}"
    echo "  audit     : ${SMOKE_TC_DIR}/workspace/crop_parameter_audit.csv"
    echo ""
    echo "Both models sowed on the same dates within each IOPT, so a phenology"
    echo "difference below is the model and not the calendar. Compare a"
    echo "matched pair (same IOPT on both sides):"
    echo ""
    echo "  python scripts/validate_germany.py --cells ${CELLS} \\"
    echo "      --torchcrop ${SMOKE_TC_DIR}/de_torchcrop_iopt3.parquet \\"
    echo "      --simplace ${SP_SMOKE_DIR}/simplace_iopt3/simplace_europe.parquet \\"
    echo "      --out-dir ${SP_SMOKE_DIR}/validation"
    echo ""
    echo "Or open evaluation/germany_smoke_evaluation.ipynb, which loads all"
    echo "three IOPT runs on both sides and plots the sweep."
    echo "=================================================="
    exit 0
fi

RUN_DIR="$(resolve_run_dir)"
SOWING_FILE="${RUN_DIR}/sowing_from_simplace.csv"

# --- Status -------------------------------------------------------------------
# Never builds: it is a question about a run, and a build would rewrite run.env
# underneath the answer.
if [ "${STATUS}" -eq 1 ]; then
    echo "=================================================="
    echo "1. SIMPLACE — ${RUN_DIR}"
    echo "=================================================="
    if [ -x "${RUN_DIR}/submit.sh" ]; then
        "${RUN_DIR}/submit.sh" --status
    else
        echo "  not built yet."
    fi

    echo ""
    echo "=================================================="
    echo "2. HANDOFF — ${SOWING_FILE}"
    echo "=================================================="
    if [ -s "${SOWING_FILE}" ]; then
        echo "  $(( $(wc -l < "${SOWING_FILE}") - 1 )) cell-seasons over" \
             "$(tail -n +2 "${SOWING_FILE}" | cut -d, -f1 | sort -u | wc -l) cells"
    else
        echo "  not written yet; torchcrop cannot start."
    fi

    echo ""
    echo "=================================================="
    echo "3. TORCHCROP — ${TC_SHARD_DIR}"
    echo "=================================================="
    "${SUBMIT_DIR}/torchcrop_status.sh"
    exit 0
fi

# --- 1. Build and submit SIMPLACE ---------------------------------------------
echo "Building the SIMPLACE run and validating the solution..."
BUILD_ARGS=(--config "${SP_CONFIG}" --lines-per-task "${SP_LINES_PER_TASK}")
[ -n "${SP_RUN_DIR}" ] && BUILD_ARGS+=(--out-dir "${SP_RUN_DIR}")

BUILD_OUT=$(cm4eu simplace build "${BUILD_ARGS[@]}" 2>&1) || {
    echo "${BUILD_OUT}" >&2
    echo "ERROR: build failed; nothing submitted." >&2
    exit 1
}
echo "${BUILD_OUT}"
RUN_DIR=$(echo "${BUILD_OUT}" | sed -n 's/^Built in *: *//p')
SOWING_FILE="${RUN_DIR}/sowing_from_simplace.csv"

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo ""
    echo "--build-only: built and validated, nothing submitted."
    echo "  ${RUN_DIR}/submit.sh"
    echo "  then: TC_SOWING_FILE=${SOWING_FILE} ./submit/submit_torchcrop.sh"
    exit 0
fi

echo ""
echo "=================================================="
echo "CROPMODELLING CHAIN"
echo "=================================================="
echo "  1. simplace  : ${RUN_DIR}"
echo "  2. handoff   : ${SOWING_FILE}"
echo "  3. torchcrop : ${TC_OUT_DIR}"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    "${RUN_DIR}/submit.sh" --dry-run
    echo ""
    TC_SOWING_FILE="${SOWING_FILE}" "${SUBMIT_DIR}/submit_torchcrop.sh" --dry-run
    echo ""
    echo "--dry-run: nothing submitted."
    exit 0
fi

SP_ARGS=()
[ "${RETRY}" -eq 1 ] && SP_ARGS+=(--retry)
SP_OUT=$("${RUN_DIR}/submit.sh" ${SP_ARGS[@]:+"${SP_ARGS[@]}"}) || {
    echo "${SP_OUT}" >&2
    echo "ERROR: SIMPLACE submission failed; nothing chained behind it." >&2
    exit 1
}
echo "${SP_OUT}"
SP_JOB=$(echo "${SP_OUT}" | sed -n 's/^Submitted \([0-9]\+\).*/\1/p' | head -1)

# --retry with nothing to retry prints no job id. The sowing table is then
# already there, so the chain simply starts at the handoff.
if [ -z "${SP_JOB}" ]; then
    echo ""
    echo "No SIMPLACE array was submitted (nothing to retry)."
    if [ ! -s "${SOWING_FILE}" ]; then
        echo "ERROR: and no sowing table exists. Collect the run first:" >&2
        echo "       cm4eu simplace collect --config ${SP_CONFIG} --out-dir ${RUN_DIR}" >&2
        exit 1
    fi
    TC_SOWING_FILE="${SOWING_FILE}" exec "${SUBMIT_DIR}/submit_torchcrop.sh" \
        ${SP_ARGS[@]:+"${SP_ARGS[@]}"}
fi

# --- 2. Handoff: collect, and write the sowing table --------------------------
mkdir -p "${SP_LOG_DIR}"
HANDOFF_JOB=$(sbatch --parsable \
    --dependency=afterany:"${SP_JOB}" \
    --partition="${SP_PARTITION}" \
    --mem="${SP_MEM}" \
    --time=01:00:00 \
    --export=ALL,SP_PROJECT_DIR,SP_CONFIG,SP_CONDA_ENV,CM_RUN_DIR="${RUN_DIR}" \
    "${SUBMIT_DIR}/cropmodelling_handoff.sh") \
    || { echo "ERROR: handoff submission failed" >&2; exit 1; }
echo "Handoff submitted     : ${HANDOFF_JOB}  (afterany:${SP_JOB})"

# --- 3. torchcrop, on SIMPLACE's dates ----------------------------------------
TC_SOWING_FILE="${SOWING_FILE}" \
TC_DEPENDENCY="afterok:${HANDOFF_JOB}" \
    "${SUBMIT_DIR}/submit_torchcrop.sh" ${SP_ARGS[@]:+"${SP_ARGS[@]}"} || exit 1

cat <<EOF

==================================================
CHAIN SUBMITTED
==================================================
  simplace : ${SP_JOB}
  handoff  : ${HANDOFF_JOB}   afterany:${SP_JOB}
  torchcrop: queued behind the handoff (see above)

  progress : ./submit/submit_cropmodelling.sh --status
  cancel   : scancel ${SP_JOB} ${HANDOFF_JOB}
==================================================
EOF
