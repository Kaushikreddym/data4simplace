#!/bin/bash
# =============================================================================
# Build a SIMPLACE run from a config, then hand over to the run's own scripts.
#
#   ./submit/submit_simplace.sh              # build, then submit the array
#   ./submit/submit_simplace.sh --build-only # build and validate, submit nothing
#   ./submit/submit_simplace.sh --dry-run    # print the plan, submit nothing
#   ./submit/submit_simplace.sh --status     # what an existing run has finished
#   ./submit/submit_simplace.sh --retry      # rebuild, then re-run what failed
#   ./submit/submit_simplace.sh --smoke      # 30 German cells, run here, then evaluate
#
# Cluster defaults live in submit/simplace_env.sh and can be overridden:
#   SP_LINES_PER_TASK=5000 SP_TIME=24:00:00 ./submit/submit_simplace.sh
#
# This script exists only to get from a *config* to a built run directory. Once
# that exists it is self-driving: `cm4eu simplace build` writes submit.sh and
# run_task.sh into it, and everything below is delegated there rather than
# reimplemented here. A second implementation of the array sizing and the retry
# set is a second thing to keep in step, and the generated pair has the better
# claim — it needs neither this package nor the conda environment, so it still
# works months later.
#
#   <run>/submit.sh --status | --retry | --local | --lines A-B
#
# The build runs **here**, on the login node, before anything is queued: a
# configuration mistake then fails in seconds rather than minutes into every
# array task.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"

BUILD_ONLY=0
SMOKE=0
PASS_THROUGH=()
for arg in "$@"; do
    case "${arg}" in
        --build-only) BUILD_ONLY=1 ;;
        --smoke)      SMOKE=1 ;;
        --dry-run|--status|--retry|--local) PASS_THROUGH+=("${arg}") ;;
        -h|--help)    sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

# --- Smoke test ---------------------------------------------------------------
# A separate run directory, a shorter window and an explicit cell list. It runs
# in the foreground because 30 cells is one task of a few minutes, and because
# the point is to look at the answer rather than to queue it.
CELLS_ARG=()
if [ "${SMOKE}" -eq 1 ]; then
    mkdir -p "${SP_SMOKE_DIR}"
    SP_RUN_DIR="${SP_SMOKE_DIR}/simplace"
    SP_CONFIG="${SP_SMOKE_DIR}/smoke.yaml"
    CELLS="${SP_SMOKE_DIR}/de_cells.csv"

    # Derived from the main config, so the smoke test cannot drift away from
    # the run it is meant to be a smoke test of.
    python - "${SP_PROJECT_DIR}/config.yaml" "${SP_CONFIG}" \
        "${SP_SMOKE_START}" "${SP_SMOKE_END}" <<'PY' || exit 1
import sys, yaml
src, dst, start, end = sys.argv[1:5]
cfg = yaml.safe_load(open(src))
cfg["run_name"] = "de_smoke"
cfg["season"] |= {"start_year": int(start), "end_year": int(end)}
# A season needs the year before it, and SIMPLACE's own window must cover both.
cfg["simplace"] |= {
    "start_date": f"{cfg['season']['start_year'] - 2}-01-01",
    "end_date": f"{cfg['season']['end_year'] + 1}-12-31",
    "lines_per_task": 10_000,
}
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

    if [ ! -f "${CELLS}" ]; then
        echo "Selecting ${SP_SMOKE_CELLS} German cells (one per NUTS-3 region)..."
        python scripts/select_german_cells.py --config "${SP_CONFIG}" \
            --n-cells "${SP_SMOKE_CELLS}" --out "${CELLS}" || exit 1
    else
        echo "Reusing the cell list at ${CELLS} ($(( $(wc -l < "${CELLS}") - 1 )) cells)."
    fi
    CELLS_ARG=(--cells-file "${CELLS}")
    PASS_THROUGH+=(--local)
fi

# --- Build and validate -------------------------------------------------------
# --status must not rebuild: it is a question about a run, and rebuilding would
# rewrite run.env underneath the answer.
BUILT_DIR="${SP_RUN_DIR:-}"
case " ${PASS_THROUGH[*]:-} " in
    *" --status "*) SKIP_BUILD=1 ;;
    *)              SKIP_BUILD=0 ;;
esac

if [ "${SKIP_BUILD}" -eq 0 ]; then
    echo "Building the run directory and validating the solution..."
    BUILD_ARGS=(--config "${SP_CONFIG}")
    [ "${SMOKE}" -eq 0 ] && BUILD_ARGS+=(--lines-per-task "${SP_LINES_PER_TASK}")
    [ -n "${BUILT_DIR}" ] && BUILD_ARGS+=(--out-dir "${BUILT_DIR}")
    BUILD_ARGS+=(${CELLS_ARG[@]:+"${CELLS_ARG[@]}"})

    BUILD_OUT=$(cm4eu simplace build "${BUILD_ARGS[@]}" 2>&1) || {
        echo "${BUILD_OUT}" >&2
        echo "ERROR: build failed; nothing submitted." >&2
        exit 1
    }
    echo "${BUILD_OUT}"
    BUILT_DIR=$(echo "${BUILD_OUT}" | sed -n 's/^Built in *: *//p')
fi

if [ -z "${BUILT_DIR}" ] || [ ! -f "${BUILT_DIR}/submit.sh" ]; then
    echo "ERROR: no built run directory. Set SP_RUN_DIR, or drop --status so" >&2
    echo "       the run is built first." >&2
    exit 1
fi

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo ""
    echo "--build-only: built and validated, nothing submitted."
    echo "  ${BUILT_DIR}/submit.sh"
    exit 0
fi

# --- Hand over ----------------------------------------------------------------
if [ "${SMOKE}" -eq 0 ]; then
    exec "${BUILT_DIR}/submit.sh" ${PASS_THROUGH[@]:+"${PASS_THROUGH[@]}"}
fi

"${BUILT_DIR}/submit.sh" ${PASS_THROUGH[@]:+"${PASS_THROUGH[@]}"} || exit 1
cm4eu simplace collect --config "${SP_CONFIG}" --out-dir "${BUILT_DIR}" || exit 1

cat <<EOF

==================================================
SMOKE TEST READY TO EVALUATE
==================================================
  cells    : ${CELLS}
  simplace : ${BUILT_DIR}/simplace_europe.parquet
  seasons  : ${SP_SMOKE_START}-${SP_SMOKE_END}

Score it against CyBench yields and PEP725 phenology:

  python scripts/run_cells_torchcrop.py --config ${SP_CONFIG} \
      --cells ${CELLS} --out ${SP_SMOKE_DIR}/de_torchcrop.parquet
  python scripts/validate_germany.py --cells ${CELLS} \
      --torchcrop ${SP_SMOKE_DIR}/de_torchcrop.parquet \
      --simplace ${BUILT_DIR}/simplace_europe.parquet \
      --out-dir ${SP_SMOKE_DIR}/validation

Or open evaluation/germany_smoke_evaluation.ipynb, which does both and plots.
==================================================
EOF
