#!/bin/bash
# =============================================================================
# Build + submit one SIMPLACE run per calibration region, each on its own
# calibrated crop.xml, --jobs regions at a time.
#
#   ./submit/submit_simplace_regions.sh --export-dir <cm4eu export-regions dir>
#   ./submit/submit_simplace_regions.sh --export-dir <dir> --dry-run
#   ./submit/submit_simplace_regions.sh --export-dir <dir> --build-only
#   ./submit/submit_simplace_regions.sh --export-dir <dir> --jobs 12
#
# --export-dir is a `cm4eu export-regions` output directory:
#   cells_region_<NN>.csv   -- that region's SimplaceID list
#   crop_region_<NN>.xml    -- the template crop.xml with that region's
#                              torchcrop fit written back in (missing for an
#                              uncalibrated region -- it still gets built, on
#                              the template's own crop.xml, unchanged)
#
# Each region is its own `cm4eu simplace build --cells-file ...`, i.e. exactly
# what submit/submit_simplace.sh does, restricted to one region's cells, with
# that region's crop.xml *copied* into the built workspace afterwards (a
# solution reads SIMPLACE's crop parameters from one project-wide file, so a
# per-region run means a per-region *build*, not a per-region flag). It is a
# copy, not a symlink to --export-dir: the Singularity container only binds
# workspace/, out/, the export and the template (Workspace.binds()), so a
# symlink pointing anywhere else resolves on the login node and dangles
# inside the container -- every task then fails with
# MissingSimResourceException, not a build-time error, so the mistake is
# invisible until a task actually runs. Everything past the build -- array
# sizing, --retry, --status -- is each region's own generated submit.sh, not
# reimplemented here; see its own --help. --build-only builds and copies in
# every region's crop and submits nothing, so the swap can be checked (diff a
# region's workspace/data/crop/<file>.xml against --export-dir) before
# anything runs.
#
# --jobs (default 6, or $SP_REGION_JOBS) caps how many regions build at once.
# Each build is its own `resolve_export` -- a read of the full soil/fertiliser
# export regardless of how few cells the region keeps -- so building all ~28
# regions one at a time pays that cost 28 times in series; every region writes
# only its own --out-dir, and nothing here shares a weather-conversion cache
# (confirm with `cm4eu inspect`: "symlinked from" rather than "cached in"
# means builds never contend on a cache write), so running them concurrently
# is safe. The array jobs the builds then submit run under SLURM's own
# concurrency (TC_MAX_CONCURRENT / SP_MAX_CONCURRENT), which --jobs does not
# touch -- this only parallelises the login-node build step.
#
# Cluster settings come from submit/simplace_env.sh, same as submit_simplace.sh.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"

EXPORT_DIR=""
OUT_ROOT="${SP_REGIONS_OUT_ROOT:-}"
JOBS="${SP_REGION_JOBS:-6}"
DRY_RUN=0
BUILD_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --export-dir) EXPORT_DIR="$2"; shift 2 ;;
        --out-root)   OUT_ROOT="$2"; shift 2 ;;
        --jobs)       JOBS="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --build-only) BUILD_ONLY=1; shift ;;
        -h|--help)    sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "${EXPORT_DIR}" ] || {
    echo "ERROR: --export-dir is required (a 'cm4eu export-regions' output)." >&2
    exit 2
}
[ -d "${EXPORT_DIR}" ] || {
    echo "ERROR: ${EXPORT_DIR} is not a directory." >&2
    exit 2
}
case "${JOBS}" in
    ''|*[!0-9]*|0) echo "ERROR: --jobs must be a positive integer, got '${JOBS}'." >&2; exit 2 ;;
esac
OUT_ROOT="${OUT_ROOT:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/${SP_RUN_NAME}/simplace_regions}"

sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

shopt -s nullglob
CELL_LISTS=("${EXPORT_DIR}"/cells_region_*.csv)
shopt -u nullglob
[ ${#CELL_LISTS[@]} -gt 0 ] || {
    echo "ERROR: no cells_region_*.csv under ${EXPORT_DIR}. Run" >&2
    echo "       'cm4eu export-regions' first." >&2
    exit 2
}

echo "=================================================="
echo "SIMPLACE PER-REGION RUN"
echo "=================================================="
echo "  export dir   : ${EXPORT_DIR}"
echo "  regions      : ${#CELL_LISTS[@]}"
echo "  out root     : ${OUT_ROOT}"
echo "  config       : ${SP_CONFIG}"
echo "  parallel     : ${JOBS} region(s) building at once"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing built."
    exit 0
fi

mkdir -p "${OUT_ROOT}"

# --- One region: build, copy in its crop, (optionally) submit. Runs as a ---
# job of *this* shell (not a subshell via xargs/parallel), so it needs no
# export -f and sees every variable and function above it as-is. Its own
# stdout+stderr go to ${OUT_ROOT}/region_<NN>.log, since N of these running at
# once would otherwise interleave into unreadable output; its outcome goes to
# ${OUT_ROOT}/region_<NN>.result for the summary after `wait` collects them all.
build_region() {
    local CELLS="$1" REGION LOG RESULT REGION_OUT REGION_XML BUILD_OUT BUILT_DIR TEMPLATE_CROP
    local CALIBRATED_FLAG
    REGION=$(basename "${CELLS}" .csv | sed -E 's/^cells_region_//')
    REGION_OUT="${OUT_ROOT}/region_${REGION}"
    REGION_XML="${EXPORT_DIR}/crop_region_${REGION}.xml"
    LOG="${OUT_ROOT}/region_${REGION}.log"
    RESULT="${OUT_ROOT}/region_${REGION}.result"
    rm -f "${RESULT}"

    {
        echo "--- region ${REGION} ---"
        if ! BUILD_OUT=$(cm4eu simplace build --config "${SP_CONFIG}" \
                --lines-per-task "${SP_LINES_PER_TASK}" \
                --out-dir "${REGION_OUT}" --cells-file "${CELLS}" 2>&1); then
            echo "${BUILD_OUT}"
            echo "FAIL ${REGION} build" > "${RESULT}"
            return 1
        fi
        echo "${BUILD_OUT}"
        BUILT_DIR=$(echo "${BUILD_OUT}" | sed -n 's/^Built in *: *//p')

        if [ -f "${REGION_XML}" ]; then
            # The build links workspace/data/crop/<name> from the template;
            # swap that one link for this region's calibrated file, named the
            # same so the solution's own <res file="data/crop/<name>"/>
            # still resolves. A *copy*, not a symlink to --export-dir: the
            # container only binds workspace/, out/, the export and the
            # template (see Workspace.binds()) -- a symlink pointing anywhere
            # else resolves on the host and dangles inside the container,
            # which is a MissingSimResourceException, not a build-time error,
            # so a broken relink is invisible until a task actually runs.
            TEMPLATE_CROP=$(find "${BUILT_DIR}/workspace/data/crop" -maxdepth 1 -name '*.xml' | head -1)
            if [ -z "${TEMPLATE_CROP}" ]; then
                echo "ERROR: ${BUILT_DIR}/workspace/data/crop has no .xml to relink."
                echo "FAIL ${REGION} relink ${BUILT_DIR}" > "${RESULT}"
                return 1
            fi
            rm -f "${TEMPLATE_CROP}"
            cp "${REGION_XML}" "${TEMPLATE_CROP}"
            echo "Copied    : $(basename "${TEMPLATE_CROP}") <- ${REGION_XML}"
            CALIBRATED_FLAG=1
        else
            echo "No calibrated crop for region ${REGION}; left on the template's own crop.xml."
            CALIBRATED_FLAG=0
        fi

        if [ "${BUILD_ONLY}" -eq 1 ]; then
            echo "OK ${REGION} ${CALIBRATED_FLAG} ${BUILT_DIR}" > "${RESULT}"
            return 0
        fi

        if ! "${BUILT_DIR}/submit.sh" > "${REGION_OUT}.submit.log" 2>&1; then
            echo "submit.sh failed; see ${REGION_OUT}.submit.log"
            echo "FAIL ${REGION} submit ${BUILT_DIR}" > "${RESULT}"
            return 1
        fi
        echo "Submitted : region ${REGION} (see ${REGION_OUT}.submit.log)"
        echo "OK ${REGION} ${CALIBRATED_FLAG} ${BUILT_DIR}" > "${RESULT}"
    } > "${LOG}" 2>&1
}

echo "Building ${#CELL_LISTS[@]} region(s), ${JOBS} at a time (see ${OUT_ROOT}/region_<NN>.log)..."
for CELLS in "${CELL_LISTS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do
        wait -n
    done
    build_region "${CELLS}" &
done
wait

BUILT=() CALIBRATED=() FALLBACK=() FAILED=()
for CELLS in "${CELL_LISTS[@]}"; do
    REGION=$(basename "${CELLS}" .csv | sed -E 's/^cells_region_//')
    RESULT="${OUT_ROOT}/region_${REGION}.result"
    if [ ! -f "${RESULT}" ]; then
        FAILED+=("${REGION} (no result file -- check ${OUT_ROOT}/region_${REGION}.log)")
        continue
    fi
    read -r STATUS RID CAL_OR_REASON DIR < "${RESULT}"
    if [ "${STATUS}" = "FAIL" ]; then
        FAILED+=("${RID} (${CAL_OR_REASON} -- see ${OUT_ROOT}/region_${RID}.log)")
        continue
    fi
    BUILT+=("${DIR}")
    if [ "${CAL_OR_REASON}" = "1" ]; then CALIBRATED+=("${RID}"); else FALLBACK+=("${RID}"); fi
done

echo ""
echo "=================================================="
echo "${#CALIBRATED[@]} region(s) on their calibrated crop, ${#FALLBACK[@]} on the template, " \
     "${#FAILED[@]} failed"
[ ${#FALLBACK[@]} -gt 0 ] && echo "  fallback: ${FALLBACK[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED:"
    printf '    %s\n' "${FAILED[@]}"
fi
echo "=================================================="

if [ ${#BUILT[@]} -eq 0 ]; then
    echo "ERROR: no region built successfully." >&2
    exit 1
fi

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "--build-only: built, crop copied in, nothing submitted."
    for DIR in "${BUILT[@]}"; do
        echo "  ${DIR}/submit.sh"
    done
    [ ${#FAILED[@]} -eq 0 ]
    exit $?
fi

cat <<EOF

MONITOR each region with its own submit.sh --status, e.g.:
  ${BUILT[0]}/submit.sh --status

COLLECT each region once finished, then concatenate:
  for d in ${OUT_ROOT}/region_*/; do
      cm4eu simplace collect --config "${SP_CONFIG}" --out-dir "\${d%/}"
  done
EOF

[ ${#FAILED[@]} -eq 0 ]
