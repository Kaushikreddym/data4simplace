#!/bin/bash
# =============================================================================
# The full per-region chain: a finished `cm4eu calibrate` run -> per-region
# SIMPLACE builds/submissions -> one region-aware torchcrop array, over the
# same calibrated crop on both sides.
#
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir>
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir> --dry-run
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir> --build-only
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir> --jobs 12
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir> --stage phenology
#   ./submit/submit_cropmodelling_regions.sh --calibration-dir <dir> --torchcrop-out-dir <dir>
#
# --calibration-dir is a `cm4eu calibrate --out-dir` directory: it holds
# regions.parquet and <stage>/crops/crop_region_<NN>.yaml (--stage, default
# yield -- the final stage phenology seeds and yield fine-tunes on top of).
#
# Three steps, the first two parallelised, the third already is:
#
#   1. cm4eu export-regions   -- per-region crop.xml (SIMPLACE) + cell lists,
#                                 on the login node, a few seconds total (it
#                                 rewrites one XML block per region; it does
#                                 not touch the export).
#   2. submit_simplace_regions.sh --jobs N
#                              -- cm4eu simplace build + submit.sh per region,
#                                 N regions building at once (default 6, see
#                                 submit_simplace_regions.sh's own header for
#                                 why that is safe). Each region's build then
#                                 submits its own SLURM array immediately, so
#                                 slower regions do not hold up faster ones.
#   3. submit_torchcrop.sh, region-aware
#                              -- one SLURM array over the whole domain
#                                 (TC_N_SHARDS shards, TC_MAX_CONCURRENT at
#                                 once -- SLURM's own parallelism, untouched
#                                 here); each shard splits its cells by region
#                                 and runs each slice on that region's own
#                                 calibrated crop (region_crop_map in
#                                 cropmodelling4eu.torchcrop.run).
#
# A region with no calibrated crop file (uncalibrated -- no observations
# reached it) runs on --template unchanged on both sides, not dropped.
#
# Tunables: submit/simplace_env.sh (SP_*) and submit/torchcrop_env.sh (TC_*),
# same as submit_cropmodelling.sh. SP_REGION_JOBS sets --jobs' default.
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/torchcrop_env.sh"

CALIBRATION_DIR=""
STAGE="yield"
# The project's own SIMPLACE template is EU SUSTAg's multi-crop
# LINTUL5_crop.xml (config.yaml: simplace.template_dir), not
# TC_SIMPLACE_TEMPLATE -- that env var defaults to the single-crop Brandenburg
# template torchcrop's *standalone* comparison runs use, and relinking a
# region onto the wrong template would silently build every region from a
# different crop than the one calibrated.
TEMPLATE="/data01/FDS/muduchuru/Data/SIMPLACE/Rotation_monocrop/data/crop/LINTUL5_crop.xml"
SIMPLACE_CROP="WW"
JOBS="${SP_REGION_JOBS:-6}"
# Keyed by TC_RUN_NAME, not SP_RUN_NAME: the two env files default those
# independently (winter_wheat_2000_2024_euval vs. winter_wheat_2000_2024), and
# --calibration-dir is a torchcrop-side artifact, so its run name is the one
# that actually matches here. Override with --out-root if that is still wrong
# for a particular calibration run.
OUT_ROOT="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/${TC_RUN_NAME}/simplace_regions"
# torchcrop's own TC_OUT_DIR/TC_SHARD_DIR default to .../${TC_RUN_NAME}/torchcrop --
# the EXISTING uncalibrated production run CALIBRATION.md's numbers and this
# calibration itself are built from. run_shard skips a shard whose output
# Parquet already exists (overwrite=False, and these submit scripts wire no
# --overwrite), so pointed at that directory a "region-aware" array finds
# 20/20 shards already done in seconds and never runs the region code at all
# -- silently, since "already done" is also what a legitimately finished rerun
# looks like. A separate tree keeps the reference run intact and makes the
# calibrated run a real one.
TC_REGION_OUT_DIR="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/${TC_RUN_NAME}/torchcrop_regions"
DRY_RUN=0
BUILD_ONLY=0
SKIP_SIMPLACE=0
SKIP_TORCHCROP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --calibration-dir) CALIBRATION_DIR="$2"; shift 2 ;;
        --stage)           STAGE="$2"; shift 2 ;;
        --template)        TEMPLATE="$2"; shift 2 ;;
        --simplace-crop)   SIMPLACE_CROP="$2"; shift 2 ;;
        --jobs)            JOBS="$2"; shift 2 ;;
        --out-root)        OUT_ROOT="$2"; shift 2 ;;
        --torchcrop-out-dir) TC_REGION_OUT_DIR="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --build-only)      BUILD_ONLY=1; shift ;;
        --no-simplace)     SKIP_SIMPLACE=1; shift ;;
        --no-torchcrop)    SKIP_TORCHCROP=1; shift ;;
        -h|--help)         sed -n '2,38p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "${CALIBRATION_DIR}" ] || {
    echo "ERROR: --calibration-dir is required (a 'cm4eu calibrate --out-dir')." >&2
    exit 2
}
[ -f "${CALIBRATION_DIR}/regions.parquet" ] || {
    echo "ERROR: ${CALIBRATION_DIR}/regions.parquet not found; is this a" >&2
    echo "       finished 'cm4eu calibrate' directory?" >&2
    exit 2
}

sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

EXPORT_DIR="${CALIBRATION_DIR}/region_export_${STAGE}"

echo "=================================================="
echo "CROPMODELLING, PER REGION"
echo "=================================================="
echo "  calibration  : ${CALIBRATION_DIR} (stage: ${STAGE})"
echo "  template     : ${TEMPLATE} (crop block: ${SIMPLACE_CROP})"
echo "  region export: ${EXPORT_DIR}"
echo "  simplace out : ${OUT_ROOT}"
echo "  torchcrop out: ${TC_REGION_OUT_DIR}"
echo "  parallel     : ${JOBS} region(s) building at once"
[ "${SKIP_SIMPLACE}" -eq 1 ]  && echo "  simplace     : skipped (--no-simplace)"
[ "${SKIP_TORCHCROP}" -eq 1 ] && echo "  torchcrop    : skipped (--no-torchcrop)"
echo "=================================================="

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: nothing run."
    exit 0
fi

# --- 1. Per-region crop.xml + cell lists --------------------------------------
echo ""
echo "--- 1. cm4eu export-regions ---"
cm4eu export-regions \
    --calibration-dir "${CALIBRATION_DIR}" \
    --stage "${STAGE}" \
    --template "${TEMPLATE}" \
    --simplace-crop "${SIMPLACE_CROP}" \
    --out-dir "${EXPORT_DIR}" || exit 1

CROPS_DIR="${CALIBRATION_DIR}/${STAGE}/crops"
REGIONS_FILE="${CALIBRATION_DIR}/regions.parquet"

# --- 2. Per-region SIMPLACE, in parallel --------------------------------------
SIMPLACE_STATUS=0
if [ "${SKIP_SIMPLACE}" -eq 0 ]; then
    echo ""
    echo "--- 2. SIMPLACE, per region (${JOBS} at a time) ---"
    SP_ARGS=(--export-dir "${EXPORT_DIR}" --jobs "${JOBS}" --out-root "${OUT_ROOT}")
    [ "${BUILD_ONLY}" -eq 1 ] && SP_ARGS+=(--build-only)
    "${SUBMIT_DIR}/submit_simplace_regions.sh" "${SP_ARGS[@]}" || SIMPLACE_STATUS=$?
    if [ "${SIMPLACE_STATUS}" -ne 0 ]; then
        echo "WARNING: one or more SIMPLACE regions failed to build/submit" \
             "(see above); continuing to torchcrop regardless." >&2
    fi
fi

# --- 3. torchcrop, region-aware, one array over the whole domain -------------
TORCHCROP_STATUS=0
if [ "${SKIP_TORCHCROP}" -eq 0 ]; then
    echo ""
    echo "--- 3. torchcrop, region-aware ---"
    TC_ARGS=()
    if [ "${BUILD_ONLY}" -eq 1 ]; then
        TC_ARGS+=(--dry-run)
        echo "(--build-only: torchcrop's own build-equivalent is --dry-run, since" \
             "its 'build' step -- the workspace crop file -- costs nothing to" \
             "redo and there is no SIMPLACE-style build/submit split to stop" \
             "between.)"
    fi
    TC_REGION_CROP_DIR="${CROPS_DIR}" TC_REGIONS_FILE="${REGIONS_FILE}" \
    TC_OUT_DIR="${TC_REGION_OUT_DIR}" TC_SHARD_DIR="${TC_REGION_OUT_DIR}/shards" \
        "${SUBMIT_DIR}/submit_torchcrop.sh" "${TC_ARGS[@]}" || TORCHCROP_STATUS=$?
fi

echo ""
echo "=================================================="
echo "DONE"
echo "=================================================="
echo "  simplace   : exit ${SIMPLACE_STATUS} ($([ "${SIMPLACE_STATUS}" -eq 0 ] && echo OK || echo "see above"))"
echo "  torchcrop  : exit ${TORCHCROP_STATUS} ($([ "${TORCHCROP_STATUS}" -eq 0 ] && echo OK || echo "see above"))"
echo ""
echo "MONITOR:"
echo "  squeue -u \$USER"
echo "  ./submit/torchcrop_status.sh"
echo "  for d in ${OUT_ROOT}/region_*/; do \"\${d}submit.sh\" --status; done"
echo "=================================================="

[ "${SIMPLACE_STATUS}" -eq 0 ] && [ "${TORCHCROP_STATUS}" -eq 0 ]
