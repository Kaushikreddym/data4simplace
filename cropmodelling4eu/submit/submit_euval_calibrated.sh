#!/bin/bash
# =============================================================================
# Full EU validation run of BOTH models on the per-region calibrated crops.
#
#   ./submit/submit_euval_calibrated.sh --dry-run        # checks + plan, submits nothing
#   ./submit/submit_euval_calibrated.sh                  # build, check, submit
#   ./submit/submit_euval_calibrated.sh --after 860756   # hold every job behind a job
#   ./submit/submit_euval_calibrated.sh --torchcrop-only | --simplace-only
#
# What runs where
#   SIMPLACE   one build per calibration region (a solution reads ONE crop.xml),
#              each with that region's crop_region_NN.xml copied in, then ONE
#              array over every (region, line chunk) -- so a single %SP_SLOTS
#              caps the whole run instead of 27 per-region arrays adding up.
#              Collected per region, merged into simplace/simplace_europe.parquet.
#   torchcrop  the standard shard array, with TC_REGION_CROP_DIR/TC_REGIONS_FILE
#              selecting each cell's calibrated crop; an uncalibrated region
#              (no observations) and any unplaced cell fall back to the SUSTAg
#              WW crop, never the Brandenburg template.
#
# Inputs, and why
#   CM_CONFIG   submit/config_nfix.yaml -> export EU_val_nfix: identical weather,
#               wide soil and site to EU_val, but a soil_long.csv with the
#               corrected mineral N (EU_val's starts SIMPLACE on ~2900 kg N/ha).
#               torchcrop reads the wide soil, so both models see the same inputs.
#   CAL_DIR     calibration_full_20261019 (phenology -> yield). The yield-stage
#               crops carry the phenology fit too.
#   TC_SOWING   the sowing table the calibration was fitted on.
#   Baselines to compare against: <nfix run>/simplace (SIMPLACE default) and
#   winter_wheat_2000_2024_euval/torchcrop (torchcrop default).
#
# Cluster use: SP_SLOTS + TC_SLOTS concurrent jobs (default 10 + 4) plus one
# short collect/maps job; keep the total, with anything else running, <= 15.
# =============================================================================
set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
DATA=/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu

CM_CONFIG="${CM_CONFIG:-${SUBMIT_DIR}/config_nfix.yaml}"
CAL_DIR="${CAL_DIR:-${DATA}/winter_wheat_2000_2024_euval/calibration_full_20261019}"
CAL_STAGE="${CAL_STAGE:-yield}"
BASE_RUN="${BASE_RUN:-${DATA}/winter_wheat_2000_2024_euval_nfix}"
OUT_ROOT="${OUT_ROOT:-${BASE_RUN}/calibrated_$(basename "${CAL_DIR}" | sed 's/^calibration_//')}"
SUSTAG_XML="${SUSTAG_XML:-/data01/FDS/muduchuru/Data/SIMPLACE/TEMPLATES/Rotation_monocrop/data/crop/LINTUL5_crop.xml}"
SUSTAG_BLOCK="${SUSTAG_BLOCK:-WW}"
TC_SOWING="${TC_SOWING:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs/sowing_from_simplace_euval.csv}"
SP_SLOTS="${SP_SLOTS:-10}"
TC_SLOTS="${TC_SLOTS:-4}"

DRY_RUN=0 AFTER="" DO_SP=1 DO_TC=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --after)          AFTER="$2"; shift 2 ;;
        --torchcrop-only) DO_SP=0; shift ;;
        --simplace-only)  DO_TC=0; shift ;;
        -h|--help)        sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

RUN_NAME=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "${CM_CONFIG}")
export SP_CONFIG="${CM_CONFIG}" SP_RUN_NAME="${RUN_NAME}"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"
sp_activate
cd "${PROJECT_DIR}" || exit 1

REGIONS_EXPORT="${OUT_ROOT}/regions_export"
SP_ROOT_DIR="${OUT_ROOT}/simplace_regions"
TC_OUT="${OUT_ROOT}/torchcrop"
LOGS="${OUT_ROOT}/logs"
DEP=()
[ -n "${AFTER}" ] && DEP=(--dependency="afterany:${AFTER//,/:}")

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=================================================="
echo "EU VALIDATION ON CALIBRATED CROPS"
echo "=================================================="
echo "  config        : ${CM_CONFIG}  (run ${RUN_NAME})"
echo "  calibration   : ${CAL_DIR}  (stage ${CAL_STAGE})"
echo "  SUSTAg crop   : ${SUSTAG_XML}  (block ${SUSTAG_BLOCK})"
echo "  outputs       : ${OUT_ROOT}"
echo "  slots         : SIMPLACE ${SP_SLOTS}, torchcrop ${TC_SLOTS}${AFTER:+, after ${AFTER}}"
echo "=================================================="

# --- 0. Provenance checks: refuse a Brandenburg crop anywhere ------------------
CROPS="${CAL_DIR}/${CAL_STAGE}/crops"
[ -d "${CROPS}" ] || die "no ${CROPS}"
[ -f "${CAL_DIR}/regions.parquet" ] || die "no ${CAL_DIR}/regions.parquet"
[ -f "${SUSTAG_XML}" ] || die "no SUSTAg crop file ${SUSTAG_XML}"
grep -q "<parameter id=\"Crop\"[^>]*>${SUSTAG_BLOCK}<" "${SUSTAG_XML}" \
    || die "${SUSTAG_XML} has no ${SUSTAG_BLOCK} block -- not the SUSTAg LINTUL5_crop.xml"
BAD=$(grep -L "^source_file: .*LINTUL5_crop.xml" "${CROPS}"/crop_region_*.yaml)
[ -z "${BAD}" ] || die "calibrated crops not derived from LINTUL5_crop.xml: ${BAD}"
echo "provenance    : $(ls "${CROPS}"/crop_region_*.yaml | wc -l) calibrated crops, all from LINTUL5_crop.xml"

[ "${DRY_RUN}" -eq 1 ] && { echo "--dry-run: checks passed, nothing built or submitted."; exit 0; }
mkdir -p "${OUT_ROOT}" "${LOGS}"

# --- 1-4. SIMPLACE ----------------------------------------------------------------
SP_COLLECT_JOB=""
if [ "${DO_SP}" -eq 1 ]; then
    cm4eu export-regions --calibration-dir "${CAL_DIR}" --stage "${CAL_STAGE}" \
        --template "${SUSTAG_XML}" --simplace-crop "${SUSTAG_BLOCK}" \
        --out-dir "${REGIONS_EXPORT}" || die "export-regions failed"

    # Every calibrated value SIMPLACE can carry must be in its XML: scalars
    # verbatim, RUETB with scale_factor_rue folded in.
    python - "${REGIONS_EXPORT}" "${CROPS}" "${SUSTAG_BLOCK}" <<'EOF' || die "region crop.xml check failed"
import sys
from pathlib import Path
from cropmodelling4eu.torchcrop.params import SCALARS, flatten_crop_yaml, load_simplace_crop
export, crops, block = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
bad = 0
for xml in sorted(export.glob("crop_region_*.xml")):
    fit = flatten_crop_yaml(crops / xml.with_suffix(".yaml").name)
    got = load_simplace_crop(xml, block)
    for sp, tc in SCALARS.items():
        if tc in fit and sp in got and not isinstance(fit[tc], list):
            bad += abs(float(got[sp]) - float(fit[tc])) > 1e-4 * max(1.0, abs(float(fit[tc])))
    scale = float(fit.get("scale_factor_rue", 1.0))
    ruetb = got.get("RUETB")
    if isinstance(ruetb, list) and "ruetb" in fit:
        want = [y * scale for _, y in fit["ruetb"]]
        have = ruetb[1::2] if not isinstance(ruetb[0], (list, tuple)) else [y for _, y in ruetb]
        bad += any(abs(a - b) > 1e-4 for a, b in zip(want, have))
print(f"region crop.xml: {len(list(export.glob('crop_region_*.xml')))} checked, {bad} mismatch(es)")
sys.exit(1 if bad else 0)
EOF

    SP_REGIONS_OUT_ROOT="${SP_ROOT_DIR}" "${SUBMIT_DIR}/submit_simplace_regions.sh" \
        --export-dir "${REGIONS_EXPORT}" --out-root "${SP_ROOT_DIR}" --build-only \
        || die "region builds failed (see ${SP_ROOT_DIR}/region_*.log)"

    TABLE="${OUT_ROOT}/simplace_task_table.txt"
    : > "${TABLE}"
    for ROOT in "${SP_ROOT_DIR}"/region_*/; do
        ROOT="${ROOT%/}"
        [ -f "${ROOT}/run.env" ] || continue
        N_LINES=$(env -u SP_N_LINES bash -c "source '${ROOT}/run.env' && echo \${SP_N_LINES}")
        for ((FIRST = 1; FIRST <= N_LINES; FIRST += SP_LINES_PER_TASK)); do
            LAST=$(( FIRST + SP_LINES_PER_TASK - 1 ))
            [ "${LAST}" -gt "${N_LINES}" ] && LAST="${N_LINES}"
            echo "${ROOT} ${FIRST} ${LAST}" >> "${TABLE}"
        done
    done
    N_TASKS=$(wc -l < "${TABLE}")
    [ "${N_TASKS}" -gt 0 ] || die "empty task table ${TABLE}"
    echo "SIMPLACE      : ${N_TASKS} tasks over $(ls -d "${SP_ROOT_DIR}"/region_*/ | wc -l) region builds"

    SP_JOB=$(sbatch --parsable --job-name=sp_euval_cal "${DEP[@]+"${DEP[@]}"}" \
        --partition="${SP_PARTITION}" --cpus-per-task="${SP_CPUS}" \
        --mem="${SP_MEM}" --time="${SP_TIME}" --array="0-$((N_TASKS - 1))%${SP_SLOTS}" \
        --output="${LOGS}/sp_%A_%a.out" --error="${LOGS}/sp_%A_%a.err" \
        --export="ALL,SP_TASK_TABLE=${TABLE}" "${SUBMIT_DIR}/simplace_task_table.sh") \
        || die "SIMPLACE array submission failed"
    echo "SIMPLACE array: ${SP_JOB}"

    SP_COLLECT_JOB=$(sbatch --parsable --job-name=sp_euval_cal_collect \
        --dependency="afterany:${SP_JOB}" --partition="${SP_PARTITION}" \
        --mem=32G --time=04:00:00 --output="${LOGS}/collect_%j.out" \
        --export="ALL,SP_CONFIG=${CM_CONFIG},CM_SUBMIT_DIR=${SUBMIT_DIR},CM_REGIONS_ROOT=${SP_ROOT_DIR},CM_MERGED=${OUT_ROOT}/simplace/simplace_europe.parquet" \
        "${SUBMIT_DIR}/simplace_regions_collect.sh") \
        || die "collect submission failed"
    echo "SIMPLACE merge: ${SP_COLLECT_JOB} -> ${OUT_ROOT}/simplace/simplace_europe.parquet"
fi

# --- 5. torchcrop -------------------------------------------------------------
if [ "${DO_TC}" -eq 1 ]; then
    [ -f "${TC_SOWING}" ] || die "no sowing table ${TC_SOWING}"
    export TC_CONFIG="${CM_CONFIG}" TC_RUN_NAME="${RUN_NAME}" TC_OUT_DIR="${TC_OUT}"
    export TC_CROP_SOURCE=simplace TC_CROP_XML="${SUSTAG_XML}" TC_SIMPLACE_CROP="${SUSTAG_BLOCK}"
    export TC_REGION_CROP_DIR="${CROPS}" TC_REGIONS_FILE="${CAL_DIR}/regions.parquet"
    export TC_SOWING_FILE="${TC_SOWING}" TC_MAX_CONCURRENT="${TC_SLOTS}"
    export TC_DEPENDENCY="${AFTER:+afterany:${AFTER//,/:}}"
    unset TC_SHARD_DIR TC_WORK_DIR TC_CROP_FILE
    # shellcheck disable=SC1091
    source "${SUBMIT_DIR}/torchcrop_env.sh"

    # Prepare the fallback crop here and inspect it before anything is queued;
    # submit_torchcrop.sh re-prepares it identically from the same inputs.
    cm4eu workspace --config "${TC_CONFIG}" \
        --out-dir "${TC_OUT}" --crop-source simplace --crop-xml "${TC_CROP_XML}" \
        --simplace-crop "${TC_SIMPLACE_CROP}" \
        --seeds-xml "${TC_SIMPLACE_TEMPLATE}/data/crop/seeds.xml" \
        --seeds-crop "${TC_SEEDS_CROP:-winter_wheat}" \
        --management-xml "${TC_SIMPLACE_TEMPLATE}/data/management/management.xml" \
        > /dev/null || die "torchcrop workspace preparation failed"
    grep -q "^source_file: .*LINTUL5_crop.xml" "${TC_CROP_FILE}" \
        || die "fallback crop ${TC_CROP_FILE} is not SUSTAg-derived"
    echo "torchcrop     : fallback crop ${TC_CROP_FILE} (SUSTAg ${TC_SIMPLACE_CROP}), region crops ${CROPS}"
    "${SUBMIT_DIR}/submit_torchcrop.sh" || die "torchcrop submission failed"
fi

echo ""
echo "Outputs when done:"
[ "${DO_SP}" -eq 1 ] && echo "  SIMPLACE  ${OUT_ROOT}/simplace/simplace_europe.parquet   (baseline ${BASE_RUN}/simplace/)"
[ "${DO_TC}" -eq 1 ] && echo "  torchcrop ${TC_OUT}/torchcrop_europe.parquet   (baseline ${DATA}/winter_wheat_2000_2024_euval/torchcrop/)"
