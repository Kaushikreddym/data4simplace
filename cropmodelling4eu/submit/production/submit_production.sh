#!/bin/bash
# =============================================================================
# Production runs of BOTH models on the per-region calibrated crops, for any
# forcing: the MSWX reanalysis and every bias-corrected, downscaled CMIP6
# model x experiment.
#
#   ./submit/production/submit_production.sh --dry-run mswx cmip6     # checks + plan only
#   ./submit/production/submit_production.sh mswx                     # the EU validation run
#   ./submit/production/submit_production.sh cmip6                    # every finished export
#   ./submit/production/submit_production.sh cmip6/MIROC6             # one model, all experiments
#   ./submit/production/submit_production.sh cmip6/MIROC6/ssp370      # one experiment
#   ./submit/production/submit_production.sh --after 861500 cmip6     # hold the chain behind a job
#   ./submit/production/submit_production.sh --torchcrop-only mswx | --simplace-only ...
#
# Naming: one directory per forcing, no dates --
#   <output_dir>/runs/<crop>/calibrated/mswx
#   <output_dir>/runs/<crop>/calibrated/cmip6/<model>/<experiment>
# each holding config.yaml (the run's own config), provenance.json (which
# calibration, export and window), simplace/, simplace_regions/, torchcrop/.
# Inputs are read from cropmodelling4eu_inputs/exports/{mswx/EU_val_nfix,
# cmip6/<model>/<experiment>}; a CMIP6 export counts once its scenario.json
# exists. Every run uses the same calibration, CAL_DIR (stage CAL_STAGE).
#
# Per forcing
#   SIMPLACE   one build per calibration region (a solution reads ONE crop.xml),
#              then ONE array over every (region, line chunk), collected per
#              region and merged into simplace/simplace_europe.parquet plus
#              simplace/sowing_from_simplace.csv.
#   torchcrop  the shard array with TC_REGION_CROP_DIR/TC_REGIONS_FILE selecting
#              each cell's calibrated crop (SUSTAg WW fallback, never Brandenburg).
#              It sows on SIMPLACE's simulated dates: for CMIP6 those of the same
#              run, so it is queued behind that run's merge; for MSWX the table
#              the calibration was fitted on (MSWX_SOWING), so it runs at once.
#   CMIP6      window and harvest years from scenario.json (a winter crop sown in
#              the first year is harvested in the second), co2 from the export's
#              site/co2.csv per year, daily torchcrop output off (CMIP6_TC_DAILY=1
#              restores it), lines_per_task scaled to the window (~9 h tasks).
#
# Cluster use: forcings are chained, not stacked. Each SIMPLACE array waits for
# the previous forcing's, each torchcrop array for the previous torchcrop
# array, so at most SP_SLOTS + TC_SLOTS (10 + 4) jobs plus a short merge run
# at once -- keep the total, with anything else running, <= 15.
# Region builds run here, on the login node, ~28 per forcing; submit a model
# at a time if that is too long a wait.
# =============================================================================
set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
INPUTS=/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs

BASE_CONFIG="${BASE_CONFIG:-${SUBMIT_DIR}/production/config_production.yaml}"
CROP_NAME="${CROP_NAME:-winter_wheat}"
OUTPUT_DIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "${BASE_CONFIG}")
CAL_DIR="${CAL_DIR:-${OUTPUT_DIR}/winter_wheat_2000_2024_euval/calibration}"
CAL_STAGE="${CAL_STAGE:-yield}"
MSWX_EXPORT="${MSWX_EXPORT:-${INPUTS}/exports/mswx/EU_val_nfix}"
MSWX_SOWING="${MSWX_SOWING:-${INPUTS}/sowing_from_simplace_euval.csv}"
CMIP6_ROOT="${CMIP6_ROOT:-${INPUTS}/exports/cmip6}"
SUSTAG_XML="${SUSTAG_XML:-/data01/FDS/muduchuru/Data/SIMPLACE/TEMPLATES/Rotation_monocrop/data/crop/LINTUL5_crop.xml}"
SUSTAG_BLOCK="${SUSTAG_BLOCK:-WW}"
SP_SLOTS="${SP_SLOTS:-10}"
TC_SLOTS="${TC_SLOTS:-4}"
# A cell costs ~72 s of SIMPLACE over 46 years; ~20000 cell-years is a ~9 h task.
CELL_YEARS_PER_TASK="${CELL_YEARS_PER_TASK:-20000}"

DRY_RUN=0 AFTER="" DO_SP=1 DO_TC=1 FORCINGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --after)          AFTER="$2"; shift 2 ;;
        --torchcrop-only) DO_SP=0; shift ;;
        --simplace-only)  DO_TC=0; shift ;;
        -h|--help)        sed -n '2,46p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)  FORCINGS+=("$1"); shift ;;
    esac
done
[ "${#FORCINGS[@]}" -gt 0 ] || { echo "Name a forcing: mswx, cmip6, cmip6/<model>[/<exp>] (--help)" >&2; exit 2; }

die() { echo "ERROR: $*" >&2; exit 1; }

# Expand each argument to forcing keys: mswx | cmip6/<model>/<experiment>.
KEYS=()
for F in "${FORCINGS[@]}"; do
    case "${F}" in
        mswx) KEYS+=(mswx) ;;
        cmip6|cmip6/*)
            SUB="${F#cmip6}"; SUB="${SUB#/}"
            FOUND=$(find "${CMIP6_ROOT}/${SUB}" -name scenario.json -printf '%h\n' 2>/dev/null \
                | sed "s|^${CMIP6_ROOT}/|cmip6/|" | awk -F/ 'NF == 3' | sort)
            [ -n "${FOUND}" ] || die "no finished CMIP6 export under ${CMIP6_ROOT}/${SUB} (no scenario.json)"
            KEYS+=(${FOUND}) ;;
        *) die "unknown forcing '${F}': mswx, cmip6, cmip6/<model>[/<exp>]" ;;
    esac
done

# shellcheck disable=SC1091
source "${SUBMIT_DIR}/utils/simplace_env.sh"
sp_activate
cd "${PROJECT_DIR}" || exit 1

# --- Provenance checks, once: refuse a Brandenburg crop anywhere ---------------
CROPS="${CAL_DIR}/${CAL_STAGE}/crops"
[ -d "${CROPS}" ] || die "no ${CROPS}"
[ -f "${CAL_DIR}/regions.parquet" ] || die "no ${CAL_DIR}/regions.parquet"
[ -f "${SUSTAG_XML}" ] || die "no SUSTAg crop file ${SUSTAG_XML}"
grep -q "<parameter id=\"Crop\"[^>]*>${SUSTAG_BLOCK}<" "${SUSTAG_XML}" \
    || die "${SUSTAG_XML} has no ${SUSTAG_BLOCK} block -- not the SUSTAg LINTUL5_crop.xml"
BAD=$(grep -L "^source_file: .*LINTUL5_crop.xml" "${CROPS}"/crop_region_*.yaml)
[ -z "${BAD}" ] || die "calibrated crops not derived from LINTUL5_crop.xml: ${BAD}"

echo "=================================================="
echo "PRODUCTION RUNS ON CALIBRATED CROPS"
echo "=================================================="
echo "  calibration : ${CAL_DIR}  (stage ${CAL_STAGE}, $(ls "${CROPS}"/crop_region_*.yaml | wc -l) region crops, all from LINTUL5_crop.xml)"
echo "  base config : ${BASE_CONFIG}"
echo "  runs under  : ${OUTPUT_DIR}/runs/${CROP_NAME}/calibrated/"
echo "  forcings    : ${#KEYS[@]}  (chained; SIMPLACE ${SP_SLOTS} + torchcrop ${TC_SLOTS} slots${AFTER:+, after ${AFTER}})"
echo "=================================================="

# Region crop XMLs depend only on the calibration: export and check them once.
REGIONS_EXPORT="${CAL_DIR}/${CAL_STAGE}/simplace_regions_export"
if [ "${DRY_RUN}" -eq 0 ] && [ "${DO_SP}" -eq 1 ]; then
    cm4eu export-regions --calibration-dir "${CAL_DIR}" --stage "${CAL_STAGE}" \
        --template "${SUSTAG_XML}" --simplace-crop "${SUSTAG_BLOCK}" \
        --out-dir "${REGIONS_EXPORT}" > /dev/null || die "export-regions failed"
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
print(f"region crops : {len(list(export.glob('crop_region_*.xml')))} crop.xml checked, {bad} mismatch(es)")
sys.exit(1 if bad else 0)
EOF
fi

# Writes the run's config + provenance and prints "run_name first last"
# (harvest years); --dry-run prints without writing.
run_config() {
    python3 - "$@" <<'EOF'
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

base, key, export, out_root, run_name, cal_dir, stage, per_task, dry = sys.argv[1:10]
config = yaml.safe_load(Path(base).read_text())
config["run_name"] = run_name
config["paths"]["export_dir"] = export
window = None
if key != "mswx":
    window = json.loads((Path(export) / "scenario.json").read_text())["time"]
    first, last = int(window["start"][:4]) + 1, int(window["end"][:4])
    years = int(window["end"][:4]) - int(window["start"][:4]) + 1
    config["season"].update(start_year=first, end_year=last)
    config["simplace"].update(
        start_date=window["start"], end_date=window["end"], co2="annual",
        lines_per_task=max(20, min(500, int(per_task) // years)),
    )
if dry != "1":
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "config.yaml").write_text(
        f"# Generated by submit_production.sh from {base}; do not edit.\n"
        + yaml.safe_dump(config, sort_keys=False)
    )
    (Path(out_root) / "provenance.json").write_text(json.dumps({
        "forcing": key, "export_dir": export, "window": window,
        "seasons": [config["season"]["start_year"], config["season"]["end_year"]],
        "calibration": cal_dir, "calibration_stage": stage, "base_config": base,
        "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")
print(config["season"]["start_year"], config["season"]["end_year"],
      config["simplace"]["lines_per_task"], config["simplace"]["start_date"],
      config["simplace"]["end_date"])
EOF
}

PREV_SP="${AFTER//,/:}" PREV_TC="${AFTER//,/:}" STATUS=0
for KEY in "${KEYS[@]}"; do
    case "${KEY}" in
        mswx) EXPORT="${MSWX_EXPORT}" ;;
        *)    EXPORT="${CMIP6_ROOT}/${KEY#cmip6/}" ;;
    esac
    RUN_NAME="runs/${CROP_NAME}/calibrated/${KEY}"
    OUT_ROOT="${OUTPUT_DIR}/${RUN_NAME}"
    CONFIG="${OUT_ROOT}/config.yaml"
    LOGS="${OUT_ROOT}/logs"
    SP_ROOT_DIR="${OUT_ROOT}/simplace_regions"
    TC_OUT="${OUT_ROOT}/torchcrop"
    [ -d "${EXPORT}" ] || die "no export ${EXPORT}"

    read -r FIRST LAST LINES WSTART WEND < <(run_config "${BASE_CONFIG}" "${KEY}" "${EXPORT}" \
        "${OUT_ROOT}" "${RUN_NAME}" "${CAL_DIR}" "${CAL_STAGE}" "${CELL_YEARS_PER_TASK}" "${DRY_RUN}") \
        || die "config generation failed for ${KEY}"
    if [ "${KEY}" = mswx ]; then SOWING="${MSWX_SOWING}"; else SOWING="${OUT_ROOT}/simplace/sowing_from_simplace.csv"; fi

    echo ""
    echo "=== ${KEY}"
    echo "  export   : ${EXPORT}"
    echo "  run      : ${OUT_ROOT}"
    echo "  window   : ${WSTART} .. ${WEND}  (harvests ${FIRST}-${LAST}, ${LINES} lines/task)"
    echo "  sowing   : ${SOWING}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        [ "${KEY}" = mswx ] && { [ -f "${MSWX_SOWING}" ] || die "no sowing table ${MSWX_SOWING}"; }
        continue
    fi
    mkdir -p "${LOGS}"

    # --- SIMPLACE: region builds, one array, merge --------------------------------
    SP_COLLECT_JOB=""
    if [ "${DO_SP}" -eq 1 ]; then
        (
            export SP_CONFIG="${CONFIG}" SP_RUN_NAME="${RUN_NAME}" SP_RUN_DIR="" SP_LINES_PER_TASK="${LINES}"
            "${SUBMIT_DIR}/utils/submit_simplace_regions.sh" --export-dir "${REGIONS_EXPORT}" \
                --out-root "${SP_ROOT_DIR}" --build-only > "${LOGS}/region_builds.log" 2>&1
        ) || { echo "  SKIP: region builds failed (see ${LOGS}/region_builds.log)" >&2; STATUS=1; continue; }

        TABLE="${OUT_ROOT}/simplace_task_table.txt"
        : > "${TABLE}"
        for ROOT in "${SP_ROOT_DIR}"/region_*/; do
            ROOT="${ROOT%/}"
            [ -f "${ROOT}/run.env" ] || continue
            N_LINES=$(env -u SP_N_LINES bash -c "source '${ROOT}/run.env' && echo \${SP_N_LINES}")
            for ((A = 1; A <= N_LINES; A += LINES)); do
                B=$(( A + LINES - 1 )); [ "${B}" -gt "${N_LINES}" ] && B="${N_LINES}"
                echo "${ROOT} ${A} ${B}" >> "${TABLE}"
            done
        done
        N_TASKS=$(wc -l < "${TABLE}")
        [ "${N_TASKS}" -gt 0 ] || { echo "  SKIP: empty task table ${TABLE}" >&2; STATUS=1; continue; }

        DEP=(); [ -n "${PREV_SP}" ] && DEP=(--dependency="afterany:${PREV_SP}")
        SP_JOB=$(sbatch --parsable --job-name="sp_${KEY//\//_}" "${DEP[@]+"${DEP[@]}"}" \
            --partition="${SP_PARTITION}" --cpus-per-task="${SP_CPUS}" \
            --mem="${SP_MEM}" --time="${SP_TIME}" --array="0-$((N_TASKS - 1))%${SP_SLOTS}" \
            --output="${LOGS}/sp_%A_%a.out" --error="${LOGS}/sp_%A_%a.err" \
            --export="ALL,SP_TASK_TABLE=${TABLE}" "${SUBMIT_DIR}/utils/simplace_task_table.sh") \
            || die "SIMPLACE array submission failed for ${KEY}"
        PREV_SP="${SP_JOB}"
        SP_COLLECT_JOB=$(sbatch --parsable --job-name="sp_${KEY//\//_}_collect" \
            --dependency="afterany:${SP_JOB}" --partition="${SP_PARTITION}" \
            --mem=32G --time=04:00:00 --output="${LOGS}/collect_%j.out" \
            --export="ALL,SP_CONFIG=${CONFIG},CM_SUBMIT_DIR=${SUBMIT_DIR},CM_REGIONS_ROOT=${SP_ROOT_DIR},CM_MERGED=${OUT_ROOT}/simplace/simplace_europe.parquet" \
            "${SUBMIT_DIR}/utils/simplace_regions_collect.sh") \
            || die "collect submission failed for ${KEY}"
        echo "  SIMPLACE : array ${SP_JOB} (${N_TASKS} tasks), merge ${SP_COLLECT_JOB}"
    fi

    # --- torchcrop ---------------------------------------------------------------
    if [ "${DO_TC}" -eq 1 ]; then
        # afterany, not afterok: the merge writes the sowing of every region that
        # collected and names the ones that did not; the shards then refuse to
        # start only if the table is missing altogether.
        WAIT=()
        [ "${KEY}" != mswx ] && [ -n "${SP_COLLECT_JOB}" ] && WAIT+=("${SP_COLLECT_JOB}")
        [ -n "${PREV_TC}" ] && WAIT+=("${PREV_TC}")
        if [ "${KEY}" != mswx ] && [ -z "${SP_COLLECT_JOB}" ] && [ ! -f "${SOWING}" ]; then
            echo "  SKIP torchcrop: no ${SOWING} and no SIMPLACE run queued to write it" >&2
            STATUS=1; continue
        fi
        TC_LOG="${LOGS}/submit_torchcrop.log"
        (
            export TC_CONFIG="${CONFIG}" TC_RUN_NAME="${RUN_NAME}" TC_OUT_DIR="${TC_OUT}" TC_DATA_DIR="${EXPORT}"
            export TC_START_YEAR="${FIRST}" TC_END_YEAR="${LAST}"
            [ "${KEY}" != mswx ] && export TC_DAILY="${CMIP6_TC_DAILY:-0}"
            export TC_CROP_SOURCE=simplace TC_CROP_XML="${SUSTAG_XML}" TC_SIMPLACE_CROP="${SUSTAG_BLOCK}"
            export TC_REGION_CROP_DIR="${CROPS}" TC_REGIONS_FILE="${CAL_DIR}/regions.parquet"
            export TC_SOWING_FILE="${SOWING}" TC_MAX_CONCURRENT="${TC_SLOTS}"
            export TC_DEPENDENCY="${WAIT[*]:+afterany:$(IFS=:; echo "${WAIT[*]}")}"
            unset TC_SHARD_DIR TC_WORK_DIR TC_CROP_FILE
            # shellcheck disable=SC1091
            source "${SUBMIT_DIR}/utils/torchcrop_env.sh"
            # Prepare first and inspect the fallback crop before anything queues;
            # the submit re-prepares it identically from the same inputs.
            "${SUBMIT_DIR}/baseline/submit_torchcrop.sh" --prepare || exit 1
            grep -q "^source_file: .*LINTUL5_crop.xml" "${TC_CROP_FILE}" \
                || { echo "fallback crop ${TC_CROP_FILE} is not SUSTAg-derived" >&2; exit 1; }
            "${SUBMIT_DIR}/baseline/submit_torchcrop.sh"
        ) > "${TC_LOG}" 2>&1 || { echo "  torchcrop submission FAILED (see ${TC_LOG})" >&2; STATUS=1; continue; }
        TC_JOB=$(sed -n 's/^Shard array submitted : *\([0-9]*\).*/\1/p' "${TC_LOG}")
        [ -n "${TC_JOB}" ] && PREV_TC="${TC_JOB}"
        echo "  torchcrop: array ${TC_JOB:-?}${WAIT[*]:+ (after ${WAIT[*]})}, log ${TC_LOG}"
    fi
done

echo ""
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "--dry-run: checks passed, nothing built or submitted."
else
    echo "Results per run: simplace/simplace_europe.parquet, torchcrop/torchcrop_europe.parquet"
    echo "Monitor: squeue -u \$USER"
fi
exit "${STATUS}"
