#!/bin/bash
# =============================================================================
# torchcrop MSWX runs of the sowing comparison: old rule-based sowing vs the
# learned sowing network, crossed with the old vs the PEP725 tsumem.
#
#   ./submit/sowing/submit_sowing_comparison.sh --dry-run          # checks + plan
#   ./submit/sowing/submit_sowing_comparison.sh                    # all five arms
#   ./submit/sowing/submit_sowing_comparison.sh A0 B0              # some arms
#   ./submit/sowing/submit_sowing_comparison.sh --after 870000 B0  # hold behind a job
#
#   arm  sowing table                           crops
#   A0   SIMPLACE rule (euval)                  calibrated (tsumem per region)  = the old method
#   A1   SIMPLACE rule (euval)                  calibrated, tsumem/tbasem from PEP725
#   B1   sowing network                         calibrated (tsumem per region)
#   B0   sowing network                         calibrated, tsumem/tbasem from PEP725 = the new method
#   C    per-cell median of the network table   calibrated, tsumem/tbasem from PEP725
#   D    yield-maximising sowing (upper bound)  calibrated, tsumem/tbasem from PEP725
#
# D is not a sowing method: it runs the network table shifted by each of
# ORACLE_OFFSETS (default -42 -28 -14 14 28 42 d) into sowing_comparison/D_grid/,
# reuses B0 as offset 0, and keeps per (cell, season) the offset with the highest
# simulated yield (`cm4eu sowing oracle`, one small job after the grid's maps).
# It reads the season's whole weather and the model's own yield, so it bounds
# what sowing can do in torchcrop. Cost: one full torchcrop run per offset.
#
# Output: <output_dir>/runs/<crop>/sowing_comparison/<arm>/{config.yaml,
# provenance.json, torchcrop/}. Nothing under runs/<crop>/calibrated/ is read
# for writing or touched: production and its fallback stay as they are.
#
# Needs a finished `cm4eu sowing all` in SOWING_DIR (tsumem.json and the two
# sowing tables). The PEP725 crop directory is derived here, once, as a copy.
# Arms are chained (afterany) so at most TC_SLOTS shards run at a time.
# =============================================================================
set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
INPUTS=/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs

BASE_CONFIG="${BASE_CONFIG:-${SUBMIT_DIR}/production/config_production.yaml}"
CROP_NAME="${CROP_NAME:-winter_wheat}"
OUTPUT_DIR=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['paths']['output_dir'])" "${BASE_CONFIG}")
CAL_DIR="${CAL_DIR:-${OUTPUT_DIR}/winter_wheat_2000_2024_euval/calibration_dm}"
CAL_STAGE="${CAL_STAGE:-yield}"
EXPORT="${EXPORT:-${INPUTS}/exports/mswx/EU_val_nfix}"
RULE_SOWING="${RULE_SOWING:-${INPUTS}/sowing_from_simplace_euval.csv}"
SOWING_DIR="${SOWING_DIR:-${OUTPUT_DIR}/sowing_network}"
NET_SOWING="${SOWING_DIR}/sowing_network.csv"
CLIM_SOWING="${SOWING_DIR}/sowing_network_climatology.csv"
SUSTAG_XML="${SUSTAG_XML:-/data01/FDS/muduchuru/Data/SIMPLACE/TEMPLATES/Rotation_monocrop/data/crop/LINTUL5_crop.xml}"
SUSTAG_BLOCK="${SUSTAG_BLOCK:-WW}"
TC_SLOTS="${TC_SLOTS:-4}"
COMPARE_ROOT="${OUTPUT_DIR}/runs/${CROP_NAME}/sowing_comparison"

DRY_RUN=0 AFTER="" ARMS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --after)   AFTER="$2"; shift 2 ;;
        -h|--help) sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 2 ;;
        *)  ARMS+=("$1"); shift ;;
    esac
done
[ "${#ARMS[@]}" -gt 0 ] || ARMS=(A0 A1 B1 B0 C)

die() { echo "ERROR: $*" >&2; exit 1; }

CROPS="${CAL_DIR}/${CAL_STAGE}/crops"
PEP_CROPS="${SOWING_DIR}/crops_pep725"
[ -d "${CROPS}" ] || die "no calibrated crops ${CROPS}"
[ -f "${CAL_DIR}/regions.parquet" ] || die "no ${CAL_DIR}/regions.parquet"

needs_network=0 needs_pep=0
for ARM in "${ARMS[@]}"; do
    case "${ARM}" in
        A0) ;; A1) needs_pep=1 ;; B1) needs_network=1 ;; B0|C|D) needs_network=1 needs_pep=1 ;;
        *) die "unknown arm ${ARM} (A0 A1 B1 B0 C D)" ;;
    esac
done
[ -f "${RULE_SOWING}" ] || die "no rule-based sowing table ${RULE_SOWING}"
if [ "${needs_network}" -eq 1 ]; then
    [ -f "${NET_SOWING}" ] || die "no ${NET_SOWING}: run 'cm4eu sowing all' first"
    [ -f "${CLIM_SOWING}" ] || die "no ${CLIM_SOWING}"
fi
TSUMEM="" TBASEM=""
if [ "${needs_pep}" -eq 1 ]; then
    [ -f "${SOWING_DIR}/tsumem.json" ] || die "no ${SOWING_DIR}/tsumem.json: run 'cm4eu sowing fit-tsumem'"
    read -r TSUMEM TBASEM < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['tsumem'], d['tbasem'])" "${SOWING_DIR}/tsumem.json")
fi

echo "=================================================="
echo "SOWING COMPARISON (torchcrop only, MSWX)"
echo "=================================================="
echo "  arms         : ${ARMS[*]}"
echo "  calibration  : ${CAL_DIR} (stage ${CAL_STAGE})"
echo "  rule sowing  : ${RULE_SOWING}"
echo "  network      : ${SOWING_DIR}"
[ -n "${TSUMEM}" ] && echo "  PEP725 clock : tsumem ${TSUMEM}, tbasem ${TBASEM} -> ${PEP_CROPS}"
echo "  output       : ${COMPARE_ROOT}/<arm>"
echo "=================================================="

if [ "${needs_pep}" -eq 1 ] && [ "${DRY_RUN}" -eq 0 ]; then
    cm4eu sowing override-crops --out-dir "${SOWING_DIR}" --crops-dir "${CROPS}" --target "${PEP_CROPS}" \
        --set "tsumem=${TSUMEM}" "tbasem=${TBASEM}" || die "override-crops failed"
fi

read -r -a OFFSETS <<< "${ORACLE_OFFSETS:--42 -28 -14 14 28 42}"
ORACLE_DIR="${SOWING_DIR}/oracle"
label() { if [ "$1" -lt 0 ]; then echo "m${1#-}"; else echo "p$1"; fi; }

# D expands into one torchcrop run per offset, each a plain arm D_grid/<label>.
RUNS=()
for ARM in "${ARMS[@]}"; do
    if [ "${ARM}" = D ]; then
        for OFF in "${OFFSETS[@]}"; do
            [ "${OFF}" -eq 0 ] && die "offset 0 is B0; leave it out of ORACLE_OFFSETS"
            RUNS+=("D_grid/$(label "${OFF}")")
        done
    else
        RUNS+=("${ARM}")
    fi
done

PREV="${AFTER}" STATUS=0 GRID_MAPS=() B0_MAPS=""
for ARM in "${RUNS[@]}"; do
    case "${ARM}" in
        A0) SOWING="${RULE_SOWING}" ARM_CROPS="${CROPS}"     OVERRIDES="" ;;
        A1) SOWING="${RULE_SOWING}" ARM_CROPS="${PEP_CROPS}" OVERRIDES="tsumem=${TSUMEM} tbasem=${TBASEM}" ;;
        B1) SOWING="${NET_SOWING}"  ARM_CROPS="${CROPS}"     OVERRIDES="" ;;
        B0) SOWING="${NET_SOWING}"  ARM_CROPS="${PEP_CROPS}" OVERRIDES="tsumem=${TSUMEM} tbasem=${TBASEM}" ;;
        C)  SOWING="${CLIM_SOWING}" ARM_CROPS="${PEP_CROPS}" OVERRIDES="tsumem=${TSUMEM} tbasem=${TBASEM}" ;;
        D_grid/*)
            LABEL="${ARM#D_grid/}" OFF="${LABEL#[mp]}"; [ "${LABEL:0:1}" = m ] && OFF="-${OFF}"
            SOWING="${ORACLE_DIR}/sowing_network_${LABEL}.csv" ARM_CROPS="${PEP_CROPS}"
            OVERRIDES="tsumem=${TSUMEM} tbasem=${TBASEM}"
            if [ "${DRY_RUN}" -eq 0 ]; then
                cm4eu sowing shift --source "${NET_SOWING}" --offset "${OFF}" --target "${SOWING}" \
                    || die "shift ${OFF}"
            fi ;;
    esac
    RUN_NAME="runs/${CROP_NAME}/sowing_comparison/${ARM}"
    OUT_ROOT="${OUTPUT_DIR}/${RUN_NAME}"
    CONFIG="${OUT_ROOT}/config.yaml"
    echo "--- ${ARM}: sowing ${SOWING}"
    echo "          crops  ${ARM_CROPS}${OVERRIDES:+ (fallback crop: ${OVERRIDES})}"
    [ "${DRY_RUN}" -eq 1 ] && continue

    python3 - "${BASE_CONFIG}" "${OUT_ROOT}" "${RUN_NAME}" "${EXPORT}" "${ARM}" "${SOWING}" \
        "${ARM_CROPS}" "${CAL_DIR}" "${CAL_STAGE}" "${SOWING_DIR}" "${OVERRIDES}" <<'EOF' || die "config for ${ARM}"
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
base, out, run_name, export, arm, sowing, crops, cal, stage, sowing_dir, overrides = sys.argv[1:12]
config = yaml.safe_load(Path(base).read_text())
config["run_name"] = run_name
config["paths"]["export_dir"] = export
Path(out).mkdir(parents=True, exist_ok=True)
(Path(out) / "config.yaml").write_text(
    f"# Generated by submit_sowing_comparison.sh from {base}; do not edit.\n"
    + yaml.safe_dump(config, sort_keys=False))
digest = hashlib.sha256(Path(sowing).read_bytes()).hexdigest()
model = Path(sowing_dir) / "model" / "model.json"
(Path(out) / "provenance.json").write_text(json.dumps({
    "arm": arm, "forcing": "mswx", "export_dir": export,
    "seasons": [config["season"]["start_year"], config["season"]["end_year"]],
    "sowing_file": sowing, "sowing_sha256": digest, "crops_dir": crops,
    "fallback_crop_overrides": overrides or None,
    "tsumem_source": "PEP725 fit" if overrides else "calibrated per region",
    "network_model": str(model) if "network" in Path(sowing).name else None,
    "calibration": cal, "calibration_stage": stage, "base_config": base,
    "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, indent=2) + "\n")
EOF
    read -r FIRST LAST < <(python3 -c "import yaml,sys; s=yaml.safe_load(open(sys.argv[1]))['season']; print(s['start_year'], s['end_year'])" "${CONFIG}")
    TC_LOG="${OUT_ROOT}/submit_torchcrop.log"
    (
        export TC_CONFIG="${CONFIG}" TC_RUN_NAME="${RUN_NAME}" TC_OUT_DIR="${OUT_ROOT}/torchcrop" TC_DATA_DIR="${EXPORT}"
        export TC_START_YEAR="${FIRST}" TC_END_YEAR="${LAST}"
        export TC_CROP_SOURCE=simplace TC_CROP_XML="${SUSTAG_XML}" TC_SIMPLACE_CROP="${SUSTAG_BLOCK}"
        export TC_REGION_CROP_DIR="${ARM_CROPS}" TC_REGIONS_FILE="${CAL_DIR}/regions.parquet"
        export TC_SOWING_FILE="${SOWING}" TC_MAX_CONCURRENT="${TC_SLOTS}" TC_CROP_OVERRIDES="${OVERRIDES}"
        export TC_SHARD_BY="${TC_SHARD_BY:-regions}" TC_CPUS="${TC_CPUS:-8}" TC_MEM="${TC_MEM:-32G}"
        export TC_DEPENDENCY="${PREV:+afterany:${PREV//,/:}}" TC_LOG_DIR="${OUT_ROOT}/logs"
        unset TC_SHARD_DIR TC_WORK_DIR TC_CROP_FILE
        # shellcheck disable=SC1091
        source "${SUBMIT_DIR}/utils/torchcrop_env.sh"
        "${SUBMIT_DIR}/baseline/submit_torchcrop.sh"
    ) > "${TC_LOG}" 2>&1 || { echo "  ${ARM}: submission FAILED (see ${TC_LOG})" >&2; STATUS=1; continue; }
    JOB=$(sed -n 's/^Shard array submitted : *\([0-9]*\).*/\1/p' "${TC_LOG}")
    MAPS=$(sed -n 's/^Maps submitted *: *\([0-9]*\).*/\1/p' "${TC_LOG}")
    [ -n "${JOB}" ] && PREV="${JOB}"
    case "${ARM}" in D_grid/*) [ -n "${MAPS}" ] && GRID_MAPS+=("${MAPS}") ;; B0) B0_MAPS="${MAPS}" ;; esac
    echo "  ${ARM}: array ${JOB:-?}, maps ${MAPS:-?}, log ${TC_LOG}"
done

# --- D: pick the yield-maximising offset per (cell, season) -------------------
if printf '%s\n' "${ARMS[@]}" | grep -qx D; then
    D_ROOT="${COMPARE_ROOT}/D"
    RUN_ARGS="--run=0=${COMPARE_ROOT}/B0/torchcrop/torchcrop_europe.parquet"
    for OFF in "${OFFSETS[@]}"; do
        RUN_ARGS+=" --run=${OFF}=${COMPARE_ROOT}/D_grid/$(label "${OFF}")/torchcrop/torchcrop_europe.parquet"
    done
    echo "--- D: oracle over offsets 0 ${OFFSETS[*]} (offset 0 = B0) -> ${D_ROOT}"
    [ "${DRY_RUN}" -eq 1 ] && exit "${STATUS}"
    [ "${STATUS}" -eq 0 ] || die "a grid run failed to submit; D not assembled"
    mkdir -p "${D_ROOT}/logs"
    python3 - "${D_ROOT}" "${PEP_CROPS}" "${TSUMEM}" "${TBASEM}" "${CAL_DIR}" "${CAL_STAGE}" "${OFFSETS[*]}" <<'PROV' || die "provenance for D"
import json, sys
from datetime import datetime, timezone
from pathlib import Path
out, crops, tsumem, tbasem, cal, stage, offsets = sys.argv[1:8]
(Path(out) / "provenance.json").write_text(json.dumps({
    "arm": "D", "forcing": "mswx", "sowing_file": str(Path(out) / "sowing_yield_max.csv"),
    "sowing_sha256": None, "sowing_rule": "argmax simulated yield_t_ha over network table + offsets",
    "offsets_days": [0, *map(int, offsets.split())], "crops_dir": crops,
    "fallback_crop_overrides": f"tsumem={tsumem} tbasem={tbasem}", "tsumem_source": "PEP725 fit",
    "calibration": cal, "calibration_stage": stage,
    "submitted": datetime.now(timezone.utc).isoformat(timespec="seconds"),
}, indent=2) + "\n")
PROV
    DEPS=("${GRID_MAPS[@]}"); [ -n "${B0_MAPS}" ] && DEPS+=("${B0_MAPS}")
    DEP_ARG=""; [ "${#DEPS[@]}" -gt 0 ] && DEP_ARG="--dependency=afterany:$(IFS=:; echo "${DEPS[*]}")"
    ORACLE_JOB=$(sbatch --parsable --job-name=sowing_oracle --partition=compute --cpus-per-task=4 \
        --mem="${ORACLE_MEM:-64G}" --time=02:00:00 ${DEP_ARG} --output="${D_ROOT}/logs/oracle_%j.out" \
        --wrap="bash -c 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate ${SOWING_CONDA_ENV:-sdba} && cm4eu sowing oracle --target ${D_ROOT} ${RUN_ARGS}'") \
        || die "oracle submission failed"
    echo "  D: oracle job ${ORACLE_JOB}${DEP_ARG:+ (${DEP_ARG#--dependency=})}"
fi
exit "${STATUS}"
