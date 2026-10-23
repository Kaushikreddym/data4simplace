#!/bin/bash
# =============================================================================
# Weather exports for the CMIP6 experiments bias-corrected onto MSWX by climdata.
#
#   ./submit/submit_cmip6.sh --dry-run           # check every experiment, submit nothing
#   ./submit/submit_cmip6.sh                     # submit every complete experiment
#   ./submit/submit_cmip6.sh --retry             # resubmit only unfinished tiles
#   CMIP6_EXPERIMENTS="MIROC6_historical MIROC6_hist-aer" ./submit/submit_cmip6.sh
#   CMIP6_SERIAL=1 D4S_MAX_CONCURRENT=15 ./submit/submit_cmip6.sh   # <=15 jobs at once
#
# One export per <model>/<experiment> under ${CMIP6_OUT_ROOT}. Each exports
# weather ONLY, for exactly the cells of ${CMIP6_BASE_EXPORT}; a dependent job
# then links that export's soil, site.csv and wide fertilizer schedule and
# writes the experiment's own site/co2.csv (data4simplace --link-base). Soil
# and management are properties of the place, not of the climate, so every
# experiment runs on identical non-weather inputs.
#
# An experiment missing any of ${CMIP6_REQUIRED} is refused unless the variable
# is in ${CMIP6_CLIMATOLOGY_FILL} (default: wind only): a
# model without rsds cannot drive a crop model, and an MSWX-filled rsds would
# erase the aerosol-radiation signal hist-aer exists to show.
#
# CO2 follows the forcing: historical and hist-GHG get the CMIP6 historical
# series, hist-aer (and hist-nat) stay at 1850, as their GCMs were forced.
#
# Afterwards, crop-model each finished export with
#   cropmodelling4eu/submit/production/submit_production.sh cmip6
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/env.sh"

export CMIP6_ROOT="${CMIP6_ROOT:-/data01/FDS/muduchuru/Atmos/CMIP6_MSWX_BC}"
# EU_val_nfix, not EU_val: its soil_long.csv carries the corrected mineral N.
export CMIP6_BASE_EXPORT="${CMIP6_BASE_EXPORT:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs/exports/mswx/EU_val_nfix}"
# One export per <model>/<experiment>, beside the observed (mswx/) ones.
export CMIP6_OUT_ROOT="${CMIP6_OUT_ROOT:-/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs/exports/cmip6}"
export CMIP6_START="${CMIP6_START:-1850-01-01}"
export CMIP6_END="${CMIP6_END:-2014-12-31}"
# yearly reads climdata's _yearly_cache: identical values, ~15x fewer opens.
# Switch to daily if that cache is cleaned up.
export CMIP6_LAYOUT="${CMIP6_LAYOUT:-yearly}"
export CMIP6_REQUIRED="${CMIP6_REQUIRED:-tasmin,tasmax,tas,rsds,pr,hurs,sfcWind}"
# Variables a crop model needs but this study does not, filled ONLY where the
# experiment has no files for them, with the per-cell MSWX day-of-year
# climatology (no interannual variability, no forcing signal). Recorded per
# experiment in scenario.json. Empty it to refuse such experiments instead.
export CMIP6_CLIMATOLOGY_FILL="${CMIP6_CLIMATOLOGY_FILL:-sfcWind}"
export CMIP6_CLIMATOLOGY_YEARS="${CMIP6_CLIMATOLOGY_YEARS:-1979,2014}"
export CMIP6_CO2_FILE="${CMIP6_CO2_FILE:-/data01/FDS/muduchuru/Atmos/CMIP6_forcing/input4MIPs_UoM-CMIP-1-2-0/co2_global_annual_UoM-CMIP-1-2-0_1750-2014.csv}"
# A 165-year tile holds ~4 GB of float32 climate and the exporter's float64
# copies on top; 24G is sized for the 46-year MSWX run.
export D4S_MEM="${CMIP6_MEM:-48G}"
export D4S_TIME="${CMIP6_TIME:-12:00:00}"
# 1 = each experiment's tiles wait for the previous experiment's link job, so
# D4S_MAX_CONCURRENT caps the whole submission, not each experiment.
export CMIP6_SERIAL="${CMIP6_SERIAL:-0}"

DRY_RUN=0
RETRY=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --retry)   RETRY=1 ;;
        -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

d4s_activate
cd "${D4S_PROJECT_DIR}" || exit 1
mkdir -p "${D4S_LOG_DIR}"

[ -f "${CMIP6_BASE_EXPORT}/soil/soil.csv" ] || {
    echo "ERROR: base export ${CMIP6_BASE_EXPORT} has no soil/soil.csv" >&2; exit 1; }

# historical and DAMIP hist-* only: the window and the CO2 series here stop in
# 2014, so an SSP experiment (2015-2100) would find no files and no forcing.
EXPERIMENTS="${CMIP6_EXPERIMENTS:-$(find "${CMIP6_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    \( -name '*_historical' -o -name '*_hist-*' \) -printf '%f\n' | sort)}"

co2_hold_year() {
    case "$1" in
        *_hist-aer|*_hist-nat) echo 1850 ;;
        *) echo null ;;
    esac
}

SUBMITTED=()
REFUSED=()
PREV_JOB=""
for EXP in ${EXPERIMENTS}; do
    OUT="${CMIP6_OUT_ROOT}/${EXP%%_*}/${EXP#*_}"     # <model>_<exp> -> <model>/<exp>
    WORK="${OUT}/_work"
    RUN_CONFIG="${WORK}/config_run.yaml"
    mkdir -p "${WORK}/configs"
    echo ""
    echo "=== ${EXP} -> ${OUT}"

    if [ "${RETRY}" -eq 0 ] || [ ! -f "${RUN_CONFIG}" ]; then
        python submit/tile_config.py \
            --config "${D4S_CONFIG}" \
            --output-dir "${OUT}" \
            --out "${RUN_CONFIG}" \
            --flags-only run_climate_processing,export_simplace_weather \
            --set climate.source=cmip6_bc \
            --set "climate.experiment=${EXP}" \
            --set "climate.layout=${CMIP6_LAYOUT}" \
            --set "climate.required=[${CMIP6_REQUIRED}]" \
            --set "climate.climatology_fill=[${CMIP6_CLIMATOLOGY_FILL}]" \
            --set "climate.climatology_years=[${CMIP6_CLIMATOLOGY_YEARS}]" \
            --set "climate.climatology_cache=${CMIP6_OUT_ROOT}/_climatology" \
            --set "paths.cmip6_bc_root=${CMIP6_ROOT}" \
            --set "paths.cell_set_from=${CMIP6_BASE_EXPORT}" \
            --set "paths.co2_file=${CMIP6_CO2_FILE}" \
            --set "site.co2_hold_year=$(co2_hold_year "${EXP}")" \
            --set "time.start=${CMIP6_START}" \
            --set "time.end=${CMIP6_END}" >/dev/null || exit 1
    fi

    if ! data4simplace --config "${RUN_CONFIG}" --dry-run 2>&1 | tail -1; then
        REFUSED+=("${EXP}")
        continue
    fi

    NTILES=$(data4simplace --config "${RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}" --count-tiles) || exit 1
    if [ "${RETRY}" -eq 1 ]; then
        SPEC=$(python submit/tile_status.py --config "${RUN_CONFIG}" \
            --tile-deg "${D4S_TILE_DEG}" --missing) || exit 1
    else
        SPEC="0-$((NTILES - 1))"
    fi
    echo "  tiles: ${NTILES}, array ${SPEC:-<none unfinished>}, co2 hold: $(co2_hold_year "${EXP}")"
    [ "${DRY_RUN}" -eq 1 ] && { SUBMITTED+=("${EXP} (dry run)"); continue; }

    EXPORTS="ALL,D4S_PROJECT_DIR,D4S_CONDA_ENV,D4S_RUN_CONFIG=${RUN_CONFIG},D4S_OUT_DIR=${OUT}"
    EXPORTS="${EXPORTS},D4S_WORK_DIR=${WORK},D4S_WCS_CACHE_ROOT=${WORK}/wcs_cache"
    EXPORTS="${EXPORTS},D4S_TILE_DEG,D4S_LOG_DIR,D4S_CPUS"

    LOGS="${D4S_LOG_DIR}/cmip6/${EXP}"
    mkdir -p "${LOGS}"
    DEP=()
    [ "${CMIP6_SERIAL}" -eq 1 ] && [ -n "${PREV_JOB}" ] && DEP=(--dependency=afterany:"${PREV_JOB}")
    if [ -n "${SPEC}" ]; then
        ARRAY_JOB=$(sbatch --parsable --job-name="d4s_${EXP}" \
            --partition="${D4S_PARTITION}" --cpus-per-task="${D4S_CPUS}" \
            --mem="${D4S_MEM}" --time="${D4S_TIME}" \
            --array="${SPEC}%${D4S_MAX_CONCURRENT}" \
            --output="${LOGS}/tile_%A_%a.log" --error="${LOGS}/tile_%A_%a.err" \
            --export="${EXPORTS}" "${DEP[@]+"${DEP[@]}"}" submit/tile_array.sh) || exit 1
        DEP=(--dependency=afterany:"${ARRAY_JOB}")
        echo "  tile array : ${ARRAY_JOB}"
    fi
    # --wrap runs /bin/sh, which aborts on ~/.bashrc's hyphenated function names.
    # afterany: the link step itself refuses an export with unfinished tiles,
    # and says which, rather than being cancelled silently by afterok.
    LINK_JOB=$(sbatch --parsable --job-name="d4s_link_${EXP}" \
        --partition="${D4S_PARTITION}" --mem=8G --time=01:00:00 \
        --output="${LOGS}/link_%j.log" --error="${LOGS}/link_%j.err" "${DEP[@]+"${DEP[@]}"}" \
        --export="${EXPORTS}" \
        --wrap="bash -c 'source ${SUBMIT_DIR}/env.sh && d4s_activate && cd ${D4S_PROJECT_DIR} && \
data4simplace --config ${RUN_CONFIG} --tile-deg ${D4S_TILE_DEG} --link-base'") || exit 1
    echo "  link base  : ${LINK_JOB}"
    PREV_JOB="${LINK_JOB}"
    echo "  logs       : ${LOGS}/"
    SUBMITTED+=("${EXP}")
done

echo ""
echo "=================================================="
echo "CMIP6 WEATHER EXPORTS"
echo "=================================================="
echo "  submitted : ${#SUBMITTED[@]}  ${SUBMITTED[*]:-}"
echo "  refused   : ${#REFUSED[@]}  ${REFUSED[*]:-}"
[ "${#REFUSED[@]}" -gt 0 ] && echo "              (missing a climate.required variable; see above)"
echo "  done when : ${CMIP6_OUT_ROOT}/<model>/<experiment>/scenario.json exists"
echo "=================================================="
