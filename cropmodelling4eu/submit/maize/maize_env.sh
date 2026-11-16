#!/bin/bash
# =============================================================================
# Grain maize on the EU_val inputs. Source this BEFORE any driver:
#
#   source submit/maize/maize_env.sh
#   submit/baseline/submit_cropmodelling.sh --smoke      # 30 German cells, both models
#   submit/baseline/submit_cropmodelling.sh              # the uncalibrated baseline run
#   submit/calibration/submit_calibration.sh             # the wheat protocol, on maize
#
# Every value is a default (:-), so anything already exported wins. The
# drivers' own env files read these names, so nothing in them changes for
# wheat.
#
# CM4EU_CROP selects the crop profile (src/cropmodelling4eu/crops.py): CLMS
# label and offsets, moisture basis, parameter specs. sbatch --export=ALL
# carries it into every job.
# =============================================================================

export CM4EU_CROP="${CM4EU_CROP:-maize}"

_CM="/data01/FDS/muduchuru/codes/GITHUB/data4simplace/cropmodelling4eu"
_EXPORT="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu_inputs/exports/mswx/EU_val_maize"
_RUNS="/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu"
export CM_MAIZE_RUN_NAME="${CM_MAIZE_RUN_NAME:-maize_2000_2024_euval}"

# --- SIMPLACE ------------------------------------------------------------------
export SP_CONFIG="${SP_CONFIG:-${_CM}/config_maize.yaml}"
export SP_RUN_NAME="${SP_RUN_NAME:-${CM_MAIZE_RUN_NAME}}"
export SP_SMOKE_DIR="${SP_SMOKE_DIR:-${_RUNS}/de_smoke_maize}"
# The 15-node cap is on everything at once; one array may not take it all.
export SP_MAX_CONCURRENT="${SP_MAX_CONCURRENT:-12}"
# 8 glibc JVMs x 4g heap per task (production-resource-tuning); 40G is the baseline default.
export SP_MEM="${SP_MEM:-48G}"

# --- torchcrop -----------------------------------------------------------------
export TC_CONFIG="${TC_CONFIG:-${_CM}/config_maize.yaml}"
export TC_DATA_DIR="${TC_DATA_DIR:-${_EXPORT}}"
export TC_RUN_NAME="${TC_RUN_NAME:-${CM_MAIZE_RUN_NAME}}"
export TC_SMOKE_DIR="${TC_SMOKE_DIR:-${_RUNS}/de_smoke_maize}"
export TC_CROP="${TC_CROP:-maize}"
# Harmonised against the SUSTAg block SIMPLACE runs. seeds.xml names grain maize
# KM (Koernermais); its values are recorded, never mapped.
export TC_CROP_XML="${TC_CROP_XML:-/data01/FDS/muduchuru/Data/SIMPLACE/TEMPLATES/Rotation_monocrop/data/crop/LINTUL5_crop.xml}"
export TC_SIMPLACE_CROP="${TC_SIMPLACE_CROP:-MAIZ}"
export TC_SEEDS_CROP="${TC_SEEDS_CROP:-KM}"
export TC_CELLS_FILE="${TC_CELLS_FILE:-${_EXPORT}/domain_maize_5pct.csv}"
export TC_MAX_CONCURRENT="${TC_MAX_CONCURRENT:-12}"

# --- Calibration ---------------------------------------------------------------
export CAL_CONFIG="${CAL_CONFIG:-${_CM}/config_maize.yaml}"
export CAL_RUN_NAME="${CAL_RUN_NAME:-${CM_MAIZE_RUN_NAME}}"
export CAL_SIMPLACE_CROP="${CAL_SIMPLACE_CROP:-MAIZ}"
export CAL_CROP_XML="${CAL_CROP_XML:-${TC_CROP_XML}}"
# The baseline SIMPLACE run's rule-based sowing, as for wheat; it exists only
# once that run is collected.
export CAL_SOWING_FILE="${CAL_SOWING_FILE:-${_RUNS}/${CM_MAIZE_RUN_NAME}/simplace/sowing_from_simplace.csv}"

unset _CM _EXPORT _RUNS
