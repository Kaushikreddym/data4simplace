#!/bin/bash
# =============================================================================
# The join between the two models: collect the finished SIMPLACE run and write
# the sowing dates torchcrop is then to sow on.
#
# Submitted by submit/submit_cropmodelling.sh with a dependency on the SIMPLACE
# array. It exists as its own job because the sowing table cannot be written
# until every SIMPLACE task has finished, and the torchcrop array cannot start
# until it has been.
#
# `cm4eu simplace collect` writes both the run Parquet and, beside it,
# sowing_from_simplace.csv. The check below is the useful part: an array that
# half-failed still collects, and the torchcrop shards would then silently run
# on a fraction of the domain.
# =============================================================================
#SBATCH --job-name=cm_handoff
#SBATCH --output=submit/logs/cm_handoff_%j.out
#SBATCH --error=submit/logs/cm_handoff_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -uo pipefail

for _d in "${SP_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/simplace_env.sh" ] && { SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/simplace_env.sh. Set SP_PROJECT_DIR." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"
sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

: "${CM_RUN_DIR:?cropmodelling_handoff.sh needs CM_RUN_DIR (the built run)}"
SOWING="${CM_RUN_DIR}/sowing_from_simplace.csv"

sp_banner "COLLECT SIMPLACE + WRITE SOWING DATES"
echo "  run dir     : ${CM_RUN_DIR}"

cm4eu simplace collect --config "${SP_CONFIG}" --out-dir "${CM_RUN_DIR}" || {
    echo "ERROR: collect failed; torchcrop has no sowing dates to run on." >&2
    exit 1
}

[ -s "${SOWING}" ] || {
    echo "ERROR: ${SOWING} was not written or is empty." >&2
    exit 1
}

# One line per (cell, season) that actually sowed. Fewer than the project's
# cells means SIMPLACE tasks failed or cells never matured, and the torchcrop
# run behind this will cover only what is here.
CELLS=$(tail -n +2 "${SOWING}" | cut -d, -f1 | sort -u | wc -l)
ROWS=$(( $(wc -l < "${SOWING}") - 1 ))
echo ""
echo "Sowing dates : ${SOWING}"
echo "  ${ROWS} cell-seasons over ${CELLS} cells"
echo "torchcrop will sow on these dates; cells absent from the file are skipped."
