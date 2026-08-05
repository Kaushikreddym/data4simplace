#!/bin/bash
# =============================================================================
# Mosaic the finished tiles into the final SIMPLACE inputs.
#
# Runs as a dependent job after the tile array (afterany, not afterok: one bad
# tile must not block the mosaic of the other 111). Weather files are already
# per-cell and globally named, so only the soil shards need concatenating.
# `--combine-only --tile-deg` also checks completeness and logs any tile whose
# .done marker is absent, so an incomplete soil.csv is never silent.
# =============================================================================
#SBATCH --job-name=d4s_combine
#SBATCH --output=submit/logs/combine_%j.out
#SBATCH --error=submit/logs/combine_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -uo pipefail

# SLURM runs a *copy* of this script from the node's spool dir
# (/var/spool/slurmd/job<id>/slurm_script), so BASH_SOURCE points there and
# cannot find env.sh next to it. Resolve the real submit dir instead.
for _d in "${D4S_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/env.sh" ] && { D4S_SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${D4S_SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/env.sh. Set D4S_PROJECT_DIR or sbatch" >&2
    echo "       from the project root so SLURM_SUBMIT_DIR resolves it." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${D4S_SUBMIT_DIR}/env.sh"
d4s_activate

d4s_banner "COMBINE TILES"
cd "${D4S_PROJECT_DIR}" || exit 1

echo "--- tile status before combining ---"
python submit/tile_status.py --config "${D4S_RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}"
echo ""

data4simplace \
    --config "${D4S_RUN_CONFIG}" \
    --tile-deg "${D4S_TILE_DEG}" \
    --combine-only
STATUS=$?

echo ""
echo "--- final state ---"
python submit/tile_status.py --config "${D4S_RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}"
echo ""
echo "combine exited ${STATUS} at $(date -Is)"
exit ${STATUS}
