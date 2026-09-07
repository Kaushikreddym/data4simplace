#!/bin/bash
# =============================================================================
# Validation datasets: one array task per CLMS tile.
#
# This array runs twice per run, under D4S_VAL_STAGE:
#
#   cuts   - reads CTY + CPMCE on a stride, accumulating the UNSPLIT emergence
#            histogram per country. Cheap, and its only job is to find where the
#            autumn and spring modes separate.
#   stats  - reads all three layers at full resolution and accumulates the
#            per-class histograms, using the cuts the first pass fixed.
#
# The two cannot be one pass: a single pass would have to assume the winter /
# spring boundary before it could measure it, which is the assumption this whole
# design exists to remove.
#
# SLURM_ARRAY_TASK_ID *is* the tile index, exactly as it is the shard index in
# cropmodelling4eu/submit/torchcrop_array.sh.
# =============================================================================
#SBATCH --job-name=d4s_val
#SBATCH --output=submit/logs/val_%A_%a.out
#SBATCH --error=submit/logs/val_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -uo pipefail

# SLURM runs a *copy* of this script from the node's spool dir, so BASH_SOURCE
# points there and cannot find env.sh beside it. Resolve the real submit dir.
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

STAGE="${D4S_VAL_STAGE:-stats}"
TILE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

d4s_banner "VALIDATION / PHENOLOGY (${STAGE}) - tile index ${TILE_INDEX}"
cd "${D4S_PROJECT_DIR}" || exit 1

python -m data4simplace.phenology.handler \
    --config "${D4S_RUN_CONFIG}" \
    --stage "${STAGE}" \
    --tile-index "${TILE_INDEX}"
STATUS=$?

echo ""
echo "validation ${STAGE} tile ${TILE_INDEX} exited ${STATUS} at $(date -Is)"
exit ${STATUS}
