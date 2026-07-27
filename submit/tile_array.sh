#!/bin/bash
# =============================================================================
# SLURM array job: one task = one tile of the Europe grid.
#
# Submitted by submit/submit_europe.sh, which supplies --array (sized from
# `data4simplace --count-tiles`). Do not sbatch this directly unless you also
# pass --array yourself.
#
# Each task is fully independent: it writes its own weather CSVs (globally
# named by SimplaceID), its own soil shard and its own .done marker, so tasks
# never contend for a file. Re-running the array skips finished tiles.
# =============================================================================
#SBATCH --job-name=d4s_tile
#SBATCH --output=submit/logs/tile_%A_%a.out
#SBATCH --error=submit/logs/tile_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
d4s_activate

TILE_INDEX="${SLURM_ARRAY_TASK_ID:?tile_array.sh must run as a SLURM array job}"
d4s_banner "TILE ${TILE_INDEX}"

cd "${D4S_PROJECT_DIR}" || exit 1

# Per-task config: identical to the run config except for a private SoilGrids
# WCS cache directory (the cache is keyed by coverage id, not by bbox - see
# submit/tile_config.py). Kept on disk so a retry of this tile reuses it.
TILE_CONFIG="${D4S_WORK_DIR}/configs/config_tile_${TILE_INDEX}.yaml"
python submit/tile_config.py \
    --config "${D4S_RUN_CONFIG}" \
    --tile-index "${TILE_INDEX}" \
    --cache-root "${D4S_WCS_CACHE_ROOT}" \
    --out "${TILE_CONFIG}" || exit 1

echo "tile config  : ${TILE_CONFIG}"
echo ""

srun --cpu-bind=none data4simplace \
    --config "${TILE_CONFIG}" \
    --tile-deg "${D4S_TILE_DEG}" \
    --tile-index "${TILE_INDEX}"
STATUS=$?

echo ""
if [ ${STATUS} -eq 0 ]; then
    echo "OK   tile ${TILE_INDEX} finished at $(date -Is)"
else
    # No .done marker is written on failure, so the tile stays in the retry set
    # reported by submit/status.sh.
    echo "FAIL tile ${TILE_INDEX} exited ${STATUS} at $(date -Is)" >&2
fi
exit ${STATUS}
