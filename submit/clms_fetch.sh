#!/bin/bash
# =============================================================================
# CLMS fetch: one array task per tile, all products and years.
#
# Downloads the CLMS HRL Croplands rasters the phenology (validation) stage
# reads: 664 tiles x 8 years x 3 layers = 15 936 objects, ~80.7 GB. A task owns
# one tile and pulls its 24 rasters through one S3 client.
#
# Needs a CDSE account. The keys live in a NAMED boto3 profile (default `cdse`,
# override with CDSE_S3_PROFILE) -- never in the run config, so a config can be
# shared or committed without carrying a secret. Check them before submitting:
#
#     python -m data4simplace.phenology.fetch
#
# Every object is verified against its own size and S3 ETag, and lands on a
# `.part` sibling that is renamed only once complete. So a file that exists is a
# file that finished, re-running costs nothing, and a killed array is resumed by
# simply resubmitting it -- which is the whole reason the fetch is split from the
# processing rather than done inline.
#
# SLURM_ARRAY_TASK_ID is the tile index, as in validation_array.sh.
# =============================================================================
#SBATCH --job-name=d4s_clms_fetch
#SBATCH --output=submit/logs/clms_fetch_%A_%a.out
#SBATCH --error=submit/logs/clms_fetch_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

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

TILE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"

d4s_banner "CLMS FETCH - tile index ${TILE_INDEX}"
cd "${D4S_PROJECT_DIR}" || exit 1

python -m data4simplace.phenology.handler \
    --config "${D4S_RUN_CONFIG}" \
    --stage clms-fetch \
    --tile-index "${TILE_INDEX}"
STATUS=$?

echo ""
echo "CLMS fetch tile ${TILE_INDEX} exited ${STATUS} at $(date -Is)"
exit ${STATUS}
