#!/bin/bash
# =============================================================================
# Combine the finished shards and render the spatial maps.
#
# Runs as a dependent job after the shard array. Unlike the data4simplace
# combine step this uses afterok-style checking *inside* the script rather than
# afterany at submit time: an incomplete shard set does not leave an obvious
# hole (shards are dealt round-robin, so a missing one speckles all of Europe),
# so torchcrop_maps.py refuses to map a partial run.
# =============================================================================
#SBATCH --job-name=tc_maps
#SBATCH --output=submit/logs/tc_maps_%j.out
#SBATCH --error=submit/logs/tc_maps_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

set -uo pipefail

for _d in "${TC_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/torchcrop_env.sh" ] && { TC_SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${TC_SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/torchcrop_env.sh. Set TC_PROJECT_DIR or" >&2
    echo "       sbatch from the project root so SLURM_SUBMIT_DIR resolves it." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${TC_SUBMIT_DIR}/torchcrop_env.sh"
tc_activate

tc_banner "TORCHCROP COMBINE + MAPS"
cd "${TC_PROJECT_DIR}" || exit 1

echo "--- shards present ---"
ls -1 "${TC_SHARD_DIR}"/torchcrop_shard_*.parquet 2>/dev/null | wc -l
echo ""

# Cartopy downloads its 50m coastlines on first use; give it a writable cache
# on shared storage so every node reuses one copy instead of re-fetching.
export CARTOPY_USER_BACKGROUNDS="${TC_OUT_DIR}/.cartopy"
export XDG_DATA_HOME="${TC_OUT_DIR}/.cartopy"
mkdir -p "${XDG_DATA_HOME}"

srun --cpu-bind=none python submit/torchcrop_maps.py \
    --shard-dir "${TC_SHARD_DIR}" \
    --out-dir "${TC_OUT_DIR}" \
    --n-shards "${TC_N_SHARDS}"
STATUS=$?

echo ""
if [ ${STATUS} -eq 0 ]; then
    echo "OK   maps written to ${TC_OUT_DIR}/maps at $(date -Is)"
    ls -1sh "${TC_OUT_DIR}/maps"
else
    echo "FAIL maps exited ${STATUS} at $(date -Is)" >&2
fi
exit ${STATUS}
