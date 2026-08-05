#!/bin/bash
# =============================================================================
# NPK / management export for the whole grid, in one job.
#
# data4simplace.tiling only covers the climate and soil stages - see
# tiling._run_tile - so the NPK/fertilizer export is not produced by the tile
# array. It does not need tiling: NPKHandler aligns coarse global rasters to the
# target grid and ManagementExporter writes a single CSV, so the whole Europe
# grid fits in memory comfortably.
#
# Skips itself when paths.npk_root is null (the pipeline would log
# "no NPK data; skipping" and write nothing).
# =============================================================================
#SBATCH --job-name=d4s_mgmt
#SBATCH --output=submit/logs/mgmt_%j.out
#SBATCH --error=submit/logs/mgmt_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

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

d4s_banner "NPK / MANAGEMENT EXPORT"
cd "${D4S_PROJECT_DIR}" || exit 1

NPK_ROOT=$(python -c "
import sys, yaml
cfg = yaml.safe_load(open('${D4S_RUN_CONFIG}'))
print((cfg.get('paths') or {}).get('npk_root') or '')
")
if [ -z "${NPK_ROOT}" ]; then
    echo "paths.npk_root is null in ${D4S_RUN_CONFIG} - nothing to export. Skipping."
    exit 0
fi
echo "npk root     : ${NPK_ROOT}"

MGMT_CONFIG="${D4S_WORK_DIR}/config_management.yaml"
python submit/tile_config.py \
    --config "${D4S_RUN_CONFIG}" \
    --flags-only "run_npk_processing,apply_agricultural_mask,export_simplace_management" \
    --out "${MGMT_CONFIG}" || exit 1

echo "mgmt config  : ${MGMT_CONFIG}"
echo ""

data4simplace --config "${MGMT_CONFIG}"
STATUS=$?

echo ""
echo "management export exited ${STATUS} at $(date -Is)"
exit ${STATUS}
