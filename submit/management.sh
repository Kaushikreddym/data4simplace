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
# 64G was not enough: the last run died with MaxRSS 67 GB. NPKHandler aligns the
# global 0.05 degree rasters to the target grid in one pass over the whole of
# Europe -- there is no tiling on this stage -- so the peak is set by the domain,
# not by anything tunable in the config.
#SBATCH --mem=160G
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

# The tile array exports management too (tiling._run_tile), and combine.sh has
# already mosaicked its shards into management/fertilizer_<crop>.csv. Re-running
# the untiled export on top of that does not add anything and actively degrades
# the file: the tiled schedule is written per tile with the soil and irrigation
# stages live, so it carries vIRR and the soil-restricted cell set, while this
# job runs with those stages off. That is what happened to the EU run of
# 2026-09-02 -- combine wrote a correct schedule at 13:00 and this job
# overwrote it at 13:13 with one that had no vIRR column (so every cell would
# have run rainfed) and 3 605 locations with no soil profile.
#
# So: only export here when the tile array did not. Set D4S_FORCE_MGMT=1 to
# override, e.g. to re-export management alone against an existing soil export.
MGMT_SHARDS="${D4S_OUT_DIR}/management/_shards"
if [ "${D4S_FORCE_MGMT:-0}" != "1" ] && \
   compgen -G "${MGMT_SHARDS}/tile_*.csv" >/dev/null 2>&1; then
    echo "The tile array already exported management ($(ls "${MGMT_SHARDS}"/tile_*.csv | wc -l) shards"
    echo "under ${MGMT_SHARDS}); combine.sh mosaicked them. Skipping the untiled"
    echo "re-export, which would drop vIRR and widen the cell set."
    echo "Set D4S_FORCE_MGMT=1 to export anyway."
    exit 0
fi

MGMT_CONFIG="${D4S_WORK_DIR}/config_management.yaml"
# run_irrigation_classification is kept on: mgmt_export appends vIRR from it,
# and without the flag the schedule silently loses the column -- which
# cropmodelling4eu reads as "no irrigation data, run every cell rainfed"
# (export.management.irrigation_flags), discarding the whole classification.
# run_soil_processing stays off (it is untiled here and would OOM); the
# soil-restricted cell set is recovered instead by spatial.export_cell_mask,
# which falls back to the soil export already in paths.output_dir.
python submit/tile_config.py \
    --config "${D4S_RUN_CONFIG}" \
    --flags-only "run_npk_processing,apply_agricultural_mask,export_simplace_management,run_irrigation_classification" \
    --out "${MGMT_CONFIG}" || exit 1

echo "mgmt config  : ${MGMT_CONFIG}"
echo ""

data4simplace --config "${MGMT_CONFIG}"
STATUS=$?

echo ""
echo "management export exited ${STATUS} at $(date -Is)"
exit ${STATUS}
