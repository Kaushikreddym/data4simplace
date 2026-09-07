#!/bin/bash
# =============================================================================
# Validation datasets: merge what the array produced, in one job.
#
#   D4S_VAL_STAGE=reduce-cuts   - sum the per-country emergence histograms and
#                                 write season_cuts.csv. Must finish before the
#                                 stats array starts, because that array reads
#                                 the cuts it fixes.
#   D4S_VAL_STAGE=reduce-stats  - sum every tile's class histograms and write
#                                 phenology_adm.parquet / phenology_grid.parquet.
#
# Summing before reducing is the point, not an optimisation: most administrative
# units straddle a tile boundary, and a median per tile then averaged is simply
# the wrong number. Adding the counts and taking one median gives what a single
# pass over the whole unit would.
#
# Skips itself when paths.clms_root is null, so it is harmless to chain
# unconditionally from submit_europe.sh.
# =============================================================================
#SBATCH --job-name=d4s_valred
#SBATCH --output=submit/logs/valred_%j.out
#SBATCH --error=submit/logs/valred_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
# The grid reduction is partitioned by year (see handler.reduce_stats), which is
# what brought this back under control after a 64 GB OOM; the headroom is for the
# widest single year.
#SBATCH --mem=128G
#SBATCH --time=04:00:00

set -uo pipefail

for _d in "${D4S_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit" \
          "$(dirname "${BASH_SOURCE[0]}")"; do
    [ -f "${_d}/env.sh" ] && { D4S_SUBMIT_DIR="${_d}"; break; }
done
if [ -z "${D4S_SUBMIT_DIR:-}" ]; then
    echo "ERROR: cannot locate submit/env.sh." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${D4S_SUBMIT_DIR}/env.sh"
d4s_activate

STAGE="${D4S_VAL_STAGE:-reduce-stats}"

d4s_banner "VALIDATION / PHENOLOGY (${STAGE})"
cd "${D4S_PROJECT_DIR}" || exit 1

CLMS_ROOT=$(python -c "
import yaml
cfg = yaml.safe_load(open('${D4S_RUN_CONFIG}'))
print((cfg.get('paths') or {}).get('clms_root') or '')
")
if [ -z "${CLMS_ROOT}" ]; then
    echo "paths.clms_root is null in ${D4S_RUN_CONFIG} - no validation data. Skipping."
    exit 0
fi
echo "clms root    : ${CLMS_ROOT}"

python -m data4simplace.phenology.handler \
    --config "${D4S_RUN_CONFIG}" \
    --stage "${STAGE}"
STATUS=$?

echo ""
echo "validation ${STAGE} exited ${STATUS} at $(date -Is)"
exit ${STATUS}
