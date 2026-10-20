#!/bin/bash
# =============================================================================
# SLURM array worker: run line N of a task table, one array index per row.
#
# Each row is "<built SIMPLACE workspace> <first line> <last line>". It lets
# many per-region builds share ONE array -- and so one %N concurrency limit --
# where each build's own submit.sh would submit an array of its own and the
# per-array limits would add up. Submitted by submit_euval_calibrated.sh.
# =============================================================================
#SBATCH --job-name=sp_tasktable
set -uo pipefail

TABLE="${SP_TASK_TABLE:?SP_TASK_TABLE must name the task table}"
INDEX="${SLURM_ARRAY_TASK_ID:?run as an array task}"
read -r ROOT FIRST LAST < <(sed -n "$((INDEX + 1))p" "${TABLE}")
[ -n "${ROOT:-}" ] || { echo "ERROR: no row ${INDEX} in ${TABLE}" >&2; exit 2; }

# run.env assigns with ${VAR:-default}, so anything inherited from the
# submitting shell's simplace_env.sh would silently replace the region's own
# settings. Each task starts from the region build alone.
while read -r var; do unset "${var}"; done < <(compgen -v SP_)

echo "task ${INDEX}: ${ROOT} lines ${FIRST}-${LAST}"
exec "${ROOT}/run_task.sh" --root "${ROOT}" "${FIRST}-${LAST}"
