#!/bin/bash
# =============================================================================
# Progress of the tiled Europe run: tile markers, queue state and failed logs.
#   ./submit/status.sh
# =============================================================================

set -uo pipefail

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
d4s_activate
cd "${D4S_PROJECT_DIR}" || exit 1

if [ ! -f "${D4S_RUN_CONFIG}" ]; then
    echo "No run config at ${D4S_RUN_CONFIG} - has ./submit/submit_europe.sh been run?"
    exit 1
fi

echo "=================================================="
echo "TILE PROGRESS"
echo "=================================================="
python submit/tile_status.py --config "${D4S_RUN_CONFIG}" --tile-deg "${D4S_TILE_DEG}"

echo ""
echo "=================================================="
echo "QUEUE"
echo "=================================================="
squeue -u "$USER" -o "%.18i %.12j %.8T %.10M %.6D %R" || true

echo ""
echo "=================================================="
echo "TASKS THAT EXITED NON-ZERO (most recent array)"
echo "=================================================="
LATEST=$(ls -t "${D4S_LOG_DIR}"/tile_*.err 2>/dev/null | head -n 1)
if [ -z "${LATEST}" ]; then
    echo "no tile logs yet under ${D4S_LOG_DIR}"
else
    JOB_ID=$(basename "${LATEST}" | cut -d_ -f2)
    FOUND=0
    for f in "${D4S_LOG_DIR}"/tile_"${JOB_ID}"_*.err; do
        [ -s "$f" ] || continue
        if grep -q "^FAIL tile" "$f"; then
            echo "--- $f"
            grep "^FAIL tile" "$f"
            tail -n 5 "$f" | sed 's/^/    /'
            FOUND=1
        fi
    done
    [ "${FOUND}" -eq 0 ] && echo "none for array ${JOB_ID}"
fi

echo ""
echo "Retry the unfinished tiles with: ./submit/submit_europe.sh --retry"
