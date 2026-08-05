#!/bin/bash
# =============================================================================
# Progress of a torchcrop Europe run: which shards are done, which are queued,
# which failed.
#
#   ./submit/torchcrop_status.sh
#   TC_RUN_NAME=other ./submit/torchcrop_status.sh
# =============================================================================

set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/torchcrop_env.sh"

echo "=================================================="
echo "TORCHCROP RUN STATUS: ${TC_RUN_NAME}"
echo "=================================================="
echo "  shard dir : ${TC_SHARD_DIR}"
echo ""

DONE=() MISSING=()
for ((s = 0; s < TC_N_SHARDS; s++)); do
    f=$(printf '%s/torchcrop_shard_%03d.parquet' "${TC_SHARD_DIR}" "${s}")
    if [ -f "${f}" ]; then
        DONE+=("${s}")
    else
        MISSING+=("${s}")
    fi
done

printf "  done      : %d / %d\n" "${#DONE[@]}" "${TC_N_SHARDS}"
if [ "${#MISSING[@]}" -gt 0 ]; then
    printf "  missing   : %s\n" "$(IFS=,; echo "${MISSING[*]}")"
else
    echo "  missing   : none"
fi
echo ""

if [ "${#DONE[@]}" -gt 0 ]; then
    echo "--- shard sizes ---"
    ls -1sh "${TC_SHARD_DIR}"/torchcrop_shard_*.parquet 2>/dev/null
    echo ""
fi

echo "--- queue ---"
squeue -u "${USER}" -o "%.10i %.12j %.8T %.10M %.6D %R" | grep -E "tc_shard|tc_maps|JOBID" \
    || echo "  no torchcrop jobs queued or running"
echo ""

echo "--- recent failures in ${TC_LOG_DIR} ---"
grep -l "^FAIL" "${TC_LOG_DIR}"/tc_shard_*.out 2>/dev/null | tail -10 \
    || echo "  none"

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo ""
    echo "Re-submit the missing shards with:"
    echo "  ./submit/submit_torchcrop.sh --retry"
fi
