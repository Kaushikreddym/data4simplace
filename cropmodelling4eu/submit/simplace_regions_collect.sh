#!/bin/bash
# =============================================================================
# Collect every per-region SIMPLACE build, then merge them into one parquet.
#
#   SP_CONFIG=<config> CM_REGIONS_ROOT=<simplace_regions> CM_MERGED=<parquet> \
#       sbatch simplace_regions_collect.sh
#
# A region that fails to collect is named and skipped rather than aborting the
# merge, so one bad task range never costs the other 26 regions their output.
# Submitted by submit_euval_calibrated.sh after the SIMPLACE array.
# =============================================================================
#SBATCH --job-name=sp_regions_collect
set -uo pipefail

: "${SP_CONFIG:?}" "${CM_REGIONS_ROOT:?}" "${CM_MERGED:?}"
SUBMIT_DIR="${CM_SUBMIT_DIR:?}"
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/simplace_env.sh"
sp_activate
cd "${SP_PROJECT_DIR}" || exit 1

FAILED=()
for ROOT in "${CM_REGIONS_ROOT}"/region_*/; do
    ROOT="${ROOT%/}"
    cm4eu simplace collect --config "${SP_CONFIG}" --out-dir "${ROOT}" \
        || FAILED+=("$(basename "${ROOT}")")
done

mkdir -p "$(dirname "${CM_MERGED}")"
python - "${CM_REGIONS_ROOT}" "${CM_MERGED}" <<'EOF'
import sys
from pathlib import Path

import pandas as pd

root, merged = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(root.glob("region_*/simplace_europe.parquet"))
table = pd.concat(
    [pd.read_parquet(f).assign(region=f.parent.name) for f in files], ignore_index=True
)
table.to_parquet(merged, index=False)
print(f"merged {len(files)} regions, {len(table):,} cell-seasons, "
      f"{table['SimplaceID'].nunique():,} cells -> {merged}")
EOF
[ "${#FAILED[@]}" -eq 0 ] || { echo "collect FAILED for: ${FAILED[*]}" >&2; exit 1; }
