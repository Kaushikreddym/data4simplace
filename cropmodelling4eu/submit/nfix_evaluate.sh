#!/bin/bash
# =============================================================================
# Collect the corrected-N SIMPLACE run and evaluate it against its baseline.
# Queued by hand behind the array with --dependency=afterany:<array job>:
#
#   sbatch --dependency=afterany:857098 submit/nfix_evaluate.sh
#
# 1. `cm4eu simplace collect` on the nfix run (tolerates failed tasks).
# 2. A paired (cell, season) table of nfix - baseline, since the two builds
#    differ only in soil_long.csv's NH4/NO3 and every delta is therefore the
#    initial mineral N.
# 3. full_run_evaluation.ipynb on both SIMPLACE runs with explicit paths. The
#    notebook's defaults point at the superseded winter_wheat_2000_2024 run,
#    which is what the earlier euval execution silently evaluated.
# =============================================================================
#SBATCH --job-name=nfix_eval
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/winter_wheat_2000_2024_euval_nfix/evaluation/nfix_eval_%j.out

set -uo pipefail

PROJECT_DIR=/data01/FDS/muduchuru/codes/GITHUB/data4simplace/cropmodelling4eu
ROOT=/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu
BASE="${ROOT}/winter_wheat_2000_2024_euval"
NFIX="${ROOT}/winter_wheat_2000_2024_euval_nfix"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/submit/simplace_env.sh"
sp_activate
cd "${PROJECT_DIR}" || exit 1
mkdir -p "${NFIX}/evaluation" "${BASE}/evaluation_fixedpaths"

"${NFIX}/simplace/submit.sh" --status
cm4eu simplace collect --config "${PROJECT_DIR}/submit/config_nfix.yaml" \
    --out-dir "${NFIX}/simplace" || { echo "ERROR: collect failed" >&2; exit 1; }

python - "${BASE}/simplace/simplace_europe.parquet" "${NFIX}/simplace/simplace_europe.parquet" \
    "${NFIX}/evaluation" <<'EOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

base_path, nfix_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
keys = ["SimplaceID", "year"]
fields = ["yield_t_ha", "biomass_g_m2", "max_lai", "anthesis_doy", "maturity_doy", "days_to_maturity"]
base = pd.read_parquet(base_path, columns=[*keys, "lat", *fields])
nfix = pd.read_parquet(nfix_path, columns=[*keys, *fields])
paired = base.merge(nfix, on=keys, suffixes=("_base", "_nfix"))
print(f"cell-seasons: base {len(base)}, nfix {len(nfix)}, paired {len(paired)}")

rows = []
for f in fields:
    d = paired[f"{f}_nfix"] - paired[f"{f}_base"]
    rows.append({
        "variable": f,
        "base_median": paired[f"{f}_base"].median(),
        "nfix_median": paired[f"{f}_nfix"].median(),
        "delta_mean": d.mean(),
        "delta_p05": d.quantile(0.05),
        "delta_median": d.median(),
        "delta_p95": d.quantile(0.95),
        "share_changed": float((d.abs() > 1e-6).mean()),
    })
summary = pd.DataFrame(rows).set_index("variable")
summary.to_csv(out / "nfix_vs_base_summary.csv", float_format="%.4f")
print(summary.round(3).to_string())

paired["lat_band"] = pd.cut(paired["lat"], np.arange(34, 74, 4))
paired["dyield"] = paired["yield_t_ha_nfix"] - paired["yield_t_ha_base"]
band = paired.groupby("lat_band", observed=True).agg(
    n=("dyield", "size"),
    base_yield=("yield_t_ha_base", "median"),
    nfix_yield=("yield_t_ha_nfix", "median"),
    dyield_median=("dyield", "median"),
    dyield_rel=("dyield", lambda d: d.sum() / paired.loc[d.index, "yield_t_ha_base"].sum()),
)
band.to_csv(out / "nfix_vs_base_by_latband.csv", float_format="%.4f")
print(band.round(3).to_string())
paired.to_parquet(out / "nfix_vs_base_paired.parquet")
EOF

run_notebook() {  # <simplace parquet> <output dir>
    mkdir -p "$2/outputs"
    SIMPLACE_PARQUET="$1" \
    SIM_PARQUET="${BASE}/torchcrop/torchcrop_europe.parquet" \
    EVALUATION_OUTPUT_DIR="$2/outputs" \
    python -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output-dir "$2" --output full_run_evaluation.executed.ipynb \
        evaluation/full_run_evaluation.ipynb
}
run_notebook "${NFIX}/simplace/simplace_europe.parquet" "${NFIX}/evaluation" || echo "ERROR: nfix notebook failed" >&2
run_notebook "${BASE}/simplace/simplace_europe.parquet" "${BASE}/evaluation_fixedpaths" || echo "ERROR: baseline notebook failed" >&2

for d in "${BASE}/evaluation_fixedpaths" "${NFIX}/evaluation"; do
    echo "== ${d}"
    cat "${d}/outputs/tables/full_run_yield_metrics_by_country.csv" 2>/dev/null | head -40
done
echo ALLDONE
