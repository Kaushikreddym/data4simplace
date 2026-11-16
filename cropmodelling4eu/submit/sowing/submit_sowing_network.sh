#!/bin/bash
# =============================================================================
# The learned-sowing pipeline as one SLURM job: PEP725 split -> season cache ->
# PEP725 tsumem -> hazard network -> sowing tables.
#
#   ./submit/sowing/submit_sowing_network.sh                 # every step
#   ./submit/sowing/submit_sowing_network.sh train predict   # some steps
#   SOWING_CONFIG=my.yaml ./submit/sowing/submit_sowing_network.sh
#
# Writes to the SowingConfig out_dir (default <output_dir>/sowing_network).
# One node: the cache build reads ~68k gzip files on SOWING_WORKERS processes,
# training runs on the CPU threads of sowing.train.threads.
# =============================================================================
set -uo pipefail

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS=("$@")
[ "${#STEPS[@]}" -gt 0 ] || STEPS=(split build-cache fit-tsumem train predict)
PARTITION="${SOWING_PARTITION:-compute}"
CPUS="${SOWING_CPUS:-64}"
MEM="${SOWING_MEM:-90G}"
LOGS="${SUBMIT_DIR}/logs"
mkdir -p "${LOGS}"

CONFIG_ARG=""
[ -n "${SOWING_CONFIG:-}" ] && CONFIG_ARG="--sowing-config ${SOWING_CONFIG}"

JOB=$(sbatch --parsable --job-name=sowing_net --partition="${PARTITION}" \
    --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${SOWING_TIME:-24:00:00}" \
    --output="${LOGS}/sowing_net_%j.out" ${SOWING_DEPENDENCY:+--dependency=${SOWING_DEPENDENCY}} <<EOF
#!/bin/bash
set -eo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${SOWING_CONDA_ENV:-sdba}"
export OMP_NUM_THREADS=${CPUS} MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for step in ${STEPS[*]}; do
    echo "=== \$(date -Is) \${step}"
    cm4eu sowing "\${step}" ${CONFIG_ARG} --workers ${CPUS}
done
echo "=== \$(date -Is) done"
EOF
) || { echo "ERROR: sbatch failed" >&2; exit 1; }
echo "sowing pipeline job ${JOB}: ${STEPS[*]} (log ${LOGS}/sowing_net_${JOB}.out)"
