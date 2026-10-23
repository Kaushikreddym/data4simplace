#!/bin/bash
# =============================================================================
# Run N independent torchcrop calibrations side by side on ONE node.
#
#   ./submit/calibration/submit_calibration_ensemble.sh --dry-run   # print the plan
#   ./submit/calibration/submit_calibration_ensemble.sh             # one SLURM job, N members
#   ./submit/calibration/submit_calibration_ensemble.sh --local     # run here (this node)
#
# One calibration is a chain of Adam steps and cannot use more than about one
# core (the day loop is sequential; splitting a batch across workers measured
# slower). Independent calibrations can: 16 single-process runs on one 80-core
# node gave 9.6x the throughput of one, each run only 1.67x slower (memory
# bandwidth). So the cores go to an ensemble -- one member per seed
# (CAL_SEEDS), which varies the train/val/test split, the per-epoch draws and
# the batch order -- and the spread across members is the fit's uncertainty.
#
# Every CAL_* setting of calibration_env.sh applies to every member, unchanged.
# Ensemble settings (environment, with defaults):
#   CAL_SEEDS          "0 1 ... 15"   one member per seed
#   CAL_MEMBER_WORKERS 1              forked workers per member (yield stage)
#   CAL_ENS_DIR        ${CAL_RUN_DIR}/calibration_ensemble
#   CAL_ENS_PARTITION  highmem        ~6 GB per phenology member: 16 fit in
#   CAL_ENS_MEM        180G           highmem's 192 GB, ~12 in compute's 95 GB
#   CAL_ENS_TIME       7-00:00:00
#
# Members share ONE weather cache, built once before they start (the first
# member would otherwise race 15 others to write it). Each writes its own
# outputs under ${CAL_ENS_DIR}/member_<seed>/, with a calibrate.log there;
# members.tsv records every member's exit status when the job ends.
# =============================================================================
set -uo pipefail

# Inside the job SLURM runs a spooled copy of this file, so its own directory
# is not under submit/; CAL_PROJECT_DIR (exported) locates it.
for _d in "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
          "${CAL_PROJECT_DIR:-}/submit" "${SLURM_SUBMIT_DIR:-}/submit"; do
    [ -f "${_d}/utils/calibration_env.sh" ] && { SUBMIT_DIR="${_d}"; break; }
done
[ -n "${SUBMIT_DIR:-}" ] || { echo "ERROR: cannot locate submit/utils/calibration_env.sh" >&2; exit 1; }
# shellcheck disable=SC1091
source "${SUBMIT_DIR}/utils/calibration_env.sh"

MODE=submit
for arg in "$@"; do
    case "${arg}" in
        --dry-run) MODE=dry ;;
        --local)   MODE=local ;;
        --run)     MODE=run ;;     # the job body, as SLURM runs it
        -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

export CAL_SEEDS="${CAL_SEEDS:-$(seq -s ' ' 0 15)}"
export CAL_MEMBER_WORKERS="${CAL_MEMBER_WORKERS:-1}"
export CAL_ENS_DIR="${CAL_ENS_DIR:-${CAL_RUN_DIR}/calibration_ensemble}"
CAL_ENS_PARTITION="${CAL_ENS_PARTITION:-highmem}"
CAL_ENS_MEM="${CAL_ENS_MEM:-180G}"
CAL_ENS_TIME="${CAL_ENS_TIME:-7-00:00:00}"
N_MEMBERS=$(wc -w <<<"${CAL_SEEDS}")
CPUS=$(( N_MEMBERS * CAL_MEMBER_WORKERS ))
[ "${CPUS}" -gt 80 ] && CPUS=80

run_members() {
    local shared="${CAL_ENS_DIR}/_shared"
    mkdir -p "${shared}"
    echo "=== building the shared caches in ${shared} $(date -Is)"
    CAL_OUT_DIR="${shared}" CAL_PREPARE_ONLY=1 CAL_WORKERS=1 \
        bash "${SUBMIT_DIR}/utils/calibrate.sh" > "${shared}/prepare.log" 2>&1 \
        || { echo "ERROR: shared prepare failed; see ${shared}/prepare.log" >&2; return 1; }

    local pids=() seed out
    for seed in ${CAL_SEEDS}; do
        out="${CAL_ENS_DIR}/member_$(printf '%02d' "${seed}")"
        mkdir -p "${out}"
        for f in weather_cache cell_to_adm.parquet regions.parquet regions.settings.json; do
            [ -e "${shared}/${f}" ] && ln -sfn "${shared}/${f}" "${out}/${f}"
        done
        CAL_OUT_DIR="${out}" CAL_SEED="${seed}" CAL_WORKERS="${CAL_MEMBER_WORKERS}" \
        CAL_PREPARE_ONLY=0 OMP_NUM_THREADS="${CAL_THREADS}" \
            bash "${SUBMIT_DIR}/utils/calibrate.sh" > "${out}/calibrate.log" 2>&1 &
        pids+=("$!:${seed}")
        echo "member seed ${seed}: pid $! -> ${out}"
    done

    printf 'seed\texit\tout_dir\n' > "${CAL_ENS_DIR}/members.tsv"
    local failed=0 entry pid
    for entry in "${pids[@]}"; do
        pid="${entry%%:*}"; seed="${entry##*:}"
        wait "${pid}"; local status=$?
        [ "${status}" -ne 0 ] && failed=$((failed + 1))
        printf '%s\t%s\t%s\n' "${seed}" "${status}" \
            "${CAL_ENS_DIR}/member_$(printf '%02d' "${seed}")" >> "${CAL_ENS_DIR}/members.tsv"
    done
    echo "=== ${N_MEMBERS} members finished, ${failed} failed $(date -Is)"
    [ "${failed}" -eq 0 ]
}

if [ "${MODE}" = "run" ]; then
    cal_activate
    run_members
    exit $?
fi

# --- At the prompt: the crop, the checks, the plan --------------------------
if [ "${MODE}" != "dry" ] && [ ! -f "${CAL_CROP_FILE}" ]; then
    cal_activate
    cal_build_crop || exit 1
elif [ "${MODE}" = "dry" ]; then
    export CAL_CROP_WILL_BUILD=1
fi
cal_check_inputs || exit 1

echo "=================================================="
echo "  torchcrop calibration ensemble"
echo "=================================================="
echo "  members     : ${N_MEMBERS} (seeds ${CAL_SEEDS})"
echo "  per member  : ${CAL_MEMBER_WORKERS} worker(s), batch ${CAL_BATCH_SIZE}${CAL_STAGE_BATCH_SIZE:+ (per stage: ${CAL_STAGE_BATCH_SIZE})}"
echo "  stage(s)    : ${CAL_STAGE}   epochs: ${CAL_EPOCHS:-optim.max_epochs}"
echo "  outputs     : ${CAL_ENS_DIR}/member_<seed>/"
echo "  resources   : ${CAL_ENS_PARTITION}, ${CPUS} cpus, ${CAL_ENS_MEM}, ${CAL_ENS_TIME}"
echo "=================================================="

case "${MODE}" in
    dry)   echo "(dry run: nothing started)"; exit 0 ;;
    local) cal_activate; run_members; exit $? ;;
esac

mkdir -p "${CAL_LOG_DIR}"
JOB=$(sbatch --parsable --job-name=cm_calib_ens --partition="${CAL_ENS_PARTITION}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${CAL_ENS_MEM}" \
    --time="${CAL_ENS_TIME}" --output="${CAL_LOG_DIR}/cm_calib_ens_%j.log" \
    --export=ALL "${SUBMIT_DIR}/calibration/submit_calibration_ensemble.sh" --run) \
    || { echo "sbatch failed" >&2; exit 1; }
echo "submitted job ${JOB}"
echo "  members : ${CAL_ENS_DIR}/member_<seed>/calibrate.log"
echo "  summary : ${CAL_ENS_DIR}/members.tsv (written when the job ends)"
