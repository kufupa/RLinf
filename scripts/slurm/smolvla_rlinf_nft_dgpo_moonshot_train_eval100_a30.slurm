#!/bin/bash
#SBATCH --job-name=dgpo-ms
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# DGPO moonshot: 30 updates, save@3, 100ep eval sweep on every checkpoint (same job).
# Hydra overrides via logs/slurm/generated/rlinf_dgpo_moonshot_${MOONSHOT_TAG}_env.sh

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_ray_local.sh
setup_rlinf_smolvla_env
setup_rlinf_slurm_ray
require_python_bin
trap archive_rlinf_slurm_ray_logs EXIT

MOONSHOT_TAG="${MOONSHOT_TAG:?set MOONSHOT_TAG}"
MOONSHOT_ENV="${RLINF_ROOT}/logs/slurm/generated/rlinf_dgpo_moonshot_${MOONSHOT_TAG}_env.sh"
[[ -f "${MOONSHOT_ENV}" ]] || { echo "missing ${MOONSHOT_ENV}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${MOONSHOT_ENV}"

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export RLINF_STAGE_DIAG=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${SLURM_JOB_ID:-local}"
LOG_ROOT="${RLINF_ROOT}/logs/results/rlinf_nft_dgpo_ms_${MOONSHOT_TAG}_${RUN_ID}"
RUN_LOG="${RLINF_ROOT}/logs/slurm/dgpo-ms_${MOONSHOT_TAG}_${RUN_ID}.driver.log"
EXPORT_DIR="${LOG_ROOT}/eval100/checkpoints"
EVAL_OUT="${LOG_ROOT}/eval100/sweep"
mkdir -p "${LOG_ROOT}" "${EXPORT_DIR}" "${EVAL_OUT}" "$(dirname "${RUN_LOG}")"
echo "${LOG_ROOT}" > "${RLINF_ROOT}/logs/slurm/rlinf_nft_dgpo_ms_${MOONSHOT_TAG}_${RUN_ID}.log_root"

nvidia-smi -L || true
echo "RLINF_DGPO_MOONSHOT tag=${MOONSHOT_TAG} job=${RUN_ID} log_root=${LOG_ROOT}"
echo "  epochs=30 save_interval=3 eval_updates=${MOONSHOT_EVAL_UPDATES}"

# --- Phase 1: train ---
# shellcheck disable=SC2086
"${PYTHON_BIN}" examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name metaworld_pushv3_nft_smolvla \
  cluster.ray.address=local \
  cluster.ray.temp_dir="${RAY_TMPDIR}" \
  cluster.ray.include_dashboard=false \
  +cluster.ray.num_cpus=16 \
  +cluster.ray.object_store_memory=6000000000 \
  runner.logger.log_path="${LOG_ROOT}" \
  runner.logger.experiment_name="${MOONSHOT_EXPERIMENT_NAME}" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=30 \
  runner.save_interval=3 \
  runner.val_check_interval=-1 \
  ${MOONSHOT_HYDRA_OVERRIDES} 2>&1 | tee "${RUN_LOG}"

grep -q "runner.run:done" "${RUN_LOG}"
echo "RLINF_DGPO_MOONSHOT_TRAIN_OK tag=${MOONSHOT_TAG} log_root=${LOG_ROOT}"

# --- Phase 2: export + 100ep sweep ---
CKPT_ROOT="${LOG_ROOT}/${MOONSHOT_EXPERIMENT_NAME}/checkpoints"
echo "=== DGPO moonshot eval100 tag=${MOONSHOT_TAG} ckpt_root=${CKPT_ROOT} ==="

"${PYTHON_BIN}" -u scripts/export_rlinf_ckpt_to_direct_ppo.py \
  --rlinf-ckpt-root "${CKPT_ROOT}" \
  --model-path "${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}" \
  --updates "${MOONSHOT_EVAL_UPDATES}" \
  --output-dir "${EXPORT_DIR}"

"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "rlinf_nft_dgpo_ms_${MOONSHOT_TAG}_eval100_${RUN_ID}" \
  --checkpoint-dir "${EXPORT_DIR}" \
  --output-dir "${EVAL_OUT}" \
  --model-path "${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 25 \
  --num-episodes 100 \
  --seed-base 1000 \
  --max-episode-steps 150 \
  --chunk-len 5 \
  --include-baseline \
  --only-updates "${MOONSHOT_EVAL_UPDATES}"

echo "RLINF_DGPO_MOONSHOT_OK tag=${MOONSHOT_TAG} sweep=${EVAL_OUT} exported=${EXPORT_DIR}"
