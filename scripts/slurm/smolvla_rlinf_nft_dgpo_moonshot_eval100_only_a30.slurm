#!/bin/bash
#SBATCH --job-name=dgpo-ms-ev100
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Eval-only: export RLinf DGPO moonshot ckpts → 100ep sweep (train already finished).
# Requires MOONSHOT_TAG + RLINF_DGPO_TRAIN_JOB (or TRAIN_LOG_ROOT).

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

MOONSHOT_TAG="${MOONSHOT_TAG:?set MOONSHOT_TAG}"
TRAIN_JOB="${RLINF_DGPO_TRAIN_JOB:?set RLINF_DGPO_TRAIN_JOB}"
MOONSHOT_ENV="${RLINF_ROOT}/logs/slurm/generated/rlinf_dgpo_moonshot_${MOONSHOT_TAG}_env.sh"
[[ -f "${MOONSHOT_ENV}" ]] || { echo "missing ${MOONSHOT_ENV}" >&2; exit 1; }
# shellcheck source=/dev/null
source "${MOONSHOT_ENV}"

RUN_ID="${SLURM_JOB_ID:-local}"
LOG_ROOT="${TRAIN_LOG_ROOT:-$(cat "${RLINF_ROOT}/logs/slurm/rlinf_nft_dgpo_ms_${MOONSHOT_TAG}_${TRAIN_JOB}.log_root")}"
CKPT_ROOT="${LOG_ROOT}/${MOONSHOT_EXPERIMENT_NAME}/checkpoints"
EXPORT_DIR="${LOG_ROOT}/eval100/checkpoints"
EVAL_OUT="${LOG_ROOT}/eval100/sweep"
mkdir -p "${EXPORT_DIR}" "${EVAL_OUT}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

nvidia-smi -L || true
echo "RLINF_DGPO_MOONSHOT_EVAL100_ONLY tag=${MOONSHOT_TAG} train_job=${TRAIN_JOB} ckpt_root=${CKPT_ROOT}"

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

echo "RLINF_DGPO_MOONSHOT_EVAL100_OK tag=${MOONSHOT_TAG} sweep=${EVAL_OUT}"
