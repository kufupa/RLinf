#!/bin/bash
#SBATCH --job-name=nft-dgpo-ev100
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Export RLinf DGPO checkpoints → direct-PPO update_*.pt, then 100ep eval (seeds 1000-1099).

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
TRAIN_JOB="${RLINF_NFT_DGPO_TRAIN_JOB:?set RLINF_NFT_DGPO_TRAIN_JOB to training job id}"

LOG_ROOT="$(cat "${RLINF_ROOT}/logs/slurm/nft_dgpo_train_${TRAIN_JOB}.log_root")"
CKPT_ROOT="${LOG_ROOT}/smolvla_rlinf_nft_dgpo_train/checkpoints"
EXPORT_DIR="${RLINF_ROOT}/logs/results/rlinf_nft_dgpo_eval100_${RUN_ID}/checkpoints"
EVAL_OUT="${RLINF_ROOT}/logs/results/rlinf_nft_dgpo_eval100_${RUN_ID}/sweep"
mkdir -p "${EXPORT_DIR}" "${EVAL_OUT}"

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12

nvidia-smi -L || true
echo "RLINF_NFT_DGPO_EVAL100 train_job=${TRAIN_JOB} ckpt_root=${CKPT_ROOT}"

"${PYTHON_BIN}" -u scripts/export_rlinf_ckpt_to_direct_ppo.py \
  --rlinf-ckpt-root "${CKPT_ROOT}" \
  --model-path "${SMOLVLA_CHECKPOINT:-jadechoghari/smolvla_metaworld}" \
  --updates "5,10,15,20" \
  --output-dir "${EXPORT_DIR}"

"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "rlinf_nft_dgpo_eval100_${RUN_ID}" \
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
  --only-updates "5,10,15,20"

echo "RLINF_NFT_DGPO_EVAL100_OK sweep=${EVAL_OUT} exported=${EXPORT_DIR}"
