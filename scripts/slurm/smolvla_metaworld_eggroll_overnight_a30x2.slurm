#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-night
#SBATCH --partition=a30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=07:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_campaign_${RUN_ID}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
mkdir -p "${OUTPUT_DIR}"

echo "SMOLVLA_EGGROLL_CAMPAIGN_ENV job=${RUN_ID} target=${TARGET_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi || true

"${PYTHON_BIN}" scripts/run_smolvla_eggroll_campaign.py \
  --output-dir "${OUTPUT_DIR}" \
  --target-name "${TARGET_NAME}" \
  --gpu-slots "0,1" \
  --max-population-soft 96 \
  --steps-per-update 120 \
  --chunk-len 5 \
  --envs-per-member 1 \
  --episodes-per-member 1 \
  --max-runtime-s 21600 \
  --cpus-per-worker 6
