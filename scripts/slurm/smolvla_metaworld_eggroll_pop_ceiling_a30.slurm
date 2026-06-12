#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-ceil
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_pop_ceiling_${RUN_ID}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
mkdir -p "${OUTPUT_DIR}"

echo "SMOLVLA_EGGROLL_CEILING_ENV job=${RUN_ID} target=${TARGET_NAME} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi -L || true
nvidia-smi || true

"${PYTHON_BIN}" scripts/run_smolvla_eggroll_campaign.py \
  --output-dir "${OUTPUT_DIR}" \
  --target-name "${TARGET_NAME}" \
  --gpu-slots "0" \
  --population-list "128,160,192,256" \
  --max-population-soft 256 \
  --steps-per-update 120 \
  --chunk-len 5 \
  --envs-per-member 1 \
  --episodes-per-member 1 \
  --max-runtime-s 13200 \
  --cpus-per-worker 14
