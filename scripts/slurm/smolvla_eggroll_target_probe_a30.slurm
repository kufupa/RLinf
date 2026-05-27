#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-target
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_target_probe_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

nvidia-smi || true

"${PYTHON_BIN}" scripts/inspect_smolvla_eggroll_targets.py \
  --run-name "smolvla_eggroll_target_probe_${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${SMOLVLA_CHECKPOINT}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 2 \
  --chunk-len 5 \
  --max-episode-steps 120 \
  --rank 1 \
  --scale 1e-3 \
  --max-targets 32 \
  --device cuda
