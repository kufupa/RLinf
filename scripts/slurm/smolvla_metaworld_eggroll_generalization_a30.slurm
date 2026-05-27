#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-b-pilot
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Phase B pilot: per-update seed rotation (default --reset-seed-mode per_update).
# Train seeds: 2000-2039 (10 updates x 4 eval seeds). Held-out band 3000+ reserved for eval script.

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

cd "${RLINF_ROOT}"

RUN_ID="${SLURM_JOB_ID:-local}"
WORKER_SEED="${SMOLVLA_EGGROLL_WORKER_SEED:-7000}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
POPULATION_SIZE="${SMOLVLA_EGGROLL_PROD_POPULATION:-24}"
TOTAL_UPDATES="${SMOLVLA_EGGROLL_PILOT_UPDATES:-10}"
EVAL_SEEDS_PER_MEMBER="${SMOLVLA_EGGROLL_EVAL_SEEDS_PER_MEMBER:-4}"
THREADS_PER_WORKER="${SMOLVLA_EGGROLL_THREADS_PER_WORKER:-14}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_generalization_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}" logs/slurm

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"

RESET_SEED_BASE=2000
TRAIN_SEED_END=$((RESET_SEED_BASE + TOTAL_UPDATES * EVAL_SEEDS_PER_MEMBER - 1))
echo "SMOLVLA_EGGROLL_GENERALIZATION_ENV job=${RUN_ID} seed=${WORKER_SEED} target=${TARGET_NAME} pop=${POPULATION_SIZE} eval_seeds=${EVAL_SEEDS_PER_MEMBER} updates=${TOTAL_UPDATES} reset_seed_mode=per_update train_seeds=${RESET_SEED_BASE}-${TRAIN_SEED_END}"
nvidia-smi -L || true
nvidia-smi || true

"${PYTHON_BIN}" scripts/run_smolvla_metaworld_eggroll.py \
  --run-name "smolvla_eggroll_generalization_${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --target-name "${TARGET_NAME}" \
  --population-size "${POPULATION_SIZE}" \
  --envs-per-member "${EVAL_SEEDS_PER_MEMBER}" \
  --episodes-per-member 1 \
  --total-updates "${TOTAL_UPDATES}" \
  --steps-per-update 120 \
  --chunk-len 5 \
  --max-episode-steps 120 \
  --rank 1 \
  --sigma 1e-4 \
  --learning-rate 1e-3 \
  --reset-seed-base "${RESET_SEED_BASE}" \
  --seed "${WORKER_SEED}"

echo "SMOLVLA_EGGROLL_GENERALIZATION_JOB_OK ${OUTPUT_DIR}"
