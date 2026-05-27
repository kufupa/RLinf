#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-fair1
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/slurm/%x_%A_%a.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

SEEDS=(5000 6000)
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${TASK_INDEX}"
WORKER_SEED="${SMOLVLA_EGGROLL_WORKER_SEED:-${SEEDS[${TASK_INDEX}]}}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
POPULATION_SIZE="${SMOLVLA_EGGROLL_PROD_POPULATION:-24}"
TOTAL_UPDATES="${SMOLVLA_EGGROLL_PROD_UPDATES:-10}"
EVAL_SEEDS_PER_MEMBER="${SMOLVLA_EGGROLL_EVAL_SEEDS_PER_MEMBER:-4}"
THREADS_PER_WORKER="${SMOLVLA_EGGROLL_THREADS_PER_WORKER:-14}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_fair_worker_${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"

echo "SMOLVLA_EGGROLL_FAIR_ARRAY_ENV job=${SLURM_JOB_ID:-local} array=${TASK_INDEX} seed=${WORKER_SEED} target=${TARGET_NAME} pop=${POPULATION_SIZE} eval_seeds=${EVAL_SEEDS_PER_MEMBER} updates=${TOTAL_UPDATES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi -L || true
nvidia-smi || true

"${PYTHON_BIN}" scripts/run_smolvla_metaworld_eggroll.py \
  --run-name "smolvla_eggroll_fair_worker_${RUN_ID}" \
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
  --reset-seed-base 2000 \
  --seed "${WORKER_SEED}"

echo "SMOLVLA_EGGROLL_FAIR_ARRAY_JOB_OK ${OUTPUT_DIR}"
