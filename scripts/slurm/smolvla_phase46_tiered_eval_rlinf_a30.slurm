#!/bin/bash
#SBATCH --job-name=ph46-tiered
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/phase46_tiered_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

CKPT_DIR="${1:?grpo checkpoint dir}"
OUT_DIR="${2:?eval output dir}"
FIRST_UPDATE="${3:-2}"
LAST_UPDATE="${4:-20}"

export OMP_NUM_THREADS=14
export MKL_NUM_THREADS=14
export OPENBLAS_NUM_THREADS=14

mkdir -p "${OUT_DIR}"
nvidia-smi -L || true
echo "phase46_tiered_eval ckpt=${CKPT_DIR} out=${OUT_DIR}"

bash /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/run_phase46_tiered_eval_rlinf.sh \
  "${CKPT_DIR}" "${OUT_DIR}" "${FIRST_UPDATE}" "${LAST_UPDATE}"

echo "PHASE46_TIERED_EVAL_RLINF_OK job=${SLURM_JOB_ID:-local}"
