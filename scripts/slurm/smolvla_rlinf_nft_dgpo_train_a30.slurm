#!/bin/bash
#SBATCH --job-name=nft-dgpo-train
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# DGPO NFT short train: 20 epochs, 32 envs, group_size=16.

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_ray_local.sh
setup_rlinf_smolvla_env
setup_rlinf_slurm_ray
require_python_bin
trap archive_rlinf_slurm_ray_logs EXIT

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export RLINF_STAGE_DIAG=1

RUN_ID="${SLURM_JOB_ID:-local}"
LOG_ROOT="${RLINF_ROOT}/logs/results/rlinf_nft_dgpo_train_${RUN_ID}"
RUN_LOG="${RLINF_ROOT}/logs/slurm/nft-dgpo-train_${RUN_ID}.driver.log"
mkdir -p "${LOG_ROOT}" "$(dirname "${RUN_LOG}")"
echo "${LOG_ROOT}" > "${RLINF_ROOT}/logs/slurm/nft_dgpo_train_${RUN_ID}.log_root"

nvidia-smi -L || true

"${PYTHON_BIN}" examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name metaworld_pushv3_nft_smolvla \
  cluster.ray.address=local \
  cluster.ray.temp_dir="${RAY_TMPDIR}" \
  cluster.ray.include_dashboard=false \
  +cluster.ray.num_cpus=12 \
  +cluster.ray.object_store_memory=6000000000 \
  runner.logger.log_path="${LOG_ROOT}" \
  runner.logger.experiment_name="smolvla_rlinf_nft_dgpo_train" \
  runner.logger.logger_backends=[] \
  runner.max_epochs=20 \
  runner.save_interval=5 \
  runner.val_check_interval=-1 \
  algorithm.dpo_beta=10 \
  env.train.total_num_envs=32 \
  env.train.max_steps_per_rollout_epoch=120 \
  env.train.max_episode_steps=120 \
  env.eval.total_num_envs=25 \
  env.eval.max_steps_per_rollout_epoch=120 \
  env.eval.max_episode_steps=120 \
  actor.micro_batch_size=16 \
  actor.global_batch_size=32 \
  actor.optim.lr=5.0e-6 2>&1 | tee "${RUN_LOG}"

grep -q "runner.run:done" "${RUN_LOG}"
echo "RLINF_NFT_DGPO_TRAIN_OK log_root=${LOG_ROOT} ckpt_root=${LOG_ROOT}/smolvla_rlinf_nft_dgpo_train/checkpoints"
