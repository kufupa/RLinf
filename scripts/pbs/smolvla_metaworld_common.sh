#!/bin/bash

setup_smolvla_metaworld_env() {
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    . /etc/profile.d/modules.sh
  fi
  module purge >/dev/null 2>&1 || true
  module load tools/prod
  module load Python/3.12.3-GCCcore-13.3.0
  module load Mesa/24.1.3-GCCcore-13.3.0

  cd "/rds/general/user/aa6622/home/project/RLinf"
  mkdir -p logs/pbs logs/ray

  export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
  export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
  export PYTHONPATH="/rds/general/user/aa6622/home/project/RLinf:/rds/general/user/aa6622/home/project/src:${PYTHONPATH:-}"
  export EMBODIED_PATH="/rds/general/user/aa6622/home/project/RLinf/examples/embodiment"
  export PYTHONUNBUFFERED=1
  export HYDRA_FULL_ERROR=1
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export SMOLVLA_POLICY_DEVICE="${SMOLVLA_POLICY_DEVICE:-cuda}"
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2
  export OPENBLAS_NUM_THREADS=2
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  PYTHON_BIN="${GRPO_PYTHON:-/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "error: missing PYTHON_BIN=${PYTHON_BIN}" >&2
    exit 2
  fi
  export PYTHON_BIN

  # shellcheck source=scripts/pbs/lib/rlinf_ray_local.sh
  source scripts/pbs/lib/rlinf_ray_local.sh
  setup_rlinf_local_ray
  trap archive_rlinf_ray_logs EXIT
}
