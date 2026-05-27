#!/bin/bash
#SBATCH --job-name=rlinf-smolvla-import
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

nvidia-smi || true
"${PYTHON_BIN}" - <<'PY'
import os
import sys

print("RLINF_SMOLVLA_IMPORT_ENV", {
    "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
    "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
    "PYTHONPATH": os.environ.get("PYTHONPATH"),
})

import gymnasium
import lerobot
import metaworld
import mujoco
import torch
import smolvla_pipeline.evaluator
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv
from rlinf.models.embodiment.smolvla.smolvla_action_model import SmolVLAForRLActionPrediction

assert hasattr(SmolVLAPolicy, "select_action_distr_params")
assert hasattr(SmolVLAPolicy, "_get_distr_params_chunk")
print("RLINF_SMOLVLA_IMPORT_OK", sys.executable, sys.version.replace("\n", " "), torch.__version__, torch.cuda.is_available())
PY
