#!/bin/bash
#SBATCH --job-name=rlinf-smolvla-ray
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

"${PYTHON_BIN}" -m pip install "ray[default]>=2.47.0"
"${PYTHON_BIN}" - <<'PY'
import ray
print("RLINF_SMOLVLA_RAY_OK", ray.__version__)
PY
