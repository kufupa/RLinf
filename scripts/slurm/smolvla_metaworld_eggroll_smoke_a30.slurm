#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-smoke
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
BASE_OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_smoke_${RUN_ID}"
TARGET_OUTPUT_DIR="${BASE_OUTPUT_DIR}/target_probe"
RUNNER_OUTPUT_DIR="${BASE_OUTPUT_DIR}/runner"
mkdir -p "${TARGET_OUTPUT_DIR}" "${RUNNER_OUTPUT_DIR}"

nvidia-smi || true

"${PYTHON_BIN}" -m pytest \
  tests/unit_tests/test_eggroll_core.py \
  tests/unit_tests/test_smolvla_eggroll_batching.py \
  tests/unit_tests/test_smolvla_eggroll_targets.py \
  tests/unit_tests/test_smolvla_eggroll_runner.py \
  -q

"${PYTHON_BIN}" scripts/inspect_smolvla_eggroll_targets.py \
  --run-name "smolvla_eggroll_smoke_target_${RUN_ID}" \
  --output-dir "${TARGET_OUTPUT_DIR}" \
  --model-path "${SMOLVLA_CHECKPOINT}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --num-envs 2 \
  --chunk-len 5 \
  --max-episode-steps 120 \
  --rank 1 \
  --scale 1e-3 \
  --max-targets 16 \
  --device cuda

TARGET_NAME="$("${PYTHON_BIN}" - "${TARGET_OUTPUT_DIR}/target_probe.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
selected = None
for line in path.read_text(encoding="utf-8").splitlines():
    payload = json.loads(line)
    if payload.get("marker") == "SMOLVLA_EGGROLL_TARGET_SELECTED":
        selected = payload["name"]
if not selected:
    raise SystemExit("missing SMOLVLA_EGGROLL_TARGET_SELECTED")
print(selected)
PY
)"
echo "SMOLVLA_EGGROLL_SMOKE_TARGET ${TARGET_NAME}"

"${PYTHON_BIN}" scripts/run_smolvla_metaworld_eggroll.py \
  --run-name "smolvla_eggroll_smoke_${RUN_ID}" \
  --output-dir "${RUNNER_OUTPUT_DIR}" \
  --model-path "${SMOLVLA_CHECKPOINT}" \
  --task-name push-v3 \
  --task-description "Push the puck to a goal" \
  --target-name "${TARGET_NAME}" \
  --population-size 2 \
  --envs-per-member 1 \
  --episodes-per-member 1 \
  --total-updates 1 \
  --steps-per-update 20 \
  --chunk-len 5 \
  --max-episode-steps 120 \
  --rank 1 \
  --sigma 1e-4 \
  --learning-rate 1e-3 \
  --seed 11 \
  --verify-batched-equivalence

"${PYTHON_BIN}" - "${RUNNER_OUTPUT_DIR}/metrics.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
markers = {json.loads(line)["marker"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
required = {"SMOLVLA_EGGROLL_EQUIV_OK", "SMOLVLA_EGGROLL_UPDATE", "SMOLVLA_EGGROLL_RUN_OK"}
missing = sorted(required - markers)
if missing:
    raise SystemExit(f"missing markers: {missing}")
print("SMOLVLA_EGGROLL_SMOKE_JOB_OK", path.parent)
PY
