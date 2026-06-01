#!/usr/bin/env bash
# Tiered RLinf fast eval: 25ep all checkpoints + 100ep baseline/first/last. seed_base=1000.
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo}"
CKPT_DIR="${1:?checkpoint dir}"
OUT_DIR="${2:?output dir}"
FIRST_UPDATE="${3:-2}"
LAST_UPDATE="${4:-20}"
MODEL_PATH="${5:-/vol/bitbucket/aa6622/.cache/huggingface/hub/models--jadechoghari--smolvla_metaworld/snapshots/ef3089ecb84eeeb7d33fedab24f6c76180a68900}"

cd "${RLINF_ROOT}"
source "${RLINF_ROOT}/scripts/slurm/rlinf_smolvla_common.sh"
setup_rlinf_smolvla_env
require_python_bin

mkdir -p "${OUT_DIR}"

COMMON=(
  --checkpoint-dir "${CKPT_DIR}"
  --model-path "${MODEL_PATH}"
  --task-name push-v3
  --task-description "Push the puck to a goal"
  --num-envs 25
  --seed-base 1000
  --max-episode-steps 150
  --chunk-len 5
)

echo "[phase46-tiered] pass A: 25ep all checkpoints"
"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "phase46_25ep" \
  --output-dir "${OUT_DIR}/eval_25ep" \
  --num-episodes 25 \
  --include-baseline \
  "${COMMON[@]}"

echo "[phase46-tiered] pass B: 100ep milestones first=${FIRST_UPDATE} last=${LAST_UPDATE}"
"${PYTHON_BIN}" -u scripts/eval_smolvla_metaworld_ckpt_sweep.py \
  --run-name "phase46_100ep" \
  --output-dir "${OUT_DIR}/eval_100ep" \
  --num-episodes 100 \
  --include-baseline \
  --only-updates "${FIRST_UPDATE},${LAST_UPDATE}" \
  "${COMMON[@]}"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

def load_rows(p: Path) -> list[dict]:
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

rows_25 = load_rows(out / "eval_25ep" / "results.jsonl")
rows_100 = load_rows(out / "eval_100ep" / "results.jsonl")
for row in rows_25 + rows_100:
    row["pc_success"] = float(row.get("success_rate", 0.0)) * 100.0

summary = {
    "protocol": {
        "backend": "rlinf_fast",
        "seed_base": 1000,
        "max_episode_steps": 150,
        "chunk_len": 5,
        "num_envs": 25,
    },
    "rows_25ep": rows_25,
    "rows_100ep": rows_100,
}
(out / "tiered_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("PHASE46_TIERED_EVAL_RLINF_OK", f"25ep={len(rows_25)}", f"100ep={len(rows_100)}")
PY

echo "PHASE46_TIERED_EVAL_RLINF_OK out=${OUT_DIR}"
