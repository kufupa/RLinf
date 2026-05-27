#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-prod
#SBATCH --partition=a30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_production_${RUN_ID}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
POPULATION_SIZE="${SMOLVLA_EGGROLL_PROD_POPULATION:-96}"
TOTAL_UPDATES="${SMOLVLA_EGGROLL_PROD_UPDATES:-50}"
STEPS_PER_UPDATE="${SMOLVLA_EGGROLL_STEPS_PER_UPDATE:-120}"
CHUNK_LEN="${SMOLVLA_EGGROLL_CHUNK_LEN:-5}"
THREADS_PER_WORKER="${SMOLVLA_EGGROLL_THREADS_PER_WORKER:-6}"
export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"

echo "SMOLVLA_EGGROLL_PRODUCTION_ENV job=${RUN_ID} target=${TARGET_NAME} population=${POPULATION_SIZE} updates=${TOTAL_UPDATES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi || true

launch_worker() {
  local slot="$1"
  local seed="$2"
  local run_dir="${OUTPUT_DIR}/seed_${seed}"
  mkdir -p "${run_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${slot}"
    export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
    "${PYTHON_BIN}" scripts/run_smolvla_metaworld_eggroll.py \
      --run-name "smolvla_eggroll_prod_${RUN_ID}_seed_${seed}" \
      --output-dir "${run_dir}" \
      --target-name "${TARGET_NAME}" \
      --population-size "${POPULATION_SIZE}" \
      --envs-per-member 1 \
      --episodes-per-member 1 \
      --total-updates "${TOTAL_UPDATES}" \
      --steps-per-update "${STEPS_PER_UPDATE}" \
      --chunk-len "${CHUNK_LEN}" \
      --max-episode-steps 120 \
      --rank 1 \
      --sigma 1e-4 \
      --learning-rate 1e-3 \
      --seed "${seed}"
  ) >"${run_dir}/worker_stdout.log" 2>&1 &
}

launch_worker 0 5000
launch_worker 1 6000
wait

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summaries = []
ok = True
for metrics_path in sorted(output_dir.glob("seed_*/metrics.jsonl")):
    run_ok = False
    updates = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("marker") == "SMOLVLA_EGGROLL_UPDATE":
            updates.append(payload)
        elif payload.get("marker") == "SMOLVLA_EGGROLL_RUN_OK":
            run_ok = True
    summaries.append(
        {
            "run_dir": str(metrics_path.parent),
            "ok": run_ok,
            "updates": len(updates),
            "last_seconds_per_member_episode": updates[-1].get("seconds_per_member_episode") if updates else None,
            "last_score_sum": sum(member["score"] for member in updates[-1].get("member_scores", [])) if updates else None,
        }
    )
    ok = ok and run_ok
payload = {"marker": "SMOLVLA_EGGROLL_PRODUCTION_OK" if ok else "SMOLVLA_EGGROLL_PRODUCTION_FAILED", "runs": summaries}
(output_dir / "production_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(payload["marker"] + " " + json.dumps(payload, sort_keys=True), flush=True)
raise SystemExit(0 if ok else 2)
PY
