#!/bin/bash
#SBATCH --job-name=smolvla-eggroll-fair
#SBATCH --partition=a30
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail

source /vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo/scripts/slurm/rlinf_smolvla_common.sh
setup_rlinf_smolvla_env
require_python_bin

RUN_ID="${SLURM_JOB_ID:-local}"
OUTPUT_DIR="${RLINF_ROOT}/logs/results/smolvla_eggroll_fair_production_${RUN_ID}"
TARGET_NAME="${SMOLVLA_EGGROLL_TARGET_NAME:-vlm_with_expert.lm_expert.layers.1.self_attn.v_proj}"
POPULATION_SIZE="${SMOLVLA_EGGROLL_PROD_POPULATION:-24}"
TOTAL_UPDATES="${SMOLVLA_EGGROLL_PROD_UPDATES:-10}"
EVAL_SEEDS_PER_MEMBER="${SMOLVLA_EGGROLL_EVAL_SEEDS_PER_MEMBER:-4}"
THREADS_PER_WORKER="${SMOLVLA_EGGROLL_THREADS_PER_WORKER:-6}"
export OUTPUT_DIR
mkdir -p "${OUTPUT_DIR}"

echo "SMOLVLA_EGGROLL_FAIR_ENV job=${RUN_ID} target=${TARGET_NAME} population=${POPULATION_SIZE} eval_seeds=${EVAL_SEEDS_PER_MEMBER} updates=${TOTAL_UPDATES} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
nvidia-smi -L || true
nvidia-smi topo -m || true
nvidia-smi || true

launch_worker() {
  local slot="$1"
  local seed="$2"
  local run_dir="${OUTPUT_DIR}/seed_${seed}"
  mkdir -p "${run_dir}"
  (
    export WORKER_SLOT="${slot}"
    export WORKER_SEED="${seed}"
    export WORKER_RUN_DIR="${run_dir}"
    export WORKER_RUN_NAME="smolvla_eggroll_fair_${RUN_ID}_seed_${seed}"
    export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
    export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${THREADS_PER_WORKER}" \
      /bin/bash -lc '
        set -euo pipefail
        cd "${RLINF_ROOT}"
        export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
        export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
        export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
        echo "SMOLVLA_EGGROLL_FAIR_WORKER slot=${WORKER_SLOT} seed=${WORKER_SEED} pid=$$ CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
        nvidia-smi || true
        "${PYTHON_BIN}" scripts/run_smolvla_metaworld_eggroll.py \
          --run-name "${WORKER_RUN_NAME}" \
          --output-dir "${WORKER_RUN_DIR}" \
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
      '
  ) >"${run_dir}/worker_stdout.log" 2>&1 &
}

launch_worker 0 5000
pid_a=$!
launch_worker 1 6000
pid_b=$!

status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summaries = []
for metrics_path in sorted(output_dir.glob("seed_*/metrics.jsonl")):
    run_ok = False
    updates = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload.get("marker") == "SMOLVLA_EGGROLL_UPDATE":
            updates.append(payload)
        elif payload.get("marker") == "SMOLVLA_EGGROLL_RUN_OK":
            run_ok = True
    last = updates[-1] if updates else {}
    summaries.append(
        {
            "run_dir": str(metrics_path.parent),
            "ok": run_ok,
            "updates": len(updates),
            "last_seconds_per_member_episode": last.get("seconds_per_member_episode"),
            "last_score_sum": sum(member["score"] for member in last.get("member_scores", [])),
            "fair_seed_layout": last.get("fair_seed_layout"),
            "resources": last.get("resources"),
        }
    )
ok = bool(summaries) and all(item["ok"] for item in summaries)
payload = {
    "marker": "SMOLVLA_EGGROLL_FAIR_PRODUCTION_OK" if ok else "SMOLVLA_EGGROLL_FAIR_PRODUCTION_FAILED",
    "runs": summaries,
}
(output_dir / "fair_production_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(payload["marker"] + " " + json.dumps(payload, sort_keys=True), flush=True)
raise SystemExit(0 if ok else 2)
PY

exit "${status}"
