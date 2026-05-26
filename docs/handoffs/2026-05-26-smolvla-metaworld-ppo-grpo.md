# Handoff: Resume SmolVLA MetaWorld PPO/GRPO

Audience: another LLM or engineer moving this work to another Imperial GPU
cluster.

## Clone

```bash
git clone https://github.com/kufupa/RLinf.git
cd RLinf
git checkout smolvla-metaworld-ppo-grpo
```

Only this RLinf fork is needed for the SmolVLA MetaWorld PPO path. The old
external `project/src/smolvla_grpo` runtime dependency was replaced by
`rlinf/envs/metaworld/lerobot_adapter.py`.

## Environment

```bash
module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

python -m venv ~/.envs/lerobot_mw_py312
source ~/.envs/lerobot_mw_py312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/smolvla-metaworld-py312-imperial.txt
```

If installing RLinf itself fails on Python 3.12 due to package metadata, run from
source:

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## Required Variables

Set these before running PBS jobs:

```bash
export RLINF_ROOT=$PWD
export PYTHON_BIN=$HOME/.envs/lerobot_mw_py312/bin/python
export HF_HOME=/path/to/hf-cache
export SMOLVLA_CHECKPOINT=/path/to/smolvla_metaworld/snapshot
export RLINF_EPHEMERAL_BASE=/scratch/$USER/rlinf_ray_runs
```

If the checkpoint is online-accessible, `SMOLVLA_CHECKPOINT` may be omitted and
the code defaults to:

```text
jadechoghari/smolvla_metaworld
```

For offline runs:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Smoke Test

```bash
qsub scripts/pbs/smolvla_metaworld_direct_ppo_smoke.pbs
```

Expected: model loads, `push-v3` env resets, one PPO-style action/logprob/value
path executes.

## Resume PPO

Stage 2 sparse resume from update 25:

```bash
export RESUME_CKPT=/path/to/stage2/checkpoints/update_000025.pt
qsub scripts/pbs/smolvla_metaworld_direct_ppo_stage2_sparse_resume50.pbs
```

Stage 3b sparse resume from update 50:

```bash
export RESUME_CKPT=/path/to/stage3b/checkpoints/update_000050.pt
qsub scripts/pbs/smolvla_metaworld_direct_ppo_stage3b_sparse_resume50.pbs
```

Fresh direct PPO runs:

```bash
qsub scripts/pbs/smolvla_metaworld_direct_ppo_stage2_sparse.pbs
qsub scripts/pbs/smolvla_metaworld_direct_ppo_stage3b_sparse_lr3e7.pbs
```

## Evaluate Checkpoints

```bash
export SMOLVLA_EVAL_CKPT_DIR=/path/to/run/checkpoints
qsub scripts/pbs/smolvla_metaworld_eval_ckpt_sweep.pbs
```

The evaluator supports:

```bash
export SMOLVLA_EVAL_ONLY_UPDATES=0,10,25,50
export SMOLVLA_EVAL_MAX_CKPTS=0
export SMOLVLA_EVAL_CKPT_STRIDE=1
```

## Bottleneck Benchmark

```bash
qsub scripts/pbs/smolvla_metaworld_component_benchmark.pbs
```

This prints `SMOLVLA_COMPONENT_BENCH` JSON for:

- policy-only inference at batch sizes `1,4,8,16,32`
- env-only stepping
- rollout-only model + env
- PPO-update-only cached rollout update

Direct PPO training prints `DIRECT_SMOLVLA_PPO_METRIC` JSON per update with
timing, call counts, shapes, and PPO metrics.

## Outputs

PBS jobs write:

```text
logs/pbs/
logs/ray/
$RLINF_EPHEMERAL_BASE/<job_id>/results/
```

Direct PPO run dirs contain:

```text
metrics.jsonl
checkpoints/update_*.pt
checkpoints/latest.pt
eval/
```

Collect available logs/artifacts:

```bash
python scripts/collect_smolvla_bottleneck_artifacts.py
```

Docs to read:

- `docs/env/smolvla-metaworld-env-build.md`
- `docs/experiments/2026-05-26-smolvla-metaworld-ppo-grpo-run-history.md`
- `docs/experiments/smolvla_metaworld_ppo_grpo_run_history.json`

## Watchouts

- Imperial PBS resource lines are site-specific. Adjust queue, `gpu_type`, and
  `Qlist` to match the destination cluster.
- The Python 3.12 venv needs the matching Python module loaded first.
- Resume scripts now require `RESUME_CKPT`; they do not contain local ephemeral
  checkpoint paths.
- Raw checkpoints/logs are not committed. Move them separately if training must
  resume from existing deltas.
