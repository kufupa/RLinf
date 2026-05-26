# SmolVLA MetaWorld PPO/GRPO Run History

This records the run identities and reproducibility handles for the SmolVLA
MetaWorld PPO/GRPO work. Raw logs/checkpoints are intentionally not committed;
use `scripts/collect_smolvla_bottleneck_artifacts.py` to collect available logs
into `logs/artifacts/smolvla_bottleneck_<UTC>/`.

## Core Runs

- `direct_ppo_4env_sparse_stage2`
  - Job: `2835191.pbs-7`
  - Script: `scripts/pbs/smolvla_metaworld_direct_ppo_stage2_sparse.pbs`
  - Runner: `scripts/run_smolvla_metaworld_direct_ppo.py`
  - GPU request: `1x RTX6000`, queue `v1_gpu72`
  - Hyperparams: `num_envs=4`, `steps_per_update=120`, `chunk_len=5`,
    `total_updates=25`, `update_epochs=2`, `minibatch_envs=4`,
    `lr=1e-6`, `value_lr=1e-4`
  - Reward: `sparse_success_delta`, relative reward enabled
  - Eval: `25` envs, updates `0,1,5,10,25`
  - Resume checkpoint produced: `checkpoints/update_000025.pt`

- `direct_ppo_4env_dense_stage2`
  - Job: `2835192.pbs-7`
  - Script: `scripts/pbs/smolvla_metaworld_direct_ppo_stage2_dense.pbs`
  - GPU request: `1x RTX6000`, queue `v1_gpu72`
  - Hyperparams: same direct PPO shape as sparse stage2
  - Reward: dense return

- `direct_ppo_8env_sparse_stage3`
  - Job: `2835766.pbs-7`
  - Script: `scripts/pbs/smolvla_metaworld_direct_ppo_stage3_sparse_scale.pbs`
  - GPU request: `1x RTX6000`, queue `v1_gpu72`
  - Hyperparams: `num_envs=8`, `steps_per_update=120`, `chunk_len=5`
  - Purpose: test scaling pressure from more envs per update

- `direct_ppo_4env_sparse_stage3b_lr3e7`
  - Job: `2836701.pbs-7`
  - Script: `scripts/pbs/smolvla_metaworld_direct_ppo_stage3b_sparse_lr3e7.pbs`
  - GPU request: `1x RTX6000`, queue `v1_gpu72`
  - Hyperparams: `num_envs=4`, `steps_per_update=120`, `chunk_len=5`,
    `total_updates=50`, `update_epochs=2`, `minibatch_envs=4`,
    `lr=3e-7`, `value_lr=1e-4`
  - Reward: `sparse_success_delta`, relative reward enabled
  - Resume checkpoint produced: `checkpoints/update_000050.pt`

## Native RLinf Baselines

- `native_rlinf_ppo_4env`
  - Job: `2842447.pbs-7`
  - Script: `scripts/pbs/native_metaworld_pushv3_learning_4env.pbs`
  - Runner: `examples/embodiment/train_embodied_agent.py`
  - Purpose: baseline through full RLinf embodied trainer stack

- `native_rlinf_ppo_8env_scaling`
  - Job: `2842187.pbs-7`
  - Script: `scripts/pbs/native_metaworld_pushv3_scaling_8env.pbs`
  - Purpose: scaling baseline through full RLinf embodied trainer stack

- `native_rlinf_ppo_12env_scaling`
  - Job: `2842188.pbs-7`
  - Script: `scripts/pbs/native_metaworld_pushv3_scaling_12env.pbs`
  - Purpose: scaling baseline through full RLinf embodied trainer stack

- `native_rlinf_grpo_4env`
  - Job: `2842501.pbs-7`
  - Script: `scripts/pbs/native_metaworld_pushv3_grpo_learning_4env.pbs`
  - Purpose: native MetaWorld GRPO learning attempt

## Bottleneck Instrumentation

- Direct PPO runner prints `DIRECT_SMOLVLA_PPO_METRIC` JSON lines.
  - Includes per-update timing totals, call counts, rollout shapes, PPO losses,
    action clipping, gradient norms, and checkpoint timing.

- Component benchmark prints `SMOLVLA_COMPONENT_BENCH` JSON lines.
  - Job: `2855898.pbs-7`
  - Script: `scripts/pbs/smolvla_metaworld_component_benchmark.pbs`
  - Runner: `scripts/benchmark_smolvla_metaworld_components.py`
  - GPU request: `1x L40S`, queue `v1_gpu72`
  - Modes: policy-only, env-only, rollout-only, PPO-update-only
  - Policy batch sizes: `1,4,8,16,32`

## Raw Artifact Collection

Run:

```bash
python scripts/collect_smolvla_bottleneck_artifacts.py
```

Expected output:

```text
logs/artifacts/smolvla_bottleneck_<UTC>/
```

That bundle copies raw logs when present, copies PBS scripts, and extracts JSONL
metric lines. Missing raw logs mean the committed run metadata remains available,
but walltime/VRAM/result values must be recovered from scheduler accounting or
the original log location.
