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

## EGGROLL Throughput Runs

- `smolvla_eggroll_smoke_244823`
  - Job: `244823`
  - Script: `scripts/slurm/smolvla_metaworld_eggroll_smoke_a30.slurm`
  - Runner: `scripts/run_smolvla_metaworld_eggroll.py`
  - GPU request: `1x A30`, partition `a30`
  - Target: `vlm_with_expert.lm_expert.layers.1.self_attn.v_proj`
  - Result: target probe, serial/batched equivalence, and one tiny EGGROLL update passed.

- `smolvla_eggroll_campaign_244828`
  - Job: `244828`
  - Script: `scripts/slurm/smolvla_metaworld_eggroll_overnight_a30x2.slurm`
  - Controller: `scripts/run_smolvla_eggroll_campaign.py`
  - GPU request: `2x A30`, `32` CPU cap, `128G` RAM
  - Task: `push-v3`, `max_episode_steps=120`, `chunk_len=5`, `envs_per_member=1`
  - Population sweep: `4,16,32,48,64,80,96`
  - Best throughput: population `96`, `1.9387s/member-episode`, peak VRAM `3.691GB`
  - Pi0.5 baseline: `10.18s/member-episode`; SmolVLA-EGGROLL best is about `5.25x` faster on the same amortized metric.
  - RCA notes: first campaign attempt failed because nested worker commands lost `PYTHONPATH` and exceeded gpucluster3 nested `srun` CPU limits; fixed by controller-pinned subprocess workers with explicit repo environment.

- `smolvla_eggroll_production_244837`
  - Job: `244837`
  - Script: `scripts/slurm/smolvla_metaworld_eggroll_production_a30x2.slurm`
  - Status: completed successfully
  - Config: two pinned one-GPU production seeds, population `64`, `50` updates, `push-v3`, `max_episode_steps=120`
  - Final metrics: both seeds completed `50/50` updates and saved `target_weight_final.npy`.
  - Last-update throughput: seed `5000` `1.246s/member-episode`, seed `6000` `1.336s/member-episode`.
  - Recent final-window throughput: about `1.26s/member-episode` and `1.32s/member-episode`; peak PyTorch VRAM about `2.85GB`.
  - Best observed per-update success sums: `18/64` and `20/64`.
  - RCA notes: production attempt `244832` with population `96` OOMed under two concurrent workers; attempt `244834` using nested `srun` isolated GPUs but serialized one worker. Production was relaunched with population `64`, which was proven concurrent-safe in the sweep.

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
