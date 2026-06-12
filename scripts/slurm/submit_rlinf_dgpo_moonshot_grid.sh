#!/bin/bash
# 8 DGPO moonshots @ 30 updates on A30 — all parallel (QoS cap 8).
# Each job: train 30ep, save@3, 100ep eval on ckpts 3,6,9,12,15,18,21,24,27,30.
# No sbatch --export (gpucluster3).

set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
cd "${RLINF_ROOT}"
mkdir -p logs/slurm/generated logs/slurm logs/results

source "${RLINF_ROOT}/scripts/slurm/eggroll_sbatch_submit.sh"

GEN="${RLINF_ROOT}/logs/slurm/generated"
TRAIN_EVAL_SLURM="${RLINF_ROOT}/scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_train_eval100_a30.slurm"
EVAL_UPDATES="3,6,9,12,15,18,21,24,27,30"

write_moonshot_env() {
  local tag="$1"
  local exp_name="$2"
  local hydra="$3"
  cat > "${GEN}/rlinf_dgpo_moonshot_${tag}_env.sh" <<EOF
export MOONSHOT_TAG="${tag}"
export MOONSHOT_EXPERIMENT_NAME="${exp_name}"
export MOONSHOT_EVAL_UPDATES="${EVAL_UPDATES}"
export MOONSHOT_HYDRA_OVERRIDES="${hydra}"
EOF
}

# Shared knobs — moonshot-specific overrides must come AFTER this (Hydra last-wins).
COMMON="env.train.max_steps_per_rollout_epoch=120 env.train.max_episode_steps=120 env.eval.max_steps_per_rollout_epoch=150 env.eval.max_episode_steps=150 env.eval.total_num_envs=25 env.train.total_num_envs=32 actor.micro_batch_size=16 actor.global_batch_size=32 actor.optim.lr=5.0e-6 actor.optim.value_lr=1.0e-4 algorithm.nft_tau=1.0"

# M1: nuclear beta — 20× preference temperature, force winner/loser separation in group sigmoid.
write_moonshot_env "m1_nuclear_beta" "smolvla_rlinf_dgpo_ms_m1_nuclear_beta" \
  "${COMMON} algorithm.dpo_beta=200"

# M2: giant group — 32-way group comparisons (DGPO group sum over bigger cohort).
write_moonshot_env "m2_giant_group" "smolvla_rlinf_dgpo_ms_m2_giant_group" \
  "${COMMON} algorithm.group_size=32 algorithm.dpo_beta=15 env.train.total_num_envs=32 actor.micro_batch_size=32 actor.global_batch_size=32 actor.nft_forward_micro_batch_size=16 actor.enable_offload=True rollout.enable_offload=True"

# M3: EMA chase — moving reference (tau=0.85) instead of frozen ref; ref tracks policy slowly.
write_moonshot_env "m3_ema_chase" "smolvla_rlinf_dgpo_ms_m3_ema_chase" \
  "${COMMON} algorithm.nft_tau=0.85 algorithm.dpo_beta=20"

# M4: raw signal — skip GRPO clip/rescale; direct sparse success as signed advantage.
write_moonshot_env "m4_raw_signal" "smolvla_rlinf_dgpo_ms_m4_raw_signal" \
  "${COMMON} algorithm.adv_type=raw algorithm.normalize_advantages=False algorithm.dpo_beta=30"

# M5: throughput beast — 2× rollout + 4× update epochs (VRAM-safe: 32env/mbs16 not 64/32).
write_moonshot_env "m5_throughput" "smolvla_rlinf_dgpo_ms_m5_throughput" \
  "${COMMON} algorithm.rollout_epoch=2 algorithm.update_epoch=4 algorithm.dpo_beta=15 actor.enable_offload=True rollout.enable_offload=True"

# M6: LR missile — 10× policy LR + 2.5× beta; aggressive weight movement.
write_moonshot_env "m6_lr_missile" "smolvla_rlinf_dgpo_ms_m6_lr_missile" \
  "${COMMON} algorithm.dpo_beta=25 actor.optim.lr=5.0e-5 actor.optim.value_lr=5.0e-4"

# M7: chaos open — no reward filtering, wide adv clip; train on full reward spectrum.
write_moonshot_env "m7_chaos_open" "smolvla_rlinf_dgpo_ms_m7_chaos_open" \
  "${COMMON} algorithm.filter_rewards=False algorithm.adv_clip_max=1.5 algorithm.dpo_beta=75"

# M8: Flow-SDE hybrid — stochastic rollouts (Flow-SDE hit 41% on push-v3) + DGPO loss.
write_moonshot_env "m8_flowsde_hybrid" "smolvla_rlinf_dgpo_ms_m8_flowsde_hybrid" \
  "${COMMON} algorithm.dpo_beta=15 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=0.25 actor.model.noise_level=0.25"

submit_moonshot() {
  local tag="$1"
  local wrap="${GEN}/rlinf_dgpo_moonshot_${tag}.slurm"
  cat > "${wrap}" <<EOF
#!/bin/bash
#SBATCH --job-name=dgpo-ms-${tag}
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
export MOONSHOT_TAG="${tag}"
source "${GEN}/rlinf_dgpo_moonshot_${tag}_env.sh"
exec bash "${TRAIN_EVAL_SLURM}"
EOF
  chmod +x "${wrap}"
  eggroll_sbatch_submit "${wrap}"
}

J_M1="$(submit_moonshot m1_nuclear_beta)"
J_M2="$(submit_moonshot m2_giant_group)"
J_M3="$(submit_moonshot m3_ema_chase)"
J_M4="$(submit_moonshot m4_raw_signal)"
J_M5="$(submit_moonshot m5_throughput)"
J_M6="$(submit_moonshot m6_lr_missile)"
J_M7="$(submit_moonshot m7_chaos_open)"
J_M8="$(submit_moonshot m8_flowsde_hybrid)"

cat <<EOF
RLINF_DGPO_MOONSHOT_GRID_OK (8 parallel @ 30ep, save@3, 100ep eval all ckpts)
  m1_nuclear_beta:   ${J_M1}  dpo_beta=200
  m2_giant_group:    ${J_M2}  group_size=32, beta=15
  m3_ema_chase:      ${J_M3}  nft_tau=0.85, beta=20
  m4_raw_signal:     ${J_M4}  adv_type=raw, beta=30
  m5_throughput:     ${J_M5}  rollout×2 update×4, 64env, beta=15
  m6_lr_missile:     ${J_M6}  lr=5e-5, beta=25
  m7_chaos_open:     ${J_M7}  filter_rewards=False, beta=75
  m8_flowsde_hybrid: ${J_M8}  flow_sde noise=0.25, beta=15

Eval ckpts: ${EVAL_UPDATES} (100ep seeds 1000-1099, baseline included)
Outputs: logs/results/rlinf_nft_dgpo_ms_<tag>_<jobid>/eval100/sweep/results.jsonl

Monitor: squeue -u \$USER -p a30
  tail -f logs/slurm/dgpo-ms-<tag>-<jobid>.out
EOF
