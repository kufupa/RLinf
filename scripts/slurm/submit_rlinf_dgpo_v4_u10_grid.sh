#!/bin/bash
# V4 DGPO grid: 24 moonshots @ 10 updates (save@1, eval u1–u10), packed 3-per-GPU (8 jobs).
# Valid Direct-DGPO (group-id fix). Code-fix flags: microbatch weights, filter-in-gw, adv clip.
#
# Usage: bash submit_rlinf_dgpo_v4_u10_grid.sh
# Env: MOONSHOT_MEM (default 50G), SBATCH_EXCLUDE (optional)

set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
cd "${RLINF_ROOT}"
mkdir -p logs/slurm/generated logs/slurm logs/results

source "${RLINF_ROOT}/scripts/slurm/eggroll_sbatch_submit.sh"

GEN="${RLINF_ROOT}/logs/slurm/generated"
TRIPLE_SLURM="${RLINF_ROOT}/scripts/slurm/smolvla_rlinf_nft_dgpo_triple_u10_a30.slurm"
EVAL_UPDATES="1,2,3,4,5,6,7,8,9,10"
MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"

write_moonshot_env() {
  local tag="$1"
  local exp_name="$2"
  local hydra="$3"
  cat > "${GEN}/rlinf_dgpo_moonshot_${tag}_env.sh" <<EOF
export MOONSHOT_TAG="${tag}"
export MOONSHOT_EXPERIMENT_NAME="${exp_name}"
export MOONSHOT_EVAL_UPDATES="${EVAL_UPDATES}"
export MOONSHOT_MAX_EPOCHS=10
export MOONSHOT_SAVE_INTERVAL=1
export MOONSHOT_HYDRA_OVERRIDES="${hydra}"
EOF
}

COMMON="env.train.max_steps_per_rollout_epoch=120 env.train.max_episode_steps=120 env.eval.max_steps_per_rollout_epoch=150 env.eval.max_episode_steps=150 env.eval.total_num_envs=25 env.train.total_num_envs=32 actor.micro_batch_size=16 actor.global_batch_size=32 actor.optim.lr=5.0e-6 actor.optim.value_lr=1.0e-4 algorithm.nft_tau=1.0"

# Code-fix shorthand
FIX_MB="algorithm.dgpo_weight_precompute=False"
FIX_GW="algorithm.dgpo_apply_loss_mask_to_group_weights=True"
FIX_AC="algorithm.dgpo_clip_signed_adv=True"
FIX_ALL="${FIX_MB} ${FIX_GW} ${FIX_AC}"

declare -A MOONSHOT_SPECS=(
  # --- A: code fixes + beta (bundle 1–2) ---
  [v4_baseline]="${COMMON}|smolvla_rlinf_dgpo_v4_baseline"
  [v4_fix_microbatch]="${COMMON} ${FIX_MB}|smolvla_rlinf_dgpo_v4_fix_microbatch"
  [v4_fix_filtergw]="${COMMON} ${FIX_GW}|smolvla_rlinf_dgpo_v4_fix_filtergw"
  [v4_fix_advclip]="${COMMON} ${FIX_AC}|smolvla_rlinf_dgpo_v4_fix_advclip"
  [v4_fix_all3]="${COMMON} ${FIX_ALL}|smolvla_rlinf_dgpo_v4_fix_all3"
  [v4_beta100]="${COMMON} algorithm.dpo_beta=100|smolvla_rlinf_dgpo_v4_beta100"
  # --- B: prior winners (bundle 3–4) ---
  [v4_ema_tau85]="${COMMON} algorithm.nft_tau=0.85 algorithm.dpo_beta=20|smolvla_rlinf_dgpo_v4_ema_tau85"
  [v4_no_filter]="${COMMON} algorithm.filter_rewards=False algorithm.adv_clip_max=1.5 algorithm.dpo_beta=75|smolvla_rlinf_dgpo_v4_no_filter"
  [v4_champion]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=50 algorithm.adv_clip_max=1.5|smolvla_rlinf_dgpo_v4_champion"
  [v4_fix_all3_ema]="${COMMON} ${FIX_ALL} algorithm.nft_tau=0.85 algorithm.dpo_beta=50|smolvla_rlinf_dgpo_v4_fix_all3_ema"
  [v4_fix_all3_open]="${COMMON} ${FIX_ALL} algorithm.filter_rewards=False algorithm.dpo_beta=75 algorithm.adv_clip_max=1.5|smolvla_rlinf_dgpo_v4_fix_all3_open"
  [v4_beta100_ema]="${COMMON} algorithm.nft_tau=0.85 algorithm.dpo_beta=100|smolvla_rlinf_dgpo_v4_beta100_ema"
  # --- C: exploration + stability (bundle 5–6) ---
  [v4_flowsde25]="${COMMON} algorithm.dpo_beta=15 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=0.25 actor.model.noise_level=0.25|smolvla_rlinf_dgpo_v4_flowsde25"
  [v4_flowsde50_champ]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=50 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=0.5 actor.model.noise_level=0.5|smolvla_rlinf_dgpo_v4_flowsde50_champ"
  [v4_giant_g32]="${COMMON} algorithm.group_size=32 algorithm.dpo_beta=15 actor.micro_batch_size=32 actor.global_batch_size=32 actor.nft_forward_micro_batch_size=16 actor.enable_offload=True rollout.enable_offload=True|smolvla_rlinf_dgpo_v4_giant_g32"
  [v4_roll2_upd2]="${COMMON} algorithm.rollout_epoch=2 algorithm.update_epoch=2 algorithm.dpo_beta=15|smolvla_rlinf_dgpo_v4_roll2_upd2"
  [v4_kl_tight]="${COMMON} algorithm.nft_beta=2.0 algorithm.max_drift=0.15|smolvla_rlinf_dgpo_v4_kl_tight"
  [v4_lr15x]="${COMMON} algorithm.filter_rewards=False algorithm.dpo_beta=25 actor.optim.lr=7.5e-6|smolvla_rlinf_dgpo_v4_lr15x"
  # --- D: combos + controls (bundle 7–8) ---
  [v4_tau80_chaos]="${COMMON} algorithm.nft_tau=0.80 algorithm.filter_rewards=False algorithm.dpo_beta=60 algorithm.adv_clip_max=2.0|smolvla_rlinf_dgpo_v4_tau80_chaos"
  [v4_beta200]="${COMMON} algorithm.dpo_beta=200|smolvla_rlinf_dgpo_v4_beta200"
  [v4_sde10_soft]="${COMMON} algorithm.nft_tau=0.95 algorithm.filter_rewards=False algorithm.dpo_beta=10 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=1.0 actor.model.noise_level=1.0|smolvla_rlinf_dgpo_v4_sde10_soft"
  [v4_peak_lite]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=40 algorithm.adv_clip_max=1.2 algorithm.rollout_epoch=2 algorithm.update_epoch=2|smolvla_rlinf_dgpo_v4_peak_lite"
  [v4_fix_mb_ema]="${COMMON} ${FIX_MB} algorithm.nft_tau=0.85 algorithm.dpo_beta=20|smolvla_rlinf_dgpo_v4_fix_mb_ema"
  [v4_fix_mb_open]="${COMMON} ${FIX_MB} algorithm.filter_rewards=False algorithm.dpo_beta=50 algorithm.adv_clip_max=1.5|smolvla_rlinf_dgpo_v4_fix_mb_open"
)

# 8 bundles × 3 tags = 24 moonshots
BUNDLES=(
  "b1:v4_baseline v4_fix_microbatch v4_fix_filtergw"
  "b2:v4_fix_advclip v4_fix_all3 v4_beta100"
  "b3:v4_ema_tau85 v4_no_filter v4_champion"
  "b4:v4_fix_all3_ema v4_fix_all3_open v4_beta100_ema"
  "b5:v4_flowsde25 v4_flowsde50_champ v4_giant_g32"
  "b6:v4_roll2_upd2 v4_kl_tight v4_lr15x"
  "b7:v4_tau80_chaos v4_beta200 v4_sde10_soft"
  "b8:v4_peak_lite v4_fix_mb_ema v4_fix_mb_open"
)

write_all_envs() {
  for tag in "${!MOONSHOT_SPECS[@]}"; do
    local spec="${MOONSHOT_SPECS[$tag]}"
    local hydra="${spec%%|*}"
    local exp_name="${spec##*|}"
    write_moonshot_env "${tag}" "${exp_name}" "${hydra}"
  done
}

submit_bundle() {
  local bundle_id="$1"
  local triplet="$2"
  local exclude_line=""
  if [[ -n "${SBATCH_EXCLUDE}" ]]; then
    exclude_line="#SBATCH --exclude=${SBATCH_EXCLUDE}"
  fi
  local wrap="${GEN}/rlinf_dgpo_v4_triple_${bundle_id}.slurm"
  cat > "${wrap}" <<EOF
#!/bin/bash
#SBATCH --job-name=dgpo-v4-${bundle_id}
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MOONSHOT_MEM}
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
${exclude_line}

set -euo pipefail
export MOONSHOT_TRIPLET="${triplet}"
export MOONSHOT_BUNDLE="${bundle_id}"
exec bash "${TRIPLE_SLURM}"
EOF
  chmod +x "${wrap}"
  eggroll_sbatch_submit "${wrap}"
}

write_all_envs

declare -a JIDS=()
for entry in "${BUNDLES[@]}"; do
  bundle_id="${entry%%:*}"
  triplet="${entry#*:}"
  for tag in ${triplet}; do
    [[ -n "${MOONSHOT_SPECS[$tag]+x}" ]] || { echo "unknown tag in ${bundle_id}: ${tag}" >&2; exit 1; }
  done
  jid="$(submit_bundle "${bundle_id}" "${triplet}")"
  JIDS+=("${bundle_id}=${jid}")
  echo "submitted ${bundle_id} job=${jid} tags=${triplet} mem=${MOONSHOT_MEM}"
done

cat <<EOF

RLINF_DGPO_V4_U10_GRID_OK (8 jobs × 3 moonshots = 24 configs)
  10 updates, save@1, eval100 on u1–u10 + baseline (seeds 1000–1099)
  Resume later: runner.resume_dir=<log_root>/<exp>/checkpoints/global_step_10/actor

Bundles:
EOF
for entry in "${BUNDLES[@]}"; do
  echo "  ${entry//:/  →  }"
done
echo ""
printf 'Job IDs: %s\n' "${JIDS[*]}"
echo "Results: logs/results/rlinf_nft_dgpo_ms_<tag>_<jobid>/eval100/sweep/results.jsonl"
echo "Plan: project/docs/dgpo_v4_u10_moonshot_plan.md"
