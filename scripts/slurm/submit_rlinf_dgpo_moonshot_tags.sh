#!/bin/bash
# Submit one or more DGPO moonshot jobs (train 30ep + 100ep eval, same job).
# Usage: bash submit_rlinf_dgpo_moonshot_tags.sh m2_giant_group m5_throughput ...
# Env: MOONSHOT_MEM (default 50G), SBATCH_EXCLUDE (optional node list, e.g. hopper)

set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
cd "${RLINF_ROOT}"
mkdir -p logs/slurm/generated logs/slurm logs/results

source "${RLINF_ROOT}/scripts/slurm/eggroll_sbatch_submit.sh"

GEN="${RLINF_ROOT}/logs/slurm/generated"
TRAIN_EVAL_SLURM="${RLINF_ROOT}/scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_train_eval100_a30.slurm"
EVAL_UPDATES="3,6,9,12,15,18,21,24,27,30"
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
export MOONSHOT_HYDRA_OVERRIDES="${hydra}"
EOF
}

COMMON="env.train.max_steps_per_rollout_epoch=120 env.train.max_episode_steps=120 env.eval.max_steps_per_rollout_epoch=150 env.eval.max_episode_steps=150 env.eval.total_num_envs=25 env.train.total_num_envs=32 actor.micro_batch_size=16 actor.global_batch_size=32 actor.optim.lr=5.0e-6 actor.optim.value_lr=1.0e-4 algorithm.nft_tau=1.0"

declare -A MOONSHOT_SPECS=(
  [m1_nuclear_beta]="${COMMON} algorithm.dpo_beta=200|smolvla_rlinf_dgpo_ms_m1_nuclear_beta"
  [m2_giant_group]="${COMMON} algorithm.group_size=32 algorithm.dpo_beta=15 env.train.total_num_envs=32 actor.micro_batch_size=32 actor.global_batch_size=32 actor.nft_forward_micro_batch_size=16 actor.enable_offload=True rollout.enable_offload=True|smolvla_rlinf_dgpo_ms_m2_giant_group"
  [m3_ema_chase]="${COMMON} algorithm.nft_tau=0.85 algorithm.dpo_beta=20|smolvla_rlinf_dgpo_ms_m3_ema_chase"
  [m4_raw_signal]="${COMMON} algorithm.adv_type=raw algorithm.normalize_advantages=False algorithm.dpo_beta=30|smolvla_rlinf_dgpo_ms_m4_raw_signal"
  [m5_throughput]="${COMMON} algorithm.rollout_epoch=2 algorithm.update_epoch=4 algorithm.dpo_beta=15 actor.enable_offload=True rollout.enable_offload=True|smolvla_rlinf_dgpo_ms_m5_throughput"
  [m6_lr_missile]="${COMMON} algorithm.dpo_beta=25 actor.optim.lr=5.0e-5 actor.optim.value_lr=5.0e-4|smolvla_rlinf_dgpo_ms_m6_lr_missile"
  [m7_chaos_open]="${COMMON} algorithm.filter_rewards=False algorithm.adv_clip_max=1.5 algorithm.dpo_beta=75|smolvla_rlinf_dgpo_ms_m7_chaos_open"
  [m8_flowsde_hybrid]="${COMMON} algorithm.dpo_beta=15 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=0.25 actor.model.noise_level=0.25|smolvla_rlinf_dgpo_ms_m8_flowsde_hybrid"
  # --- V2: evidence-driven (wave-1 winners + Flow-SDE bridge) ---
  [v2_champion_fusion]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=50 algorithm.adv_clip_max=1.5|smolvla_rlinf_dgpo_v2_champion_fusion"
  [v2_ema_flowsde_full]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=20 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=1.0 actor.model.noise_level=1.0|smolvla_rlinf_dgpo_v2_ema_flowsde_full"
  [v2_flowsde_flow_lr]="${COMMON} algorithm.nft_tau=0.9 algorithm.filter_rewards=False algorithm.dpo_beta=25 algorithm.update_epoch=2 actor.optim.lr=7.5e-6 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=0.5 actor.model.noise_level=0.5|smolvla_rlinf_dgpo_v2_flowsde_flow_lr"
  [v2_peak_hunter]="${COMMON} algorithm.nft_tau=0.85 algorithm.filter_rewards=False algorithm.dpo_beta=40 algorithm.adv_clip_max=1.2 algorithm.rollout_epoch=2 algorithm.update_epoch=2|smolvla_rlinf_dgpo_v2_peak_hunter"
  [v2_chaos_tau80]="${COMMON} algorithm.nft_tau=0.80 algorithm.filter_rewards=False algorithm.dpo_beta=60 algorithm.adv_clip_max=2.0|smolvla_rlinf_dgpo_v2_chaos_tau80"
  [v2_sde_soft_ema]="${COMMON} algorithm.nft_tau=0.95 algorithm.filter_rewards=False algorithm.dpo_beta=10 actor.model.noise_method=flow_sde actor.model.flow_sde_noise_level=1.0 actor.model.noise_level=1.0|smolvla_rlinf_dgpo_v2_sde_soft_ema"
  # --- V3: post group-id fix validation (Direct-DGPO, mean DSM energy) ---
  [v3_groupfix_baseline]="${COMMON}|smolvla_rlinf_dgpo_v3_groupfix_baseline"
)

submit_moonshot() {
  local tag="$1"
  local spec="${MOONSHOT_SPECS[$tag]:-}"
  [[ -n "${spec}" ]] || { echo "unknown tag: ${tag}" >&2; return 1; }
  local hydra="${spec%%|*}"
  local exp_name="${spec##*|}"
  write_moonshot_env "${tag}" "${exp_name}" "${hydra}"

  local exclude_line=""
  if [[ -n "${SBATCH_EXCLUDE}" ]]; then
    exclude_line="#SBATCH --exclude=${SBATCH_EXCLUDE}"
  fi

  local wrap="${GEN}/rlinf_dgpo_moonshot_${tag}.slurm"
  cat > "${wrap}" <<EOF
#!/bin/bash
#SBATCH --job-name=dgpo-ms-${tag}
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=${MOONSHOT_MEM}
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
${exclude_line}

set -euo pipefail
export MOONSHOT_TAG="${tag}"
source "${GEN}/rlinf_dgpo_moonshot_${tag}_env.sh"
exec bash "${TRAIN_EVAL_SLURM}"
EOF
  chmod +x "${wrap}"
  eggroll_sbatch_submit "${wrap}"
}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <tag> [tag...]" >&2
  exit 1
fi

declare -a JIDS=()
for tag in "$@"; do
  jid="$(submit_moonshot "${tag}")"
  JIDS+=("${tag}=${jid}")
  echo "submitted ${tag} job=${jid} mem=${MOONSHOT_MEM} exclude=${SBATCH_EXCLUDE:-none}"
done

printf 'RLINF_DGPO_MOONSHOT_SUBMIT_OK %s\n' "${JIDS[*]}"
