#!/bin/bash
# DGPO moonshot V2 — 6 evidence-driven runs @ 30ep, save@3, 100ep eval (same job).
set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
cd "${RLINF_ROOT}"

MOONSHOT_MEM="${MOONSHOT_MEM:-50G}" \
  bash "${RLINF_ROOT}/scripts/slurm/submit_rlinf_dgpo_moonshot_tags.sh" \
  v2_champion_fusion \
  v2_ema_flowsde_full \
  v2_flowsde_flow_lr \
  v2_peak_hunter \
  v2_chaos_tau80 \
  v2_sde_soft_ema

cat <<'EOF'

RLINF_DGPO_MOONSHOT_V2_GRID_OK
  v2_champion_fusion  m3+m7 fusion (tau0.85, no-filter, beta50)
  v2_ema_flowsde_full EMA + Flow-SDE noise=1.0
  v2_flowsde_flow_lr  Flow-SDE lr7.5e-6 + SDE0.5 + update×2
  v2_peak_hunter      mid-peak hunter (beta40, rollout×2 update×2)
  v2_chaos_tau80      aggressive chaos (tau0.80, beta60)
  v2_sde_soft_ema     soft DGPO + full SDE (tau0.95, beta10)

Protocol: 30 updates, save@3, 100ep eval all ckpts, 50G RAM
EOF
