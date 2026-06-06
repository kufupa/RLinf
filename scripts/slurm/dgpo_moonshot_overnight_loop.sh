#!/bin/bash
# 15-minute wake loop for DGPO moonshot grid (50G RAM, auto-resubmit/eval-only).
set -euo pipefail
RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
INTERVAL="${DGPO_MOONSHOT_INTERVAL_SEC:-900}"
export MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"

while true; do
  bash "${RLINF}/scripts/slurm/dgpo_moonshot_overnight_monitor.sh" || true
  pending="$(squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -c "dgpo-ms" || echo 0)"
  done_count=0
  for tag in m1_nuclear_beta m2_giant_group m3_ema_chase m4_raw_signal m5_throughput m6_lr_missile m7_chaos_open m8_flowsde_hybrid \
    v2_champion_fusion v2_ema_flowsde_full v2_flowsde_flow_lr v2_peak_hunter v2_chaos_tau80 v2_sde_soft_ema; do
    out="$(ls -1t "${RLINF}/logs/slurm/dgpo-ms-"${tag}"_*.out "${RLINF}/logs/slurm/dgpo-ms_${tag}"_*.out 2>/dev/null | head -1 || true)"
    [[ -n "${out}" ]] && grep -qE "RLINF_DGPO_MOONSHOT_OK|RLINF_DGPO_MOONSHOT_EVAL100_OK" "${out}" 2>/dev/null && done_count=$((done_count + 1)) || true
  done
  echo "DGPO_MOONSHOT_LOOP pending=${pending} complete=${done_count}/8 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "${done_count}" -ge 8 && "${pending}" -eq 0 ]]; then
    echo "DGPO_MOONSHOT_GRID_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    break
  fi
  sleep "${INTERVAL}"
done
