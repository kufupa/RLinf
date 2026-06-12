#!/bin/bash
# Wake every 60s until v3_groupfix_baseline train+eval100 completes.
set -euo pipefail

RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
INTERVAL="${DGPO_V3_INTERVAL_SEC:-60}"
export MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"
LOOP_LOG="${RLINF}/logs/slurm/dgpo_v3_groupfix_loop.log"

while true; do
  if bash "${RLINF}/scripts/slurm/dgpo_v3_groupfix_overnight_monitor.sh" >> "${LOOP_LOG}" 2>&1; then
    if grep -q "DGPO_V3_GROUPFIX_COMPLETE" "${RLINF}/logs/slurm/dgpo_v3_groupfix_monitor.log" 2>/dev/null; then
      echo "DGPO_V3_GROUPFIX_LOOP_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOOP_LOG}"
      break
    fi
  fi
  sleep "${INTERVAL}"
done
