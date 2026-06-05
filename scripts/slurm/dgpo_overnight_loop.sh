#!/bin/bash
# 20-minute overnight wake loop for DGPO NFT pipeline monitoring.
set -euo pipefail
RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
INTERVAL="${DGPO_OVERNIGHT_INTERVAL_SEC:-1200}"

while true; do
  sleep "${INTERVAL}"
  bash "${RLINF}/scripts/slurm/dgpo_overnight_monitor.sh" || true
  echo "DGPO_AGENT_LOOP_WAKE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
