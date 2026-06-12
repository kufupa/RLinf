#!/bin/bash
# Monitor DGPO NFT slurm jobs; resubmit failed smoke/train/eval on a30 (no --export).
set -euo pipefail

RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
LOG="${RLINF}/logs/slurm/dgpo_overnight_monitor.log"
OVERNIGHT_LOG="/vol/bitbucket/aa6622/project/docs/dgpo_overnight_log.md"
source "${RLINF}/scripts/slurm/eggroll_sbatch_submit.sh"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

append_log() {
  echo "- **$(ts)** $1" >> "${OVERNIGHT_LOG}"
}

latest_out() {
  ls -1t "${RLINF}"/logs/slurm/nft-dgpo-*_*.driver.log 2>/dev/null | head -1 || true
}

check_job_done() {
  local pattern="$1"
  local out
  out="$(ls -1t "${RLINF}"/logs/slurm/${pattern} 2>/dev/null | head -1 || true)"
  [[ -n "${out}" ]] && grep -q "runner.run:done" "${out}" 2>/dev/null
}

{
  echo "=== dgpo_overnight_monitor $(ts) ==="
  squeue -u "${USER}" -o "%.18i %.9P %.30j %.8T %.10M %R" | rg "nft-dgpo|JOBID" || true
  out="$(latest_out)"
  if [[ -n "${out}" ]]; then
    echo "latest_driver=${out}"
    tail -n 30 "${out}" || true
  fi
} | tee -a "${LOG}"

if ! check_job_done "nft-dgpo-smoke_*"; then
  if ! squeue -u "${USER}" -h -o "%j" | rg -q "nft-dgpo-smoke"; then
    jid="$(eggroll_sbatch_submit "${RLINF}/scripts/slurm/smolvla_rlinf_nft_dgpo_smoke_a30.slurm")"
    append_log "resubmit smoke job_id=${jid}"
    echo "resubmitted smoke ${jid}"
  fi
fi

if check_job_done "nft-dgpo-smoke_*" && ! check_job_done "nft-dgpo-train_*"; then
  if ! squeue -u "${USER}" -h -o "%j" | rg -q "nft-dgpo-train"; then
    jid="$(eggroll_sbatch_submit "${RLINF}/scripts/slurm/smolvla_rlinf_nft_dgpo_train_a30.slurm")"
    append_log "submit train job_id=${jid}"
    echo "submitted train ${jid}"
  fi
fi
