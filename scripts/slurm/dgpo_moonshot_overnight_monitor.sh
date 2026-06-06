#!/bin/bash
# Autonomous DGPO moonshot monitor: resubmit failed, eval-only if train OK, 50G RAM.
set -euo pipefail

RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
LOG="${RLINF}/logs/slurm/dgpo_moonshot_monitor.log"
OVERNIGHT_LOG="/vol/bitbucket/aa6622/project/docs/dgpo_overnight_log.md"
source "${RLINF}/scripts/slurm/eggroll_sbatch_submit.sh"

MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"
MAX_DGPO_JOBS="${MAX_DGPO_JOBS:-8}"
TAGS=(m1_nuclear_beta m2_giant_group m3_ema_chase m4_raw_signal m5_throughput m6_lr_missile m7_chaos_open m8_flowsde_hybrid
  v2_champion_fusion v2_ema_flowsde_full v2_flowsde_flow_lr v2_peak_hunter v2_chaos_tau80 v2_sde_soft_ema)

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

append_log() {
  echo "- **$(ts)** $1" >> "${OVERNIGHT_LOG}"
}

occupied_nodes() {
  squeue -u "${USER}" -h -o "%N" 2>/dev/null | grep -v '^$' | sort -u | paste -sd, - | sed 's/^,//'
}

latest_driver_for_tag() {
  local tag="$1"
  ls -1t "${RLINF}"/logs/slurm/dgpo-ms_"${tag}"_*.driver.log 2>/dev/null | head -1 || true
}

latest_out_for_tag() {
  local tag="$1"
  ls -1t "${RLINF}"/logs/slurm/dgpo-ms-"${tag}"_*.out "${RLINF}"/logs/slurm/dgpo-ms_"${tag}"_*.out 2>/dev/null | head -1 || true
}

train_done_for_tag() {
  local tag="$1"
  local drv out
  drv="$(latest_driver_for_tag "${tag}")"
  out="$(latest_out_for_tag "${tag}")"
  { [[ -n "${drv}" ]] && grep -q "runner.run:done" "${drv}" 2>/dev/null; } \
    || { [[ -n "${out}" ]] && grep -q "runner.run:done" "${out}" 2>/dev/null; }
}

eval_done_for_tag() {
  local tag="$1"
  local out drv
  out="$(latest_out_for_tag "${tag}")"
  drv="$(latest_driver_for_tag "${tag}")"
  { [[ -n "${out}" ]] && grep -qE "RLINF_DGPO_MOONSHOT_OK|RLINF_DGPO_MOONSHOT_EVAL100_OK" "${out}" 2>/dev/null; } \
    || { [[ -n "${drv}" ]] && grep -qE "RLINF_DGPO_MOONSHOT_OK|RLINF_DGPO_MOONSHOT_EVAL100_OK" "${drv}" 2>/dev/null; }
}

job_active_for_tag() {
  local tag="$1"
  squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -Eq "dgpo-ms-${tag}|dgpo-ms-ev100-${tag}"
}

dgpo_jobs_queued() {
  squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -c "dgpo-ms" || true
}

latest_train_job_for_tag() {
  local tag="$1"
  sacct -u "${USER}" -n -X -o JobID,JobName --starttime=2026-06-04 2>/dev/null \
    | grep "dgpo-ms-${tag}" | grep -v "ev100" | awk '{print $1}' | tail -1
}

submit_train_eval() {
  local tag="$1"
  local exclude="${2:-}"
  SBATCH_EXCLUDE="${exclude}" MOONSHOT_MEM="${MOONSHOT_MEM}" \
    bash "${RLINF}/scripts/slurm/submit_rlinf_dgpo_moonshot_tags.sh" "${tag}"
}

submit_eval_only() {
  local tag="$1"
  local train_job="$2"
  local wrap="${RLINF}/logs/slurm/generated/dgpo_eval100_${tag}_${train_job}.slurm"
  cat > "${wrap}" <<EOF
#!/bin/bash
#SBATCH --job-name=dgpo-ms-ev100-${tag}
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=${MOONSHOT_MEM}
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
export MOONSHOT_TAG="${tag}"
export RLINF_DGPO_TRAIN_JOB="${train_job}"
exec bash "${RLINF}/scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_eval100_only_a30.slurm"
EOF
  chmod +x "${wrap}"
  eggroll_sbatch_submit "${wrap}"
}

rca_from_log() {
  local tag="$1"
  local out
  out="$(latest_out_for_tag "${tag}")"
  [[ -n "${out}" ]] || { echo "no_log"; return 0; }
  if grep -qE "ConnectionError.*multiple active Ray" "${out}" 2>/dev/null; then
    echo "ray_collision"
  elif grep -qE "OutOfMemoryError|CUDA out of memory|oom-kill|Killed" "${out}" 2>/dev/null; then
    echo "cuda_oom"
  elif grep -qE "QOSMaxMemoryPerUser" "${out}" 2>/dev/null; then
    echo "memory_qos"
  elif grep -qE "micro-batch size.*not a multiple of.*group_size" "${out}" 2>/dev/null; then
    echo "group_size_mismatch"
  else
    echo "unknown"
  fi
}

{
  echo "=== dgpo_moonshot_monitor $(ts) mem=${MOONSHOT_MEM} ==="
  squeue -u "${USER}" -o "%.10i %.24j %.8T %.10m %R" | grep -E "dgpo-ms|JOBID" || true
  echo "queued_dgpo_jobs=$(dgpo_jobs_queued)"
} | tee -a "${LOG}"

EXCLUDE="$(occupied_nodes)"
echo "occupied_nodes=${EXCLUDE:-none}" | tee -a "${LOG}"

for tag in "${TAGS[@]}"; do
  if job_active_for_tag "${tag}"; then
    echo "skip ${tag}: active in queue" | tee -a "${LOG}"
    continue
  fi

  if eval_done_for_tag "${tag}"; then
    echo "skip ${tag}: eval done" | tee -a "${LOG}"
    continue
  fi

  if train_done_for_tag "${tag}"; then
    train_job="$(latest_train_job_for_tag "${tag}")"
    if [[ -n "${train_job}" ]]; then
      if [[ "$(dgpo_jobs_queued)" -ge "${MAX_DGPO_JOBS}" ]]; then
        echo "defer eval-only ${tag}: at job limit" | tee -a "${LOG}"
        continue
      fi
      jid="$(submit_eval_only "${tag}" "${train_job}")"
      append_log "eval-only ${tag} train_job=${train_job} new_eval_job=${jid}"
      echo "eval-only ${tag} job=${jid}" | tee -a "${LOG}"
    fi
    continue
  fi

  if [[ "$(dgpo_jobs_queued)" -ge "${MAX_DGPO_JOBS}" ]]; then
    echo "defer resubmit ${tag}: at job limit ($(dgpo_jobs_queued)/${MAX_DGPO_JOBS})" | tee -a "${LOG}"
    continue
  fi

  rca="$(rca_from_log "${tag}")"
  exclude=""
  if [[ "${rca}" == "ray_collision" ]]; then
    exclude="${EXCLUDE}"
  fi
  jid="$(submit_train_eval "${tag}" "${exclude}")"
  append_log "resubmit ${tag} job_id=${jid} rca=${rca} mem=${MOONSHOT_MEM} exclude=${exclude:-none}"
  echo "resubmit ${tag} job=${jid} rca=${rca}" | tee -a "${LOG}"
done

done_count=0
for tag in "${TAGS[@]}"; do
  eval_done_for_tag "${tag}" && done_count=$((done_count + 1)) || true
done
echo "DGPO_MOONSHOT_MONITOR_OK complete=${done_count}/8 queued=$(dgpo_jobs_queued) $(ts)" | tee -a "${LOG}"
