#!/bin/bash
# Autonomous monitor: v3_groupfix_baseline smoke → train30 + eval100 (post group-id fix).
set -euo pipefail

RLINF="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
LOG="${RLINF}/logs/slurm/dgpo_v3_groupfix_monitor.log"
OVERNIGHT_LOG="/vol/bitbucket/aa6622/project/docs/dgpo_overnight_log.md"
TAG="v3_groupfix_baseline"
MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"
SMOKE_CPUS="${SMOKE_CPUS:-16}"

source "${RLINF}/scripts/slurm/eggroll_sbatch_submit.sh"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

append_log() {
  echo "- **$(ts)** $1" >> "${OVERNIGHT_LOG}"
}

occupied_nodes() {
  squeue -u "${USER}" -h -o "%N" 2>/dev/null | grep -v '^$' | sort -u | paste -sd, - | sed 's/^,//'
}

latest_smoke_log() {
  ls -1t "${RLINF}"/logs/slurm/nft-dgpo-smoke_*.out \
    "${RLINF}"/logs/slurm/nft-dgpo-smoke_*.driver.log 2>/dev/null | head -1 || true
}

latest_tag_log() {
  ls -1t "${RLINF}"/logs/slurm/dgpo-ms-${TAG}_*.out \
    "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.out \
    "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.driver.log 2>/dev/null | head -1 || true
}

log_has() {
  local f="$1"
  local pat="$2"
  [[ -n "${f}" && -f "${f}" ]] && grep -qE "${pat}" "${f}" 2>/dev/null
}

smoke_ok() {
  local f
  f="$(latest_smoke_log)"
  log_has "${f}" "RLINF_NFT_DGPO_SMOKE_OK|runner.run:done"
}

train_ok() {
  local drv out
  drv="$(ls -1t "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.driver.log 2>/dev/null | head -1 || true)"
  out="$(ls -1t "${RLINF}"/logs/slurm/dgpo-ms-${TAG}_*.out "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.out 2>/dev/null | head -1 || true)"
  log_has "${drv}" "runner.run:done" || log_has "${out}" "RLINF_DGPO_MOONSHOT_TRAIN_OK|runner.run:done"
}

eval_ok() {
  local drv out
  drv="$(ls -1t "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.driver.log 2>/dev/null | head -1 || true)"
  out="$(ls -1t "${RLINF}"/logs/slurm/dgpo-ms-${TAG}_*.out "${RLINF}"/logs/slurm/dgpo-ms_${TAG}_*.out 2>/dev/null | head -1 || true)"
  log_has "${drv}" "RLINF_DGPO_MOONSHOT_OK|RLINF_DGPO_MOONSHOT_EVAL100_OK" \
    || log_has "${out}" "RLINF_DGPO_MOONSHOT_OK|RLINF_DGPO_MOONSHOT_EVAL100_OK"
}

job_active() {
  local pat="$1"
  squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -q "${pat}"
}

latest_failed_smoke_state() {
  sacct -u "${USER}" -n -X -o JobID,JobName,State --starttime=2026-06-05 2>/dev/null \
    | grep "nft-dgpo-smoke" | awk '$3 ~ /FAILED|TIMEOUT|CANCELLED/ {print $3}' | tail -1 || true
}

latest_failed_train_state() {
  sacct -u "${USER}" -n -X -o JobID,JobName,State --starttime=2026-06-05 2>/dev/null \
    | grep "dgpo-ms-${TAG}" | grep -v ev100 | awk '$3 ~ /FAILED|TIMEOUT|CANCELLED/ {print $3}' | tail -1 || true
}

latest_train_job_id() {
  sacct -u "${USER}" -n -X -o JobID,JobName,State --starttime=2026-06-05 2>/dev/null \
    | grep "dgpo-ms-${TAG}" | grep -v ev100 | awk '{print $1}' | tail -1
}

rca_from_log() {
  local f="$1"
  [[ -n "${f}" && -f "${f}" ]] || { echo "no_log"; return 0; }
  if grep -qE "ConnectionError.*multiple active Ray" "${f}" 2>/dev/null; then
    echo "ray_collision"
  elif grep -qE "OutOfMemoryError|CUDA out of memory|oom-kill|Killed" "${f}" 2>/dev/null; then
    echo "cuda_oom"
  elif grep -qE "QOSMaxMemoryPerUser" "${f}" 2>/dev/null; then
    echo "memory_qos"
  elif grep -qE "DGPO group integrity failed|requires dgpo_group_id" "${f}" 2>/dev/null; then
    echo "dgpo_grouping"
  elif grep -qE "world_size=1 until distributed" "${f}" 2>/dev/null; then
    echo "multi_gpu_dgpo"
  elif grep -qE "micro-batch size.*not a multiple of.*group_size" "${f}" 2>/dev/null; then
    echo "group_size_mismatch"
  else
    echo "unknown"
  fi
}

submit_smoke() {
  sbatch --parsable --mem="${MOONSHOT_MEM}" --cpus-per-task="${SMOKE_CPUS}" --time=01:30:00 \
    "${RLINF}/scripts/slurm/smolvla_rlinf_nft_dgpo_smoke_a30.slurm"
}

submit_train_eval() {
  local exclude="${1:-}"
  SBATCH_EXCLUDE="${exclude}" MOONSHOT_MEM="${MOONSHOT_MEM}" \
    bash "${RLINF}/scripts/slurm/submit_rlinf_dgpo_moonshot_tags.sh" "${TAG}" \
    | grep -oE "${TAG}=[0-9]+" | cut -d= -f2
}

submit_train_eval_dep() {
  local smoke_jid="$1"
  local exclude="${2:-}"
  local wrap="${RLINF}/logs/slurm/generated/rlinf_dgpo_moonshot_${TAG}.slurm"
  [[ -f "${wrap}" ]] || submit_train_eval "${exclude}" >/dev/null
  local args=(--parsable --mem="${MOONSHOT_MEM}" --dependency="afterok:${smoke_jid}")
  if [[ -n "${exclude}" ]]; then
    args+=(--exclude="${exclude}")
  fi
  sbatch "${args[@]}" "${wrap}"
}

submit_eval_only() {
  local train_job="$1"
  local wrap="${RLINF}/logs/slurm/generated/dgpo_eval100_${TAG}_${train_job}.slurm"
  cat > "${wrap}" <<EOF
#!/bin/bash
#SBATCH --job-name=dgpo-ms-ev100-${TAG}
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=${MOONSHOT_MEM}
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.out

set -euo pipefail
export MOONSHOT_TAG="${TAG}"
export RLINF_DGPO_TRAIN_JOB="${train_job}"
exec bash "${RLINF}/scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_eval100_only_a30.slurm"
EOF
  chmod +x "${wrap}"
  eggroll_sbatch_submit "${wrap}"
}

{
  echo "=== dgpo_v3_groupfix_monitor $(ts) mem=${MOONSHOT_MEM} ==="
  squeue -u "${USER}" -o "%.10i %.24j %.8T %.10m %R" | grep -E "nft-dgpo-smoke|dgpo-ms-${TAG}|JOBID" || true
  echo "smoke_ok=$(smoke_ok && echo yes || echo no) train_ok=$(train_ok && echo yes || echo no) eval_ok=$(eval_ok && echo yes || echo no)"
} | tee -a "${LOG}"

if eval_ok; then
  echo "DGPO_V3_GROUPFIX_COMPLETE $(ts)" | tee -a "${LOG}"
  exit 0
fi

EXCLUDE="$(occupied_nodes)"

if job_active "nft-dgpo-smoke" || job_active "dgpo-ms-${TAG}" || job_active "dgpo-ms-ev100-${TAG}"; then
  train_log="$(latest_tag_log)"
  step_line=""
  if [[ -n "${train_log}" && -f "${train_log}" ]]; then
    step_line="$(grep -E "Global Step:" "${train_log}" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')"
  fi
  if [[ -n "${step_line}" ]]; then
    echo "jobs still running; ${step_line} $(ts)" | tee -a "${LOG}"
  else
    echo "jobs still running; monitor only $(ts)" | tee -a "${LOG}"
  fi
  exit 0
fi

if train_ok && ! eval_ok; then
  train_job="$(latest_train_job_id)"
  if [[ -n "${train_job}" ]]; then
    jid="$(submit_eval_only "${train_job}")"
    append_log "v3 eval-only train_job=${train_job} new_job=${jid}"
    echo "eval-only job=${jid}" | tee -a "${LOG}"
  fi
  exit 0
fi

if smoke_ok && ! train_ok; then
  rca="$(rca_from_log "$(latest_tag_log)")"
  exclude=""
  [[ "${rca}" == "ray_collision" ]] && exclude="${EXCLUDE}"
  jid="$(submit_train_eval "${exclude}")"
  append_log "v3 train+eval resubmit job=${jid} rca=${rca} (smoke already ok)"
  echo "train+eval resubmit job=${jid}" | tee -a "${LOG}"
  exit 0
fi

# Smoke not ok or never ran — chain smoke → train
smoke_fail="$(latest_failed_smoke_state)"
train_fail="$(latest_failed_train_state)"
rca="$(rca_from_log "$(latest_smoke_log)")"
exclude=""
[[ "${rca}" == "ray_collision" ]] && exclude="${EXCLUDE}"

smoke_jid="$(submit_smoke)"
append_log "v3 smoke resubmit job=${smoke_jid} rca=${rca} smoke_state=${smoke_fail:-none} train_state=${train_fail:-none} exclude=${exclude:-none}"
train_jid="$(submit_train_eval_dep "${smoke_jid}" "${exclude}")"
append_log "v3 train+eval chained job=${train_jid} afterok=${smoke_jid}"
echo "smoke=${smoke_jid} train=${train_jid} rca=${rca}" | tee -a "${LOG}"
