#!/bin/bash
# Submit remaining V4 bundles (b7, b8) after QoS submit cap frees slots.
set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
cd "${RLINF_ROOT}"
source "${RLINF_ROOT}/scripts/slurm/eggroll_sbatch_submit.sh"
GEN="${RLINF_ROOT}/logs/slurm/generated"
TRIPLE_SLURM="${RLINF_ROOT}/scripts/slurm/smolvla_rlinf_nft_dgpo_triple_u10_a30.slurm"
MOONSHOT_MEM="${MOONSHOT_MEM:-50G}"
SBATCH_EXCLUDE="${SBATCH_EXCLUDE:-}"

submit_one_bundle() {
  local bundle_id="$1"
  local triplet="$2"
  local dep="${3:-}"
  local exclude_line=""
  [[ -n "${SBATCH_EXCLUDE}" ]] && exclude_line="#SBATCH --exclude=${SBATCH_EXCLUDE}"
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
  if [[ -n "${dep}" ]]; then
    eggroll_sbatch_submit --dependency="${dep}" "${wrap}"
  else
    eggroll_sbatch_submit "${wrap}"
  fi
}

# Wait until user has submit quota (<=6 pending+running dgpo-v4 jobs or general cap)
while true; do
  pending_v4="$(squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -c "dgpo-v4" || true)"
  if [[ "${pending_v4}" -lt 6 ]]; then
    break
  fi
  echo "waiting for dgpo-v4 slot (current=${pending_v4}) $(date -u +%H:%M:%S)"
  sleep 120
done

J7="$(submit_one_bundle b7 "v4_tau80_chaos v4_beta200 v4_sde10_soft" "")"
echo "submitted b7 job=${J7}"

while true; do
  pending_v4="$(squeue -u "${USER}" -h -o "%j" 2>/dev/null | grep -c "dgpo-v4" || true)"
  if [[ "${pending_v4}" -lt 7 ]]; then
    break
  fi
  echo "waiting for b8 slot (current=${pending_v4}) $(date -u +%H:%M:%S)"
  sleep 120
done

J8="$(submit_one_bundle b8 "v4_peak_lite v4_fix_mb_ema v4_fix_mb_open" "")"
echo "submitted b8 job=${J8}"
echo "RLINF_DGPO_V4_REMAINING_OK b7=${J7} b8=${J8}"
