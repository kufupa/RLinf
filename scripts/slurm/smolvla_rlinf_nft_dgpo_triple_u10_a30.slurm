#!/bin/bash
#SBATCH --job-name=dgpo-v4-triple
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out

# Run 3 independent DGPO moonshots (10u save@1 + eval100 each) sequentially in one GPU job.
# Tags via MOONSHOT_TRIPLET="tag_a tag_b tag_c" (written by submit_rlinf_dgpo_v4_u10_grid.sh).

set -euo pipefail

RLINF_ROOT="/vol/bitbucket/aa6622/RLinf-smolvla-metaworld-ppo-grpo"
TRAIN_EVAL="${RLINF_ROOT}/scripts/slurm/smolvla_rlinf_nft_dgpo_moonshot_u10_train_eval100_a30.slurm"
MOONSHOT_TRIPLET="${MOONSHOT_TRIPLET:?set MOONSHOT_TRIPLET}"
MOONSHOT_BUNDLE="${MOONSHOT_BUNDLE:-unknown}"

read -r -a TAGS <<< "${MOONSHOT_TRIPLET}"
if [[ "${#TAGS[@]}" -ne 3 ]]; then
  echo "MOONSHOT_TRIPLET must contain exactly 3 tags, got ${#TAGS[@]}: ${MOONSHOT_TRIPLET}" >&2
  exit 1
fi

echo "RLINF_DGPO_V4_TRIPLE_START bundle=${MOONSHOT_BUNDLE} tags=${MOONSHOT_TRIPLET} job=${SLURM_JOB_ID:-local}"
seg=1
for tag in "${TAGS[@]}"; do
  echo "=== triple segment ${seg}/3 tag=${tag} ==="
  export MOONSHOT_TAG="${tag}"
  export MOONSHOT_SEGMENT="${seg}"
  unset MOONSHOT_RESUME_DIR || true
  bash "${TRAIN_EVAL}"
  seg=$((seg + 1))
done
echo "RLINF_DGPO_V4_TRIPLE_OK bundle=${MOONSHOT_BUNDLE} tags=${MOONSHOT_TRIPLET}"
