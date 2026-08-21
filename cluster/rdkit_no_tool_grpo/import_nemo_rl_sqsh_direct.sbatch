#!/usr/bin/env bash
#SBATCH --job-name=rdkit-nemo-rl-import
#SBATCH --partition=cpu
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
BASE_CONTAINER_IMAGE=${BASE_CONTAINER_IMAGE:-docker://nvcr.io#nvidia/nemo-rl:v0.6.0}
SQSH_PATH=${SQSH_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
TMP_SQSH="${SQSH_PATH}.tmp-${SLURM_JOB_ID}"

mkdir -p "${BUNDLE_DIR}/logs" "${BUNDLE_DIR}/sqsh"
if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-nemo-rl-import-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-nemo-rl-import-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-nemo-rl-import-${USER}-${SLURM_JOB_ID}"
fi
mkdir -p \
  "${NODE_LOCAL_ROOT}/cache" \
  "${NODE_LOCAL_ROOT}/data" \
  "${NODE_LOCAL_ROOT}/runtime" \
  "${NODE_LOCAL_ROOT}/tmp"

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-nemo-rl-import-*|/tmp/rdkit-nemo-rl-import-*|*/rdkit-nemo-rl-import-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
  rm -f "${TMP_SQSH}"
}
trap cleanup EXIT

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/tmp"
export https_proxy= http_proxy= all_proxy= no_proxy="*"

echo "Base container image: ${BASE_CONTAINER_IMAGE}"
echo "Output sqsh: ${SQSH_PATH}"
echo "Started: $(date -Is)"
cd "${ENROOT_TEMP_PATH}"
enroot import -o "${TMP_SQSH}" "${BASE_CONTAINER_IMAGE}"
mv "${TMP_SQSH}" "${SQSH_PATH}"
ls -lh "${SQSH_PATH}"
echo "Finished: $(date -Is)"
