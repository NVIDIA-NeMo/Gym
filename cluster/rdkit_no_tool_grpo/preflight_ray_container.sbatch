#!/usr/bin/env bash
#SBATCH --job-name=rdkit-ray-image-check
#SBATCH --partition=cpu
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
SOURCE_CONTAINER=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-ray-image-check-${SLURM_JOB_ID}"
LOCAL_CONTAINER="${NODE_LOCAL_ROOT}/$(basename "${SOURCE_CONTAINER}")"
CONTAINER_NAME="rdkit-ray-image-check-${SLURM_JOB_ID}"

cleanup() {
  enroot remove --force "${CONTAINER_NAME}" 2>/dev/null || true
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-ray-image-check-*) rm -rf "${NODE_LOCAL_ROOT}" ;;
  esac
}
trap cleanup EXIT

mkdir -p \
  "${BUNDLE_DIR}/logs" \
  "${NODE_LOCAL_ROOT}/enroot/cache" \
  "${NODE_LOCAL_ROOT}/enroot/data" \
  "${NODE_LOCAL_ROOT}/enroot/runtime" \
  "${NODE_LOCAL_ROOT}/enroot/tmp"
rsync -a "${SOURCE_CONTAINER}" "${LOCAL_CONTAINER}"

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/enroot/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/enroot/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/enroot/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/enroot/tmp"
export ENROOT_MOUNT_HOME=no

enroot create --name "${CONTAINER_NAME}" "${LOCAL_CONTAINER}"
srun \
  --nodes=1 \
  --ntasks=1 \
  /usr/bin/env \
  ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH}" \
  ENROOT_DATA_PATH="${ENROOT_DATA_PATH}" \
  ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH}" \
  ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH}" \
  ENROOT_MOUNT_HOME="${ENROOT_MOUNT_HOME}" \
  enroot start \
  "${CONTAINER_NAME}" \
  /bin/bash -c '
    set -euo pipefail
    test -x /opt/nemo_rl_venv/bin/python
    test -x /opt/nemo_rl_venv/bin/ray
    /opt/nemo_rl_venv/bin/python -c "import ray, ray.scripts.scripts; print(ray.__version__, ray.__file__)"
    /opt/nemo_rl_venv/bin/ray --version
  '

echo "Ray image preflight passed: ${SOURCE_CONTAINER}"
