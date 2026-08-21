#!/usr/bin/env bash
#SBATCH --account=healthcareeng_research
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=rdkit-server-preflight
#SBATCH --output=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo/cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
CONTAINER_IMAGE_PATH=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
TRAIN_PYTHON=${TRAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}
PINNED_GYM_COMMIT=1a4912e231bb2795b062f7de97496caaf382c7f6
SOURCE_GYM_COMMIT=$(git -C "${GYM_DIR}" rev-parse HEAD)
git -C "${GYM_DIR}" merge-base --is-ancestor "${PINNED_GYM_COMMIT}" "${SOURCE_GYM_COMMIT}"

if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-resource-preflight-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-resource-preflight-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-resource-preflight-${USER}-${SLURM_JOB_ID}"
fi
LOCAL_GYM_DIR="${NODE_LOCAL_ROOT}/Gym"
LOCAL_CONTAINER="${NODE_LOCAL_ROOT}/container/$(basename "${CONTAINER_IMAGE_PATH}")"
LOCAL_STAMP="${NODE_LOCAL_ROOT}/resource_server_container.json"
PERSISTED_STAMP="${BUNDLE_DIR}/preflight/resource_server_container.json"
mkdir -p \
  "${LOCAL_GYM_DIR}" \
  "${NODE_LOCAL_ROOT}/container" \
  "${NODE_LOCAL_ROOT}/enroot/cache" \
  "${NODE_LOCAL_ROOT}/enroot/data" \
  "${NODE_LOCAL_ROOT}/enroot/runtime" \
  "${NODE_LOCAL_ROOT}/enroot/tmp" \
  "${NODE_LOCAL_ROOT}/uv-cache" \
  "$(dirname "${PERSISTED_STAMP}")"

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-resource-preflight-*|/tmp/rdkit-resource-preflight-*|*/rdkit-resource-preflight-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
}
trap cleanup EXIT

rsync -a \
  --exclude=.git/ \
  --exclude=artifacts/ \
  --exclude=cluster/rdkit_no_tool_grpo/enroot/ \
  --exclude=cluster/rdkit_no_tool_grpo/logs/ \
  --exclude=cluster/rdkit_no_tool_grpo/results/ \
  --exclude=cluster/rdkit_no_tool_grpo/sqsh/ \
  --exclude=cluster/rdkit_no_tool_grpo/venvs/ \
  "${GYM_DIR}/" "${LOCAL_GYM_DIR}/"
rsync -a "${CONTAINER_IMAGE_PATH}" "${LOCAL_CONTAINER}"

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/enroot/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/enroot/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/enroot/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/enroot/tmp"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --no-container-mount-home \
  --container-image="${LOCAL_CONTAINER}" \
  --container-mounts="${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT},/lustre:/lustre" \
  --container-workdir="${LOCAL_GYM_DIR}" \
  --chdir="${LOCAL_GYM_DIR}" \
  /usr/bin/env \
    GYM_DIR="${LOCAL_GYM_DIR}" \
    NODE_LOCAL_ROOT="${NODE_LOCAL_ROOT}" \
    CONTAINER_IMAGE_PATH="${CONTAINER_IMAGE_PATH}" \
    SOURCE_GYM_COMMIT="${SOURCE_GYM_COMMIT}" \
    PREFLIGHT_OUTPUT="${LOCAL_STAMP}" \
    UV_CACHE_DIR="${NODE_LOCAL_ROOT}/uv-cache" \
    PYTHONDONTWRITEBYTECODE=1 \
    "${TRAIN_PYTHON}" \
    "${LOCAL_GYM_DIR}/cluster/rdkit_no_tool_grpo/preflight_resource_server.py"

install -m 0644 "${LOCAL_STAMP}" "${PERSISTED_STAMP}"
echo "Published container preflight stamp: ${PERSISTED_STAMP}"
