#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <container-image-or-sqsh>" >&2
  exit 2
fi

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
SLURM_ACCOUNT=${SLURM_ACCOUNT:-healthcareeng_research}
SLURM_PARTITION=${SLURM_PARTITION:-pool0}
TRAIN_PYTHON=${TRAIN_PYTHON:-${BUNDLE_DIR}/venvs/nemo-rl-gym-peft/bin/python}
PROBE_TIME=${PROBE_TIME:-00:30:00}
CONTAINER_IMAGE=$1

if [[ -n "${NODE_LOCAL_SCRATCH:-}" ]]; then
  NODE_LOCAL_BASE="${NODE_LOCAL_SCRATCH}"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  NODE_LOCAL_BASE="${SLURM_TMPDIR}"
elif [[ -d /raid && -w /raid ]]; then
  NODE_LOCAL_BASE="/raid/enroot/rdkit-nemo-rl-${USER:-unknown}-probe-$$"
else
  NODE_LOCAL_BASE="/tmp/enroot/rdkit-nemo-rl-${USER:-unknown}-probe-$$"
fi

mkdir -p \
  "${BUNDLE_DIR}/enroot/cache"

export ENROOT_CACHE_PATH="${BUNDLE_DIR}/enroot/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_BASE}/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_BASE}/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_BASE}/tmp"
export GYM_DIR

echo "Probing container: ${CONTAINER_IMAGE}"
echo "Gym mount: ${GYM_DIR}:${GYM_DIR}"
echo "Enroot cache: ${ENROOT_CACHE_PATH}"
echo "Enroot data: ${ENROOT_DATA_PATH}"
echo "Enroot runtime: ${ENROOT_RUNTIME_PATH}"
echo "Enroot temp: ${ENROOT_TEMP_PATH}"

cd "${GYM_DIR}"
srun \
  --account="${SLURM_ACCOUNT}" \
  --partition="${SLURM_PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --time="${PROBE_TIME}" \
  --no-container-mount-home \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${GYM_DIR}:${GYM_DIR}" \
  --chdir=/tmp \
  --container-workdir=/tmp \
  /bin/bash -lc "cd \"${GYM_DIR}\" && PYTHONDONTWRITEBYTECODE=1 \"${TRAIN_PYTHON}\" cluster/rdkit_no_tool_grpo/nemo_rl_assets/probe_v06_runner.py"
