#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

if [[ "${SLURM_ACCOUNT}" == "<ACCOUNT>" || "${SLURM_PARTITION}" == "<PARTITION>" ]]; then
  echo "Edit ${SCRIPT_DIR}/config.env and set SLURM_ACCOUNT and SLURM_PARTITION first." >&2
  exit 2
fi

if [[ ! -f "${GYM_IMAGE}" ]]; then
  echo "GYM_IMAGE does not exist: ${GYM_IMAGE}" >&2
  exit 2
fi
if [[ ! -f "${VLLM_IMAGE}" ]]; then
  echo "VLLM_IMAGE does not exist: ${VLLM_IMAGE}" >&2
  exit 2
fi

echo "Requesting interactive Slurm allocation."
echo "GYM_IMAGE=${GYM_IMAGE}"
echo "VLLM_IMAGE=${VLLM_IMAGE}"
echo "After it starts, run: cd ${RUN_DIR} && ./04_check_compute_env_and_judge.sh"
salloc -A "${SLURM_ACCOUNT}" -p "${SLURM_PARTITION}" -N 1 --gpus-per-node "${GPU_COUNT}" -t "${WALLTIME}"
