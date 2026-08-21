#!/usr/bin/env bash
#SBATCH --job-name=rdkit-peft-venv
#SBATCH --partition=cpu
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
CONTAINER=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
TRAIN_PYTHON=${TRAIN_PYTHON:-/opt/nemo_rl_venv/bin/python}
PEFT_VERSION=${PEFT_VERSION:-0.17.1}
PEFT_VENV=${PEFT_VENV:-${BUNDLE_DIR}/venvs/peft-${PEFT_VERSION}}
UV_CACHE_DIR=${UV_CACHE_DIR:-${BUNDLE_DIR}/enroot/uv-cache}

mkdir -p "${BUNDLE_DIR}/logs" "$(dirname "${PEFT_VENV}")" "${UV_CACHE_DIR}"
srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
  --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="/lustre:/lustre" \
  --container-workdir="${GYM_DIR}" \
  --chdir="${GYM_DIR}" \
  /bin/bash -lc "
    set -euo pipefail
    if [[ ! -x '${PEFT_VENV}/bin/python' ]]; then
      '${TRAIN_PYTHON}' -m venv --system-site-packages '${PEFT_VENV}'
    fi
    UV_CACHE_DIR='${UV_CACHE_DIR}' uv pip install \
      --python '${PEFT_VENV}/bin/python' --no-deps 'peft==${PEFT_VERSION}'
    '${PEFT_VENV}/bin/python' -c \
      'import importlib.metadata; from peft import PeftModel; print(\"peft\", importlib.metadata.version(\"peft\")); print(\"PeftModel\", PeftModel)'
  "
