#!/usr/bin/env bash
#SBATCH --job-name=rdkit-integration-sqsh
#SBATCH --partition=cpu
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
BASE_CONTAINER=${BASE_CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
OUTPUT_CONTAINER=${CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0-gym-peft.sqsh}
TMP_CONTAINER="${OUTPUT_CONTAINER}.tmp-${SLURM_JOB_ID}"
PEFT_VERSION=${PEFT_VERSION:-0.17.1}
OPENAI_VERSION=${OPENAI_VERSION:-2.6.1}
DEVTOOLS_VERSION=${DEVTOOLS_VERSION:-0.12.2}
ORJSON_VERSION=${ORJSON_VERSION:-3.11.3}
PINNED_GYM_COMMIT=1a4912e231bb2795b062f7de97496caaf382c7f6
SOURCE_GYM_COMMIT=$(git -C "${GYM_DIR}" rev-parse HEAD)
git -C "${GYM_DIR}" merge-base --is-ancestor "${PINNED_GYM_COMMIT}" "${SOURCE_GYM_COMMIT}"

if [[ ! -f "${BASE_CONTAINER}" ]]; then
  echo "Missing base container: ${BASE_CONTAINER}" >&2
  exit 1
fi
if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-integration-sqsh-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-integration-sqsh-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-integration-sqsh-${USER}-${SLURM_JOB_ID}"
fi
mkdir -p \
  "${BUNDLE_DIR}/logs" \
  "${BUNDLE_DIR}/sqsh" \
  "${NODE_LOCAL_ROOT}/enroot-cache" \
  "${NODE_LOCAL_ROOT}/enroot-data" \
  "${NODE_LOCAL_ROOT}/enroot-runtime" \
  "${NODE_LOCAL_ROOT}/enroot-tmp" \
  "${NODE_LOCAL_ROOT}/container" \
  "${NODE_LOCAL_ROOT}/Gym" \
  "${NODE_LOCAL_ROOT}/uv-cache"
LOCAL_BASE_CONTAINER="${NODE_LOCAL_ROOT}/container/$(basename "${BASE_CONTAINER}")"
rsync -a "${BASE_CONTAINER}" "${LOCAL_BASE_CONTAINER}"
rsync -a \
  --exclude=.git/ \
  --exclude=artifacts/ \
  --exclude=cluster/rdkit_no_tool_grpo/enroot/ \
  --exclude=cluster/rdkit_no_tool_grpo/logs/ \
  --exclude=cluster/rdkit_no_tool_grpo/results/ \
  --exclude=cluster/rdkit_no_tool_grpo/sqsh/ \
  --exclude=cluster/rdkit_no_tool_grpo/venvs/ \
  "${GYM_DIR}/" "${NODE_LOCAL_ROOT}/Gym/"

cleanup() {
  rm -f "${TMP_CONTAINER}"
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-integration-sqsh-*|/tmp/rdkit-integration-sqsh-*|*/rdkit-integration-sqsh-*)
      rm -rf "${NODE_LOCAL_ROOT}"
      ;;
  esac
}
trap cleanup EXIT

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/enroot-cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/enroot-data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/enroot-runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/enroot-tmp"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --no-container-mount-home \
  --container-image="${LOCAL_BASE_CONTAINER}" \
  --container-save="${TMP_CONTAINER}" \
  --container-mounts="${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT}" \
  --container-workdir="${NODE_LOCAL_ROOT}" \
  --chdir="${NODE_LOCAL_ROOT}" \
  /bin/bash -lc "
    set -euo pipefail
    export UV_CACHE_DIR='${NODE_LOCAL_ROOT}/uv-cache'
    uv pip install --python /opt/nemo_rl_venv/bin/python \
      'openai==${OPENAI_VERSION}' 'devtools==${DEVTOOLS_VERSION}' \
      'orjson==${ORJSON_VERSION}'
    uv pip install --python /opt/nemo_rl_venv/bin/python --no-deps \
      'peft==${PEFT_VERSION}'
    uv pip install --python /opt/nemo_rl_venv/bin/python --no-deps \
      '${NODE_LOCAL_ROOT}/Gym'
    /opt/nemo_rl_venv/bin/python - <<'PY'
import importlib.metadata
from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
from nemo_gym.rollout_collection import RolloutCollectionHelper
from peft import PeftModel

print('nemo-gym', importlib.metadata.version('nemo-gym'))
print('openai', importlib.metadata.version('openai'))
print('peft', importlib.metadata.version('peft'))
print(GlobalConfigDictParserConfig, RunHelper, RolloutCollectionHelper, PeftModel)
PY
  "

mv "${TMP_CONTAINER}" "${OUTPUT_CONTAINER}"
ls -lh "${OUTPUT_CONTAINER}"
