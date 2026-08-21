#!/usr/bin/env bash
#SBATCH --job-name=rdkit-dependency-preflight
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu-017
#SBATCH --account=healthcareeng_research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/rdkit_no_tool_grpo/logs/%x-%j.out
#SBATCH --error=cluster/rdkit_no_tool_grpo/logs/%x-%j.err

set -euo pipefail

MIN_ENROOT_FREE_BYTES=$((50 * 1024 * 1024 * 1024))
SCRATCH_FREE_BYTES=$(df -PB1 /raid/scratch | awk 'NR == 2 {print $4}')
if (( SCRATCH_FREE_BYTES < MIN_ENROOT_FREE_BYTES )); then
  echo "Insufficient /raid/scratch space: ${SCRATCH_FREE_BYTES} bytes free" >&2
  exit 1
fi

GYM_DIR=${GYM_DIR:-/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_research/users/dcorneil/code/Gym-rdkit-es140-lora-grpo}
BUNDLE_DIR="${GYM_DIR}/cluster/rdkit_no_tool_grpo"
BASE_CONTAINER=${BASE_CONTAINER_IMAGE_PATH:-${BUNDLE_DIR}/sqsh/nemo-rl-v0.6.0.sqsh}
PEFT_VERSION=${PEFT_VERSION:-0.18.1}
OPENAI_VERSION=${OPENAI_VERSION:-2.6.1}
SOURCE_GYM_COMMIT=$(git -C "${GYM_DIR}" rev-parse HEAD)

if mkdir -p "/raid/scratch/${USER}" 2>/dev/null && [[ -w "/raid/scratch/${USER}" ]]; then
  NODE_LOCAL_ROOT="/raid/scratch/${USER}/rdkit-dependency-preflight-${SLURM_JOB_ID}"
elif [[ -n "${SLURM_TMPDIR:-}" && -w "${SLURM_TMPDIR}" ]]; then
  NODE_LOCAL_ROOT="${SLURM_TMPDIR}/rdkit-dependency-preflight-${SLURM_JOB_ID}"
else
  NODE_LOCAL_ROOT="/tmp/rdkit-dependency-preflight-${USER}-${SLURM_JOB_ID}"
fi
LOCAL_GYM_DIR="${NODE_LOCAL_ROOT}/Gym"
ENROOT_CONTAINER_NAME="rdkit-dependency-preflight-${SLURM_JOB_ID}"
LOCAL_STAMP="${NODE_LOCAL_ROOT}/dependencies.json"
PERSISTED_STAMP="${BUNDLE_DIR}/preflight/dependencies.json"
mkdir -p \
  "${LOCAL_GYM_DIR}" \
  "${NODE_LOCAL_ROOT}/enroot/cache" \
  "${NODE_LOCAL_ROOT}/enroot/data" \
  "${NODE_LOCAL_ROOT}/enroot/runtime" \
  "${NODE_LOCAL_ROOT}/enroot/tmp" \
  "${NODE_LOCAL_ROOT}/uv-cache" \
  "$(dirname "${PERSISTED_STAMP}")"

cleanup() {
  case "${NODE_LOCAL_ROOT}" in
    /raid/scratch/*/rdkit-dependency-preflight-*|/tmp/rdkit-dependency-preflight-*|*/rdkit-dependency-preflight-*)
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

export ENROOT_CACHE_PATH="${NODE_LOCAL_ROOT}/enroot/cache"
export ENROOT_DATA_PATH="${NODE_LOCAL_ROOT}/enroot/data"
export ENROOT_RUNTIME_PATH="${NODE_LOCAL_ROOT}/enroot/runtime"
export ENROOT_TEMP_PATH="${NODE_LOCAL_ROOT}/enroot/tmp"
export ENROOT_MOUNT_HOME=no

enroot create --name "${ENROOT_CONTAINER_NAME}" "${BASE_CONTAINER}"

srun \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
  /usr/bin/env \
  ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH}" \
  ENROOT_DATA_PATH="${ENROOT_DATA_PATH}" \
  ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH}" \
  ENROOT_TEMP_PATH="${ENROOT_TEMP_PATH}" \
  ENROOT_MOUNT_HOME="${ENROOT_MOUNT_HOME}" \
  enroot start \
  --rw \
  --mount "${NODE_LOCAL_ROOT}:${NODE_LOCAL_ROOT}" \
  "${ENROOT_CONTAINER_NAME}" \
  /bin/bash -c "
    set -euo pipefail
    cd '${LOCAL_GYM_DIR}'
    export UV_CACHE_DIR='${NODE_LOCAL_ROOT}/uv-cache'
    /root/.local/bin/uv pip install --python /opt/nemo_rl_venv/bin/python \
      '${LOCAL_GYM_DIR}' 'openai==${OPENAI_VERSION}'
    /root/.local/bin/uv pip install --python /opt/nemo_rl_venv/bin/python --no-deps \
      'peft==${PEFT_VERSION}'
    export PYTHONPATH='${LOCAL_GYM_DIR}'
    export DEPENDENCY_STAMP='${LOCAL_STAMP}'
    export SOURCE_GYM_COMMIT='${SOURCE_GYM_COMMIT}'
    /opt/nemo_rl_venv/bin/python - <<'PY'
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import safetensors
import torch
from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
from nemo_gym.rollout_collection import RolloutCollectionHelper
from nemo_gym.server_utils import HeadServer, ServerClient
from nemo_gym.train_data_utils import TrainDataProcessor
from peft import PeftModel
from safetensors.torch import load_file, save_file

for module_name in (
    'resources_servers.rdkit_chemistry.app',
    'responses_api_agents.simple_agent.app',
    'responses_api_models.vllm_model.app',
):
    importlib.import_module(module_name)

merge_path = Path('cluster/rdkit_no_tool_grpo/merge_es_adapter.py').resolve()
spec = importlib.util.spec_from_file_location('rdkit_direct_merge', merge_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Unable to load {merge_path}')
merge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_module)

versions = {
    name: importlib.metadata.version(name)
    for name in ('nemo-gym', 'openai', 'peft', 'safetensors', 'torch', 'transformers')
}
expected = {'openai': '${OPENAI_VERSION}', 'peft': '${PEFT_VERSION}', 'transformers': '5.3.0'}
for package, expected_version in expected.items():
    if versions[package] != expected_version:
        raise RuntimeError(
            f'{package} version {versions[package]} does not match {expected_version}'
        )

stamp = {
    'schema_version': 1,
    'gym_commit': os.environ['SOURCE_GYM_COMMIT'],
    'versions': versions,
    'imports': {
        'gym_cli': True,
        'gym_rollout_collection': True,
        'gym_server_utils': True,
        'gym_train_data': True,
        'peft': True,
        'rdkit_verifier': True,
        'simple_agent': True,
        'vllm_model_server': True,
        'direct_safetensors_merge': True,
    },
}
Path(os.environ['DEPENDENCY_STAMP']).write_text(json.dumps(stamp, indent=2, sort_keys=True) + '\n')
print(json.dumps(stamp, indent=2, sort_keys=True))
print(
    GlobalConfigDictParserConfig,
    RunHelper,
    RolloutCollectionHelper,
    HeadServer,
    ServerClient,
    TrainDataProcessor,
    PeftModel,
    load_file,
    save_file,
    safetensors.__version__,
    torch.__version__,
)
PY
  "

install -m 0644 "${LOCAL_STAMP}" "${PERSISTED_STAMP}"
echo "Dependency preflight passed: ${PERSISTED_STAMP}"
