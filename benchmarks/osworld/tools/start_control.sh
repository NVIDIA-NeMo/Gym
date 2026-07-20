#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT=${1:-${OSWORLD_RUN_ROOT:-${GYM_ROOT}}}
RUN_ID=${OSWORLD_RUN_ID:?set OSWORLD_RUN_ID}
CONTROL_HOST=${NEMO_GYM_CONTROL_HOST:-127.0.0.1}
GYM_BIN=${GYM_BIN:-${GYM_ROOT}/.venv/bin/gym}
ENV_FILE=${GYM_ROOT}/benchmarks/osworld/env.yaml

[[ -x "${GYM_BIN}" ]] || { echo "Gym executable is not available: ${GYM_BIN}" >&2; exit 2; }
[[ -r "${ENV_FILE}" ]] || { echo "prepared Gym environment is not readable: ${ENV_FILE}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results/${RUN_ID}"
exec >>"${RUN_ROOT}/logs/control-${RUN_ID}.log" 2>&1

export PYTHONPATH="${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OSWORLD_RUN_ID=${RUN_ID}
export OSWORLD_TASK_ARTIFACT_ROOT=${OSWORLD_TASK_ARTIFACT_ROOT:-${RUN_ROOT}/results/${RUN_ID}/tasks}
export OSWORLD_MODEL_IO_LOG=${OSWORLD_MODEL_IO_LOG:-${RUN_ROOT}/results/${RUN_ID}/model-io.jsonl}
export OSWORLD_RESOURCES_IO_LOG=${OSWORLD_RESOURCES_IO_LOG:-${RUN_ROOT}/results/${RUN_ID}/resources-io.jsonl}
export OSWORLD_VM_EXEC_LOG=${OSWORLD_VM_EXEC_LOG:-${RUN_ROOT}/results/${RUN_ID}/vm-exec.jsonl}

cd "${GYM_ROOT}/benchmarks/osworld"
exec "${GYM_BIN}" env start \
  +use_absolute_ip=false \
  +default_host="${CONTROL_HOST}"
