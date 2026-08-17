#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT=${1:-${OSWORLD_RUN_ROOT:-${GYM_ROOT}}}
RUN_ID=${OSWORLD_RUN_ID:?set OSWORLD_RUN_ID}
CONTROL_HOST=${NEMO_GYM_CONTROL_HOST:-127.0.0.1}
GYM_BIN=${GYM_BIN:-${GYM_ROOT}/.venv/bin/gym}
ENV_FILE=${GYM_ROOT}/benchmarks/osworld/env.yaml
STATE_DIR=${RUN_ROOT}/run/osworld/${RUN_ID}
PID_FILE=${STATE_DIR}/eval.pid

[[ -x "${GYM_BIN}" ]] || { echo "Gym executable is not available: ${GYM_BIN}" >&2; exit 2; }
[[ -r "${ENV_FILE}" ]] || { echo "prepared Gym environment is not readable: ${ENV_FILE}" >&2; exit 2; }

umask 077
mkdir -p "${RUN_ROOT}/logs" "${STATE_DIR}"
printf '%s\n' "$$" >"${PID_FILE}"
exec >>"${RUN_ROOT}/logs/eval-${RUN_ID}.log" 2>&1

export PYTHONPATH="${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OSWORLD_RUN_ID=${RUN_ID}

cd "${GYM_ROOT}/benchmarks/osworld"
exec "${GYM_BIN}" eval run --no-serve \
  +use_absolute_ip=false \
  +default_host="${CONTROL_HOST}"
