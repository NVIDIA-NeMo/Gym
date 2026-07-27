#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

cd "${GYM_DIR}"
export RAY_TMPDIR

if [[ "${1:-}" == "--tail" ]]; then
  echo "Tailing ${SERVER_LOG}. Ctrl-C only stops tail, not the servers."
  tail -f "${SERVER_LOG}"
  exit 0
fi

if [[ "${1:-}" == "--tail-vllm" ]]; then
  echo "Tailing ${VLLM_LOG}. Ctrl-C only stops tail, not vLLM."
  tail -f "${VLLM_LOG}"
  exit 0
fi

if [[ -f "${VLLM_PID_FILE}" ]]; then
  echo "vLLM PID: $(<"${VLLM_PID_FILE}")"
fi
echo "vLLM endpoint: ${POLICY_BASE_URL}"
curl -fsS --max-time 5 "${POLICY_BASE_URL}/models" >/dev/null && echo "vLLM: ready" || echo "vLLM: not ready"

if [[ -f "${SERVER_PID_FILE}" ]]; then
  echo "Server PID: $(<"${SERVER_PID_FILE}")"
fi

if [[ "${RUN_GYM_IN_CONTAINER}" == "true" ]]; then
  srun --overlap \
    --container-image "${GYM_IMAGE}" \
    --container-mounts "${CONTAINER_MOUNTS}" \
    --container-workdir "${GYM_CONTAINER_DIR}" \
    bash -lc "cd \"${GYM_CONTAINER_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_CONTAINER_DIR}:\${PYTHONPATH:-}\" && gym env status"
else
  srun --overlap bash -lc "cd \"${GYM_DIR}\" && source .venv/bin/activate && export PYTHONPATH=\"${GYM_DIR}:\${PYTHONPATH:-}\" && gym env status"
fi
