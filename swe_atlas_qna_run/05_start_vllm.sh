#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

mkdir -p "${RUN_DIR}"

if [[ ! -f "${VLLM_IMAGE}" ]]; then
  echo "VLLM_IMAGE does not exist: ${VLLM_IMAGE}" >&2
  exit 2
fi
if [[ ! -d "${POLICY_CKPT}" ]]; then
  echo "POLICY_CKPT does not exist: ${POLICY_CKPT}" >&2
  exit 2
fi

if [[ -f "${VLLM_PID_FILE}" ]] && kill -0 "$(<"${VLLM_PID_FILE}")" 2>/dev/null; then
  echo "vLLM already appears to be running with PID $(<"${VLLM_PID_FILE}")."
  echo "Log: ${VLLM_LOG}"
  exit 0
fi

echo "Starting vLLM. Log: ${VLLM_LOG}"
echo "VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"
echo "VLLM_TMPDIR=${VLLM_TMPDIR}"
echo "HF_HOME=${HF_HOME}"
echo "VLLM_CACHE_DIR=${VLLM_CACHE_DIR}"

container_cmd=$(cat <<EOF
mkdir -p "${VLLM_TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${VLLM_CACHE_DIR}" "${TORCH_HOME}"
export HOME="/workspace"
export TMPDIR="${VLLM_TMPDIR}"
export HF_HOME="${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export XDG_CACHE_HOME="${VLLM_CACHE_DIR}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_DIR}/vllm"
export TORCH_HOME="${TORCH_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${VLLM_CACHE_DIR}/torchinductor"
export TRITON_CACHE_DIR="${VLLM_CACHE_DIR}/triton"
echo "== vLLM cache environment inside container =="
env | grep -E '^(HOME|TMPDIR|HF_HOME|HF_HUB_CACHE|TRANSFORMERS_CACHE|XDG_CACHE_HOME|VLLM_CACHE_ROOT|TORCH_HOME|TORCHINDUCTOR_CACHE_DIR|TRITON_CACHE_DIR)=' | sort
echo "== cache filesystem space =="
df -h /workspace "${VLLM_TMPDIR}" "${VLLM_CACHE_DIR}" || true
vllm serve "${POLICY_CKPT}" --host 0.0.0.0 --port "${VLLM_PORT}" --served-model-name "${POLICY_MODEL_NAME}" ${VLLM_EXTRA_ARGS}
EOF
)

srun \
  --overlap \
  --container-image "${VLLM_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${GYM_CONTAINER_DIR}" \
  bash -lc "${container_cmd}" \
  > "${VLLM_LOG}" 2>&1 &

echo $! > "${VLLM_PID_FILE}"
echo "Started vLLM PID $(<"${VLLM_PID_FILE}")"

echo "Waiting for vLLM at ${POLICY_BASE_URL}/models ..."
for _ in $(seq 1 120); do
  if curl -fsS --max-time 5 "${POLICY_BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM is ready."
    exit 0
  fi
  if ! kill -0 "$(<"${VLLM_PID_FILE}")" 2>/dev/null; then
    echo "vLLM process exited early. Last log lines:" >&2
    tail -n 80 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  sleep 10
done

echo "Timed out waiting for vLLM. Last log lines:" >&2
tail -n 80 "${VLLM_LOG}" >&2 || true
exit 1
