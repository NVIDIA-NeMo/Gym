#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.env"

usage() {
  cat <<'EOF'
Usage: 10_stop_servers.sh [--gym|--vllm|--all]

  --gym    Stop only Gym servers (default)
  --vllm   Stop only the external vLLM server
  --all    Stop both Gym servers and vLLM
EOF
}

stop_pid_file() {
  local label="$1"
  local pid_file="$2"
  if [[ ! -f "${pid_file}" ]]; then
    echo "No ${label} PID file found: ${pid_file}"
    return
  fi
  local pid
  pid="$(<"${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} process ${pid}"
    kill "${pid}"
  else
    echo "${label} process ${pid} is not running"
  fi
  rm -f "${pid_file}"
}

target="${1:---gym}"
case "${target}" in
  --gym)
    stop_pid_file "Gym server" "${SERVER_PID_FILE}"
    ;;
  --vllm)
    stop_pid_file "vLLM" "${VLLM_PID_FILE}"
    ;;
  --all)
    stop_pid_file "Gym server" "${SERVER_PID_FILE}"
    stop_pid_file "vLLM" "${VLLM_PID_FILE}"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
