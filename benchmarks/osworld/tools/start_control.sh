#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT=${1:-${OSWORLD_RUN_ROOT:-${GYM_ROOT}}}
RUN_ID=${OSWORLD_RUN_ID:?set OSWORLD_RUN_ID}
CONTROL_HOST=${NEMO_GYM_CONTROL_HOST:-127.0.0.1}
GYM_BIN=${GYM_BIN:-${GYM_ROOT}/.venv/bin/gym}
GYM_PYTHON=${GYM_PYTHON:-$(dirname "${GYM_BIN}")/python}
ENV_FILE=${GYM_ROOT}/benchmarks/osworld/env.yaml
RUNTIME_DEPS_CHECKER=${GYM_ROOT}/responses_api_agents/osworld_agent/runtime_dependencies.py
RUNTIME_DEPS_INSTALLER=${GYM_ROOT}/responses_api_agents/osworld_agent/install_optional_runtime_deps.sh
STATE_DIR=${RUN_ROOT}/run/osworld/${RUN_ID}
PID_FILE=${STATE_DIR}/control.pid

[[ -x "${GYM_BIN}" ]] || { echo "Gym executable is not available: ${GYM_BIN}" >&2; exit 2; }
[[ -x "${GYM_PYTHON}" ]] || { echo "Gym Python is not available: ${GYM_PYTHON}" >&2; exit 2; }
[[ -r "${ENV_FILE}" ]] || { echo "prepared Gym environment is not readable: ${ENV_FILE}" >&2; exit 2; }
[[ -r "${RUNTIME_DEPS_CHECKER}" ]] || {
    echo "OSWorld runtime dependency checker is not readable: ${RUNTIME_DEPS_CHECKER}" >&2
    exit 2
}
command -v cc >/dev/null 2>&1 || {
    echo "A C compiler is required to build the OSWorld agent environment" >&2
    exit 2
}
python_include=$("${GYM_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("include"))')
[[ -r "${python_include}/Python.h" ]] || {
    echo "Python development headers are required (for example: apt install python3-dev)" >&2
    exit 2
}

if [[ -z "${OSWORLD_AGENT_VENV:-}" ]]; then
    OSWORLD_AGENT_VENV=$("${GYM_PYTHON}" "${RUNTIME_DEPS_CHECKER}" resolve-venv \
        --gym-root "${GYM_ROOT}" --env-file "${ENV_FILE}")
fi
osworld_agent_python=${OSWORLD_AGENT_VENV}/bin/python
print_runtime_setup() {
    echo "Prepare the OSWorld agent venv and explicitly opt in to its runtime packages:" >&2
    printf '  cd %q\n' "$(dirname "${ENV_FILE}")" >&2
    echo "  gym env prefetch" >&2
    printf '  bash %q %q\n' "${RUNTIME_DEPS_INSTALLER}" "${OSWORLD_AGENT_VENV}" >&2
}
if [[ ! -x "${osworld_agent_python}" ]]; then
    echo "Managed OSWorld agent Python is not executable: ${osworld_agent_python}" >&2
    print_runtime_setup
    exit 2
fi
if ! "${osworld_agent_python}" "${RUNTIME_DEPS_CHECKER}" check; then
    print_runtime_setup
    exit 2
fi

case "${DOCKER_HOST:-}" in
    ssh://*|tcp://*)
        [[ -n "${OSWORLD_SANDBOX_PUBLISH_HOST:-}" ]] || {
            echo "remote DOCKER_HOST requires OSWORLD_SANDBOX_PUBLISH_HOST" >&2
            exit 2
        }
        command -v docker >/dev/null 2>&1 || {
            echo "Docker CLI is required for the remote Gym Docker Sandbox" >&2
            exit 2
        }
        docker info >/dev/null || {
            echo "remote Docker daemon is not reachable through DOCKER_HOST=${DOCKER_HOST}" >&2
            exit 2
        }
        ;;
    *)
        export OSWORLD_SANDBOX_PUBLISH_HOST=${OSWORLD_SANDBOX_PUBLISH_HOST:-127.0.0.1}
        ;;
esac

umask 077
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results/${RUN_ID}" "${STATE_DIR}"
printf '%s\n' "$$" >"${PID_FILE}"
exec >>"${RUN_ROOT}/logs/control-${RUN_ID}.log" 2>&1

export PYTHONPATH="${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OSWORLD_RUN_ID=${RUN_ID}
export NEMO_GYM_RUN_ID=${NEMO_GYM_RUN_ID:-${RUN_ID}}
export OSWORLD_TASK_ARTIFACT_ROOT=${OSWORLD_TASK_ARTIFACT_ROOT:-${RUN_ROOT}/results/${RUN_ID}/tasks}
export OSWORLD_RESOURCES_IO_LOG=${OSWORLD_RESOURCES_IO_LOG:-${RUN_ROOT}/results/${RUN_ID}/resources-io.jsonl}
export OSWORLD_VM_EXEC_LOG=${OSWORLD_VM_EXEC_LOG:-${RUN_ROOT}/results/${RUN_ID}/vm-exec.jsonl}

cd "${GYM_ROOT}/benchmarks/osworld"
exec "${GYM_BIN}" env start \
  +use_absolute_ip=false \
  +default_host="${CONTROL_HOST}"
