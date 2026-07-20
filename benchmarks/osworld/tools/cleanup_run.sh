#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT=${1:-${OSWORLD_RUN_ROOT:-${GYM_ROOT}}}
RUN_ID=${OSWORLD_RUN_ID:?set OSWORLD_RUN_ID}
STATE_DIR=${RUN_ROOT}/run/osworld/${RUN_ID}
GRACE_SECONDS=${OSWORLD_CLEANUP_GRACE_SECONDS:-30}
FORCE=${OSWORLD_CLEANUP_FORCE:-0}

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "OSWORLD_RUN_ID contains unsafe characters: ${RUN_ID}" >&2
    exit 2
fi
if [[ ! "${GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "OSWORLD_CLEANUP_GRACE_SECONDS must be a non-negative integer" >&2
    exit 2
fi

process_belongs_to_run() {
    local pid=$1
    local expected_command=$2
    local cmdline

    [[ -r "/proc/${pid}/environ" && -r "/proc/${pid}/cmdline" ]] || return 1
    tr '\0' '\n' <"/proc/${pid}/environ" | grep -Fqx "OSWORLD_RUN_ID=${RUN_ID}" || return 1
    cmdline=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
    [[ "${cmdline}" == *"${expected_command}"* ]]
}

stop_role() {
    local role=$1
    local expected_command=$2
    local pid_file=${STATE_DIR}/${role}.pid
    local pid
    local attempt

    if [[ ! -s "${pid_file}" ]]; then
        echo "No ${role} PID file for run ${RUN_ID}"
        return
    fi
    pid=$(<"${pid_file}")
    if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
        echo "Refusing invalid ${role} PID file: ${pid_file}" >&2
        return 1
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "${role} process ${pid} is already stopped"
        rm -f "${pid_file}"
        return
    fi
    if ! process_belongs_to_run "${pid}" "${expected_command}"; then
        echo "Refusing to signal PID ${pid}: it is not ${role} for run ${RUN_ID}" >&2
        return 1
    fi

    echo "Stopping ${role} process ${pid} for run ${RUN_ID}"
    kill -INT "${pid}"
    for ((attempt = 0; attempt < GRACE_SECONDS * 2; attempt++)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            rm -f "${pid_file}"
            return
        fi
        sleep 0.5
    done

    echo "${role} process ${pid} exceeded the ${GRACE_SECONDS}s grace period; sending SIGTERM" >&2
    kill -TERM "${pid}"
    sleep 2
    if kill -0 "${pid}" 2>/dev/null; then
        if [[ "${FORCE}" == "1" ]]; then
            echo "Force-killing ${role} process ${pid}" >&2
            kill -KILL "${pid}"
        else
            echo "${role} process ${pid} is still running; set OSWORLD_CLEANUP_FORCE=1 to use SIGKILL" >&2
            return 1
        fi
    fi
    rm -f "${pid_file}"
}

# Stop new work before removing an orphaned run-owned Sandbox.
stop_role eval "eval run"
stop_role control "env start"

if command -v docker >/dev/null 2>&1; then
    container_ids=$(docker ps -aq \
        --filter "label=nemo-gym.sandbox=1" \
        --filter "label=nemo-gym.workload=osworld" \
        --filter "label=nemo-gym.run-id=${RUN_ID}")
    if [[ -n "${container_ids}" ]]; then
        echo "Removing run-owned OSWorld Sandbox container(s):"
        while IFS= read -r container_id; do
            [[ -n "${container_id}" ]] || continue
            echo "  ${container_id}"
            docker rm -f "${container_id}"
        done <<<"${container_ids}"
    else
        echo "No run-owned OSWorld Sandbox containers for ${RUN_ID}"
    fi
else
    echo "Docker is unavailable; no Sandbox containers inspected" >&2
fi

echo "Cleanup complete for ${RUN_ID}; logs and results were preserved."
