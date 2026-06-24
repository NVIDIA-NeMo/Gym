#!/usr/bin/env bash
# Prepare a Colossus git checkout for running Gym + OSWorld.
#
# This script intentionally does not put secrets or VM images in git. It creates
# stable runtime directories under ~/osworld-run, symlinks the repo to those
# private assets, and performs a preflight check before ng_run/ng_collect_rollouts.
#
# Expected workflow:
#   1. Desktop: edit, commit, push gym-osworld.
#   2. Colossus: git clone/pull feature/osworld2.
#   3. Colossus: copy env.yaml to ~/osworld-run/private/env.yaml out-of-band.
#   4. Colossus: keep Ubuntu.qcow2 at ~/osworld-run/osworld-vm-data/Ubuntu.qcow2.
#   5. Colossus: run this script from the git checkout.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

RUN_ROOT="${OSWORLD_RUN_ROOT:-${HOME}/osworld-run}"
PRIVATE_DIR="${OSWORLD_PRIVATE_DIR:-${RUN_ROOT}/private}"
VM_DATA_DIR="${OSWORLD_VM_DATA_DIR:-${RUN_ROOT}/osworld-vm-data}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${RUN_ROOT}/uv-cache}"
ENV_YAML="${OSWORLD_ENV_YAML:-${PRIVATE_DIR}/env.yaml}"
QCOW2="${OSWORLD_QCOW2:-${VM_DATA_DIR}/Ubuntu.qcow2}"
INSTALL_HOST_DEPS="${INSTALL_HOST_DEPS:-0}"
SYNC_VENV="${SYNC_VENV:-1}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra dev}"
PULL_DOCKER_IMAGE="${PULL_DOCKER_IMAGE:-0}"
DOCKER_IMAGE="${OSWORLD_DOCKER_IMAGE:-happysixd/osworld-docker:latest}"

# uv's installer writes to ~/.local/bin. Non-interactive SSH shells often do
# not source shell rc files, so make that path visible for this script and for
# commands that source .colossus-runtime.env later.
export PATH="${HOME}/.local/bin:${PATH}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

note() {
    echo "--- $* ---"
}

link_file() {
    local target="$1"
    local link="$2"
    mkdir -p "$(dirname "${link}")"
    if [[ -L "${link}" ]]; then
        ln -sfn "${target}" "${link}"
    elif [[ -e "${link}" ]]; then
        die "${link} exists and is not a symlink; move it aside before rerunning"
    else
        ln -s "${target}" "${link}"
    fi
}

check() {
    local label="$1"
    local value="$2"
    local status="$3"
    printf '  %-24s %s\n' "${label}:" "${value}"
    if [[ "${status}" != "ok" ]]; then
        return 1
    fi
}

cd "${GYM_ROOT}"

note "Colossus OSWorld runtime setup"
echo "repo:      ${GYM_ROOT}"
echo "run root:  ${RUN_ROOT}"
echo "env yaml:  ${ENV_YAML}"
echo "qcow2:     ${QCOW2}"
echo "uv cache:  ${UV_CACHE_DIR}"
echo

if [[ "${INSTALL_HOST_DEPS}" == "1" ]]; then
    note "host dependency bringup"
    bash "${SCRIPT_DIR}/bringup_local_host.sh"
    echo
fi

if [[ "${SYNC_VENV}" == "1" ]]; then
    note "top-level Gym venv"
    if [[ -n "${UV_SYNC_ARGS}" ]]; then
        read -r -a uv_sync_args <<< "${UV_SYNC_ARGS}"
        uv sync "${uv_sync_args[@]}"
    else
        uv sync
    fi
    echo
fi

note "runtime directories"
mkdir -p "${PRIVATE_DIR}" "${VM_DATA_DIR}" "${UV_CACHE_DIR}" "${GYM_ROOT}/results" "${GYM_ROOT}/cache" "${GYM_ROOT}/docker_vm_data"
echo "created/verified runtime dirs"
echo

if [[ -n "${ENV_YAML_SRC:-}" ]]; then
    [[ -f "${ENV_YAML_SRC}" ]] || die "ENV_YAML_SRC does not exist: ${ENV_YAML_SRC}"
    install -m 600 "${ENV_YAML_SRC}" "${ENV_YAML}"
    echo "copied ENV_YAML_SRC to ${ENV_YAML}"
fi

note "repo symlinks"
link_file "${ENV_YAML}" "${GYM_ROOT}/env.yaml"
link_file "${QCOW2}" "${GYM_ROOT}/docker_vm_data/Ubuntu.qcow2"
echo "env.yaml -> ${ENV_YAML}"
echo "docker_vm_data/Ubuntu.qcow2 -> ${QCOW2}"
echo

note "runtime env file"
cat > "${GYM_ROOT}/.colossus-runtime.env" <<EOF
export OSWORLD_RUN_ROOT="${RUN_ROOT}"
export OSWORLD_PRIVATE_DIR="${PRIVATE_DIR}"
export OSWORLD_VM_DATA_DIR="${VM_DATA_DIR}"
export OSWORLD_ENV_YAML="${ENV_YAML}"
export OSWORLD_QCOW2="${QCOW2}"
export UV_CACHE_DIR="${UV_CACHE_DIR}"
export PATH="${GYM_ROOT}/.venv/bin:\${HOME}/.local/bin:\${PATH}"
EOF
chmod 600 "${GYM_ROOT}/.colossus-runtime.env"
echo "wrote ${GYM_ROOT}/.colossus-runtime.env"
echo

note "preflight"
fail=0

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    check "git branch" "$(git branch --show-current)" ok || fail=$((fail + 1))
    check "git commit" "$(git rev-parse --short HEAD)" ok || fail=$((fail + 1))
else
    check "git checkout" "not a git worktree" fail || fail=$((fail + 1))
fi

command -v uv >/dev/null 2>&1 && check "uv" "$(uv --version)" ok || { check "uv" "MISSING" fail || true; fail=$((fail + 1)); }
[[ -x "${GYM_ROOT}/.venv/bin/ng_run" ]] && check "ng_run" "${GYM_ROOT}/.venv/bin/ng_run" ok || { check "ng_run" "missing; run uv sync" fail || true; fail=$((fail + 1)); }
[[ -x "${GYM_ROOT}/.venv/bin/ng_collect_rollouts" ]] && check "ng_collect_rollouts" "${GYM_ROOT}/.venv/bin/ng_collect_rollouts" ok || { check "ng_collect_rollouts" "missing; run uv sync" fail || true; fail=$((fail + 1)); }
command -v docker >/dev/null 2>&1 && check "docker" "$(docker --version)" ok || { check "docker" "MISSING" fail || true; fail=$((fail + 1)); }
docker info >/dev/null 2>&1 && check "docker daemon" "OK" ok || { check "docker daemon" "not responding" fail || true; fail=$((fail + 1)); }
[[ -c /dev/kvm ]] && check "kvm" "PRESENT" ok || check "kvm" "MISSING; qemu will use slow TCG fallback" ok
[[ -s "${ENV_YAML}" ]] && check "env.yaml" "present" ok || { check "env.yaml" "missing at ${ENV_YAML}" fail || true; fail=$((fail + 1)); }
[[ -s "${QCOW2}" ]] && check "Ubuntu.qcow2" "$(du -hL "${QCOW2}" | awk '{print $1}')" ok || { check "Ubuntu.qcow2" "missing at ${QCOW2}" fail || true; fail=$((fail + 1)); }
[[ -L "${GYM_ROOT}/env.yaml" ]] && check "repo env symlink" "$(readlink "${GYM_ROOT}/env.yaml")" ok || { check "repo env symlink" "missing" fail || true; fail=$((fail + 1)); }
[[ -L "${GYM_ROOT}/docker_vm_data/Ubuntu.qcow2" ]] && check "repo qcow2 symlink" "$(readlink "${GYM_ROOT}/docker_vm_data/Ubuntu.qcow2")" ok || { check "repo qcow2 symlink" "missing" fail || true; fail=$((fail + 1)); }
echo

if [[ "${PULL_DOCKER_IMAGE}" == "1" ]]; then
    note "docker image"
    docker pull "${DOCKER_IMAGE}"
    echo
fi

if [[ "${fail}" -gt 0 ]]; then
    cat <<EOF
Setup completed with ${fail} failing preflight check(s).

Expected private assets:
  env.yaml:      ${ENV_YAML}
  Ubuntu.qcow2:  ${QCOW2}

Copy env.yaml from desktop out-of-band, for example:
  scp /path/to/env.yaml $(whoami)@$(hostname -f):${ENV_YAML}

Then rerun:
  bash responses_api_agents/osworld_agent/scripts/setup_colossus_runtime.sh
EOF
    exit 1
fi

cat <<EOF
Colossus runtime checkout is ready.

For native PromptAgent smoke:
  bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh

For command preview only:
  DRY_RUN=1 bash responses_api_agents/osworld_agent/scripts/run_native_prompt_agent_smoke.sh
EOF
