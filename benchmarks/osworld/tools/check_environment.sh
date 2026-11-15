#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_VM_SHA256=${EXPECTED_VM_SHA256:-6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313}
EXPECTED_VM_SIZE=${EXPECTED_VM_SIZE:-24460197888}
MIN_FREE_GIB=${MIN_FREE_GIB:-30}

if [[ ${1:-} == "--ssh" ]]; then
    [[ $# == 3 ]] || {
        echo "usage: $0 --ssh REMOTE_USER@ENV_HOST /absolute/path/to/Ubuntu.qcow2" >&2
        exit 2
    }
    exec ssh -o BatchMode=yes "$2" bash -s -- "$3" <"${BASH_SOURCE[0]}"
fi

VM_PATH=${1:?usage: check_environment.sh /absolute/path/to/Ubuntu.qcow2}
failures=0

fail() {
    printf 'error: %s\n' "$*" >&2
    failures=$((failures + 1))
}

[[ $(uname -s) == Linux ]] || fail "environment host must run Linux"
[[ $(uname -m) == x86_64 ]] || fail "the pinned OSWorld VM requires x86_64"
command -v docker >/dev/null 2>&1 || fail "Docker CLI is missing"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is missing"

if command -v docker >/dev/null 2>&1; then
    docker info >/dev/null 2>&1 || fail "Docker daemon is not reachable"
fi
[[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]] || fail "/dev/kvm must be readable and writable"
[[ -f ${VM_PATH} && -r ${VM_PATH} ]] || fail "qcow2 is not readable: ${VM_PATH}"

if [[ -r ${VM_PATH} ]]; then
    observed_size=$(stat -c '%s' "${VM_PATH}")
    [[ ${observed_size} == "${EXPECTED_VM_SIZE}" ]] || \
        fail "qcow2 size mismatch: expected ${EXPECTED_VM_SIZE}, got ${observed_size}"
    observed_sha256=$(sha256sum "${VM_PATH}" | awk '{print $1}')
    [[ ${observed_sha256} == "${EXPECTED_VM_SHA256}" ]] || \
        fail "qcow2 SHA-256 mismatch: expected ${EXPECTED_VM_SHA256}, got ${observed_sha256}"
fi

check_free_space() {
    local check_path=$1
    local free_kib
    local required_kib=$((MIN_FREE_GIB * 1024 * 1024))

    while [[ ! -e ${check_path} && ${check_path} != / ]]; do
        check_path=$(dirname "${check_path}")
    done
    free_kib=$(df -Pk "${check_path}" | awk 'NR == 2 {print $4}')
    [[ ${free_kib} =~ ^[0-9]+$ && ${free_kib} -ge ${required_kib} ]] || \
        fail "less than ${MIN_FREE_GIB} GiB free at ${check_path}"
}

check_free_space "${VM_PATH}"
if command -v docker >/dev/null 2>&1 && docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null); then
    [[ -n ${docker_root} ]] && check_free_space "${docker_root}"
fi

if ((failures)); then
    printf 'environment_check=FAILED failures=%d\n' "${failures}" >&2
    exit 1
fi

printf 'environment_check=READY vm_path=%s vm_sha256=%s\n' "${VM_PATH}" "${EXPECTED_VM_SHA256}"
