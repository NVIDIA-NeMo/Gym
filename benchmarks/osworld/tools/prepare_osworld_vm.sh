#!/usr/bin/env bash
# Download and verify the x86_64 OSWorld VM used by the Docker provider.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_ROOT="${GYM_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
VM_DIR="${VM_DIR:-${GYM_ROOT}/docker_vm_data}"
VM_PATH="${VM_PATH:-${VM_DIR}/Ubuntu.qcow2}"
VM_ARCHIVE="${VM_ARCHIVE:-${VM_DIR}/Ubuntu.qcow2.zip}"
VM_URL="${VM_URL:-https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip}"
VM_SHA256="${VM_SHA256:-6bf667a852b3c307f61d9f09c42559351f45e0607e428b4997becf534cf4d313}"
VM_SIZE_BYTES="${VM_SIZE_BYTES:-24460197888}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

die() { echo "ERROR: $*" >&2; exit 1; }
verify_vm() {
    [[ -f "${VM_PATH}" ]] || return 1
    actual_size="$(stat -c%s "${VM_PATH}")"
    [[ "${actual_size}" == "${VM_SIZE_BYTES}" ]] || \
        die "unexpected VM size at ${VM_PATH}: ${actual_size}"
    actual_sha="$(sha256sum "${VM_PATH}" | awk '{print $1}')"
    [[ "${actual_sha}" == "${VM_SHA256}" ]] || \
        die "unexpected VM SHA-256 at ${VM_PATH}: ${actual_sha}"
}

for command in curl unzip sha256sum stat; do
    command -v "${command}" >/dev/null 2>&1 || die "missing command: ${command}"
done

if [[ -e "${VM_PATH}" ]]; then
    verify_vm
    echo "vm=${VM_PATH} sha256=${VM_SHA256} status=already-present"
    exit 0
fi

mkdir -p "${VM_DIR}"
free_kib="$(df -Pk "${VM_DIR}" | awk 'NR==2 {print $4}')"
(( free_kib >= 35 * 1024 * 1024 )) || die "at least 35 GiB free is required in ${VM_DIR}"

curl -L --fail --retry 20 --retry-all-errors --continue-at - \
    --output "${VM_ARCHIVE}" "${VM_URL}"
unzip -n "${VM_ARCHIVE}" -d "${VM_DIR}"
verify_vm

if [[ "${KEEP_ARCHIVE}" != "1" ]]; then
    rm -f "${VM_ARCHIVE}"
fi

cat <<EOF
vm=${VM_PATH}
sha256=${VM_SHA256}
size_bytes=${VM_SIZE_BYTES}
source=${VM_URL}
status=verified
EOF
