#!/usr/bin/env bash
# Non-mutating checks for a local OSWorld Docker rollout host.

set -Eeuo pipefail

CHECK_PATH="${CHECK_PATH:-${HOME}}"
MIN_FREE_GIB="${MIN_FREE_GIB:-40}"
failures=0
warnings=0

ok() { printf 'ok: %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; warnings=$((warnings + 1)); }
fail() { printf 'error: %s\n' "$*" >&2; failures=$((failures + 1)); }

need_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        ok "$1=$(command -v "$1")"
    else
        fail "missing command: $1"
    fi
}

[[ "$(uname -s)" == "Linux" ]] || fail "OSWorld Docker rollouts require Linux"
[[ "$(uname -m)" == "x86_64" ]] || fail "the default OSWorld image and VM require x86_64"
for command in bash git curl unzip sha256sum docker python3 df stat; do
    need_cmd "${command}"
done

if command -v docker >/dev/null 2>&1; then
    docker info >/dev/null 2>&1 && ok "Docker daemon is reachable" || \
        fail "Docker daemon is not reachable by $(id -un)"
fi

if [[ -c /dev/kvm && -r /dev/kvm && -w /dev/kvm ]]; then
    ok "/dev/kvm is readable and writable"
else
    warn "/dev/kvm is unavailable; software emulation is supported but substantially slower"
fi

while [[ ! -e "${CHECK_PATH}" && "${CHECK_PATH}" != "/" ]]; do
    CHECK_PATH="$(dirname "${CHECK_PATH}")"
done
free_kib="$(df -Pk "${CHECK_PATH}" | awk 'NR==2 {print $4}')"
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if [[ "${free_kib}" =~ ^[0-9]+$ ]] && (( free_kib >= required_kib )); then
    ok "free space at ${CHECK_PATH} is at least ${MIN_FREE_GIB} GiB"
else
    fail "less than ${MIN_FREE_GIB} GiB free at ${CHECK_PATH}"
fi

if (ulimit -n 65536) 2>/dev/null; then
    ok "open-file limit can be raised to 65536"
else
    fail "cannot raise open-file limit to 65536"
fi

printf 'failures=%d warnings=%d\n' "${failures}" "${warnings}"
(( failures == 0 ))
