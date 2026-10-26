#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify or install a transferred Claude binary with bash and coreutils; no network access.
#
# usage: install_claude_runtime.sh --version V --runtime DIR --sha256 HEX [--check-only | --source PATH [--gzip]]
# Prints the version of the verified runtime. --check-only exits 3 when the runtime is absent or unhealthy.

set -euo pipefail
version="" runtime="" expected="" source="" check_only="" compressed="" temporary="" lock_dir=""
while (($#)); do
    case $1 in
        --version) version=$2 && shift 2 ;;
        --runtime) runtime=$2 && shift 2 ;;
        --sha256) expected=${2,,} && shift 2 ;;
        --source) source=$2 && shift 2 ;;
        --check-only) check_only=1 && shift ;;
        --gzip) compressed=1 && shift ;;
        *) echo "Unknown argument: $1" >&2 && exit 2 ;;
    esac
done
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Claude runtime requires an exact version" >&2; exit 2; }
[[ -n $expected ]] || { echo "A transferred Claude binary requires a trusted SHA-256" >&2; exit 2; }

# Print the version when the executable has the trusted bytes and reports the expected version.
verify() {
    local digest actual
    [[ -f $1 ]] || return 1
    read -r digest _ < <(sha256sum "$1")
    [[ $digest == "$expected" ]] || { echo "Claude executable SHA-256 mismatch" >&2; return 1; }
    actual=$(timeout 30 "$1" --version) || return 1
    [[ $actual == "$version "* ]] || { echo "Claude version mismatch: expected $version, got $actual" >&2; return 1; }
    echo "$actual"
}

trap 'rm -f -- "$temporary"; [[ -z $lock_dir ]] || rmdir -- "$lock_dir"' EXIT
mkdir -p -- "$(dirname -- "$runtime")"
# Concurrent sessions in one sandbox share the runtime; only one may replace it.
if command -v flock >/dev/null 2>&1; then
    exec 9>>"$runtime.lock"
    flock 9
else
    until mkdir -- "$runtime.lock.d" 2>/dev/null; do sleep 0.1; done
    lock_dir=$runtime.lock.d
fi

verify "$runtime/claude" 2>/dev/null && exit 0
[[ -z $check_only ]] || exit 3
[[ -n $source ]] || { echo "Claude runtime is absent; provide a transferred source binary" >&2; exit 1; }
mkdir -p -m 700 -- "$runtime"
temporary=$(mktemp "$runtime/.claude-XXXXXX")
if [[ -n $compressed ]]; then
    gzip -dc -- "$source" >"$temporary"
else
    cat -- "$source" >"$temporary"
fi
chmod 700 "$temporary"
verify "$temporary"
mv -f -- "$temporary" "$runtime/claude"
