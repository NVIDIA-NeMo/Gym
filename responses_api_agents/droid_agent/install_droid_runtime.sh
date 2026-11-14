#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify or install Droid from a transferred npm platform package with bash, coreutils, gzip and tar; no network.
#
# usage: install_droid_runtime.sh --version V --runtime DIR --sha256 HEX [--check-only | --source PACKAGE]
# --sha256 names the package the runtime must come from. --check-only exits 3 when the runtime is absent or unhealthy.

set -euo pipefail
version="" runtime="" expected="" source="" check_only="" staging="" queue="" ticket=""
while (($#)); do
    case $1 in
        --version) version=$2 && shift 2 ;;
        --runtime) runtime=$2 && shift 2 ;;
        --sha256) expected=$(tr A-F a-f <<<"$2") && shift 2 ;;
        --source) source=$2 && shift 2 ;;
        --check-only) check_only=1 && shift ;;
        *) echo "Unknown argument: $1" >&2 && exit 2 ;;
    esac
done
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Droid runtime requires an exact version" >&2; exit 2; }
[[ $expected =~ ^[0-9a-f]{64}$ ]] || { echo "A transferred Droid package requires a trusted SHA-256" >&2; exit 2; }

# Succeed when DIR holds a binary from the expected package, unchanged since installation, that reports the version.
# Droid writes state under its home on every start, so the check gives it a throwaway one.
verify() {
    local package binary digest actual home
    [[ -x $1/droid && -f $1/receipt ]] || return 1
    read -r package binary <"$1/receipt" || return 1
    [[ $package == "$expected" ]] || { echo "Droid runtime comes from another package" >&2; return 1; }
    read -r digest _ < <(sha256sum "$1/droid")
    [[ $digest == "$binary" ]] || { echo "Droid executable changed since installation" >&2; return 1; }
    home=$(mktemp -d)
    actual=$(FACTORY_HOME_OVERRIDE=$home timeout 60 "$1/droid" --version) || { rm -rf -- "$home"; return 1; }
    rm -rf -- "$home"
    [[ $actual == "$version" ]] || { echo "Droid version mismatch: expected $version, got $actual" >&2; return 1; }
}

# A process as "PID START" from /proc, so a reused PID never matches its former owner; PID alone
# where /proc is missing. A zombie or dead process has none: it can never finish its turn.
process_identity() {
    local stat=
    IFS= read -r -d '' stat 2>/dev/null <"/proc/$1/stat" || true
    if [[ $stat == *') '* ]]; then
        set -- "$1" ${stat##*) }
        case $2 in Z | X) return ;; esac
        echo "$1 ${21}"
    elif kill -0 "$1" 2>/dev/null; then
        echo "$1"
    fi
}

trap 'rm -rf -- "$staging"; [[ -z $ticket ]] || : >"$queue/$ticket.done"' EXIT
mkdir -p -- "$(dirname -- "$runtime")"
# Concurrent sessions in one sandbox share the runtime; only one may replace it.
if command -v flock >/dev/null 2>&1; then
    exec 9>>"$runtime.lock"
    flock 9
else
    # Without flock, installers queue: each takes the next number in the queue directory as a
    # symlink naming its owner, then waits until every lower number is done or its owner is gone.
    # Numbers are never removed or reused, so no installer, however it exits, can release another.
    queue=$runtime.lock.d
    mkdir -p "$queue"
    owner=$(process_identity $$)
    while :; do
        last=0
        for entry in "$queue"/*; do
            number=${entry##*/}
            number=${number%.done}
            if [[ $number =~ ^[0-9]+$ ]] && ((number > last)); then last=$number; fi
        done
        # Hold the number only once its link is ours: on exit the trap marks this ticket done.
        ln -s "$owner" "$queue/$((last + 1))" 2>/dev/null && { ticket=$((last + 1)); break; }
    done
    until
        ahead=
        for ((number = 1; number < ticket; number++)); do
            [[ -e $queue/$number.done ]] && continue
            holder=$(readlink "$queue/$number" 2>/dev/null) || continue
            [[ $(process_identity "${holder%% *}") != "$holder" ]] || { ahead=1; break; }
        done
        [[ -z $ahead ]]
    do
        sleep 0.1
    done
fi

verify "$runtime" 2>/dev/null && exit 0
[[ -z $check_only ]] || exit 3
[[ -n $source ]] || { echo "Droid runtime is absent; provide a transferred package" >&2; exit 1; }
read -r digest _ < <(sha256sum "$source")
[[ $digest == "$expected" ]] || { echo "Droid package SHA-256 mismatch" >&2; exit 1; }
staging=$(mktemp -d "$runtime.staging-XXXXXX")
tar -xzf "$source" -C "$staging" package/bin/droid
mv -- "$staging/package/bin/droid" "$staging/droid"
rm -rf -- "$staging/package"
chmod 700 "$staging" && chmod 755 "$staging/droid"
read -r binary _ < <(sha256sum "$staging/droid")
echo "$expected $binary" >"$staging/receipt"
verify "$staging"
rm -rf -- "$runtime"
mv -- "$staging" "$runtime"
staging=""
rm -f -- "$source"
