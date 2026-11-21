#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Extract only the standalone CLI from its pinned wheel; do not alter task Python or packages.
set -euo pipefail
runtime=$1
version=$2
override=${3:-}
test "$version" = 0.11.0 || { echo "Unsupported Relay version: $version" >&2; exit 1; }
verify() {
    local actual
    actual=$("$1" --version)
    test "$actual" = "nemo-relay $version" || { echo "Relay version mismatch: $actual" >&2; return 1; }
}
if [[ -n $override ]]; then
    executable=$(command -v -- "$override") || { echo "Relay executable not found: $override" >&2; exit 1; }
    verify "$executable"
    printf '%s\n' "$executable"
    exit 0
fi
directory="$runtime/relay/$version"
mkdir -p "$directory"
lock="$directory/install.lock"
for ((attempt=0; ; attempt++)); do
    if mkdir "$lock" 2>/dev/null; then break; fi
    test "$attempt" -lt 120 || { echo 'Timed out waiting for Relay installation' >&2; exit 1; }
    sleep 1
done
trap 'rm -rf "$lock"' EXIT
if [[ ! -x $directory/nemo-relay ]]; then
    test "$(uname -s)" = Linux || { echo 'Relay sandbox installation requires Linux' >&2; exit 1; }
    architecture=$(uname -m)
    case "$architecture" in x86_64|aarch64) ;; *) echo "Unsupported Relay architecture: $architecture" >&2; exit 1 ;; esac
    libc=glibc
    compgen -G '/lib/ld-musl-*.so.1' >/dev/null && libc=musl
    case "$libc/$architecture" in
        glibc/aarch64) package=4f/7d/bac56c9cb3ba32aafa8771f11ef0ca2ddca3498a84d3443edf85719597d7; tag=manylinux_2_17_aarch64; digest=ab7a02f1804c14cc5ba8f47b8f59959d4db2944a12378abcf72908fe870b6069 ;;
        glibc/x86_64) package=9e/11/22c9d48f80577a2f06f9e7848b8323286321a784dc497a50427a0bd39dc5; tag=manylinux_2_17_x86_64; digest=a3fb596ae155ffae1f881df66c7425a20c9ea23807cc611f3c43f773e1b2f7b3 ;;
        musl/aarch64) package=b0/0d/3da199267f7384d18f4b5d6836b8930ddf289e99794789c54eafc20c989c; tag=musllinux_1_2_aarch64; digest=3a7ad67c594dc931102244925d02a643c2107f590ff68b21bf268f13bc09b18d ;;
        musl/x86_64) package=68/1f/77b59ec0b00103d45d562b35a413f17ec32b2f4277d0390f98b886e62a71; tag=musllinux_1_2_x86_64; digest=032531b965c7e1875dc5bc6b7993ebf9656ff0495111038f2e4def92a2418d93 ;;
    esac
    if ! command -v unzip >/dev/null 2>&1; then
        test "$(id -u)" = 0 || { echo 'Preinstall unzip or set relay.executable' >&2; exit 1; }
        if command -v apk >/dev/null 2>&1; then apk add --no-cache unzip >&2
        elif command -v apt-get >/dev/null 2>&1; then apt-get update >&2 && apt-get install -y --no-install-recommends unzip >&2
        else echo 'Preinstall unzip or set relay.executable' >&2; exit 1
        fi
    fi
    archive="$lock/relay.whl"
    curl -fsSL --retry 3 --max-time 120 \
        "https://files.pythonhosted.org/packages/$package/nemo_relay_cli_bin-$version-py3-none-$tag.whl" -o "$archive"
    printf '%s  %s\n' "$digest" "$archive" | sha256sum -c - >&2
    unzip -p "$archive" "nemo_relay_cli_bin-$version.data/scripts/nemo-relay" > "$lock/nemo-relay"
    chmod 700 "$lock/nemo-relay"
    verify "$lock/nemo-relay"
    mv "$lock/nemo-relay" "$directory/nemo-relay"
fi
verify "$directory/nemo-relay"
printf '%s\n' "$directory/nemo-relay"
