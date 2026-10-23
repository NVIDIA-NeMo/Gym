#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
root=$1
version=$2
# The supervisor needs Python; install prerequisites inside the sandbox only.
set --
command -v python3 >/dev/null 2>&1 || set -- "$@" python3
command -v curl >/dev/null 2>&1 || set -- "$@" curl ca-certificates
command -v bash >/dev/null 2>&1 || set -- "$@" bash
command -v tar >/dev/null 2>&1 || set -- "$@" tar
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "Pool requires python3, bash, curl, and tar in the sandbox" >&2; exit 1; }
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    else
        echo "Preinstall python3, bash, curl and tar in the sandbox image" >&2
        exit 1
    fi
fi
mkdir -p "$root/bin"
curl --connect-timeout 30 --max-time 120 -fsSL https://downloads.poolside.ai/pool/install.sh -o "$root/install.sh"
POOL_INSTALL_ACCEPT_EULA=1 POOL_INSTALL_UPDATE_PATH=0 POOL_INSTALL_SPLASH=0 \
    POOL_INSTALL_DIR="$root/bin" sh "$root/install.sh" "$version"
installed=$("$root/bin/pool" --version)
[ "$installed" = "${version#v}" ] || { echo "Pool version mismatch: expected $version, got $installed" >&2; exit 1; }
