#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eu
root=$1
version=$2
accept_eula=${3:-false}
[ "$accept_eula" = true ] || { echo "Accept https://poolside.ai/eula with accept_pool_eula=true before installing Pool" >&2; exit 1; }
# The supervisor needs Python; install prerequisites inside the sandbox only.
set --
command -v python3 >/dev/null 2>&1 || set -- "$@" python3
command -v curl >/dev/null 2>&1 || set -- "$@" curl ca-certificates
command -v bash >/dev/null 2>&1 || set -- "$@" bash
if [ "$#" -gt 0 ]; then
    [ "$(id -u)" = 0 ] || { echo "Pool requires python3, bash, and curl in the sandbox" >&2; exit 1; }
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    elif command -v apk >/dev/null 2>&1; then
        apk add --no-cache "$@"
    else
        echo "Preinstall python3, bash and curl in the sandbox image" >&2
        exit 1
    fi
fi
python3 -c 'import sys; assert sys.version_info >= (3, 8), "Pool requires Python >=3.8 in the sandbox"'
python3 -I "$root/install_pool.py" "$root" "$version" "$accept_eula"
