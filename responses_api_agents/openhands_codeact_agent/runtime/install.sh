#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 || $1 != --archive ]]; then
    echo 'usage: install.sh --archive /absolute/path/runtime.tar.gz' >&2
    exit 2
fi
archive=$2
prefix=/opt/openhands-codeact
expected=$(awk 'NR == 1 {print $1}' "$archive.sha256")
[[ $expected =~ ^[[:xdigit:]]{64}$ ]] || { echo 'Invalid runtime checksum' >&2; exit 1; }
actual=$(sha256sum "$archive" | awk '{print $1}')
[[ $actual == "$expected" ]] || { echo 'Runtime checksum mismatch' >&2; exit 1; }
if [[ -f $prefix/.installed-sha256 ]]; then
    [[ $(cat "$prefix/.installed-sha256") == "$expected" ]] || {
        echo 'Installed runtime does not match this archive' >&2
        exit 1
    }
    "$prefix/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13)'
    exit 0
fi
[[ ! -e $prefix ]] || { echo 'Refusing an existing incomplete runtime directory' >&2; exit 1; }
mkdir -p "$prefix"
tar -xzf "$archive" -C "$prefix"
sed -i "s|^home = .*|home = $prefix/python/bin|" "$prefix/venv/pyvenv.cfg"
sed -i "s|^executable = .*|executable = $prefix/python/bin/python3.13|" "$prefix/venv/pyvenv.cfg"
"$prefix/venv/bin/python" -c 'from openhands.agenthub import nemo_gym_client; from nemo_gym.client import request'
printf '%s\n' "$expected" > "$prefix/.installed-sha256"
