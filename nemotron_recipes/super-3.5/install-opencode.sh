#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Installs OpenCode inside a SWE-bench sandbox from your local copy (OPENCODE_DIR, mounted
# read-only at /opt/opencode) instead of downloading it, so it also works in task images
# without curl. Gym's OpenCode agent runs it before each task:
#
#   bash install-opencode.sh --binary <path>                              # Verified, Multilingual
#   bash install-opencode.sh --glibc-binary <path> --musl-binary <path>   # Pro (some images are Alpine)
#
# Same result as OpenCode's own installer with --binary: the binary in ~/.opencode/bin.

set -eu

glibc= musl=
while [ $# -gt 0 ]; do
    case "$1" in
        --binary | --glibc-binary) glibc=$2; shift 2 ;;
        --musl-binary) musl=$2; shift 2 ;;
        *) echo "install-opencode.sh: unknown option $1" >&2; exit 1 ;;
    esac
done

binary=$glibc
if [ -n "$musl" ] && { [ -f /etc/alpine-release ] || ldd --version 2>&1 | grep -qi musl; }; then
    binary=$musl
fi
[ -f "$binary" ] || { echo "install-opencode.sh: $binary not found; is OPENCODE_DIR set?" >&2; exit 1; }

mkdir -p "$HOME/.opencode/bin"
cp "$binary" "$HOME/.opencode/bin/opencode"
chmod 755 "$HOME/.opencode/bin/opencode"
echo "Installed OpenCode from $binary"
