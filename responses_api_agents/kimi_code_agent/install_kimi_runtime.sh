#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned Kimi Code release binary, uploaded by the agent server, into a private runtime directory inside
# the task sandbox, or reuse one that already reports the pinned version. Only Kimi Code is installed; the task's own
# tools are untouched. Needs bash, sha256sum, timeout and mktemp. Exits 3 on an image without glibc.
#
# usage: install_kimi_runtime.sh RUNTIME VERSION TARGET SHA256 BINARY
set -euo pipefail
runtime=$1
version=$2
sha256=$4
binary=$5
# Native assets are extracted here instead of the task's HOME.
export KIMI_CODE_CACHE_DIR="$runtime/cache"

[ -x "$runtime/kimi" ] && echo "$sha256  $runtime/kimi" | sha256sum -c --status - \
    && [ "$(timeout 60 "$runtime/kimi" --version 2>/dev/null)" = "$version" ] && exit 0
# Kimi Code ships glibc builds only.
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    echo "Kimi Code $version needs a glibc sandbox image" >&2
    exit 3
fi
mkdir -p "$runtime"
staging=$(mktemp -d "$runtime/.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
echo "$sha256  $binary" | sha256sum -c -
cp "$binary" "$staging/kimi"
chmod 0755 "$staging/kimi"
test "$(timeout 60 "$staging/kimi" --version)" = "$version"
# A rename within one directory is atomic, so a concurrent seed never sees a partial binary.
mv -f "$staging/kimi" "$runtime/kimi"
