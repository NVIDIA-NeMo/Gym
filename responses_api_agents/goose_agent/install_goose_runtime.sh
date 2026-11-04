#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned static Goose release, uploaded by the agent server, into a private runtime directory inside the
# task sandbox, or reuse one that already reports the pinned version. Only Goose is installed; the task's own tools are
# untouched. Needs bash, sha256sum, tar, gzip and mktemp.
#
# usage: install_goose_runtime.sh RUNTIME VERSION ARCH SHA256 ARCHIVE
set -euo pipefail
runtime=$1
version=$2
sha256=$4
archive=$5

installed_version() { "$1" --version 2>/dev/null | tr -d ' '; }

[ -x "$runtime/goose" ] && [ "$(installed_version "$runtime/goose")" = "$version" ] && exit 0
mkdir -p "$runtime"
staging=$(mktemp -d "$runtime/.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
echo "$sha256  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$staging"
test "$(installed_version "$staging/goose")" = "$version"
# A rename within one directory is atomic, so a concurrent seed never sees a partial binary.
mv -f "$staging/goose" "$runtime/goose"
