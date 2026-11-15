#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned static Goose release, uploaded by the agent server, into a private runtime directory inside the
# task sandbox. Every seed verifies the archive and publishes it again, so a binary already at the runtime path is never
# trusted. Only Goose is installed. The task's own tools are untouched. Needs bash, sha256sum, tar, gzip, mktemp and
# timeout.
#
# usage: install_goose_runtime.sh RUNTIME VERSION ARCH SHA256 ARCHIVE
set -euo pipefail
runtime=$1
version=$2
sha256=$4
archive=$5

mkdir -p "$runtime"
staging=$(mktemp -d "$runtime/.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
# Goose sets up logging under GOOSE_PATH_ROOT even for --version, so keep it out of the task's HOME.
export GOOSE_PATH_ROOT="$staging/root"
echo "$sha256  $archive" | sha256sum -c -
tar -xzf "$archive" -C "$staging"
test "$(timeout 30 "$staging/goose" --version 2>/dev/null | tr -d ' ')" = "$version"
# A rename within one directory is atomic, so a concurrent seed never sees a partial binary.
mv -f "$staging/goose" "$runtime/goose"
