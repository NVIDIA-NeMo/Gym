#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install a private Node.js plus the pinned Qwen Code package into a runtime directory inside the task sandbox, or
# reuse one that already reports the pinned version. The task's own Node, npm and PATH are untouched: Qwen Code runs
# as RUNTIME/bin/node RUNTIME/lib/node_modules/@qwen-code/qwen-code/cli-entry.js. The agent server uploads the
# Node.js build (or a complete runtime archive from build_runtime.sh); npm installs Qwen Code from the registry.
# Needs bash, flock, sha256sum, tar, gzip and mktemp, plus a glibc image (the agent server checks it first).
#
# usage: install_qwen_runtime.sh RUNTIME QWEN_VERSION NODE_SHA256 NODE_ARCHIVE RUNTIME_ARCHIVE
#   Exactly one of NODE_ARCHIVE and RUNTIME_ARCHIVE is nonempty.
set -euo pipefail
runtime=$1
qwen_version=$2
node_sha256=$3
node_archive=$4
runtime_archive=$5
cli=lib/node_modules/@qwen-code/qwen-code/cli-entry.js
# The task image's Node settings must not reach the private Node.
unset NODE_OPTIONS NODE_PATH

# QWEN_HOME keeps the version check from writing Qwen Code state under the task's HOME.
installed() { QWEN_HOME="$1/.version-home" "$1/bin/node" "$1/$cli" --version 2>/dev/null | tail -n 1; }

mkdir -p "$(dirname "$runtime")"
# Seeds sharing a sandbox install one at a time, so a ready runtime is never replaced while in use.
exec 9>"$runtime.lock"
flock 9
[ -f "$runtime/ready" ] && [ "$(installed "$runtime")" = "$qwen_version" ] && exit 0
staging=$(mktemp -d "$runtime.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
if [ -n "$runtime_archive" ]; then
    tar -xzf "$runtime_archive" -C "$staging"
else
    echo "$node_sha256  $node_archive" | sha256sum -c -
    tar -xzf "$node_archive" -C "$staging" --strip-components=1
    # npm's cache and logs stay in the staging directory, out of the task's HOME.
    HOME="$staging/.npm-home" npm_config_cache="$staging/.npm-cache" PATH="$staging/bin:$PATH" \
        "$staging/bin/node" "$staging/lib/node_modules/npm/bin/npm-cli.js" install --global --prefix "$staging" \
        --ignore-scripts --no-audit --no-fund "@qwen-code/qwen-code@${qwen_version}"
    rm -rf "$staging/.npm-home" "$staging/.npm-cache"
fi
test "$(installed "$staging")" = "$qwen_version"
touch "$staging/ready"
rm -rf "$runtime"
mv "$staging" "$runtime"
