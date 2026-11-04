#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install a private Node.js plus the pinned Prime Agent package, both uploaded by the agent server, into a runtime
# directory inside the task sandbox, or reuse one that already reports the pinned version. npm installs the package's
# dependencies from the registry. The task's own Node, npm and PATH are untouched: Prime Agent runs as
# RUNTIME/bin/node RUNTIME/lib/node_modules/prime-agent/dist/bundle/cli.js. Needs bash, sha256sum, tar, gzip and
# mktemp, plus a glibc image.
#
# usage: install_prime_runtime.sh RUNTIME VERSION NODE_ARCHIVE NODE_SHA256 PACKAGE PACKAGE_SHA256
set -euo pipefail
runtime=$1
version=$2
node_archive=$3
node_sha256=$4
package=$5
package_sha256=$6
cli=lib/node_modules/prime-agent/dist/bundle/cli.js

# A private agent directory keeps the version check from writing Prime Agent state under the task's HOME. Prime Agent
# 0.7.0 prints its version on stderr.
installed() {
    PRIME_AGENT_CODING_AGENT_DIR="$1/.version-check" PI_SKIP_VERSION_CHECK=1 \
        "$1/bin/node" "$1/$cli" --version 2>&1 | tail -n 1 | sed 's/^v//'
}

[ -f "$runtime/ready" ] && [ "$(installed "$runtime")" = "$version" ] && exit 0
# The official Node.js Linux builds need glibc.
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    echo "Prime Agent's Node.js runtime needs a glibc sandbox image" >&2
    exit 1
fi
mkdir -p "$(dirname "$runtime")"
staging=$(mktemp -d "$runtime.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
echo "$node_sha256  $node_archive" | sha256sum -c -
echo "$package_sha256  $package" | sha256sum -c -
tar -xzf "$node_archive" -C "$staging" --strip-components=1
# npm's cache and logs stay in the staging directory, out of the task's HOME. The package's optional postinstall only
# pre-downloads tools that Prime Agent otherwise fetches on first use.
HOME="$staging/.npm-home" npm_config_cache="$staging/.npm-cache" PATH="$staging/bin:$PATH" \
    "$staging/bin/node" "$staging/lib/node_modules/npm/bin/npm-cli.js" install --global --prefix "$staging" \
    --ignore-scripts --no-audit --no-fund "$package"
rm -rf "$staging/.npm-home" "$staging/.npm-cache"
test "$(installed "$staging")" = "$version"
touch "$staging/ready"
# Publish with one rename; a concurrent seed that lost the race keeps the published runtime.
rm -rf "$runtime.old"
[ -d "$runtime" ] && mv "$runtime" "$runtime.old"
mv "$staging" "$runtime" || [ -f "$runtime/ready" ]
rm -rf "$runtime.old"
