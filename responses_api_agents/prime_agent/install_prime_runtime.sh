#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install a private Node.js plus the pinned Prime Agent package, both uploaded by the agent server, into a runtime
# directory inside the task sandbox, or reuse one that already reports the pinned version. npm installs the package's
# dependencies from the registry. The task's own Node, npm and PATH are untouched: Prime Agent runs as
# RUNTIME/bin/node RUNTIME/lib/node_modules/prime-agent/dist/bundle/cli.js. Needs bash, sha256sum, tar, gzip, mktemp
# and flock, plus a glibc image.
#
# Prime Agent's only built-in tool is an IPython kernel. RUNTIME/kernel is a private Python, also uploaded, with the
# packages Prime Agent's own kernel bootstrap installs (pinned here) and the package's prime-agent-runtime, installed
# by the uploaded uv from the package index.
#
# usage: install_prime_runtime.sh RUNTIME VERSION NODE_ARCHIVE NODE_SHA256 PACKAGE PACKAGE_SHA256 PYTHON_ARCHIVE
#        PYTHON_SHA256 UV_ARCHIVE
set -euo pipefail
runtime=$1
version=$2
node_archive=$3
node_sha256=$4
package=$5
package_sha256=$6
python_archive=$7
python_sha256=$8
uv_archive=$9
cli=lib/node_modules/prime-agent/dist/bundle/cli.js
kernel_packages=(
    ipykernel==7.4.0 dill==0.4.1 nest-asyncio==1.6.0 tyro==1.0.16 requests==2.34.2 httpx==0.28.1 pyyaml==6.0.3
    tomli==2.5.0 python-dotenv==1.2.4 pandas==3.1.0 numpy==2.5.4 scipy==1.18.1 beautifulsoup4==4.15.0 lxml==6.1.3
    pydantic==2.14.2
)

# A private agent directory keeps the version check from writing Prime Agent state under the task's HOME. Prime Agent
# 0.7.0 prints its version on stderr.
installed() {
    PRIME_AGENT_CODING_AGENT_DIR="$1/.version-check" PI_SKIP_VERSION_CHECK=1 \
        "$1/bin/node" "$1/$cli" --version 2>&1 | tail -n 1 | sed 's/^v//'
}
# The imports Prime Agent checks before it uses a configured kernel Python.
kernel_ready() {
    "$1/kernel/bin/python3" -c "import ipykernel, rlm, dill, requests, httpx, yaml, tomli, dotenv, pandas, numpy, \
scipy, bs4, lxml, pydantic, tyro"
}

ready() { [ -f "$runtime/ready" ] && [ "$(installed "$runtime")" = "$version" ] && kernel_ready "$runtime" 2>/dev/null; }
ready && exit 0
# The official Node.js Linux builds need glibc.
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    echo "Prime Agent's Node.js runtime needs a glibc sandbox image" >&2
    exit 1
fi
mkdir -p "$(dirname "$runtime")"
# Sessions sharing a filesystem install one at a time. A session that waited reuses the runtime the winner published.
exec 9>"$runtime.lock"
flock -x 9
ready && exit 0
staging=$(mktemp -d "$runtime.install.XXXXXX")
trap 'rm -rf "$staging"' EXIT
echo "$node_sha256  $node_archive" | sha256sum -c -
echo "$package_sha256  $package" | sha256sum -c -
echo "$python_sha256  $python_archive" | sha256sum -c -
tar -xzf "$node_archive" -C "$staging" --strip-components=1
# npm's cache and logs stay in the staging directory, out of the task's HOME. The package's optional postinstall only
# pre-downloads tools that Prime Agent otherwise fetches on first use.
HOME="$staging/.npm-home" npm_config_cache="$staging/.npm-cache" PATH="$staging/bin:$PATH" \
    "$staging/bin/node" "$staging/lib/node_modules/npm/bin/npm-cli.js" install --global --prefix "$staging" \
    --ignore-scripts --no-audit --no-fund "$package"
rm -rf "$staging/.npm-home" "$staging/.npm-cache"
test "$(installed "$staging")" = "$version"
mkdir "$staging/kernel" "$staging/uv"
tar -xzf "$python_archive" -C "$staging/kernel" --strip-components=1
tar -xzf "$uv_archive" -C "$staging/uv" --strip-components=1
UV_CACHE_DIR="$staging/.uv-cache" "$staging/uv/uv" pip install --python "$staging/kernel/bin/python3" \
    --break-system-packages "${kernel_packages[@]}" "$staging/lib/node_modules/prime-agent/dist/prime-agent-runtime"
rm -rf "$staging/uv" "$staging/.uv-cache"
kernel_ready "$staging"
touch "$staging/ready"
rm -rf "$runtime.old"
if [ -d "$runtime" ]; then mv "$runtime" "$runtime.old"; fi
mv "$staging" "$runtime"
rm -rf "$runtime.old"
