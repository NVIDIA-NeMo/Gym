#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build the runtime archive the agent uploads into each sandbox: a checksum-verified Node.js
# build for the sandbox architecture plus Qwen Code, as bin/node and bin/qwen.
# Needs curl, tar, sha256sum (or shasum) and npm >= 10 on the build host, which may be any platform:
# npm installs Qwen Code's optional native packages for the target (linux, glibc, ARCH).
#
# usage: build_runtime.sh OUTPUT.tar.gz [QWEN_VERSION] [x64|arm64]
set -euo pipefail

out="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
qwen_version=${2:-0.25.1}
arch=${3:-x64}
node_version=22.23.2
node="node-v${node_version}-linux-${arch}"
# sha256 of the .tar.xz builds, from https://nodejs.org/dist/v22.23.2/SHASUMS256.txt.
case "$arch" in
  x64) node_sha256=d60acfe00a2932254bb0ad20e01b0d74397a0875595de719654b214f4b03f307 ;; # pragma: allowlist secret
  arm64) node_sha256=fff4078c5def658577f92c88db7db3bc0072924bfb93fe52c1e744a54e94abb8 ;; # pragma: allowlist secret
  *) echo "unsupported arch: $arch" >&2; exit 1 ;;
esac

# The unpacked runtime stays next to the archive for inspection.
work="${out%.tar.gz}.build"
mkdir -p "$work"
cd "$work"
curl -fsSLO "https://nodejs.org/dist/v${node_version}/${node}.tar.xz"
echo "${node_sha256}  ${node}.tar.xz" | { sha256sum -c - 2>/dev/null || shasum -a 256 -c -; }

mkdir -p runtime
tar xJf "${node}.tar.xz" -C runtime --strip-components=1
npm install --global --prefix runtime --ignore-scripts --no-audit --no-fund --os linux --cpu "$arch" --libc glibc \
  "@qwen-code/qwen-code@${qwen_version}"
test -e runtime/bin/qwen
# Node headers and the bundled npm are not needed at run time.
tar czf "$out" --no-xattrs -C runtime --exclude ./include --exclude ./share --exclude ./bin/npm --exclude ./bin/npx \
  --exclude ./bin/corepack --exclude ./lib/node_modules/npm --exclude ./lib/node_modules/corepack .
echo "wrote $out (Node ${node_version} linux-${arch}, Qwen Code ${qwen_version})"
