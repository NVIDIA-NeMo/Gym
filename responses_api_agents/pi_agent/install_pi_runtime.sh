#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Install only Pi's runtime inside an existing Resources-owned task sandbox.
# Do not prepare task dependencies or replace runtimes on the task's PATH.
runtime=$1
pi_version=$2
node_version=22.19.0
test "$(uname -s)" = Linux
getconf GNU_LIBC_VERSION >/dev/null
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Native Pi requires Python >=3.9"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Native Pi supports Linux x86_64/aarch64 glibc sandboxes only' >&2; exit 1 ;;
esac

if [ ! -f "$runtime/ready" ]; then
  # Slim task images may omit the downloader needed to bootstrap Node.
  if ! command -v curl >/dev/null 2>&1; then
    if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
      echo 'Native Pi requires curl: preinstall curl and ca-certificates in the task image (automatic installation requires root and apt-get).' >&2
      exit 1
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates
  fi
  mkdir -p "$runtime"
  cd "$runtime"
  archive="node-v${node_version}-linux-${arch}.tar.xz"
  curl -fsSL --retry 3 "https://nodejs.org/dist/v${node_version}/${archive}" -o "$archive"
  curl -fsSL --retry 3 "https://nodejs.org/dist/v${node_version}/SHASUMS256.txt" -o SHASUMS256.txt
  awk -v archive="$archive" '$2 == archive' SHASUMS256.txt | sha256sum --check --strict -
  mkdir -p node
  tar -xJf "$archive" --strip-components=1 -C node
  PATH="$runtime/node/bin:$PATH" "$runtime/node/bin/node" \
    "$runtime/node/lib/node_modules/npm/bin/npm-cli.js" install \
    --prefix "$runtime/pi" --no-audit --no-fund \
    "@earendil-works/pi-coding-agent@${pi_version}"
  "$runtime/node/bin/node" "$runtime/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" --version
  touch "$runtime/ready"
fi
