#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Install only Pi's runtime inside an existing Resources-owned task sandbox.
# Do not prepare task dependencies or replace runtimes on the task's PATH.
runtime=$1
pi_version=$2
node_version=22.19.0

install_packages() {
  [ "$#" -gt 0 ] || return 0
  if [ "$(id -u)" -ne 0 ]; then
    echo "Pi requires $*: preinstall these packages in the task image (automatic installation requires root)." >&2
    exit 1
  fi
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  else
    echo "Pi requires $*: preinstall these packages in the task image (automatic installation requires apt-get or apk)." >&2
    exit 1
  fi
}

test "$(uname -s)" = Linux
if ! command -v python3 >/dev/null 2>&1; then
  install_packages python3
fi
python3 -c 'import sys; assert sys.version_info >= (3, 8), "Pi requires Python >=3.8"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Pi supports Linux x86_64/aarch64 sandboxes only' >&2; exit 1 ;;
esac
platform="linux-${arch}"
node_dist="https://nodejs.org/dist/v${node_version}"
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  :
elif [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
  if [ "$arch" != x64 ]; then
    echo "Pi's pinned Node ${node_version} musl build supports x86_64 only" >&2
    exit 1
  fi
  platform="linux-x64-musl"
  # Same binary source used by nodejs/docker-node's Alpine images.
  node_dist="https://unofficial-builds.nodejs.org/download/release/v${node_version}"
else
  echo 'Pi requires glibc or musl; could not identify the sandbox libc' >&2
  exit 1
fi

if [ ! -f "$runtime/ready" ]; then
  packages=()
  if ! command -v curl >/dev/null 2>&1; then
    packages+=(curl ca-certificates)
  fi
  if [ "$platform" = linux-x64-musl ] && ! python3 -c 'import ctypes; ctypes.CDLL("libstdc++.so.6")' 2>/dev/null; then
    packages+=(libstdc++)
  fi
  install_packages "${packages[@]}"
  mkdir -p "$runtime"
  cd "$runtime"
  # gzip and sha256sum -c work with Alpine's BusyBox; xz and --strict may not.
  archive="node-v${node_version}-${platform}.tar.gz"
  curl -fsSL --retry 3 "${node_dist}/${archive}" -o "$archive"
  curl -fsSL --retry 3 "${node_dist}/SHASUMS256.txt" -o SHASUMS256.txt
  awk -v archive="$archive" '$2 == archive' SHASUMS256.txt | sha256sum -c -
  mkdir -p node
  tar -xzf "$archive" --strip-components=1 -C node
  if [ "$platform" = linux-x64-musl ] && ! "$runtime/node/bin/node" --version; then
    # Older Alpine images have libstdc++, but lack symbols needed by Node 22.
    # Give only this Node binary a private library; leave task libraries and env intact.
    command -v patchelf >/dev/null 2>&1 || install_packages patchelf
    curl -fsSL --retry 3 \
      'https://dl-cdn.alpinelinux.org/alpine/v3.19/main/x86_64/libstdc++-13.2.1_git20231014-r0.apk' \
      -o libstdc++.apk
    echo '3cf66a7164240ef590106496d3c75f486bac46cba9cf2198c0c3b318c53ad027  libstdc++.apk' | sha256sum -c -
    mkdir -p libstdcpp
    tar -xzf libstdc++.apk -C libstdcpp usr/lib
    patchelf --set-rpath '$ORIGIN/../../libstdcpp/usr/lib' "$runtime/node/bin/node"
  fi
  "$runtime/node/bin/node" --version
  PATH="$runtime/node/bin:$PATH" "$runtime/node/bin/node" \
    "$runtime/node/lib/node_modules/npm/bin/npm-cli.js" install \
    --prefix "$runtime/pi" --ignore-scripts --no-audit --no-fund \
    "@earendil-works/pi-coding-agent@${pi_version}"
  "$runtime/node/bin/node" "$runtime/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js" --version
  touch "$runtime/ready"
fi
