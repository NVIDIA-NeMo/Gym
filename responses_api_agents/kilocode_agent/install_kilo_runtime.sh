#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; echo "Kilo installer failed (exit $status) at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR

# Install a pinned Node and Kilo CLI into a private per-session directory inside the task sandbox.
# Task files and the task's PATH are left alone; packages are added only when download tools are missing.
runtime=$1
kilo_version=$2
node_version=22.19.0
[[ "$kilo_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'An exact Kilo version is required' >&2; exit 1; }
[ "$(uname -s)" = Linux ] || { echo 'Kilo sandbox sessions require Linux' >&2; exit 1; }
# The session runner is a standalone Python script.
python3 -c 'import sys; assert sys.version_info >= (3, 8), "Kilo sandbox sessions require Python >=3.8"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Kilo supports Linux x86_64/aarch64 sandboxes only' >&2; exit 1 ;;
esac
platform="linux-${arch}"
node_dist="https://nodejs.org/dist/v${node_version}"
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  :
elif [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
  [ "$arch" = x64 ] || { echo 'Pinned Node musl build supports x86_64 only' >&2; exit 1; }
  platform=linux-x64-musl
  node_dist="https://unofficial-builds.nodejs.org/download/release/v${node_version}"
else
  echo 'Kilo requires glibc or musl; could not identify sandbox libc' >&2
  exit 1
fi

install_packages() {
  [ "$(id -u)" = 0 ] || { echo "Kilo requires $*: preinstall them or use a root image." >&2; exit 1; }
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  else
    echo "Kilo requires $*: preinstall them (automatic installation requires apt-get or apk)." >&2
    exit 1
  fi
}

# gzip and sha256sum -c also work with Alpine's BusyBox.
missing=0
for command in curl tar gzip sha256sum awk; do
  if ! command -v "$command" >/dev/null 2>&1; then missing=1; fi
done
if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then missing=1; fi
if [ "$missing" = 1 ]; then
  install_packages curl ca-certificates tar gzip coreutils gawk
fi
if [ "$platform" = linux-x64-musl ] && ! python3 -c 'import ctypes; ctypes.CDLL("libstdc++.so.6")' 2>/dev/null; then
  install_packages libstdc++
fi

mkdir -p "$runtime/home" "$runtime/cache"
export HOME="$runtime/home" XDG_CACHE_HOME="$runtime/cache" npm_config_cache="$runtime/cache/npm"
cd "$runtime"
archive="node-v${node_version}-${platform}.tar.gz"
curl -fsSL --retry 3 "${node_dist}/${archive}" -o "$archive"
curl -fsSL --retry 3 "${node_dist}/SHASUMS256.txt" -o SHASUMS256.txt
awk -v archive="$archive" '$2 == archive' SHASUMS256.txt | sha256sum -c -
mkdir -p node
tar -xzf "$archive" --strip-components=1 -C node
if [ "$platform" = linux-x64-musl ] && ! "$runtime/node/bin/node" --version; then
  # Old Alpine's C++ runtime lacks symbols required by Node 22. Only the private
  # Node binary sees this pinned library; task libraries and LD_LIBRARY_PATH stay intact.
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

# Kilo's postinstall selects the native binary for this CPU/libc and runs `node`, so Node is on PATH here only.
PATH="$runtime/node/bin:$PATH" "$runtime/node/bin/node" \
  "$runtime/node/lib/node_modules/npm/bin/npm-cli.js" install \
  --prefix "$runtime/kilo" --no-audit --no-fund "@kilocode/cli@${kilo_version}"
actual=$("$runtime/node/bin/node" "$runtime/kilo/node_modules/@kilocode/cli/bin/kilo" --version)
grep -qxF -- "$kilo_version" <<<"$actual" || { echo "Kilo version mismatch: $actual" >&2; exit 1; }
printf '%s\n' "$kilo_version"
