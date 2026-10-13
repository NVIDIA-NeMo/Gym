#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; echo "Codex installer failed (exit $status) at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR

runtime=$1
codex_version=$2
node_version=22.19.0
[[ "$codex_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'An exact Codex version is required' >&2; exit 1; }
[ "$(uname -s)" = Linux ] || { echo 'Native Codex requires Linux' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 8), "Native Codex requires Python >=3.8"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Native Codex supports Linux x86_64/aarch64 sandboxes only' >&2; exit 1 ;;
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
  echo 'Native Codex requires glibc or musl; could not identify sandbox libc' >&2
  exit 1
fi

install_packages() {
  [ "$(id -u)" = 0 ] || { echo "Native Codex requires $*: preinstall them or use a root image." >&2; exit 1; }
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  else
    echo "Native Codex requires $*: preinstall them (automatic installation requires apt-get or apk)." >&2
    exit 1
  fi
}

# Serialize cache population for sessions borrowing the same sandbox/runtime.
if ! command -v flock >/dev/null 2>&1; then
  install_packages util-linux
fi
mkdir -p "$runtime"
exec 9>"$runtime/install.lock"
# BusyBox flock lacks -w; preserve the bounded wait using its nonblocking flag.
lock_deadline=$((SECONDS + 600))
until flock -n 9; do
  [ "$SECONDS" -lt "$lock_deadline" ] || { echo 'Timed out waiting for Codex runtime lock' >&2; exit 1; }
  sleep 1
done
mkdir -p "$runtime/home" "$runtime/cache"
export HOME="$runtime/home" XDG_CACHE_HOME="$runtime/cache" npm_config_cache="$runtime/cache/npm"
if [ ! -f "$runtime/ready" ]; then
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
  PATH="$runtime/node/bin:$PATH" "$runtime/node/bin/node" \
    "$runtime/node/lib/node_modules/npm/bin/npm-cli.js" install \
    --prefix "$runtime/codex" --no-audit --no-fund "@openai/codex@${codex_version}"
fi
actual=$("$runtime/node/bin/node" "$runtime/codex/node_modules/@openai/codex/bin/codex.js" --version)
[ "$actual" = "codex-cli $codex_version" ] || { echo "Codex version mismatch: $actual" >&2; exit 1; }
touch "$runtime/ready"
printf '%s\n' "$actual"
