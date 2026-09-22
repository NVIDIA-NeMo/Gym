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
getconf GNU_LIBC_VERSION >/dev/null
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Native Codex requires Python >=3.9"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Native Codex supports Linux x86_64/aarch64 glibc sandboxes only' >&2; exit 1 ;;
esac

# Serialize cache population for sessions borrowing the same sandbox/runtime.
if ! command -v flock >/dev/null 2>&1; then
  if [ "$(id -u)" != 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    echo 'Native Codex requires flock (util-linux); preinstall it or use a root/apt-get image.' >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends util-linux
fi
mkdir -p "$runtime"
exec 9>"$runtime/install.lock"
flock -w 600 9
mkdir -p "$runtime/home" "$runtime/cache"
export HOME="$runtime/home" XDG_CACHE_HOME="$runtime/cache" npm_config_cache="$runtime/cache/npm"
if [ ! -f "$runtime/ready" ]; then
  # Slim Debian-family images can bootstrap prerequisites; other images need them preinstalled.
  missing=0
  for command in curl tar xz sha256sum awk; do
    if ! command -v "$command" >/dev/null 2>&1; then missing=1; fi
  done
  if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then missing=1; fi
  if [ "$missing" = 1 ]; then
    if [ "$(id -u)" != 0 ] || ! command -v apt-get >/dev/null 2>&1; then
      echo 'Native Codex needs curl, CA certificates, tar, xz, sha256sum, awk; preinstall them (automatic installation requires root and apt-get).' >&2
      exit 1
    fi
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates tar xz-utils coreutils gawk
  fi

  cd "$runtime"
  archive="node-v${node_version}-linux-${arch}.tar.xz"
  curl -fsSL --retry 3 "https://nodejs.org/dist/v${node_version}/${archive}" -o "$archive"
  curl -fsSL --retry 3 "https://nodejs.org/dist/v${node_version}/SHASUMS256.txt" -o SHASUMS256.txt
  awk -v archive="$archive" '$2 == archive' SHASUMS256.txt | sha256sum --check --strict -
  mkdir -p node
  tar -xJf "$archive" --strip-components=1 -C node
  PATH="$runtime/node/bin:$PATH" "$runtime/node/bin/node" \
    "$runtime/node/lib/node_modules/npm/bin/npm-cli.js" install \
    --prefix "$runtime/codex" --no-audit --no-fund "@openai/codex@${codex_version}"
fi
actual=$("$runtime/node/bin/node" "$runtime/codex/node_modules/@openai/codex/bin/codex.js" --version)
[ "$actual" = "codex-cli $codex_version" ] || { echo "Codex version mismatch: $actual" >&2; exit 1; }
touch "$runtime/ready"
printf '%s\n' "$actual"
