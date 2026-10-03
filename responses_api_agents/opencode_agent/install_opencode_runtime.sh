#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; printf "OpenCode setup failed: command=%s exit=%s\n" "$BASH_COMMAND" "$status" >&2; exit "$status"' ERR
runtime=$1
version=$2
staged_binary=${3:-}
installer=${4:-}
musl_binary=${5:-}

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "Native OpenCode requires $1 in the task image" >&2; exit 1; }
}
require python3
python3 -c 'import sys; assert sys.platform == "linux" and sys.version_info >= (3, 8), "Native OpenCode requires Linux and Python >=3.8 in the task image"'
# Serialize preparation of the version-scoped cache in a shared task sandbox.
# The lock descriptor remains open across exec and releases on installer exit.
if [ "${NG_OPENCODE_INSTALL_LOCKED:-0}" != 1 ]; then
  exec python3 -c 'import fcntl,os,sys; fd=os.open(sys.argv[1]+".lock",os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX); os.set_inheritable(fd,True); os.environ["NG_OPENCODE_INSTALL_LOCKED"]="1"; os.execvp("bash",["bash",*sys.argv[2:]])' "$runtime" "$0" "$@"
fi
case "$(uname -m)" in
  x86_64) arch=x64-baseline ;;
  aarch64) arch=arm64 ;;
  *) echo 'Native OpenCode supports Linux x86_64/aarch64 task images' >&2; exit 1 ;;
esac
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  :
elif [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
  arch="${arch}-musl"
else
  echo 'Native OpenCode requires glibc or musl; could not identify the sandbox libc' >&2
  exit 1
fi
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/data" "$runtime/config"
export HOME="$runtime/home" XDG_CACHE_HOME="$runtime/cache" XDG_DATA_HOME="$runtime/data" XDG_CONFIG_HOME="$runtime/config"
if [ -x "$runtime/opencode" ] && [ "$("$runtime/opencode" --version)" = "$version" ]; then
  exit 0
fi
if [ -n "$staged_binary" ]; then
  if [ -n "$installer" ]; then
    if [ -n "$musl_binary" ]; then
      bash "$installer" --glibc-binary "$staged_binary" --musl-binary "$musl_binary"
    else
      bash "$installer" --binary "$staged_binary"
    fi
    cp "$HOME/.opencode/bin/opencode" "$runtime/opencode"
  else
    cp "$staged_binary" "$runtime/opencode"
  fi
else
  if ! command -v curl >/dev/null 2>&1 || [ ! -s /etc/ssl/certs/ca-certificates.crt ]; then
    if [ "$(id -u)" != 0 ]; then
      echo 'Native OpenCode needs curl and ca-certificates; preinstall them in the task image (automatic installation requires root)' >&2
      exit 1
    fi
    if command -v apk >/dev/null 2>&1; then
      apk add --no-cache curl ca-certificates
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates
    else
      echo 'Native OpenCode needs curl and ca-certificates; preinstall them in the task image (automatic installation requires apt-get or apk)' >&2
      exit 1
    fi
  fi
  require tar
  require gzip
  curl -fSL --retry 3 "https://github.com/anomalyco/opencode/releases/download/v${version}/opencode-linux-${arch}.tar.gz" -o "$runtime/opencode.tar.gz"
  tar -xzf "$runtime/opencode.tar.gz" -C "$runtime" opencode
fi
chmod +x "$runtime/opencode"
actual=$("$runtime/opencode" --version)
if [ "$actual" != "$version" ]; then
  echo "OpenCode version mismatch: requested $version, found $actual" >&2
  exit 1
fi
