#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; printf "OpenCode setup failed: command=%s exit=%s\n" "$BASH_COMMAND" "$status" >&2; exit "$status"' ERR
runtime=$1
version=$2
source_binary=${3:-}
source_sha256=${4:-}
python_executable=${5:-python3}
path_prefix=${6:-}
if [ -n "$path_prefix" ]; then export PATH="$path_prefix:$PATH"; fi

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "OpenCode sandbox execution requires $1 in the task image" >&2; exit 1; }
}
require "$python_executable"
"$python_executable" -c 'import sys; assert sys.platform == "linux" and sys.version_info >= (3, 8), "OpenCode sandbox execution requires Linux and Python >=3.8 in the task image"'
# Serialize preparation of the version-scoped cache in a shared task sandbox.
# The lock descriptor remains open across exec and releases on installer exit.
if [ "${NG_OPENCODE_INSTALL_LOCKED:-0}" != 1 ]; then
  exec "$python_executable" -c 'import fcntl,os,sys; fd=os.open(sys.argv[1]+".lock",os.O_CREAT|os.O_RDWR,0o600); fcntl.flock(fd,fcntl.LOCK_EX); os.set_inheritable(fd,True); os.environ["NG_OPENCODE_INSTALL_LOCKED"]="1"; os.execvp("bash",["bash",*sys.argv[2:]])' "$runtime" "$0" "$@"
fi
case "$(uname -m)" in
  x86_64) arch=x64-baseline ;;
  aarch64) arch=arm64 ;;
  *) echo 'OpenCode sandbox execution supports Linux x86_64/aarch64 task images' >&2; exit 1 ;;
esac
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  :
elif [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
  arch="${arch}-musl"
else
  echo 'OpenCode sandbox execution requires glibc or musl; could not identify the sandbox libc' >&2
  exit 1
fi
mkdir -p "$runtime/home" "$runtime/cache" "$runtime/data" "$runtime/config"
export HOME="$runtime/home" XDG_CACHE_HOME="$runtime/cache" XDG_DATA_HOME="$runtime/data" XDG_CONFIG_HOME="$runtime/config"
if [ -n "$source_binary" ]; then
  [ -n "$source_sha256" ] || { echo 'A preinstalled OpenCode binary requires its SHA-256' >&2; exit 1; }
  # Publish a verified new inode so active sessions can keep executing the old one.
  "$python_executable" -I - "$source_binary" "$source_sha256" "$runtime/opencode" "$version" <<'PY'
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

source, expected, target, version = sys.argv[1:]
if not Path(source).is_file():
    raise ValueError("OpenCode executable digest mismatch")
fd, staged = tempfile.mkstemp(prefix=".opencode-", dir=Path(target).parent)
os.close(fd)
try:
    shutil.copyfile(source, staged)
    if hashlib.sha256(Path(staged).read_bytes()).hexdigest() != expected:
        raise ValueError("OpenCode executable digest mismatch")
    os.chmod(staged, 0o755)
    if subprocess.check_output([staged, "--version"], text=True).rstrip("\n") != version:
        raise ValueError("OpenCode executable version mismatch")
    os.replace(staged, target)
finally:
    Path(staged).unlink(missing_ok=True)
PY
fi
cached=false
if [ -x "$runtime/opencode" ] && [ "$("$runtime/opencode" --version)" = "$version" ]; then
  cached=true
fi
missing=()
if [ "$cached" = false ]; then
  for tool in curl tar gzip; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then
    missing+=(ca-certificates)
  fi
fi
if [ "${#missing[@]}" -gt 0 ]; then
    if [ "$(id -u)" != 0 ]; then
      echo "OpenCode sandbox execution needs ${missing[*]}; preinstall them in the task image (automatic installation requires root)" >&2
      exit 1
    fi
    if command -v apk >/dev/null 2>&1; then
      apk add --no-cache "${missing[@]}"
    elif command -v apt-get >/dev/null 2>&1; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
    else
      echo "OpenCode sandbox execution needs ${missing[*]}; preinstall them in the task image (automatic installation requires apt-get or apk)" >&2
      exit 1
    fi
fi
# Stock OpenCode can download ripgrep itself. A system copy avoids per-session
# downloads, but must not make cached runtimes or non-root/RHEL images unusable.
if ! command -v rg >/dev/null 2>&1; then
  if [ "$(id -u)" = 0 ]; then
    if command -v apk >/dev/null 2>&1; then
      apk add --no-cache ripgrep || true
    elif command -v apt-get >/dev/null 2>&1; then
      (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ripgrep) || true
    fi
  fi
  command -v rg >/dev/null 2>&1 || echo 'Warning: optional ripgrep is unavailable; OpenCode can download it on demand' >&2
fi
if [ "$cached" = true ]; then
  exit 0
fi
curl -fSL --retry 3 "https://github.com/anomalyco/opencode/releases/download/v${version}/opencode-linux-${arch}.tar.gz" -o "$runtime/opencode.tar.gz"
tar -xzf "$runtime/opencode.tar.gz" -C "$runtime" opencode
chmod +x "$runtime/opencode"
actual=$("$runtime/opencode" --version)
if [ "$actual" != "$version" ]; then
  echo "OpenCode version mismatch: requested $version, found $actual" >&2
  exit 1
fi
