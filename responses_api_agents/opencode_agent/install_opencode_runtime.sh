#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; printf "OpenCode setup failed: command=%s exit=%s\n" "$BASH_COMMAND" "$status" >&2; exit "$status"' ERR
runtime=$1
version=$2
source_binary=${3:-}
source_sha256=${4:-}
path_prefix=${5:-}
if [ -n "$path_prefix" ]; then export PATH="$path_prefix:$PATH"; fi

[ "$(uname -s)" = Linux ] || { echo 'OpenCode sandbox execution requires a Linux task image' >&2; exit 1; }
# A process as "PID START" from /proc, so a reused PID never matches its former owner; PID alone
# where /proc is missing. A zombie or dead process has none: it can never finish its turn.
process_identity() {
  local stat=
  IFS= read -r -d '' stat 2>/dev/null <"/proc/$1/stat" || true
  if [[ $stat == *') '* ]]; then
    set -- "$1" ${stat##*) }
    case $2 in Z | X) return ;; esac
    echo "$1 ${21}"
  elif kill -0 "$1" 2>/dev/null; then
    echo "$1"
  fi
}
staged=
ticket=
# However the installer exits, remove a partly staged binary and leave the installer queue.
trap '[ -z "$staged" ] || rm -f "$staged"; [ -z "$ticket" ] || : >"$queue/$ticket.done"' EXIT
# Serialize preparation of the version-scoped cache in a shared task sandbox.
if command -v flock >/dev/null 2>&1; then
  # The kernel releases the lock when the installer exits, however it exits.
  exec 9>"$runtime.lock"
  flock 9
else
  # Without flock, installers queue: each takes the next number in the queue directory as a
  # symlink naming its owner, then waits until every lower number is done or its owner is gone.
  # Numbers are never removed or reused, so no installer, however it exits, can release another.
  queue=$runtime.lock.d
  mkdir -p "$queue"
  owner=$(process_identity $$)
  while :; do
    last=0
    for entry in "$queue"/*; do
      number=${entry##*/}
      number=${number%.done}
      if [[ $number =~ ^[0-9]+$ ]] && ((number > last)); then last=$number; fi
    done
    ln -s "$owner" "$queue/$((last + 1))" 2>/dev/null && { ticket=$((last + 1)); break; }
  done
  until
    ahead=
    for ((number = 1; number < ticket; number++)); do
      [[ -e $queue/$number.done ]] && continue
      holder=$(readlink "$queue/$number" 2>/dev/null) || continue
      [[ $(process_identity "${holder%% *}") != "$holder" ]] || { ahead=1; break; }
    done
    [[ -z $ahead ]]
  do
    sleep 0.1
  done
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
  command -v sha256sum >/dev/null 2>&1 || { echo 'Verifying a preinstalled OpenCode binary requires sha256sum' >&2; exit 1; }
  [ -f "$source_binary" ] || { echo 'OpenCode executable digest mismatch' >&2; exit 1; }
  staged=$(mktemp "$runtime/.opencode-XXXXXX")
  cp "$source_binary" "$staged"
  digest=$(sha256sum "$staged")
  if [ "${digest%% *}" != "$source_sha256" ]; then
    echo 'OpenCode executable digest mismatch' >&2
    exit 1
  fi
  chmod 755 "$staged"
  # The probe must exit successfully as well as print the exact version.
  if ! actual=$("$staged" --version) || [ "$actual" != "$version" ]; then
    echo 'OpenCode executable version mismatch' >&2
    exit 1
  fi
  mv -f "$staged" "$runtime/opencode"
  staged=
fi
cached=false
# A cached binary must also exit successfully, not only print the version.
if [ -x "$runtime/opencode" ] && actual=$("$runtime/opencode" --version) && [ "$actual" = "$version" ]; then
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
