#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
umask 077

fail() { printf 'NOOA runtime: %s\n' "$*" >&2; exit 1; }
require() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || fail "requires '$tool'; use a task image with it preinstalled"
  done
}
probe() {
  require uname
  [[ "$(uname -s)" = Linux ]] || fail 'requires a Linux task sandbox'
  local arch libc loader
  arch=$(uname -m)
  case "$arch" in x86_64|aarch64) ;; *) fail "unsupported architecture: $arch" ;; esac
  if command -v getconf >/dev/null 2>&1 && getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    libc=gnu
  elif command -v ldd >/dev/null 2>&1 && [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
    libc=musl
  else
    libc=
    for loader in /lib/ld-musl-*.so.1; do
      [[ ! -e "$loader" ]] || libc=musl
    done
    [[ -n "$libc" ]] || fail 'cannot identify glibc or musl; use a supported task image'
  fi
  printf '%s/%s\n' "$arch" "$libc"
}

mode=${1:-}
case "$mode" in
  probe) [[ $# = 1 ]] || fail 'usage: probe'; probe; exit 0 ;;
  check) [[ $# = 4 ]] || fail 'usage: check <runtime> <fingerprint> <health code>' ;;
  install) [[ $# = 6 || $# = 7 ]] || fail 'usage: install <runtime> <fingerprint> <source sha256> <python sha256> <health code> [wheels sha256]' ;;
  *) fail 'expected probe, check, or install' ;;
esac
runtime=$2
fingerprint=$3
[[ "$runtime" = /* && "$runtime" != */ && "${runtime##*/}" != . && "${runtime##*/}" != .. ]] \
  || fail 'runtime must be an absolute directory path without a trailing slash'
[[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || fail 'fingerprint must be a SHA-256 digest'
script=${BASH_SOURCE[0]}
[[ "$script" = /*.sh ]] || fail 'invoke the uploaded installer by its absolute .sh path'
scratch=${script%.sh}
require mkdir tee
mkdir -p "$scratch/home" "$scratch/cache" "$scratch/tmp"
export HOME="$scratch/home" XDG_CACHE_HOME="$scratch/cache" TMPDIR="$scratch/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
cd "$scratch"
exec > >(tee -a "$scratch/$mode.log") 2>&1
publish=
on_exit() {
  local status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$scratch/$mode.exit-code"
  if [[ -n "$publish" && -L "$publish" ]]; then rm -f -- "$publish"; fi
  if [[ "$status" != 0 ]]; then printf 'NOOA runtime %s failed; diagnostics: %s\n' "$mode" "$scratch" >&2; fi
  exit "$status"
}
trap on_exit EXIT
trap 'status=$?; printf "NOOA runtime command failed (exit %s), line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2; exit "$status"' ERR
health_code=${4:-}
[[ "$mode" != install ]] || health_code=$6
health() { "$1/python/bin/python3" -I -B -c "$health_code" 9>&-; }
ready() {
  [[ -f "$runtime/ready" && "$(< "$runtime/ready")" = "$fingerprint" ]] && health "$runtime"
}
if [[ "$mode" = check ]]; then
  if ready; then exit 0; fi
  printf 'NOOA runtime is absent or unhealthy: %s\n' "$runtime" >&2
  exit 1
fi

require bash tar gzip sha256sum flock sleep mktemp ln mv rm
probe >/dev/null
source_sha256=$4
python_sha256=$5
wheels_sha256=${7:-}
[[ -z "$wheels_sha256" || "$wheels_sha256" =~ ^[0-9a-f]{64}$ ]] || fail 'wheel checksum must be a SHA-256 digest'
[[ "$source_sha256" =~ ^[0-9a-f]{64}$ && "$python_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || fail 'source and Python checksums must be SHA-256 digests'
parent=${runtime%/*}
[[ -n "$parent" ]] || parent=/
name=${runtime##*/}
mkdir -p "$parent"
# Keep the lock outside the published runtime; healthy generations remain read-only.
exec 9>"$parent/.$name.install.lock"
lock_deadline=$((SECONDS + 600))
until flock -n 9; do
  [[ "$SECONDS" -lt "$lock_deadline" ]] || fail 'timed out waiting for the runtime installation lock'
  sleep 1
done
if ready; then exit 0; fi
[[ ! -e "$runtime" || -L "$runtime" ]] \
  || fail 'publication path is a real directory; choose a fresh runtime fingerprint without removing it'
# No archive is extracted until every expected digest has been verified.
printf '%s  %s\n' "$source_sha256" "$scratch/source.tar.gz" "$python_sha256" "$scratch/python.tar.gz" \
  | sha256sum -c -
if [[ -n "$wheels_sha256" ]]; then
  printf '%s  %s\n' "$wheels_sha256" "$scratch/wheels.tar.gz" | sha256sum -c -
fi
mkdir -p "$parent/.builds"
generation=$(mktemp -d "$parent/.builds/$name.XXXXXX")
mkdir -p "$generation/python" "$generation/source/cache"
tar -xzf "$scratch/python.tar.gz" -C "$generation/python" --strip-components=1
tar -xzf "$scratch/source.tar.gz" -C "$generation/source"
# Installed console scripts reference this permanent path, so never rename the generation.
# The installer alone owns the lock; an orphaned pip process cannot retain it after a kill.
if [[ -n "$wheels_sha256" ]]; then
  mkdir -p "$generation/wheels"
  tar -xzf "$scratch/wheels.tar.gz" -C "$generation/wheels"
  # Every package, including Gym and pinned NOOA, was built outside the task.
  # Explicit wheel paths and --no-deps prevent direct-URL metadata from causing downloads.
  "$generation/python/bin/python3" -I -B -m pip --isolated install --no-cache-dir \
    --no-index --no-deps --only-binary=:all: "$generation/wheels/"*.whl 9>&-
  # Offline health checks and workers must use LiteLLM's bundled model metadata.
  # A private-prefix startup hook also applies to later isolated (-I) interpreters,
  # without changing the task's Python or process environment.
  site_packages="$generation/python/lib/python3.13/site-packages"
  mkdir -p "$site_packages"
  printf '%s\n' 'import os; os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"' \
    > "$site_packages/nooa_offline_runtime.pth"
else
  "$generation/python/bin/python3" -I -B -m pip --isolated install --no-cache-dir \
    "$generation/source" -r "$generation/source/runtime-requirements.txt" 9>&-
fi
health "$generation"
printf '%s\n' "$fingerprint" > "$generation/ready"
publish="$parent/.${generation##*/}.publish"
ln -s "$generation" "$publish"
# -T replaces the symlink itself, rather than moving into its target directory.
mv -Tf -- "$publish" "$runtime" || fail 'atomic publication requires mv with -T support'
publish=
printf 'NOOA runtime ready: %s\n' "$runtime"
