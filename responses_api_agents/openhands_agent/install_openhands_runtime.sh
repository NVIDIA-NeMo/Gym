#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the private OpenHands runtime in a task sandbox. The Agent Server uploads this script, a
# checksum-verified standalone CPython archive and a hash-locked requirements file, then runs:
#
#   install_openhands_runtime.sh probe
#   install_openhands_runtime.sh check   <runtime> <fingerprint>
#   install_openhands_runtime.sh install <runtime> <fingerprint> <python sha256> <requirements sha256>
#
# The runtime lives outside the task repository and the task's home. The task's own interpreters and
# libraries are never touched. pip downloads the locked packages, so the sandbox needs access to PyPI.
set -Eeuo pipefail
umask 022

fail() { printf 'OpenHands runtime: %s\n' "$*" >&2; exit 1; }
require() {
  local tool
  for tool in "$@"; do
    command -v "$tool" >/dev/null 2>&1 || fail "requires '$tool'; use a task image with it preinstalled"
  done
}
probe() {
  require uname
  [[ "$(uname -s)" = Linux ]] || fail 'requires a Linux task sandbox'
  local arch
  arch=$(uname -m)
  case "$arch" in x86_64|aarch64) ;; *) fail "unsupported architecture: $arch" ;; esac
  # The standalone CPython builds need glibc; musl images are not supported.
  if command -v getconf >/dev/null 2>&1 && getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
    printf '%s/gnu\n' "$arch"
  elif command -v ldd >/dev/null 2>&1 && [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
    fail 'musl task images are unsupported'
  else
    printf '%s/gnu\n' "$arch"
  fi
}

mode=${1:-}
case "$mode" in
  probe) [[ $# = 1 ]] || fail 'usage: probe'; probe; exit 0 ;;
  check) [[ $# = 3 ]] || fail 'usage: check <runtime> <fingerprint>' ;;
  install) [[ $# = 5 ]] || fail 'usage: install <runtime> <fingerprint> <python sha256> <requirements sha256>' ;;
  *) fail 'expected probe, check, or install' ;;
esac
runtime=$2
fingerprint=$3
[[ "$runtime" = /* && "$runtime" != */ ]] || fail 'runtime must be an absolute directory path without a trailing slash'
[[ "$fingerprint" =~ ^[0-9a-f]{64}$ ]] || fail 'fingerprint must be a SHA-256 digest'
interpreter=interp/bin/openhands-gym
# Imports the runner needs, at the pinned version; a directory alone does not prove the runtime works.
health_code='import openhands.sdk, openhands.tools.file_editor, openhands.tools.task_tracker, openhands.tools.terminal, openhands.tools.preset.default; assert openhands.sdk.__version__ == "1.59.0", openhands.sdk.__version__'
health() { OPENHANDS_SUPPRESS_BANNER=1 "$1/$interpreter" -I -B -c "$health_code" >/dev/null 9>&-; }
ready() { [[ -f "$runtime/ready" && "$(< "$runtime/ready")" = "$fingerprint" ]] && health "$runtime"; }
if [[ "$mode" = check ]]; then
  if ready; then exit 0; fi
  printf 'OpenHands runtime is absent or unhealthy: %s\n' "$runtime" >&2
  exit 1
fi

script=${BASH_SOURCE[0]}
[[ "$script" = /*.sh ]] || fail 'invoke the uploaded installer by its absolute .sh path'
scratch=${script%.sh}
python_sha256=$4
requirements_sha256=$5
[[ "$python_sha256" =~ ^[0-9a-f]{64}$ && "$requirements_sha256" =~ ^[0-9a-f]{64}$ ]] \
  || fail 'checksums must be SHA-256 digests'
require bash tar gzip sha256sum flock sleep mktemp ln mv rm tee mkdir
probe >/dev/null
mkdir -p "$scratch/home" "$scratch/cache" "$scratch/tmp"
# Installer state stays in the scratch directory, away from the task's home.
export HOME="$scratch/home" XDG_CACHE_HOME="$scratch/cache" TMPDIR="$scratch/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE
cd "$scratch"
exec > >(tee -a "$scratch/install.log") 2>&1
publish=
on_exit() {
  local status=$?
  trap - EXIT
  if [[ -n "$publish" && -L "$publish" ]]; then rm -f -- "$publish"; fi
  if [[ "$status" != 0 ]]; then printf 'OpenHands runtime install failed (exit %s); diagnostics: %s\n' "$status" "$scratch" >&2; fi
  exit "$status"
}
trap on_exit EXIT

parent=${runtime%/*}
[[ -n "$parent" ]] || parent=/
name=${runtime##*/}
mkdir -p "$parent"
# The lock lives beside the runtime, so concurrent seeds in one sandbox share one installation.
exec 9>"$parent/.$name.install.lock"
lock_deadline=$((SECONDS + 900))
until flock -n 9; do
  [[ "$SECONDS" -lt "$lock_deadline" ]] || fail 'timed out waiting for the runtime installation lock'
  sleep 1
done
if ready; then exit 0; fi
[[ ! -e "$runtime" || -L "$runtime" ]] || fail "publication path is a real directory: $runtime"
# Nothing is extracted or installed until both uploads match the digests the Agent Server verified.
printf '%s  %s\n%s  %s\n' "$python_sha256" "$scratch/python.tar.gz" "$requirements_sha256" "$scratch/requirements.txt" \
  | sha256sum -c -
mkdir -p "$parent/.builds"
generation=$(mktemp -d "$parent/.builds/$name.XXXXXX")
mkdir -p "$generation/interp"
tar -xzf "$scratch/python.tar.gz" -C "$generation/interp" --strip-components=1
# Every package is pinned with hashes, so the dependency set cannot change between episodes.
"$generation/interp/bin/python3" -I -B -m pip --isolated install --no-cache-dir --no-deps --require-hashes \
  --no-warn-script-location -r "$scratch/requirements.txt" 9>&-
# The runner runs under this name, so 'pkill python3' or 'pkill -f python' in a task cannot stop it.
ln -s "$(readlink "$generation/interp/bin/python3" || echo python3)" "$generation/$interpreter"
health "$generation"
printf '%s\n' "$fingerprint" > "$generation/ready"
publish="$parent/.${generation##*/}.publish"
ln -s "$generation" "$publish"
# -T replaces the symlink itself instead of moving into its target directory.
mv -Tf -- "$publish" "$runtime" || fail 'atomic publication requires mv with -T support'
publish=
printf 'OpenHands runtime ready: %s\n' "$runtime"
