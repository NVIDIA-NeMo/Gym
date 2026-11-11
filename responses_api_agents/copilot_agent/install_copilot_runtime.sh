#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install the pinned Copilot CLI in a task sandbox, outside the task repository, with bash and coreutils.
#
# usage: install_copilot_runtime.sh RUNTIME VERSION PACKAGE TARBALL SHA512
#
# The Agent Server selects PACKAGE, the npm platform package for this sandbox, and reads the TARBALL URL and its
# SHA-512 (hex) from the registry. The standalone binary unpacks its bundled package into $HOME/.cache on first run,
# so the installer runs it once with a private HOME and records that directory for COPILOT_CLI_DIST_DIR; the task's
# HOME is never written. Prints the runtime record as JSON on the last line of stdout.

set -euo pipefail
runtime=$1 version=$2 package=$3 tarball=$4 sha512=$5
[[ $runtime == /* ]] || { echo "Copilot runtime requires an absolute path: $runtime" >&2; exit 2; }
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Copilot runtime requires an exact version" >&2; exit 2; }
[[ $package =~ ^copilot-linux(musl)?-(x64|arm64)$ ]] || { echo "Unsupported Copilot package: $package" >&2; exit 2; }
[[ $tarball =~ ^https?:// ]] || { echo "Copilot package URL must be http(s): $tarball" >&2; exit 2; }
[[ $sha512 =~ ^[0-9a-f]{128}$ ]] || { echo "Copilot package requires a hex SHA-512 digest" >&2; exit 2; }
staging=$runtime.partial-$$
trap 'rm -rf -- "$staging"' EXIT

install_packages() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "Copilot requires $*: preinstall these packages in the task image (automatic installation requires root)." >&2
    exit 1
  fi
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  else
    echo "Copilot requires $*: preinstall these packages in the task image (automatic installation requires apt-get or apk)." >&2
    exit 1
  fi
}

# Succeed when the binary reports the pinned version; with a third argument, as activation runs it.
healthy() {
  local output
  output=$(env -i HOME="$2" PATH=/usr/bin:/bin COPILOT_AUTO_UPDATE=false ${3:+"COPILOT_CLI_DIST_DIR=$3"} "$1" --version) \
    || return 1
  [[ $output == *"$version"* ]]
}

# The directory the first run unpacked; a usable runtime holds exactly one for the pinned version.
dist_dir() {
  local found=() dir
  for dir in "$1"/home/.cache/copilot/pkg/*/"$version"; do
    if [[ -d $dir ]]; then found+=("$dir"); fi
  done
  if ((${#found[@]} != 1)); then
    echo "Copilot did not unpack exactly one $version package under $1/home/.cache/copilot/pkg" >&2
    return 1
  fi
  echo "${found[0]}"
}

# One JSON line; the paths are the adapter's and the binary's, which need no escaping.
record() {
  [[ $1 != *[\"\\]* ]] || { echo "Copilot runtime path cannot be recorded as JSON: $1" >&2; return 1; }
  printf '{"binary": "%s", "dist_dir": "%s", "package": "%s", "version": "%s", "sha512": "%s"}\n' \
    "$runtime/copilot" "$1" "$package" "$version" "$sha512"
}

# A directory alone does not prove a runtime: it needs the ready marker and a passing health check.
reuse() {
  local dist
  [[ -f $runtime/ready ]] && dist=$(dist_dir "$runtime" 2>/dev/null) \
    && healthy "$runtime/copilot" "$runtime/home" "$dist" 2>/dev/null && record "$dist"
}

mkdir -p "$(dirname "$runtime")"
# Concurrent seeds in one sandbox share one installation.
locked=0
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$runtime.lock"
  flock -w 60 9 || { echo "Timed out waiting for Copilot runtime lock: $runtime.lock" >&2; exit 1; }
  locked=1
  # No other installer is staging now; remove what interrupted ones left.
  rm -rf -- "$runtime".partial-*
fi
reuse && exit 0

for tool in sha512sum tar; do
  command -v "$tool" >/dev/null 2>&1 || { echo "Copilot installation requires $tool in the task image" >&2; exit 1; }
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  install_packages curl ca-certificates
fi
mkdir -p "$staging/home"
archive=$staging/package.tgz
if command -v curl >/dev/null 2>&1; then
  curl -fsSL --retry 3 -o "$archive" "$tarball"
else
  wget -q -O "$archive" "$tarball"
fi
printf '%s  %s\n' "$sha512" "$archive" | sha512sum -c - >&2 \
  || { echo "Copilot package $package@$version does not match the registry's SHA-512" >&2; exit 1; }
tar -xzf "$archive" -C "$staging" package/copilot
mv "$staging/package/copilot" "$staging/copilot"
rm -rf -- "$staging/package" "$archive"
chmod 755 "$staging/copilot"
healthy "$staging/copilot" "$staging/home" || { echo "Copilot $version failed its health check" >&2; exit 1; }
dist_dir "$staging" >/dev/null

# Without flock, another installer may have published a runtime meanwhile.
reuse && exit 0
if [ "$locked" = 1 ]; then
  rm -rf -- "$runtime"
  mv -T -- "$staging" "$runtime"
elif ! mv -T -- "$staging" "$runtime" 2>/dev/null; then
  # Never remove a runtime another installer may have published since the readiness check.
  reuse && exit 0
  echo "Cannot replace stale Copilot runtime safely without flock: $runtime" >&2
  exit 1
fi
dist=$(dist_dir "$runtime")
healthy "$runtime/copilot" "$runtime/home" "$dist" \
  || { echo "Copilot runtime failed its health check after installation" >&2; exit 1; }
record "$dist" >"$runtime/runtime.json"
touch "$runtime/ready"
cat "$runtime/runtime.json"
