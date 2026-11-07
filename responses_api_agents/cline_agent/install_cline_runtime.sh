#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
trap 'status=$?; echo "Cline installer failed (exit $status) at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR

# Install the pinned Cline CLI into a private runtime directory inside the task sandbox.
# Cline ships a self-contained glibc binary per platform in @cline/cli-linux-<arch>; Node is not needed.
# Task files and the task's PATH are left alone; packages are added only when download tools are missing.
#
# usage: install_cline_runtime.sh RUNTIME VERSION SHA512_X64 SHA512_ARM64 [BINARY SHA512_BINARY]
# The SHA-512 digests (hex) are those of the npm tarballs the adapter pins for VERSION.
# With BINARY, the Agent Server verified that tarball and uploaded only its binary; no network is needed here.
runtime=$1
version=$2
sha512_x64=$3
sha512_arm64=$4
source_binary=${5:-}
binary_sha512=${6:-}
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo 'An exact Cline version is required' >&2; exit 1; }
[ "$(uname -s)" = Linux ] || { echo 'Cline sandbox sessions require Linux' >&2; exit 1; }
case "$(uname -m)" in
  x86_64) arch=x64 expected=$sha512_x64 ;;
  aarch64) arch=arm64 expected=$sha512_arm64 ;;
  *) echo 'Cline supports Linux x86_64/aarch64 sandboxes only' >&2; exit 1 ;;
esac
[[ "$expected" =~ ^[0-9a-f]{128}$ ]] || { echo "No pinned SHA-512 for Cline $version on $arch" >&2; exit 1; }
if ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  echo "Cline $version ships only glibc Linux binaries; this sandbox has another libc" >&2
  exit 1
fi

# The runtime is reusable across sessions; it counts only once the marker exists and the binary runs.
ready() {
  [ -f "$runtime/ready" ] && [ "$(CLINE_NO_AUTO_UPDATE=1 "$runtime/cline" --version 2>/dev/null)" = "$version" ]
}
if ready; then
  printf '%s\n' "$version"
  exit 0
fi

install_packages() {
  [ "$(id -u)" = 0 ] || { echo "Cline requires $*: preinstall them or use a root image." >&2; exit 1; }
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y "$@"
  elif command -v yum >/dev/null 2>&1; then
    yum install -y "$@"
  else
    echo "Cline requires $*: preinstall them (automatic installation requires apt-get, dnf or yum)." >&2
    exit 1
  fi
}

if [ -z "$source_binary" ]; then
  missing=0
  for command in curl tar gzip sha512sum; do
    if ! command -v "$command" >/dev/null 2>&1; then missing=1; fi
  done
  if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then missing=1; fi
  if [ "$missing" = 1 ]; then
    install_packages curl ca-certificates tar gzip coreutils
  fi
else
  [[ "$binary_sha512" =~ ^[0-9a-f]{128}$ ]] || { echo 'Missing verified Cline binary SHA-512' >&2; exit 1; }
fi

mkdir -p "$(dirname "$runtime")"
staging=$(mktemp -d "$runtime.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
if [ -n "$source_binary" ]; then
  cp -- "$source_binary" "$staging/cline"
  echo "$binary_sha512  $staging/cline" | sha512sum -c -
else
  archive="$staging/cli-linux-${arch}-${version}.tgz"
  curl -fsSL --retry 3 "https://registry.npmjs.org/@cline/cli-linux-${arch}/-/cli-linux-${arch}-${version}.tgz" -o "$archive"
  echo "$expected  $archive" | sha512sum -c -
  # Only the binary: the package also carries the hub dashboard's web assets, which headless runs never serve.
  tar -xzf "$archive" -C "$staging" package/bin/cline
  mv "$staging/package/bin/cline" "$staging/cline"
  rm -rf "$staging/package" "$archive"
fi
chmod 755 "$staging/cline"
actual=$(CLINE_NO_AUTO_UPDATE=1 "$staging/cline" --version)
[ "$actual" = "$version" ] || { echo "Cline version mismatch: $actual" >&2; exit 1; }
touch "$staging/ready"
# Publish the runtime with one rename, so a concurrent session never sees a partial directory. A runtime
# that another installer published meanwhile is kept; only a stale or partial one is replaced, under a lock
# so that two installers replacing it cannot remove the runtime the first one published.
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$runtime.lock"
  flock -w 60 9 || { echo "Timed out waiting for Cline runtime lock: $runtime.lock" >&2; exit 1; }
  if ! ready; then
    rm -rf -- "$runtime"
    mv -T -- "$staging" "$runtime"
  fi
elif ! mv -T -- "$staging" "$runtime" 2>/dev/null && ! ready; then
  # A cold install and ready-runtime reuse need no lock. Replacing a stale directory does:
  # never delete a runtime another installer may have published since our last readiness check.
  echo "Cannot replace stale Cline runtime safely without flock: $runtime. Install flock or use a fresh sandbox." >&2
  exit 1
fi
printf '%s\n' "$version"
