#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
trap 'status=$?; echo "OpenClaw installer failed (exit $status) at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR

runtime=$1
openclaw_version=$2
node_version=22.19.0
if [ "$(uname -s)" != Linux ]; then
  echo 'OpenClaw sandbox execution requires a Linux sandbox.' >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo 'OpenClaw sandbox execution requires Python >=3.8 in the task image for descendant supervision.' >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info >= (3, 8), "OpenClaw sandbox execution requires Python >=3.8"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'OpenClaw sandbox execution supports Linux x86_64/aarch64 only.' >&2; exit 1 ;;
esac
platform="linux-${arch}"
node_dist="https://nodejs.org/dist/v${node_version}"
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  :
elif [[ "$(ldd --version 2>&1 || true)" == *musl* ]]; then
  if [ "$arch" != x64 ]; then
    echo "OpenClaw sandbox execution's pinned Node ${node_version} musl build supports x86_64 only" >&2
    exit 1
  fi
  platform=linux-x64-musl
  node_dist="https://unofficial-builds.nodejs.org/download/release/v${node_version}"
else
  echo 'OpenClaw sandbox execution requires glibc or musl; could not identify the sandbox libc' >&2
  exit 1
fi

install_packages() {
  if [ "$(id -u)" -ne 0 ]; then
    echo "OpenClaw sandbox execution requires $*: preinstall these packages (automatic installation requires root)." >&2
    exit 1
  fi
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache "$@"
  elif command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
  else
    echo "OpenClaw sandbox execution requires $*: preinstall these packages (automatic installation requires apt-get or apk)." >&2
    exit 1
  fi
}

# Never change the task repository or its dependency environment.
python3 - "$runtime" "$PWD" <<'PY'
import pathlib, sys
runtime, workdir = (pathlib.Path(p).resolve() for p in sys.argv[1:])
if workdir == runtime or workdir in runtime.parents or runtime in workdir.parents:
    raise SystemExit('OpenClaw runtime must be outside the task repository')
PY
# One sandbox can receive overlapping session setup attempts. Keep the ready
# check and all cache writes under a version-scoped advisory lock.
if ! command -v flock >/dev/null 2>&1; then
  install_packages util-linux
fi
mkdir -p "$runtime"
exec 9>"$runtime/.install.lock"
flock -x 9
mkdir -p "$runtime/home" "$runtime/cache"
export HOME="$runtime/home"
export npm_config_cache="$runtime/cache/npm"
export XDG_CACHE_HOME="$runtime/cache"
verify_version() {
  "$runtime/node/bin/node" -e \
    'const p=require(process.argv[1]); if(p.version!==process.argv[2])throw Error("OpenClaw version mismatch")' \
    "$runtime/openclaw/node_modules/openclaw/package.json" "$openclaw_version"
}
if [ -f "$runtime/ready" ]; then
  verify_version
  "$runtime/node/bin/node" "$runtime/openclaw/node_modules/openclaw/openclaw.mjs" --version
  exit 0
fi

missing=()
for prerequisite in curl tar gzip sha256sum awk; do
  command -v "$prerequisite" >/dev/null 2>&1 || missing+=("$prerequisite")
done
if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then
  missing+=(ca-certificates)
fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "OpenClaw sandbox execution prerequisites missing: ${missing[*]}. Preinstall curl, ca-certificates, tar, gzip, coreutils, gawk." >&2
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
  # Confine the newer C++ runtime to our Node binary, preserving task libraries.
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
  --prefix "$runtime/openclaw" --no-audit --no-fund "openclaw@${openclaw_version}"
"$runtime/node/bin/node" -e \
  'const p=require(process.argv[1]); if(p.version!==process.argv[2])throw Error("OpenClaw version mismatch")' \
  "$runtime/openclaw/node_modules/openclaw/package.json" "$openclaw_version"
"$runtime/node/bin/node" "$runtime/openclaw/node_modules/openclaw/openclaw.mjs" --version
printf '%s\n' "$openclaw_version" > "$runtime/ready"
