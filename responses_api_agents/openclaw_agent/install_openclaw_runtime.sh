#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
trap 'status=$?; echo "OpenClaw installer failed (exit $status) at line $LINENO: $BASH_COMMAND" >&2; exit "$status"' ERR

runtime=$1
openclaw_version=$2
node_version=22.19.0
if [ "$(uname -s)" != Linux ] || ! getconf GNU_LIBC_VERSION >/dev/null 2>&1; then
  echo 'Native OpenClaw requires a Linux glibc sandbox; Alpine/musl is not supported.' >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Native OpenClaw requires Python >=3.9 in the task image for descendant supervision.' >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info >= (3, 9), "Native OpenClaw requires Python >=3.9"'
case "$(uname -m)" in
  x86_64) arch=x64 ;;
  aarch64) arch=arm64 ;;
  *) echo 'Native OpenClaw supports Linux x86_64/aarch64 glibc only.' >&2; exit 1 ;;
esac

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
  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    echo 'Native OpenClaw requires flock: preinstall util-linux in the task image (automatic installation requires root and apt-get).' >&2
    exit 1
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends util-linux
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
for prerequisite in curl tar xz sha256sum awk; do
  command -v "$prerequisite" >/dev/null 2>&1 || missing+=("$prerequisite")
done
if [ ! -s /etc/ssl/certs/ca-certificates.crt ] && [ ! -s /etc/pki/tls/certs/ca-bundle.crt ]; then
  missing+=(ca-certificates)
fi
if [ "${#missing[@]}" -gt 0 ]; then
  if [ "$(id -u)" -ne 0 ] || ! command -v apt-get >/dev/null 2>&1; then
    echo "Native OpenClaw prerequisites missing: ${missing[*]}. Preinstall curl, ca-certificates, tar, xz-utils, coreutils, gawk (automatic installation requires root and apt-get)." >&2
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
  --prefix "$runtime/openclaw" --no-audit --no-fund "openclaw@${openclaw_version}"
"$runtime/node/bin/node" -e \
  'const p=require(process.argv[1]); if(p.version!==process.argv[2])throw Error("OpenClaw version mismatch")' \
  "$runtime/openclaw/node_modules/openclaw/package.json" "$openclaw_version"
"$runtime/node/bin/node" "$runtime/openclaw/node_modules/openclaw/openclaw.mjs" --version
printf '%s\n' "$openclaw_version" > "$runtime/ready"
