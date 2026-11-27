#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate docker/<stack>/package-lock.json from the package.json heredoc in docker/<stack>.Dockerfile.
#
# The manifest lives only inside the Dockerfile; this script extracts it, resolves the tree with npm
# (no install), writes the lock back and fails if the lock root no longer mirrors the manifest.
#   usage: regen_locks.sh [ui-next|ui-vite ...]   (default: both)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
stacks=("$@"); [ ${#stacks[@]} -eq 0 ] && stacks=(ui-next ui-vite)
for stack in "${stacks[@]}"; do
  df=$HERE/$stack.Dockerfile
  work=$(mktemp -d "${TMPDIR:-/tmp}/regen-$stack.XXXXXX")
  trap 'rm -rf "$work"' EXIT
  awk "/<<'EOF' \/app\/package.json\$/{on=1; next} on && /^EOF\$/{exit} on" "$df" > "$work/package.json"
  cp "$HERE/$stack/package-lock.json" "$work/"
  (cd "$work" && npm install --package-lock-only --legacy-peer-deps --ignore-scripts --no-audit --no-fund --loglevel=error)
  python3 - "$work" <<'PY'
import json, sys
w = sys.argv[1]
manifest = json.load(open(f"{w}/package.json"))
root = json.load(open(f"{w}/package-lock.json"))["packages"][""]
# npm mirrors the manifest's dependencies into the lock root (overrides are not recorded there).
if manifest.get("dependencies") != root.get("dependencies"):
    sys.exit("lock root dependencies differ from the Dockerfile manifest")
PY
  cp "$work/package-lock.json" "$HERE/$stack/package-lock.json"
  rm -rf "$work"; trap - EXIT
  echo "$stack: package-lock.json regenerated"
done
