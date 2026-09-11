#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare once outside task containers; bind the result read-only at /opt/hermes.
set -euo pipefail
: "${DEPS_DIR:?Set DEPS_DIR to an empty runtime directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DEPS_DIR"
exec 9>"$DEPS_DIR/.prepare.lock"
flock 9
HERMES_COMMIT=29112bef099274229cadff79cdff7bf7b99c4b77
if [[ -f "$DEPS_DIR/hermes-runtime.json" && -x "$DEPS_DIR/bin/python3" ]] &&
    [[ "$(git -C "$DEPS_DIR/hermes-src" rev-parse HEAD)" == "$HERMES_COMMIT" ]] &&
    "$DEPS_DIR/bin/python3" -I - "$DEPS_DIR" "$HERMES_COMMIT" <<'PY'
import json
import pathlib
import sys
import run_agent
root = pathlib.Path(sys.argv[1]).resolve()
assert json.loads((root / "hermes-runtime.json").read_text())["hermes_commit"] == sys.argv[2]
assert pathlib.Path(run_agent.__file__).resolve().is_relative_to(root / "hermes-src")
PY
then
    echo "Pinned Hermes runtime is ready: $DEPS_DIR"
    exit 0
fi
source "$SCRIPT_DIR/../anyswe_agent/setup_scripts/_portable_python.sh"
install_portable_python
if [ ! -d "$DEPS_DIR/hermes-src/.git" ]; then
    git clone --depth=1 --branch=v2026.8.31 https://github.com/NousResearch/hermes-agent.git "$DEPS_DIR/hermes-src"
fi
git -C "$DEPS_DIR/hermes-src" checkout --detach "$HERMES_COMMIT"
# This release intentionally rejects wheels; retain its source assets and use
# setuptools' simple .pth editable mode, then make that path relocatable.
install_python_packages -e "$DEPS_DIR/hermes-src" --config-settings editable_mode=compat
"$DEPS_DIR/bin/python3" - "$DEPS_DIR/hermes-src" <<'PY'
import os
import pathlib
import sys
import sysconfig
source = pathlib.Path(sys.argv[1]).resolve()
site = pathlib.Path(sysconfig.get_path("purelib"))
matched = False
for path in site.glob("__editable__*hermes*.pth"):
    if path.read_text().strip() != str(source):
        raise RuntimeError(f"Unexpected editable layout: {path}")
    path.write_text(os.path.relpath(source, site) + "\n")
    matched = True
if not matched:
    raise RuntimeError("Hermes editable .pth was not installed")
PY
"$DEPS_DIR/bin/python3" -I -c 'from run_agent import AIAgent; import inspect; assert "request_overrides" in inspect.signature(AIAgent).parameters'
"$DEPS_DIR/bin/python3" -m pip freeze > "$DEPS_DIR/requirements.freeze.txt"
printf '{"hermes_commit":"%s","tag":"v2026.8.31"}\n' "$HERMES_COMMIT" > "$DEPS_DIR/hermes-runtime.json"
