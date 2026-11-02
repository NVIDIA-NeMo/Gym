#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Prepare once outside task containers; bind the result read-only at /opt/hermes.
set -euo pipefail
# The runtime is mounted without the build host's uv cache.
export UV_LINK_MODE=copy
: "${DEPS_DIR:?Set DEPS_DIR to an empty runtime directory}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DEPS_DIR"
exec 9>"$DEPS_DIR/.prepare.lock"
flock 9
source "$SCRIPT_DIR/../anyswe_agent/setup_scripts/_portable_python.sh"
# uv's --python-version controls resolution, but --prefix uses the host
# interpreter's directory layout. Select the matching host Python explicitly.
export UV_PYTHON="$PYTHON_VERSION"
mkdir -p "$DEPS_DIR/bin"
install -m 755 "$SCRIPT_DIR/runtime_python.sh" "$DEPS_DIR/bin/hermes-python"
HERMES_VERSION="${HERMES_VERSION:-2237be355906fbe6065ce1815711eee52b2d646e}"
HERMES_REPO_URL="${HERMES_REPO_URL:-https://github.com/NousResearch/hermes-agent.git}"
if [ ! -d "$DEPS_DIR/hermes-src/.git" ]; then
    git init "$DEPS_DIR/hermes-src"
    git -C "$DEPS_DIR/hermes-src" remote add origin "$HERMES_REPO_URL"
else
    git -C "$DEPS_DIR/hermes-src" remote set-url origin "$HERMES_REPO_URL"
fi
git -C "$DEPS_DIR/hermes-src" fetch --depth=1 origin "$HERMES_VERSION"
HERMES_COMMIT="$(git -C "$DEPS_DIR/hermes-src" rev-parse 'FETCH_HEAD^{commit}')"
# Hermes's grep fallback requires GNU options absent from Alpine's BusyBox.
# Keep this static binary separate from Python so task tools retain their interpreter.
if [[ ! -x "$DEPS_DIR/tools/bin/rg" ]]; then
    rg_tmp=$(mktemp -d "$DEPS_DIR/.ripgrep.XXXXXX")
    trap 'rm -rf "$rg_tmp"' EXIT
    curl -fsSL --retry 3 -o "$rg_tmp/rg.tar.gz" \
        https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz
    echo "4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e  $rg_tmp/rg.tar.gz" | sha256sum -c -
    tar xzf "$rg_tmp/rg.tar.gz" -C "$rg_tmp" --strip-components=1
    mkdir -p "$DEPS_DIR/tools/bin"
    install -m 644 "$rg_tmp/COPYING" "$rg_tmp/LICENSE-MIT" "$rg_tmp/UNLICENSE" "$DEPS_DIR/tools/"
    install -m 755 "$rg_tmp/rg" "$DEPS_DIR/tools/bin/rg"
fi
# A musl build can be prepared with uv on a glibc host; its imports are checked
# when the runner starts in the target container. Keep the native import check.
validate_runtime_import() {
    if [[ "$ARCH" == *-musl ]] && ! portable_python_can_run; then
        return 0
    fi
    "$DEPS_DIR/bin/python3" -I - "$DEPS_DIR" <<'PY'
import pathlib
import sys
import run_agent
root = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(run_agent.__file__).resolve().is_relative_to(root / "hermes-src")
PY
}
if [[ -f "$DEPS_DIR/hermes-runtime.json" && -x "$DEPS_DIR/bin/python3" && -d "$DEPS_DIR/hermes-src/.git" ]] &&
    [[ "$(git -c safe.directory="$DEPS_DIR/hermes-src" -C "$DEPS_DIR/hermes-src" rev-parse HEAD)" == "$HERMES_COMMIT" ]] &&
    python3 -I - "$DEPS_DIR" "$HERMES_COMMIT" "$HERMES_REPO_URL" "$ARCH" "$PYTHON_VERSION" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
manifest = json.loads((root / "hermes-runtime.json").read_text())
assert manifest["hermes_commit"] == sys.argv[2]
assert manifest.get("hermes_repo_url") == sys.argv[3]
assert manifest.get("arch", "x86_64-unknown-linux-gnu") == sys.argv[4]
assert manifest.get("uv_link_mode") == "copy"
assert manifest.get("python_version") == sys.argv[5]
PY
then
    if validate_runtime_import; then
        echo "Pinned Hermes runtime is ready: $DEPS_DIR"
        exit 0
    fi
fi
install_portable_python
git -C "$DEPS_DIR/hermes-src" checkout --detach "$HERMES_COMMIT"
# This release intentionally rejects wheels; retain its source assets and use
# setuptools' simple .pth editable mode, then make that path relocatable.
# Reinstall dependencies too: older runtimes can contain uv-cache symlinks.
install_python_packages --force-reinstall -e "$DEPS_DIR/hermes-src" --config-settings editable_mode=compat
python3 -I - "$DEPS_DIR" "${PYTHON_VERSION%.*}" <<'PY'
import os
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
source = root / "hermes-src"
site = root / "lib" / f"python{sys.argv[2]}" / "site-packages"
matched = False
for path in site.glob("__editable__*hermes*.pth"):
    if path.is_symlink():
        raise RuntimeError(f"Hermes editable path still links to the build cache: {path}")
    if path.read_text().strip() != str(source):
        raise RuntimeError(f"Unexpected editable layout: {path}")
    path.write_text(os.path.relpath(source, site) + "\n")
    matched = True
if not matched:
    raise RuntimeError("Hermes editable .pth was not installed")
PY
validate_runtime_import
if [[ "$ARCH" != *-musl ]] || portable_python_can_run; then
    "$DEPS_DIR/bin/python3" -I -c 'from run_agent import AIAgent; import inspect; assert "request_overrides" in inspect.signature(AIAgent).parameters'
    "$DEPS_DIR/bin/python3" -m pip freeze > "$DEPS_DIR/requirements.freeze.txt"
else
    uv pip freeze --path "$DEPS_DIR/lib/python${PYTHON_VERSION%.*}/site-packages" > "$DEPS_DIR/requirements.freeze.txt"
fi
python3 -I - "$DEPS_DIR/hermes-runtime.json" "$HERMES_COMMIT" "$HERMES_REPO_URL" "$ARCH" "$PYTHON_VERSION" <<'PY'
import json
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "hermes_commit": sys.argv[2], "hermes_repo_url": sys.argv[3], "arch": sys.argv[4],
    "uv_link_mode": "copy",
    "python_version": sys.argv[5],
}) + "\n")
PY
