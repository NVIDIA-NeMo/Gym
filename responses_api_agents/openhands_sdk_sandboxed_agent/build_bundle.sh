#!/bin/bash
# Build the offline OpenHands bundle (standalone Python 3.12.14 + openhands-sdk/tools 1.50.0 + static tmux 3.7b) used by
# openhands_sdk_sandboxed_agent when `bundle_path` is set. Needed for agent_network_mode=no_internet_preinstalled (the per-task images are
# python:3.11-slim without tmux and have no egress) AND for block_target_v2 (that guard blocks `python -m venv`, which the pip-install path needs).
# Run on any x86_64 Linux machine with internet and `uv` (https://docs.astral.sh/uv/); the tarball is uploaded into each rollout's sandbox.
#
# Usage: bash build_bundle.sh /path/to/outdir      -> /path/to/outdir/openhands_bundle.tar.gz (~150 MB)
# Then pass it to a run:
#   ++nl2repobench_opencode_sandboxed_agent.responses_api_agents.openhands_sdk_sandboxed_agent.bundle_path=/path/to/outdir/openhands_bundle.tar.gz
# (the default `bundle_python_relpath` in app.py matches the Python version pinned below.)
set -euo pipefail
# Ignore the user site-packages: uv treats packages found in ~/.local/lib/python3.12 as "already installed" and skips them, but the
# bundle runs with an isolated interpreter (-I) that cannot see them (cryptography and wcwidth went missing this way).
export PYTHONNOUSERSITE=1
OUT="$(realpath -m "${1:?usage: build_bundle.sh OUTDIR}")"
PYVER="3.12.14"
mkdir -p "$OUT/bin"
uv python install "$PYVER" --install-dir "$OUT/py"
PY="$(ls -d "$OUT"/py/cpython-${PYVER}-linux-x86_64-gnu/bin/python3.12)"
# Install the EXACT package set the evals were run with (bundle_requirements.lock.txt = `pip freeze` of the working bundle). Do not replace this with
# unpinned installs: transitive dependencies drift (a fresh unpinned resolve failed at import).
LOCK="$(dirname "$(realpath "$0")")/bundle_requirements.lock.txt"
uv pip install --python "$PY" --break-system-packages -r "$LOCK"
# `uv pip install -r` silently skipped two locked packages (cryptography, wcwidth) here (it reports them as already installed although they are not
# in the interpreter), which breaks `import openhands.sdk` in the isolated sandbox interpreter. Install anything the interpreter does not really
# have with its own pip (--no-deps: the lock is complete).
for name in $(comm -13 <("$PY" -m pip freeze 2>/dev/null | sed 's/==.*//' | tr 'A-Z_' 'a-z-' | sort) <(grep -v '^#' "$LOCK" | sed 's/==.*//' | tr 'A-Z_' 'a-z-' | sort)); do
    "$PY" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --break-system-packages --no-deps "$(grep -i "^${name//-/[-_]}==" "$LOCK" | head -1)"
done
"$PY" -c "import openhands.sdk, openhands.tools.terminal; print('openhands import ok')"
curl -fsSL https://github.com/mjakob-gh/build-static-tmux/releases/download/v3.7b/tmux.linux-amd64.gz | gunzip > "$OUT/bin/tmux"
chmod +x "$OUT/bin/tmux"; "$OUT/bin/tmux" -V
find "$OUT/py" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
# uv creates an absolute `cpython-3.12-linux-x86_64-gnu` symlink into $OUT; it must not be in the tarball (dangling in the sandbox)
tar -czf "$OUT/openhands_bundle.tar.gz" -C "$OUT" --exclude='py/cpython-3.12-linux-x86_64-gnu' py bin
ls -la "$OUT/openhands_bundle.tar.gz"
