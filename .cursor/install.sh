#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for NeMo Gym.
# Installs uv (astral.sh egress is blocked, so we use PyPI), then materializes
# the Python 3.13.14 project virtualenv and dev dependencies from uv.lock.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing from PyPI..."
    python3 -m pip install --user --upgrade uv
fi

echo "uv version: $(uv --version)"

# `uv sync` creates .venv using the pinned interpreter in .python-version
# (3.13.14), downloading it if necessary, and installs locked dependencies.
# It is a no-op when the environment already matches uv.lock.
uv sync --extra dev

# Install git hooks so linting/format checks match CI. Safe to re-run and
# non-fatal: some managed environments set core.hooksPath, which makes
# pre-commit refuse to install hooks. The venv is fully usable either way.
uv run pre-commit install || echo "pre-commit hook install skipped (non-fatal)."

echo "NeMo Gym environment ready."
