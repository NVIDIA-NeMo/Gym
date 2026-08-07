#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#   bash scripts/setup_dev_env.sh                     # uses versions from repo config
#   UV_VERSION=0.11.29 bash scripts/setup_dev_env.sh  # override uv version
#
# This script is safe to re-run. It:
#   1. Updates uv to the version pinned in CI (or PYTHON_VERSION/UV_VERSION env vars)
#   2. Installs the Python version pinned in .python-version
#   3. Removes stale .venv directories (root + per-server venvs under resources_servers/)
#   4. Re-creates the venv and syncs all dependencies

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Resolve target versions from repo config, allow env var override
# ---------------------------------------------------------------------------

# Python: .python-version is the sole source of truth
if [[ ! -f ".python-version" ]]; then
    echo "ERROR: .python-version not found." >&2
    exit 1
fi
TARGET_PYTHON="$(tr -d '[:space:]' < .python-version)"

# uv: parse version from CI's UV_INSTALL_URL (single source of truth for pinned uv)
TARGET_UV="${UV_VERSION:-}"
if [[ -z "$TARGET_UV" ]]; then
    CI_YAML=".github/workflows/unit-tests.yml"
    if [[ -f "$CI_YAML" ]]; then
        TARGET_UV="$(grep -oE 'astral\.sh/uv/[0-9]+\.[0-9]+\.[0-9]+' "$CI_YAML" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    fi
fi

echo "==> Target Python : ${TARGET_PYTHON}"
echo "==> Target uv     : ${TARGET_UV:-latest}"

# ---------------------------------------------------------------------------
# 2. Update / install uv
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "==> uv not found; installing..."
    curl -LsSf "https://astral.sh/uv/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

CURRENT_UV="$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
echo "==> Current uv    : ${CURRENT_UV:-unknown}"

if [[ -n "$TARGET_UV" && "$CURRENT_UV" != "$TARGET_UV" ]]; then
    echo "==> Updating uv to ${TARGET_UV}..."
    uv self update "$TARGET_UV" || echo "==> Warning: could not update uv (may be managed by system package manager); continuing with ${CURRENT_UV}"
else
    echo "==> uv is up to date (${CURRENT_UV})"
fi

# ---------------------------------------------------------------------------
# 3. Install target Python via uv
# ---------------------------------------------------------------------------
echo "==> Installing Python ${TARGET_PYTHON}..."
uv python install "$TARGET_PYTHON"

# ---------------------------------------------------------------------------
# 4. Remove stale virtual environments
# ---------------------------------------------------------------------------
echo "==> Removing stale .venv directories..."

# Root venv
if [[ -d ".venv" ]]; then
    rm -rf .venv
    echo "    Removed .venv"
fi

# Per-server venvs under resources_servers/ (created by 'gym env test')
while IFS= read -r -d '' venv_dir; do
    rm -rf "$venv_dir"
    echo "    Removed $venv_dir"
done < <(find resources_servers -maxdepth 2 -name ".venv" -type d -print0 2>/dev/null)

# ---------------------------------------------------------------------------
# 5. Re-create venv and sync dependencies
# ---------------------------------------------------------------------------
echo "==> Creating venv with Python ${TARGET_PYTHON}..."
uv venv --python "$TARGET_PYTHON"

echo "==> Syncing dependencies..."
uv sync --extra dev

echo ""
echo "Done! Activate your venv with:"
echo "  source .venv/bin/activate"
