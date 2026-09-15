#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment lint
unset -f gym_ci_sanitize_environment

cd "${repo_root}"
# pre-commit is a dev-extra dependency in uv.lock. The CI image installs that
# extra into the project environment (on PATH) at build time, and local/online
# setups get it from `uv sync --extra dev`. There is intentionally no ad-hoc
# pip/uv install here so the executable always comes from the lockfile.
if ! command -v pre-commit >/dev/null 2>&1; then
    echo "pre-commit not found on PATH. Sync the dev environment first (uv sync --extra dev)." >&2
    exit 1
fi
pre-commit install
exec pre-commit run --all-files --show-diff-on-failure --color=always
