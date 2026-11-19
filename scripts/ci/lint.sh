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
if ! command -v pre-commit >/dev/null 2>&1; then
    # Online runner: install the pinned pre-commit (version from uv.lock) into
    # an isolated venv. Offline/container: the dev environment baked into the
    # image already provides pre-commit on PATH, so this branch is skipped.
    pre_commit_version="$(awk '/^name = "pre-commit"$/{f=1} f && /^version = /{gsub(/"/, "", $3); print $3; exit}' uv.lock)"
    tool_venv="${repo_root}/.cache/nemo-gym-ci/pre-commit-${pre_commit_version}"
    python -m venv "${tool_venv}"
    "${tool_venv}/bin/python" -m pip install --disable-pip-version-check "pre-commit==${pre_commit_version}"
    PATH="${tool_venv}/bin:${PATH}"
fi
pre-commit install
exec pre-commit run --all-files --show-diff-on-failure --color=always
