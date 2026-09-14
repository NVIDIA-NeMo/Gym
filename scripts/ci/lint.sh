#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root
readonly pre_commit_version="3.6.0"
readonly tool_venv="${repo_root}/.cache/nemo-gym-ci/pre-commit-${pre_commit_version}"

# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment lint
unset -f gym_ci_sanitize_environment

cd "${repo_root}"
if [[ "${NEMO_GYM_CONTAINER:-}" == "1" ]] && command -v pre-commit >/dev/null 2>&1; then
    # Offline container environment: the image already provides pre-commit, so install nothing.
    pre_commit_bin="pre-commit"
else
    # Online environment (e.g. GitHub Actions): install the pinned pre-commit into a tool venv.
    python -m venv "${tool_venv}"
    "${tool_venv}/bin/python" -m pip install --disable-pip-version-check "pre-commit==${pre_commit_version}"
    pre_commit_bin="${tool_venv}/bin/pre-commit"
fi
"${pre_commit_bin}" install
exec "${pre_commit_bin}" run --all-files --show-diff-on-failure --color=always
