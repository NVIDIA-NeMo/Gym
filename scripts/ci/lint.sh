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
python -m venv "${tool_venv}"
"${tool_venv}/bin/python" -m pip install --disable-pip-version-check "pre-commit==${pre_commit_version}"
"${tool_venv}/bin/pre-commit" install
exec "${tool_venv}/bin/pre-commit" run --all-files --show-diff-on-failure --color=always
