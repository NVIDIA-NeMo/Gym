#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root
readonly pre_commit_version="4.3.0"  # exact version pinned in uv.lock (dev extra)
readonly pre_commit_sha256_sdist="499fe450cc9d42e9d58e606262795ecb64dd05438943c62b66f6a8673da30b16"  # uv.lock sdist
readonly pre_commit_sha256_wheel="2b0747ad7e6e967169136edffee14c16e148a778a54e4f967921aa1ebf2308d8"  # uv.lock wheel
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
    "${tool_venv}/bin/python" -m pip install --disable-pip-version-check --require-hashes \
        --no-deps "pre-commit==${pre_commit_version}" \
        --hash "sha256:${pre_commit_sha256_sdist}" --hash "sha256:${pre_commit_sha256_wheel}"
    pre_commit_bin="${tool_venv}/bin/pre-commit"
fi
"${pre_commit_bin}" install
exec "${pre_commit_bin}" run --all-files --show-diff-on-failure --color=always
