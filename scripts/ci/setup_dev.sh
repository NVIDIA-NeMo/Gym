#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

gym_ci_setup_dev() {
    local setup_ci_dir
    local setup_repo_root
    local setup_uv_bin_dir
    local setup_uv_install_url

    setup_ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    setup_repo_root="$(cd "${setup_ci_dir}/../.." && pwd)"
    # 0.11.20 has a resolver regression that silently drops pinned requirements.
    setup_uv_install_url="https://astral.sh/uv/0.11.19/install.sh"
    setup_uv_bin_dir="${setup_repo_root}/.cache/nemo-gym-ci/uv-0.11.19"

    cd "${setup_repo_root}"
    curl -LsSf "${setup_uv_install_url}" | env UV_UNMANAGED_INSTALL="${setup_uv_bin_dir}" sh
    export PATH="${setup_uv_bin_dir}:${PATH}"
    test "$(uv --version | awk '{print $2}')" = "0.11.19"
    if [[ ! -x .venv/bin/python ]]; then
        uv venv --python 3.12
    fi
    uv sync --extra dev
}

gym_ci_setup_dev
unset -f gym_ci_setup_dev
