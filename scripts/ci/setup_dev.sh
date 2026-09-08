#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

gym_ci_setup_dev() {
    local setup_ci_dir
    local setup_dev_venv_dir
    local setup_python_version
    local setup_repo_root
    local setup_uv_cache_dir
    local setup_uv_bin_dir
    local setup_uv_install_url

    setup_ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    setup_repo_root="$(cd "${setup_ci_dir}/../.." && pwd)"
    setup_python_version="$(<"${setup_repo_root}/.python-version")"
    setup_uv_install_url="https://astral.sh/uv/0.11.29/install.sh"
    setup_uv_bin_dir="${setup_repo_root}/.cache/nemo-gym-ci/uv-0.11.29"
    setup_dev_venv_dir="${GYM_CI_DEV_VENV_DIR:-${setup_repo_root}/.venv}"
    if [[ "${setup_dev_venv_dir}" != /* || "${setup_dev_venv_dir}" == "/" ]]; then
        echo "GYM_CI_DEV_VENV_DIR must be an absolute non-root path: ${setup_dev_venv_dir}" >&2
        return 2
    fi

    cd "${setup_repo_root}"
    curl -LsSf "${setup_uv_install_url}" | env UV_UNMANAGED_INSTALL="${setup_uv_bin_dir}" sh
    export PATH="${setup_uv_bin_dir}:${PATH}"
    test "$(uv --version | awk '{print $2}')" = "0.11.29"
    # Resolve uv's default when the CI provider did not supply a cache directory, then export the
    # same path for nested per-server installs.
    setup_uv_cache_dir="$(uv cache dir)"
    mkdir -p "${setup_uv_cache_dir}"
    setup_uv_cache_dir="$(cd "${setup_uv_cache_dir}" && pwd -P)"
    export UV_CACHE_DIR="${setup_uv_cache_dir}"
    if [[ ! -x "${setup_dev_venv_dir}/bin/python" ]]; then
        uv venv --python "${setup_python_version}" "${setup_dev_venv_dir}"
    fi
    UV_PROJECT_ENVIRONMENT="${setup_dev_venv_dir}" uv sync --extra dev
    # Keep the original Actions contract: callers run the environment's commands directly.
    # shellcheck disable=SC1091
    source "${setup_dev_venv_dir}/bin/activate"
}

gym_ci_setup_dev
unset -f gym_ci_setup_dev
