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
    local setup_uv_version
    local setup_uv_arch
    local setup_uv_sha256
    local setup_uv_archive
    local setup_uv_sync_args

    setup_ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    setup_repo_root="$(cd "${setup_ci_dir}/../.." && pwd)"
    setup_python_version="$(<"${setup_repo_root}/.python-version")"
    # The pinned uv version comes from the Dockerfile (single source of truth, also
    # used by cicd-main.yml's "Install uv" step).
    setup_uv_version="$(sed -n 's/^ARG UV_VERSION=//p' "${setup_repo_root}/docker/Dockerfile")"
    if [[ ! "${setup_uv_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Could not read a valid UV_VERSION from docker/Dockerfile: ${setup_uv_version}" >&2
        return 2
    fi
    setup_uv_bin_dir="${setup_repo_root}/.cache/nemo-gym-ci/uv-${setup_uv_version}"
    setup_dev_venv_dir="${GYM_CI_DEV_VENV_DIR:-${setup_repo_root}/.venv}"
    if [[ "${setup_dev_venv_dir}" != /* || "${setup_dev_venv_dir}" == "/" ]]; then
        echo "GYM_CI_DEV_VENV_DIR must be an absolute non-root path: ${setup_dev_venv_dir}" >&2
        return 2
    fi

    cd "${setup_repo_root}"
    if command -v uv >/dev/null 2>&1 && [[ "$(uv --version | awk '{print $2}')" == "${setup_uv_version}" ]]; then
        # uv is already present at the pinned version (the baked CI image, or a
        # runner/image that ships it): do not download or install anything.
        # In the container (NEMO_GYM_CONTAINER=1) resolve from the pre-populated
        # cache offline; elsewhere resolve from the package index.
        if [[ "${NEMO_GYM_CONTAINER:-}" == "1" ]]; then
            setup_uv_sync_args=(--offline)
        else
            setup_uv_sync_args=()
        fi
    else
        # Online environment (e.g. GitHub Actions) without the pinned uv. Download the
        # standalone uv binary archive and verify its pinned SHA-256 BEFORE use — we do
        # NOT pipe a remote install script to a shell. Mirrors docker/Dockerfile, whose
        # release image installs uv the same way. The digests pin uv 0.11.29.
        case "$(uname -m)" in
            x86_64)
                setup_uv_arch="x86_64"
# pragma: allowlist nextline secret
                setup_uv_sha256="04f8b82f5d47f0512dcd32c67a4a6f16a0ea27c81537c338fd0ad6b23cebe829"
                ;;
            aarch64 | arm64)
                setup_uv_arch="aarch64"
# pragma: allowlist nextline secret
                setup_uv_sha256="94500fb064ae3c971a873cba64d94694c50677e0a4dbf78735c80509e7429919"
                ;;
            *)
                echo "Unsupported architecture for uv: $(uname -m)" >&2
                return 2
                ;;
        esac
        setup_uv_archive="uv-${setup_uv_arch}-unknown-linux-gnu.tar.gz"
        mkdir -p "${setup_uv_bin_dir}"
        curl -fLSs --retry 5 --retry-all-errors --retry-max-time 300 \
            --connect-timeout 30 --max-time 120 \
            -o "${setup_uv_bin_dir}/${setup_uv_archive}" \
            "https://github.com/astral-sh/uv/releases/download/${setup_uv_version}/${setup_uv_archive}"
        echo "${setup_uv_sha256}  ${setup_uv_bin_dir}/${setup_uv_archive}" | sha256sum -c -
        tar -xzf "${setup_uv_bin_dir}/${setup_uv_archive}" -C "${setup_uv_bin_dir}"
        install -m 0755 "${setup_uv_bin_dir}/uv-${setup_uv_arch}-unknown-linux-gnu/uv" "${setup_uv_bin_dir}/uv"
        rm -rf "${setup_uv_bin_dir}/${setup_uv_archive}" "${setup_uv_bin_dir}/uv-${setup_uv_arch}-unknown-linux-gnu"
        export PATH="${setup_uv_bin_dir}:${PATH}"
        test "$(uv --version | awk '{print $2}')" = "${setup_uv_version}"
        setup_uv_sync_args=()
    fi
    # Resolve uv's default when the CI provider did not supply a cache directory, then export the
    # same path for nested per-server installs.
    setup_uv_cache_dir="$(uv cache dir)"
    mkdir -p "${setup_uv_cache_dir}"
    setup_uv_cache_dir="$(cd "${setup_uv_cache_dir}" && pwd -P)"
    export UV_CACHE_DIR="${setup_uv_cache_dir}"
    if [[ ! -x "${setup_dev_venv_dir}/bin/python" ]]; then
        uv venv --python "${setup_python_version}" "${setup_dev_venv_dir}"
    fi
    UV_PROJECT_ENVIRONMENT="${setup_dev_venv_dir}" uv sync --extra dev "${setup_uv_sync_args[@]}"
    # Keep the original Actions contract: callers run the environment's commands directly.
    # shellcheck disable=SC1091
    source "${setup_dev_venv_dir}/bin/activate"
}

gym_ci_setup_dev
unset -f gym_ci_setup_dev
