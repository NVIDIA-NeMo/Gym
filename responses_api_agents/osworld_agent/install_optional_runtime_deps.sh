#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install OSWorld runtime packages that NeMo Gym deliberately excludes from
# shipped packages and containers. Run this only after `gym env prefetch` has
# created the managed OSWorld agent venv.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/osworld-agent-venv" >&2
    exit 2
fi

venv_path=$1
venv_python="${venv_path}/bin/python"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_checker="${script_dir}/runtime_dependencies.py"
if [[ ! -x "${venv_python}" ]]; then
    echo "OSWorld agent Python is not executable: ${venv_python}" >&2
    exit 2
fi
if [[ ! -r "${runtime_checker}" ]]; then
    echo "OSWorld runtime dependency checker is not readable: ${runtime_checker}" >&2
    exit 2
fi

# uv interprets a relative UV_PROJECT_ENVIRONMENT from the selected project.
# Normalize first so a caller running from benchmarks/osworld targets its venv.
venv_path="$(cd "${venv_path}" && pwd)"
venv_python="${venv_path}/bin/python"
runtime_project="${script_dir}/runtime"
for filename in pyproject.toml uv.lock; do
    if [[ ! -r "${runtime_project}/${filename}" ]]; then
        echo "Missing locked OSWorld runtime project: ${runtime_project}/${filename}" >&2
        exit 2
    fi
done

if "${venv_python}" "${runtime_checker}" check --quiet; then
    echo "[osworld-runtime-deps] Required versions are already installed and importable; skipping."
    exit 0
fi

echo "[osworld-runtime-deps] Installing opt-in runtime dependencies..."
# This explicitly opted-in layer is independently locked. Preserve the base
# agent packages while installing the exact runtime lock; never re-resolve them.
# The runtime project binds both torch and torchvision to the CPU index.
UV_PROJECT_ENVIRONMENT="${venv_path}" uv sync --locked --inexact \
    --project "${runtime_project}" --python "${venv_python}" --no-dev

"${venv_python}" "${runtime_checker}" check
echo "[osworld-runtime-deps] Done."
