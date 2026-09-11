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
torch_backend_file="${script_dir}/uv-torch-backend.txt"
if [[ ! -x "${venv_python}" ]]; then
    echo "OSWorld agent Python is not executable: ${venv_python}" >&2
    exit 2
fi
if [[ ! -r "${runtime_checker}" ]]; then
    echo "OSWorld runtime dependency checker is not readable: ${runtime_checker}" >&2
    exit 2
fi
if [[ ! -r "${torch_backend_file}" ]]; then
    echo "OSWorld agent Torch backend marker is not readable: ${torch_backend_file}" >&2
    exit 2
fi

torch_backend="$(tr -d '[:space:]' < "${torch_backend_file}")"
if [[ -z "${torch_backend}" ]]; then
    echo "OSWorld agent Torch backend marker is empty: ${torch_backend_file}" >&2
    exit 2
fi

if "${venv_python}" "${runtime_checker}" check --quiet; then
    echo "[osworld-runtime-deps] Required versions are already installed and importable; skipping."
    exit 0
fi

echo "[osworld-runtime-deps] Installing opt-in runtime dependencies..."
# Match the backend selected by `gym env prefetch`. Without this flag, uv may
# combine the CPU torch already in this venv with a CUDA torchvision wheel from
# the default package index. The versions then look compatible while native
# operators such as torchvision::nms cannot be loaded.
uv pip install --no-config --torch-backend "${torch_backend}" --python "${venv_python}" \
    "numpy>=2.1,<2.5" \
    "cryptography~=46.0" \
    "opencv-python-headless~=4.10.0.84" \
    "torchvision==0.26.0"

"${venv_python}" "${runtime_checker}" check
echo "[osworld-runtime-deps] Done."
