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

if "${venv_python}" "${runtime_checker}" check --quiet; then
    echo "[osworld-runtime-deps] Required versions are already installed and importable; skipping."
    exit 0
fi

echo "[osworld-runtime-deps] Installing opt-in runtime dependencies..."
uv pip install --no-config --python "${venv_python}" \
    "numpy<2" \
    "cryptography~=46.0" \
    "opencv-python-headless~=4.8.1.78" \
    "torchvision==0.26.0"

"${venv_python}" "${runtime_checker}" check
echo "[osworld-runtime-deps] Done."
