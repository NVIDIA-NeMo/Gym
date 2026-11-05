#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install packages that are excluded from the NeMo-Gym container (codec-bearing
# packages, plus pycountry which vLLM's Mistral tokenizer import chain needs
# when serving VLMs such as Qwen3-VL).
#
# Run this before using VLM or audio/video benchmarks inside the container:
#
#   bash docker/install_excluded_deps.sh
#
# This opt-in layer is independently locked and intentionally bypasses the
# root project's exclusion policy. Preserve the existing codec versions; this
# layer is not a dependency-upgrade mechanism.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="${script_dir}/excluded_deps"
for filename in pyproject.toml uv.lock; do
    if [[ ! -r "${project_dir}/${filename}" ]]; then
        echo "Missing locked codec project: ${project_dir}/${filename}" >&2
        exit 2
    fi
done

# Sync the environment selected by the caller's Python, not a second .venv in
# docker/. Never let uv replace an OS-managed Python environment.
python_executable="$(python -c 'import sys; print(sys.executable)')"
venv_path="$(python -c 'import sys; print(sys.prefix) if sys.prefix != sys.base_prefix else sys.exit("Activate the Gym virtual environment before installing optional codecs.")')"
# Refuse the known codec/model Torch conflict before uv can mutate the runtime.
"${python_executable}" "${project_dir}/check_runtime.py" "${project_dir}/uv.lock"
echo "[excluded-deps] Syncing locked opt-in packages..."
UV_PROJECT_ENVIRONMENT="${venv_path}" uv sync --locked --inexact \
    --project "${project_dir}" --python "${python_executable}" --no-dev
"${python_executable}" -c 'import cv2, PyNvVideoCodec, torchcodec, torchvision, torchaudio, pycountry'
echo "[excluded-deps] Done."
