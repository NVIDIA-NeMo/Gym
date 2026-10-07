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
# Safe to call multiple times — exits immediately if already installed.
# Versions are pinned to match the container's uv.lock. Update these pins
# manually when uv lock bumps them (e.g. after a vllm upgrade).
# --no-config bypasses the project's sys_platform=='never' overrides.
set -euo pipefail

if python -c "import cv2, PyNvVideoCodec, torchcodec, torchvision, torchaudio, pycountry" 2>/dev/null; then
    echo "[excluded-deps] Already installed, skipping."
    exit 0
fi

echo "[excluded-deps] Installing excluded packages..."
uv pip install --no-config \
    "opencv-python-headless==5.0.0.93" \
    "pynvvideocodec==2.0.4" \
    "torchcodec==0.16.0" \
    "torchvision==0.26.0" \
    "torchaudio==2.11.0" \
    "pycountry==26.2.16"

echo "[excluded-deps] Done."
