#!/usr/bin/env bash
# Install codec-bearing packages that are excluded from the NeMo-Gym container.
#
# Run this before using VLM or audio/video benchmarks inside the container:
#
#   bash docker/install_codec_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
# Versions are pinned to match uv.lock so there are no ABI surprises.
# --no-config bypasses the project's [tool.uv] overrides.
set -euo pipefail

if python -c "import cv2, torchvision, torchaudio" 2>/dev/null; then
    echo "[codec-deps] Already installed, skipping."
    exit 0
fi

echo "[codec-deps] Installing codec-bearing packages..."
uv pip install --no-config \
    "opencv-python-headless==5.0.0.93" \
    "torchvision==0.26.0" \
    "torchaudio==2.11.0"

echo "[codec-deps] Done."
