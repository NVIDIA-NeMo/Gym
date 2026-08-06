#!/usr/bin/env bash
# Install codec-bearing packages that are NOT shipped in the NeMo-Gym container.
#
# Run this script before using VLM/audio/video features or running benchmarks
# that require OpenCV, torchvision, torchaudio, PyAV, decord2, or ffmpeg wrappers:
#
#   bash docker/install_codec_deps.sh
#
# Safe to call multiple times — exits immediately if already installed.
# --no-config prevents the project's [tool.uv] overrides from interfering.
set -euo pipefail

if python -c "import cv2, torchvision, torchaudio, av, decord" 2>/dev/null; then
    echo "[codec-deps] Already installed, skipping."
    exit 0
fi

echo "[codec-deps] Installing codec-bearing packages..."
uv pip install --no-config \
    "opencv-python-headless" \
    "opencv-python" \
    "torchvision" \
    "torchaudio" \
    "av" \
    "decord2" \
    "ffmpeg-python" \
    "pydub"

echo "[codec-deps] Done."
