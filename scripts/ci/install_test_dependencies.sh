#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${RUNNER_TEMP:?}" "${GITHUB_ENV:?}"
# Some runtime tests drop privileges; their interpreter must be outside /root.
install -d -m755 "$RUNNER_TEMP/uv-python"
printf 'UV_PYTHON_INSTALL_DIR=%s/uv-python\n' "$RUNNER_TEMP" >> "$GITHUB_ENV"

# ARC is leaner than the hosted Ubuntu image. Native extensions and Chromium
# need these host libraries; the runner already supplies the S3 dataset mount.
packages=(
    git curl ca-certificates build-essential cmake
    libgl1 libglib2.0-0t64 libsm6 libxext6 libxrender1
    libnspr4 libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64
    libcups2t64 libdbus-1-3 libdrm2 libxkbcommon0 libx11-6
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1
    libpango-1.0-0 libcairo2 libasound2t64 fonts-liberation ffmpeg
)
for attempt in 1 2 3 4; do
    if timeout 5m sudo bash -c \
        'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"' \
        bash "${packages[@]}"; then
        exit 0
    fi
    echo "apt-get attempt $attempt/4 failed or timed out after 5m; retrying"
done
exit 1
