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
# Workflow timeouts bound installation. A shorter per-attempt timeout
# can kill dpkg during configuration and leave every subsequent install blocked.
# Complete any interrupted configuration before retrying the package install.
for attempt in 1 2 3 4; do
    if sudo bash -c '
        export DEBIAN_FRONTEND=noninteractive
        dpkg --configure -a &&
            apt-get update &&
            apt-get install -y --no-install-recommends "$@"
        ' bash "${packages[@]}"; then
        exit 0
    fi
    echo "System dependency install attempt $attempt/4 failed"
done
exit 1
