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
# Bound network operations separately so slow mirrors can be retried without
# killing dpkg during configuration. Refresh indexes only until successful:
# container update hooks can otherwise clear completed and partial downloads.
# The workflow timeout still bounds the complete installation.
# Keep tools already supplied by the runner. Upgrading Git can select its
# git-core PPA and make unrelated tests depend on that service being available.
indexes_updated=false
for attempt in 1 2 3 4; do
    if [[ "$indexes_updated" == false ]]; then
        if sudo timeout 5m apt-get update; then
            indexes_updated=true
        else
            echo "Package index update attempt $attempt/4 failed"
            continue
        fi
    fi
    if sudo bash -c '
        export DEBIAN_FRONTEND=noninteractive
        dpkg --configure -a &&
            timeout 5m apt-get install --download-only -y --no-install-recommends --no-upgrade "$@" &&
            apt-get install --no-download -y --no-install-recommends --no-upgrade "$@"
        ' bash "${packages[@]}"; then
        exit 0
    fi
    echo "System dependency install attempt $attempt/4 failed"
done
exit 1
