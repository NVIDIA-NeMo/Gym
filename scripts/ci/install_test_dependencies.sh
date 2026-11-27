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
    bubblewrap
)
# Bound network operations separately so slow mirrors can be retried without
# killing dpkg during configuration. Refresh indexes only until successful:
# container update hooks can otherwise clear completed and partial downloads.
# The workflow timeout still bounds the complete installation.
# Keep tools already supplied by the runner. Upgrading Git can select its
# git-core PPA and make unrelated tests depend on that service being available.
# The bwrap sandbox tests need unprivileged user namespaces, which Ubuntu 24.04's AppArmor default refuses
# ("setting up uid map: Permission denied"). Lift that for the job; NEMO_GYM_REQUIRE_BWRAP makes a runner where
# it cannot be lifted fail those tests instead of skipping them.
if [[ -e /proc/sys/kernel/apparmor_restrict_unprivileged_userns ]]; then
    sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0 ||
        echo "Could not allow unprivileged user namespaces; the bwrap tests will fail"
fi

# archive.ubuntu.com can be too slow to deliver this download within the step timeout. The first attempt uses
# Ubuntu's mirror inside AWS through private copies of the APT sources; a failed attempt returns to the runner's own.
apt_etc="${GYM_CI_APT_ETC:-/etc/apt}"
apt_mirror="${GYM_CI_APT_MIRROR-http://us-east-1.ec2.archive.ubuntu.com/ubuntu}"
apt_config=""
if [[ -n "$apt_mirror" ]] &&
    grep -rqsE 'https?://archive\.ubuntu\.com/ubuntu' "$apt_etc/sources.list" "$apt_etc/sources.list.d"; then
    mirror_dir="$RUNNER_TEMP/apt-mirror-sources"
    mkdir -p "$mirror_dir/sources.list.d"
    : > "$mirror_dir/sources.list"
    for source in "$apt_etc/sources.list" "$apt_etc"/sources.list.d/*; do
        if [[ ! -f "$source" ]]; then continue; fi
        target="$mirror_dir/sources.list.d/${source##*/}"
        if [[ "$source" == "$apt_etc/sources.list" ]]; then target="$mirror_dir/sources.list"; fi
        sed -E "s#https?://archive\.ubuntu\.com/ubuntu#${apt_mirror}#g" "$source" > "$target"
    done
    printf 'Dir::Etc::SourceList "%s";\nDir::Etc::SourceParts "%s";\n' \
        "$mirror_dir/sources.list" "$mirror_dir/sources.list.d" > "$mirror_dir/apt.conf"
    apt_config="$mirror_dir/apt.conf"
    echo "Using APT mirror $apt_mirror"
fi

use_runner_sources() {
    if [[ -n "$apt_config" ]]; then
        echo "Falling back to the runner's APT sources"
        apt_config=""
        indexes_updated=false
    fi
}

indexes_updated=false
for attempt in 1 2 3 4; do
    if [[ "$indexes_updated" == false ]]; then
        if sudo env ${apt_config:+"APT_CONFIG=$apt_config"} timeout 5m apt-get update; then
            indexes_updated=true
        else
            echo "Package index update attempt $attempt/4 failed"
            use_runner_sources
            continue
        fi
    fi
    if sudo bash -c '
        if [[ -n "$1" ]]; then export APT_CONFIG="$1"; fi
        shift
        export DEBIAN_FRONTEND=noninteractive
        dpkg --configure -a &&
            timeout 5m apt-get install --download-only -y --no-install-recommends --no-upgrade "$@" &&
            apt-get install --no-download -y --no-install-recommends --no-upgrade "$@"
        ' bash "$apt_config" "${packages[@]}"; then
        exit 0
    fi
    echo "System dependency install attempt $attempt/4 failed"
    use_runner_sources
done
exit 1
