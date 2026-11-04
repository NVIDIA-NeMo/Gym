#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
    echo "Disk reclamation is only needed on GitHub-hosted runners; skipping."
    exit 0
fi

readonly workspace="${GITHUB_WORKSPACE:-/}"
readonly minimum_free_kb="${GYM_CI_MIN_FREE_DISK_KB:-10485760}"
if [[ ! "${minimum_free_kb}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GYM_CI_MIN_FREE_DISK_KB must be a positive integer: ${minimum_free_kb}" >&2
    exit 2
fi

echo "Disk usage before reclaiming hosted-runner images:"
df -h "${workspace}"

# Gym's CPU test jobs do not use these large hosted-runner SDKs. Removing them before
# dependency caches and per-environment virtualenvs are expanded avoids mid-test ENOSPC.
sudo rm -rf -- \
    /opt/ghc \
    /usr/local/.ghcup \
    /usr/local/lib/android \
    /usr/local/share/powershell \
    /usr/share/dotnet
sudo apt-get clean
if command -v docker >/dev/null 2>&1; then
    sudo docker system prune --all --force
fi

echo "Disk usage after reclaiming hosted-runner images:"
df -h "${workspace}"
available_kb="$(df -Pk "${workspace}" | awk 'NR == 2 {print $4}')"
if [[ ! "${available_kb}" =~ ^[0-9]+$ ]]; then
    echo "Could not determine free disk space for ${workspace}" >&2
    exit 1
fi
if ((available_kb < minimum_free_kb)); then
    echo "Insufficient runner disk space: ${available_kb} KiB available; ${minimum_free_kb} KiB required" >&2
    exit 1
fi

echo "Runner has ${available_kb} KiB free for the test launch."
