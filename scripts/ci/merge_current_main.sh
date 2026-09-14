#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

cd "${repo_root}"
if [[ "${GITHUB_REF:-}" != refs/heads/pull-request/* ]]; then
    echo "Not a mirrored pull-request branch; testing the checked-out commit."
    exit 0
fi

git fetch --no-tags origin main
readonly current_main_ref="refs/remotes/origin/main"
if git merge-base --is-ancestor "${current_main_ref}" HEAD; then
    echo "The pull-request branch already contains current main."
    exit 0
fi

# Test the exact PR change combined with current main without rewriting the contributor's branch.
# A conflict is a validation failure and requires the PR branch to be refreshed.
git \
    -c user.name="NeMo Gym CI" \
    -c user.email="nemo-gym-ci@nvidia.com" \
    merge --no-edit --no-ff "${current_main_ref}"
echo "Testing synthetic merge $(git rev-parse HEAD) against current main $(git rev-parse "${current_main_ref}")."
