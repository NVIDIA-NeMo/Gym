#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

cd "${repo_root}"
case "${GYM_CI_USE_EXISTING_ENV:-0}" in
    0)
        # shellcheck source=scripts/ci/setup_dev.sh
        source "${ci_dir}/setup_dev.sh"
        ;;
    1)
        if [[ ! -x .venv/bin/python ]]; then
            echo "GYM_CI_USE_EXISTING_ENV=1 requires an existing .venv" >&2
            exit 2
        fi
        # GitHub pre-installs its public-only sandbox extra before the core coverage pass.
        # shellcheck disable=SC1091
        source .venv/bin/activate
        ;;
    *)
        echo "GYM_CI_USE_EXISTING_ENV must be 0 or 1" >&2
        exit 2
        ;;
esac
export PYTEST_ADDOPTS='-m "not sandbox" --cov-report= --cov-fail-under=0'
exec ng_dev_test
