#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

cd "${repo_root}"
# shellcheck source=scripts/ci/setup_dev.sh
source "${ci_dir}/setup_dev.sh"
export PYTEST_ADDOPTS='-m "not sandbox" --cov-report= --cov-fail-under=0'
exec uv run ng_dev_test
