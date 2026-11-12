#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

# No arguments retains the portable, unsharded core contract.
if [[ $# -ne 0 && ( $# -ne 2 || ! "$1" =~ ^[0-9]+$ || ! "$2" =~ ^[0-9]+$ ) ]]; then
    echo "Usage: $0 [SHARD_INDEX NUM_SHARDS]" >&2
    exit 2
fi
if [[ $# -eq 2 ]] && (( 10#$2 == 0 || 10#$1 >= 10#$2 )); then
    echo "Expected 0 <= SHARD_INDEX < NUM_SHARDS" >&2
    exit 2
fi

cd "${repo_root}"
# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment core
unset -f gym_ci_sanitize_environment
# shellcheck source=scripts/ci/test_worker_env.sh
source "${ci_dir}/test_worker_env.sh"
# shellcheck source=scripts/ci/setup_dev.sh
source "${ci_dir}/setup_dev.sh"
# The telemetry extra pulls in nemo-lens so the lens-present unit tests under
# tests/unit_tests/telemetry (pytest.mark tests/unit_tests/telemetry/conftest.py::requires_lens)
# actually run in CI instead of silently skipping, which would leave that code permanently
# uncovered by the coverage gate below. mlflow and wandb are optional extras (SDKs now opt-in
# to keep a core install lean); this suite covers both their installed- and missing-dependency
# paths (tests/unit_tests/artifacts/test_exporters.py, tests/unit_tests/data/test_gitlab_utils.py), so both
# extras must be present here.
uv sync --extra dev --extra ray --extra telemetry --extra mlflow --extra wandb
pytest_addopts='-m "not sandbox" -n 4 --dist=loadfile --cov-report= --cov-fail-under=0 --color=yes'
if [[ $# -eq 2 ]]; then
    pytest_addopts+=" --core-shard-index=$1 --core-num-shards=$2"
fi
if [[ -n "${GYM_CI_JUNIT_DIR:-}" ]]; then
    mkdir -p "${GYM_CI_JUNIT_DIR}"
    printf -v junit_xml_path '%q' "${GYM_CI_JUNIT_DIR%/}/core.xml"
    pytest_addopts+=" --junitxml=${junit_xml_path} --junit-prefix=nemo_gym.core -o junit_family=xunit1"
fi
export PYTEST_ADDOPTS="${pytest_addopts}"
exec ng_dev_test
