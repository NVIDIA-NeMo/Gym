#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ci_dir
repo_root="$(cd "${ci_dir}/../.." && pwd)"
readonly repo_root

cd "${repo_root}"
# shellcheck source=scripts/ci/sanitize_env.sh
source "${ci_dir}/sanitize_env.sh"
gym_ci_sanitize_environment core
unset -f gym_ci_sanitize_environment
# shellcheck source=scripts/ci/setup_dev.sh
source "${ci_dir}/setup_dev.sh"
# The telemetry extra pulls in nemo-lens so the lens-present unit tests under
# tests/unit_tests/telemetry (pytest.mark tests/unit_tests/telemetry/conftest.py::requires_lens)
# actually run in CI instead of silently skipping, which would leave that code permanently
# uncovered by the coverage gate below. mlflow and wandb are optional extras (SDKs now opt-in
# to keep a core install lean); this suite covers both their installed- and missing-dependency
# paths (tests/unit_tests/test_exporters.py, tests/unit_tests/test_gitlab_utils.py), so both
# extras must be present here.
uv sync --extra dev --extra telemetry --extra mlflow --extra wandb
pytest_addopts='-m "not sandbox" --cov-report= --cov-fail-under=0 --color=yes'
if [[ -n "${GYM_CI_JUNIT_DIR:-}" ]]; then
    mkdir -p "${GYM_CI_JUNIT_DIR}"
    printf -v junit_xml_path '%q' "${GYM_CI_JUNIT_DIR%/}/core.xml"
    pytest_addopts+=" --junitxml=${junit_xml_path} --junit-prefix=nemo_gym.core"
fi
export PYTEST_ADDOPTS="${pytest_addopts}"
exec ng_dev_test
