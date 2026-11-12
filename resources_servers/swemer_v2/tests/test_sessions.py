# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise swemer_v2's episode adapter and failure accounting."""

import pytest

from tests.helpers.swe_resource_sessions import (
    check_swe_resource_episode,
    check_swe_session_config,
    check_swe_verifier_interruption,
)


@pytest.mark.parametrize(
    "scenario",
    [
        "correct",
        "incorrect",
        "timeout",
        "legacy_timeout",
        "provider_error",
        "logs_disabled",
        "incomplete",
        "capture_unavailable",
        "git_failure",
        "stop_retry",
        "close_without_verify",
        "task_mismatch",
        "legacy",
        "conformance",
    ],
)
def test_episode(monkeypatch, tmp_path, scenario):
    check_swe_resource_episode(monkeypatch, tmp_path, "swemer_v2", "SwemerV2", scenario)


@pytest.mark.parametrize("error_type", ["sandbox", "timeout"])
async def test_interrupted_grading(tmp_path, error_type):
    await check_swe_verifier_interruption(tmp_path, "swemer_v2", "SwemerV2", error_type)


def test_session_config():
    check_swe_session_config("swemer_v2")
