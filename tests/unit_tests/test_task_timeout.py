# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout


@pytest.mark.parametrize(
    "metadata,expected",
    [
        (None, 1800),
        ({}, 1800),
        ({"instance_id": "task"}, 1800),
        ({"agent_timeout_sec": "900.0"}, 900),
        ({"agent_timeout_sec": "12.5"}, 12.5),
        ({"agent_timeout_sec": "3600"}, 1800),
    ],
)
def test_task_timeout_respects_task_budget_and_server_ceiling(
    metadata: dict[str, str] | None, expected: float
) -> None:
    assert resolve_agent_timeout(metadata, ceiling=1800) == expected


@pytest.mark.parametrize("raw", ["", "invalid", "0", "-1", "nan", "inf", "-inf", "1e1000"])
def test_invalid_task_timeout_keeps_server_ceiling_and_warns(raw: str, caplog: pytest.LogCaptureFixture) -> None:
    assert resolve_agent_timeout({"agent_timeout_sec": raw}, ceiling=1800) == 1800
    assert "Ignoring invalid agent_timeout_sec" in caplog.text
