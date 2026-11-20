# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import pytest

from nemo_gym.utils import ClockContext, ToolCallRecorder
from responses_api_agents.hermes_agent.sandbox_observer import TOOL_EVENTS_FILE


@pytest.fixture
def tool_events_path(tmp_path: Path) -> Path:
    return tmp_path / TOOL_EVENTS_FILE


@pytest.fixture
def tool_recorder(tool_events_path: Path) -> Callable[..., ToolCallRecorder]:
    """Per-invocation, per-boundary recorders as the sandbox runner builds them, writing to ``tool_events_path``."""
    clock = ClockContext(location="sandbox", owner="test")

    def recorder(invocation_id: str, boundary: str = "lifecycle") -> ToolCallRecorder:
        return ToolCallRecorder(
            path=tool_events_path,
            producer_id=f"test:{invocation_id}:{boundary}",
            invocation_id=invocation_id,
            agent_instance=None,
            clock=clock,
            boundary=boundary,
            timing_source="hermes.invoke_tool" if boundary == "executor" else "hermes.tool_callback",
        )

    return recorder
