# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.core.operation_timing import OperationTiming
from nemo_gym.trajectory.tool_timing import normalize_tool_calls
from nemo_gym.trajectory.trajectory_record import AgentInvocation, ObservationGap, ToolCallObservation


@pytest.mark.parametrize("completed_at", [None, 12])
def test_terminal_status_resolves_completion_gap_only_with_end_evidence(completed_at):
    observed = ToolCallObservation(
        invocation_id="root",
        tool_call_id="call",
        span_id="attempt",
        status="completed",
        timings=[
            OperationTiming(boundary="lifecycle", timing_source="test", started_at=10, completed_at=completed_at)
        ],
    )
    _, gaps = normalize_tool_calls(
        [observed],
        capture_key="capture",
        producer_id="test",
        model_calls=[],
        invocations=[AgentInvocation(invocation_id="root")],
        observation_gaps=[ObservationGap(code="tool_completion_unavailable", invocation_id="root", span_id="attempt")],
    )
    assert any(gap.code == "tool_completion_unavailable" for gap in gaps) == (completed_at is None)
