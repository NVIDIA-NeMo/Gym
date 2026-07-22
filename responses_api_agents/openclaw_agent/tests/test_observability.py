# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.openai_utils import NeMoGymFunctionCallOutput, NeMoGymResponseFunctionToolCall
from responses_api_agents.openclaw_agent.observability import build_openclaw_observations


def test_builds_root_observation_and_pairs_tools() -> None:
    bundle = build_openclaw_observations(
        "session-1",
        [
            NeMoGymResponseFunctionToolCall(
                arguments='{"query":"AAPL"}',
                call_id="call-1",
                name="web_search",
                id="call-1",
                status="completed",
            ),
            NeMoGymFunctionCallOutput(call_id="call-1", output="result", status="completed"),
        ],
        transcript_available=True,
    )

    assert bundle.invocations[0].invocation_id == "session-1"
    assert bundle.invocations[0].conversation
    assert bundle.tool_calls[0].tool_call_id == "call-1"
    assert bundle.tool_calls[0].status == "unknown"
    assert {gap.code for gap in bundle.gaps} == {
        "subagent_hierarchy_unavailable",
        "model_call_ownership_unavailable",
        "context_compaction_unavailable",
        "tool_timing_unavailable",
    }


def test_marks_missing_transcript_and_incomplete_tool() -> None:
    bundle = build_openclaw_observations(
        "fallback",
        [
            NeMoGymResponseFunctionToolCall(
                arguments="{}",
                call_id="call-1",
                name="tool",
                id="call-1",
                status="completed",
            )
        ],
        transcript_available=False,
    )

    assert bundle.tool_calls[0].status == "incomplete"
    assert "agent_transcript_unavailable" in {gap.code for gap in bundle.gaps}


def test_result_presence_does_not_claim_success_and_preserves_source() -> None:
    bundle = build_openclaw_observations(
        "run-1",
        [
            NeMoGymResponseFunctionToolCall(arguments="{}", call_id="call-1", name="tool"),
            NeMoGymFunctionCallOutput(call_id="call-1", output="partial", status="incomplete"),
        ],
        transcript_available=True,
        source="pinchbench",
    )

    assert bundle.source == "pinchbench"
    assert bundle.tool_calls[0].status == "unknown"
    assert {gap.source for gap in bundle.gaps} == {"pinchbench"}
