# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermes stopping on repeated unknown tool calls is a graded stop, not a harness failure."""

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.hermes_sandboxed_agent.app import trajectory_response
from responses_api_agents.hermes_sandboxed_agent.runner import classify_stop


MESSAGES = [
    {"role": "user", "content": "fix the bug"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Unknown tool 'bash'"},
]


def _hermes_partial(error):
    return {"completed": False, "partial": True, "error": error, "n_input": 1, "messages": list(MESSAGES)}


def test_invalid_tool_call_stop_is_graded_not_failed():
    result = classify_stop(_hermes_partial("Model generated invalid tool call: bash"))
    assert result["stop_reason"] == "invalid_tool_call"
    assert result["failed"] is False and result["completed"] is False
    assert "error" not in result and result["stop_detail"] == "Model generated invalid tool call: bash"
    # The worker exits 0 for such a result, so the agent sees no error_type and grades it.
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="fix the bug"), "policy")
    assert response.status == "incomplete"
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]
    assert response.metadata["stop_reason"] == "invalid_tool_call"
    assert response.metadata["budget_exhausted"] == "false"


def test_other_partial_errors_stay_failures():
    result = classify_stop(_hermes_partial("Provider returned 500"))
    assert "stop_reason" not in result
    response = trajectory_response(result, NeMoGymResponseCreateParamsNonStreaming(input="fix the bug"), "policy")
    assert response.status == "failed"
