# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessContext, MiniSWEConfig


async def make_harness(tmp_path, query, runner_factory):
    harness = await runner_factory(
        context=HarnessContext(session_id="limit-test", instruction="Inspect and submit"),
        config=MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
        observability_enabled=True,
    )
    return harness


@pytest.mark.parametrize("recover", [True, False])
@pytest.mark.parametrize("length_limited", [True, False])
@pytest.mark.parametrize("malformed_call", [True, False])
async def test_length_limit_recovery_and_terminal_classification(
    tmp_path, runner_factory, recover, length_limited, malformed_call
):
    requests = []

    async def query(params):
        requests.append(params)
        submit = recover and len(requests) == 2
        output = []
        if submit or malformed_call:
            output = [
                {
                    "type": "function_call",
                    "call_id": f"call-{len(requests)}",
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})
                    if submit
                    else "{}",
                }
            ]
        return NeMoGymResponse(
            id=f"response-{len(requests)}",
            created_at=0,
            object="response",
            model="model",
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
            status="incomplete" if length_limited and not submit else "completed",
            incomplete_details={"reason": "max_output_tokens"} if length_limited and not submit else None,
            output=output,
        )

    harness = await make_harness(tmp_path, query, runner_factory)
    _, outcome, extra = await harness.execute(15)
    assert len(requests) == (2 if recover else 3)
    recovery_prompt = requests[1]["input"][-1]["content"]
    assert ("output token limit" in recovery_prompt) == length_limited
    assert ("Respond more concisely" in recovery_prompt) == length_limited
    expected = "Submitted" if recover else "OutputTokenLimitExceeded" if length_limited else "RepeatedFormatError"
    assert extra["mini_swe_trajectory"]["info"]["exit_status"] == expected
    assert json.loads((tmp_path / "trajectory.json").read_text())["info"]["exit_status"] == expected
    assert outcome.reason == ("completed" if recover else "nonzero_exit")
    if not recover:
        assert outcome.detail == expected
    assert len(extra["ng_trajectory"]["turns"]) == len(requests)


@pytest.mark.parametrize(
    "status,message,is_context_overflow",
    [
        (400, "This model's maximum context length is 262144 tokens", True),
        (400, "max_tokens is too large: input length and max_tokens exceed the context", True),
        (400, "context_length_exceeded", True),
        (400, "max_tokens must be positive", False),
        (400, "Invalid tool schema", False),
        (503, "context length service unavailable", False),
    ],
)
async def test_context_overflow_stops_without_format_retries(
    tmp_path, runner_factory, status, message, is_context_overflow
):
    error = ClientResponseError(MagicMock(real_url="http://model/v1/responses"), (), status=status)
    error.response_content = json.dumps({"error": {"message": message}}).encode()
    query = AsyncMock(side_effect=error)
    harness = await make_harness(tmp_path, query, runner_factory)
    _, outcome, extra = await harness.execute(15)
    query.assert_awaited_once()
    assert outcome.reason == ("nonzero_exit" if is_context_overflow else "infrastructure_error")
    expected = "ContextWindowExceeded" if is_context_overflow else "RuntimeError"
    assert extra["mini_swe_trajectory"]["info"]["exit_status"] == expected
    assert json.loads((tmp_path / "trajectory.json").read_text())["info"]["exit_status"] == expected
    if is_context_overflow:
        assert outcome.detail == "ContextWindowExceeded"
