# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.token_id_capture.delivery import TERMINAL_RESPONSE_ID_KEY
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


@pytest.mark.parametrize("cap", [None, 8, 16, 128])
async def test_native_runner_preserves_output_token_cap(tmp_path, runner_factory, cap):
    requests = []

    async def query(params):
        requests.append(params)
        return NeMoGymResponse(
            id="submitted",
            created_at=0,
            object="response",
            model="model",
            tools=[],
            tool_choice="auto",
            parallel_tool_calls=False,
            output=[
                {
                    "type": "function_call",
                    "call_id": "submit",
                    "name": "bash",
                    "arguments": json.dumps({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}),
                }
            ],
        )

    params = NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=cap)
    harness = await runner_factory(
        context=HarnessContext(session_id="token-cap", instruction="Submit"),
        config=MiniSWEConfig(step_limit=1),
        params=params,
        query=query,
        model_name="model",
        directory=tmp_path / "artifacts",
    )
    _, outcome, _ = await harness.execute(15)
    assert outcome.reason == "completed"
    assert len(requests) == 1
    if cap is None:
        assert "max_output_tokens" not in requests[0]
    else:
        assert requests[0]["max_output_tokens"] == cap
    assert params.max_output_tokens == cap


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
    recovery_prompt = requests[1]["input"][-1]["content"][0]["text"]
    assert ("output token limit" in recovery_prompt) == length_limited
    assert ("Respond more concisely" in recovery_prompt) == length_limited
    expected = "Submitted" if recover else "RepeatedFormatError"
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
    expected = (
        "ContextWindowExceededError"
        if "maximum context length" in message
        else "BadRequestError"
        if status == 400
        else "ServiceUnavailableError"
    )
    assert extra["mini_swe_trajectory"]["info"]["exit_status"] == expected
    assert json.loads((tmp_path / "trajectory.json").read_text())["info"]["exit_status"] == expected
    if is_context_overflow:
        assert outcome.detail == "ContextWindowExceededError"


WINDOW = 393216


def usage(input_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": output_tokens},
        "total_tokens": input_tokens + output_tokens,
    }


def model_response(index, *, command=None, cut_at=None):
    """A kept bash call (``command``), or a reply the server cut so that input + output == ``cut_at``."""
    cut = command is None
    output = []
    if not cut:
        output = [
            {
                "type": "function_call",
                "call_id": f"call-{index}",
                "name": "bash",
                "arguments": json.dumps({"command": command}),
            }
        ]
    input_tokens = 392_000 + 58 * index
    return NeMoGymResponse(
        id=f"response-{index}",
        created_at=0,
        object="response",
        model="model",
        tools=[],
        tool_choice="auto",
        parallel_tool_calls=False,
        status="incomplete" if cut else "completed",
        incomplete_details={"reason": "max_output_tokens"} if cut else None,
        output=output,
        usage=usage(input_tokens, (cut_at - input_tokens) if cut else 40),
    )


async def run_scripted(tmp_path, runner_factory, script, *, window):
    requests = []

    async def query(params):
        requests.append(params)
        return script(len(requests))

    harness = await runner_factory(
        context=HarnessContext(session_id="window-test", instruction="Inspect and submit"),
        config=MiniSWEConfig(context_window_tokens=window),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=query,
        model_name="model",
        directory=tmp_path,
    )
    _, outcome, extra = await harness.execute(15)
    return requests, outcome, extra


@pytest.mark.parametrize(
    "window,cut_at,expected",
    [
        (WINDOW, WINDOW, "ContextWindowExceededError"),  # every retry was cut at the window
        (None, WINDOW, "RepeatedFormatError"),  # no window configured: the plain label
        (WINDOW, WINDOW - 1_000, "RepeatedFormatError"),  # cut below the window: a format-error quit
    ],
)
async def test_format_errors_cut_at_the_context_window_are_a_context_overflow(
    tmp_path, runner_factory, window, cut_at, expected
):
    requests, outcome, extra = await run_scripted(
        tmp_path, runner_factory, lambda n: model_response(n, cut_at=cut_at), window=window
    )
    assert len(requests) == 3
    assert extra["mini_swe_trajectory"]["info"]["exit_status"] == "RepeatedFormatError"
    assert outcome.reason == "nonzero_exit" and outcome.detail == expected


def kept_then_cut(n):
    return model_response(n, command="echo kept") if n == 1 else model_response(n, cut_at=WINDOW)


def cut_then_submit(n):
    if n == 1:
        return model_response(n, cut_at=WINDOW)
    return model_response(n, command="echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


@pytest.mark.parametrize(
    "script,declared",
    [
        (kept_then_cut, "response-1"),  # three cut replies after a kept call: train on the kept call
        (cut_then_submit, "response-2"),  # the submission is the last kept call
        (lambda n: model_response(n, cut_at=WINDOW), None),  # nothing kept: declare nothing
    ],
)
async def test_harness_declares_its_last_kept_reply_as_the_terminal_call(tmp_path, runner_factory, script, declared):
    _, _, extra = await run_scripted(tmp_path, runner_factory, script, window=WINDOW)
    assert extra.get(TERMINAL_RESPONSE_ID_KEY) == declared
