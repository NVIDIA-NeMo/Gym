# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the graph algorithms independently of the host agent adapter."""

import asyncio
import importlib
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseUsage


MODEL_URL = "http://model.example/ng-rollout/episode/training-token-capture/v1/responses"


@pytest.fixture
def workflows():
    pytest.importorskip("langgraph", reason="Install runtime-requirements.txt to test the sandbox graph runtime")
    pytest.importorskip("langchain_core", reason="Install runtime-requirements.txt to test the sandbox graph runtime")
    return importlib.import_module("responses_api_agents.langgraph_agent.workflows")


def _usage(input_tokens: int, output_tokens: int, reasoning_tokens: int, cached_tokens: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens, "audio_tokens": input_tokens % 3},
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
            "accepted_prediction_tokens": output_tokens % 5,
        },
        "total_tokens": input_tokens + output_tokens,
    }


def _model_response(text: str, usage: dict | None = None):
    payload = {
        "id": "response",
        "created_at": 1770000000.0,
        "model": "test",
        "object": "response",
        "output": [
            {
                "id": "message",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }
    response = AsyncMock()
    response.read.return_value = json.dumps(payload).encode()
    response.cookies = {}
    response.ok = True
    return response


def _body() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input="What is 6 * 7?", temperature=0.0, max_output_tokens=42)


async def _run(workflows, workflow: str, config: dict | None = None):
    return await workflows.run_workflow(
        workflow=workflow, config=config or {}, request_body=_body(), model_url=MODEL_URL
    )


def test_usage_preserves_provider_fields_and_unknowns(workflows) -> None:
    first = NeMoGymResponseUsage.model_validate(_usage(10, 20, 7, cached_tokens=2))
    second = NeMoGymResponseUsage.model_validate(_usage(11, 21, 8, cached_tokens=3))
    usage = workflows.aggregate_response_usage([first, second])
    assert usage.input_tokens == 21
    assert usage.input_tokens_details.cached_tokens == 5
    assert usage.input_tokens_details.audio_tokens == 3
    assert usage.output_tokens == 41
    assert usage.output_tokens_details.reasoning_tokens == 15
    assert usage.output_tokens_details.accepted_prediction_tokens == 1
    assert usage.total_tokens == 62
    assert workflows.aggregate_response_usage([first, None]) is None
    assert workflows.aggregate_response_usage([]) is None


async def test_reflection_stops_on_answer_and_keeps_full_model_route(workflows, monkeypatch) -> None:
    model = AsyncMock(return_value=_model_response("<answer>42</answer>"))
    monkeypatch.setattr(workflows, "request", model)
    response = await _run(workflows, "reflection")
    assert model.await_count == 1
    assert response.output[0].content[0].text == "<answer>42</answer>"
    call = model.call_args
    assert call.args == ("POST", MODEL_URL)
    assert call.kwargs["cookies"] == {}
    assert call.kwargs["json"]["temperature"] == 0.0
    assert call.kwargs["json"]["max_output_tokens"] == 42


async def test_reflection_inputs_are_chronological_without_duplicates(workflows, monkeypatch) -> None:
    model = AsyncMock(
        side_effect=[
            _model_response("initial draft", _usage(10, 20, 7, 1)),
            _model_response("specific critique", _usage(11, 21, 8, 2)),
            _model_response("revised <answer>42</answer>", _usage(12, 22, 9, 3)),
        ]
    )
    monkeypatch.setattr(workflows, "request", model)
    response = await _run(workflows, "reflection")
    inputs = [call.kwargs["json"]["input"] for call in model.call_args_list]
    assert [[message["role"] for message in items] for items in inputs] == [
        ["user"],
        ["user", "assistant", "user"],
        ["user", "assistant", "user"],
    ]
    assert [message["content"] for message in inputs[1]] == [
        "What is 6 * 7?",
        "initial draft",
        "Critique your solution. What could be wrong?",
    ]
    assert [message["content"] for message in inputs[2]] == [
        "What is 6 * 7?",
        "initial draft",
        "specific critique",
    ]
    assert [item.content[0].text for item in response.output] == [
        "initial draft",
        "specific critique",
        "revised <answer>42</answer>",
    ]
    assert response.usage.input_tokens == 33
    assert response.usage.input_tokens_details.cached_tokens == 6
    assert response.usage.output_tokens == 63
    assert response.usage.output_tokens_details.reasoning_tokens == 24
    assert response.usage.total_tokens == 96


@pytest.mark.parametrize(
    "workflow, replies, config",
    [
        ("orchestrator", ["SUBTASK 1: first\nSUBTASK 2: second", "one", "two", "<answer>42</answer>"], {}),
        ("parallel_thinking", ["one", "two", "<answer>42</answer>"], {"num_parallel_paths": 2}),
        ("rewoo", ["Plan: solve #E1 = LLM[first]", "one", "<answer>42</answer>"], {}),
    ],
)
async def test_stages_send_only_current_prompt_and_return_only_model_outputs(
    workflows, monkeypatch, workflow, replies, config
) -> None:
    model = AsyncMock(side_effect=[_model_response(text) for text in replies])
    monkeypatch.setattr(workflows, "request", model)
    response = await _run(workflows, workflow, config)
    assert [item.content[0].text for item in response.output] == replies
    assert response.usage is None
    for call in model.call_args_list:
        assert call.args == ("POST", MODEL_URL)
        assert len(call.kwargs["json"]["input"]) == 1
        assert call.kwargs["json"]["input"][0]["role"] == "user"
    if workflow == "orchestrator":
        prompts = [call.kwargs["json"]["input"][0]["content"] for call in model.call_args_list]
        assert prompts[1].endswith("Sub-task: first")
        assert prompts[2].endswith("Sub-task: second")
        assert "Result: one" in prompts[-1] and "Result: two" in prompts[-1]


async def test_rewoo_no_steps_falls_back_to_solve(workflows, monkeypatch) -> None:
    replies = ["I can solve this directly.", "<answer>42</answer>"]
    model = AsyncMock(side_effect=[_model_response(text) for text in replies])
    monkeypatch.setattr(workflows, "request", model)
    response = await _run(workflows, "rewoo")
    assert model.await_count == 2
    assert [item.content[0].text for item in response.output] == replies


@pytest.mark.parametrize("missing_usage", [False, True])
async def test_parallel_usage_includes_each_path_and_synthesis(workflows, monkeypatch, missing_usage) -> None:
    model = AsyncMock(
        side_effect=[
            _model_response("one", _usage(10, 20, 7, 1)),
            _model_response("two", None if missing_usage else _usage(11, 21, 8, 2)),
            _model_response("<answer>42</answer>", _usage(12, 22, 9, 3)),
        ]
    )
    monkeypatch.setattr(workflows, "request", model)
    response = await _run(workflows, "parallel_thinking", {"num_parallel_paths": 2})
    assert model.await_count == 3
    if missing_usage:
        assert response.usage is None
    else:
        assert response.usage.input_tokens == 33
        assert response.usage.input_tokens_details.cached_tokens == 6
        assert response.usage.output_tokens == 63
        assert response.usage.output_tokens_details.reasoning_tokens == 24
        assert response.usage.total_tokens == 96


@pytest.mark.parametrize(
    "tool_inputs,evidence,expected",
    [
        (
            [f"Get evidence {i}" for i in range(1, 12)] + ["Compare #E1, #E10, and #E1; leave #E99"],
            [f"result({i})" for i in range(1, 13)],
            "Compare result(1), result(10), and result(1); leave #E99",
        ),
        (
            ["Get first evidence", "Get second evidence", "Combine #E1 with #E2; leave #E99"],
            [r"first \1 \g<name> contains #E2", r"second \2", "combined evidence"],
            r"Combine first \1 \g<name> contains #E2 with second \2; leave #E99",
        ),
    ],
)
async def test_rewoo_substitutes_complete_references_once(
    workflows, monkeypatch, tool_inputs, evidence, expected
) -> None:
    plan = "\n".join(
        f"Plan: Collect evidence {i}. #E{i} = LLM[{tool_input}]" for i, tool_input in enumerate(tool_inputs, start=1)
    )
    model = AsyncMock(
        side_effect=[
            _model_response(plan),
            *[_model_response(text) for text in evidence],
            _model_response("<answer>42</answer>"),
        ]
    )
    monkeypatch.setattr(workflows, "request", model)
    await _run(workflows, "rewoo")
    prompts = [call.kwargs["json"]["input"][0]["content"] for call in model.call_args_list]
    assert len(prompts) == len(tool_inputs) + 2
    assert prompts[-2] == expected
    assert f"#E{len(tool_inputs)} = LLM[{expected}]" in prompts[-1]


async def test_parallel_failure_cancels_and_drains_siblings(workflows, monkeypatch) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    active = 0

    async def model(*args, **kwargs):
        nonlocal active
        prompt = kwargs["json"]["input"][0]["content"]
        if prompt.startswith("fail"):
            await started.wait()
            raise RuntimeError("model failed")
        active += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            cancelled.set()

    monkeypatch.setattr(workflows, "request", model)
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await _run(
            workflows,
            "parallel_thinking",
            {
                "num_parallel_paths": 2,
                "perspective_prompts": ["fail", "wait"],
            },
        )
    assert cancelled.is_set()
    assert active == 0
