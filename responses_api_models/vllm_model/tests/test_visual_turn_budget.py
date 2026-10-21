# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.rollout_correlation import rollout_context
from responses_api_agents.opencode_multimodal_agent.eval_budget import FINAL_TURN_NOTICE, EvalTurnBudget
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.visual_eval import VisualEvalModel


def request():
    return Request({"type": "http", "state": {}, "method": "POST", "path": "/v1/chat/completions", "headers": []})


def body():
    return NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(
        {"messages": [{"role": "user", "content": "Keep reading"}], "tool_choice": "auto"}
    )


def completion(tools=3):
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tools else "length",
                    "message": {
                        "role": "assistant",
                        "content": "Answer: A",
                        "tool_calls": [
                            {"id": f"t{i}", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                            for i in range(tools)
                        ]
                        or None,
                    },
                }
            ],
        }
    )


def gateway(monkeypatch, budget):
    monkeypatch.setattr(VisualEvalModel, "model_post_init", lambda self, context: None)
    model = VisualEvalModel.model_construct(config=SimpleNamespace(turn_budget=budget))
    # The startup patch also bypasses Pydantic's private-attribute initialization.
    model.__pydantic_private__ = {"_turn_states": {}}
    return model


@pytest.mark.asyncio
async def test_55_plus_10_enforced_even_if_client_and_model_ignore_limits(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    seen = []

    async def upstream(self, req, params):
        seen.append(params)
        # Adversarial provider: always returns three tools, including when none
        # was requested. The client below also refuses to stop after a final.
        return completion()

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    responses = []
    with rollout_context("episode-a"):
        for _ in range(65):
            responses.append(await model.chat_completions(request(), body()))
        for _ in range(3):
            with pytest.raises(HTTPException) as error:
                await model.chat_completions(request(), body())
            assert error.value.status_code == 400
    assert len(seen) == 65
    assert all(FINAL_TURN_NOTICE not in str(p.messages) for p in seen[:54])
    assert all(p.messages[-1]["content"] == FINAL_TURN_NOTICE for p in seen[54:])
    assert [p.tool_choice for p in seen[55:57]] == ["auto", "auto"]
    assert [p.tool_choice for p in seen[57:]] == ["none"] * 8
    assert [len(r.choices[0].message.tool_calls or []) for r in responses[55:]] == [1, 1] + [0] * 8
    assert responses[-1].choices[0].finish_reason == "stop"
    assert responses[-1].choices[0].message.content == "Answer: A"


@pytest.mark.asyncio
async def test_concurrent_requests_cannot_race_past_hard_limit_and_rollouts_are_isolated(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget(regular_turns=1, grace_turns=1, grace_tool_turns=0))
    upstream_calls = []

    async def upstream(self, req, params):
        await asyncio.sleep(0)
        upstream_calls.append(params)
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("a"):
        results = await asyncio.gather(
            *(model.chat_completions(request(), body()) for _ in range(8)), return_exceptions=True
        )
    assert sum(isinstance(r, HTTPException) for r in results) == 6
    assert len(upstream_calls) == 2
    with rollout_context("b"):
        await model.chat_completions(request(), body())
    assert len(upstream_calls) == 3


@pytest.mark.asyncio
async def test_missing_correlation_fails_before_upstream(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    with rollout_context(None), pytest.raises(HTTPException, match="rollout id"):
        await model.chat_completions(request(), body())


def test_config_overrides_cannot_reenable_tools(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    model.config.max_images = None
    monkeypatch.setattr(
        VLLMModel,
        "_preprocess_chat_completion_create_params",
        lambda self, req, data: dict(data, tool_choice="required", n=4, parallel_tool_calls=True),
    )
    req = request()
    req.state.visual_eval_budget = {"allowance": 0}
    result = model._preprocess_chat_completion_create_params(req, {"messages": []})
    assert result["tool_choice"] == "none"
    assert result["parallel_tool_calls"] is False
    assert result["n"] == 1


@pytest.mark.asyncio
async def test_budget_is_opt_in(monkeypatch):
    model = gateway(monkeypatch, None)
    expected = completion()

    async def upstream(self, req, params):
        assert params.tool_choice == "auto"
        return expected

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    assert await model.chat_completions(request(), body()) is expected
