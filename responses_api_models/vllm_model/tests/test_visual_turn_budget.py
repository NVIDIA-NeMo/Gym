# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectionError
from fastapi import HTTPException, Request

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, rollout_context
from responses_api_agents.opencode_multimodal_agent.eval_budget import FINAL_TURN_NOTICE, EvalTurnBudget
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.hosted_visual_eval import HostedVisualEvalModel
from responses_api_models.vllm_model.visual_eval import VisualEvalModel, VisualEvalModelConfig


def request(session_id="root"):
    headers = [(b"x-opencode-session-id", session_id.encode())] if session_id else []
    return Request({"type": "http", "state": {}, "method": "POST", "path": "/v1/chat/completions", "headers": headers})


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


def gateway(monkeypatch, budget, model_class=VisualEvalModel):
    monkeypatch.setattr(VLLMModel, "_post_init", lambda self: None)
    config = VisualEvalModelConfig(
        host="localhost",
        port=8081,
        name="policy",
        entrypoint="visual_eval.py",
        base_url="http://localhost/v1",
        api_key="unused",
        model="policy",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        turn_budget=budget,
    )
    return model_class(config=config, server_client=MagicMock(spec=ServerClient))


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
            assert error.value.status_code == 422
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
@pytest.mark.parametrize("model_class", [VisualEvalModel, HostedVisualEvalModel])
@pytest.mark.parametrize(
    "error_factory",
    [
        pytest.param(lambda: HTTPException(429, "rate limited"), id="429"),
        pytest.param(lambda: HTTPException(503, "unavailable"), id="503"),
        pytest.param(lambda: ClientConnectionError("disconnected"), id="transport"),
        pytest.param(lambda: TimeoutError("upstream timed out"), id="timeout"),
    ],
)
async def test_failed_calls_then_success_count_as_one_turn(monkeypatch, model_class, error_factory):
    model = gateway(monkeypatch, EvalTurnBudget(), model_class)
    attempts = 0

    async def upstream(self, req, params):
        nonlocal attempts
        attempts += 1
        if attempts <= 12:
            raise error_factory()
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        for _ in range(12):
            with pytest.raises(type(error_factory())):
                await model.chat_completions(request(), body())
            assert model._turn_states[("episode", "root")].turns == 0
        await model.chat_completions(request(), body())
    assert attempts == 13
    assert model._turn_states[("episode", "root")].turns == 1


@pytest.mark.asyncio
async def test_failed_boundary_attempts_preserve_notice_and_grace_allowance(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    fail_next = False
    seen = []

    async def upstream(self, req, params):
        nonlocal fail_next
        seen.append(params)
        if fail_next:
            fail_next = False
            raise HTTPException(503, "unavailable")
        return completion()

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    successful = []
    with rollout_context("episode"):
        for turn in range(1, 66):
            if turn in (54, 55, 56, 57, 58, 65):
                state = model._turn_states[("episode", "root")]
                before = (state.turns, state.grace_tool_calls)
                fail_next = True
                with pytest.raises(HTTPException):
                    await model.chat_completions(request(), body())
                assert (state.turns, state.grace_tool_calls) == before
            successful.append(await model.chat_completions(request(), body()))
            if turn in (54, 55, 56, 57, 58, 65):
                assert seen[-2].model_dump() == seen[-1].model_dump()
        with pytest.raises(HTTPException) as error:
            await model.chat_completions(request(), body())
    assert error.value.status_code == 422
    assert model._turn_states[("episode", "root")].turns == 65
    assert model._turn_states[("episode", "root")].grace_tool_calls == 2
    assert [len(r.choices[0].message.tool_calls or []) for r in successful[55:]] == [1, 1] + [0] * 8
    assert successful[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_cancelled_call_does_not_spend_a_turn_or_hold_the_lock(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    entered = asyncio.Event()

    async def upstream(self, req, params):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        task = asyncio.create_task(model.chat_completions(request(), body()))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert model._turn_states[("episode", "root")].turns == 0

        async def success(self, req, params):
            return completion(0)

        monkeypatch.setattr(VLLMModel, "chat_completions", success)
        await asyncio.wait_for(model.chat_completions(request(), body()), 1)
    assert model._turn_states[("episode", "root")].turns == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("choice_count", [0, 2])
async def test_invalid_completion_does_not_spend_a_turn(monkeypatch, choice_count):
    model = gateway(monkeypatch, EvalTurnBudget())

    async def upstream(self, req, params):
        response = completion(0)
        return response.model_copy(update={"choices": response.choices * choice_count})

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"), pytest.raises(ValueError, match="exactly one returned completion"):
        await model.chat_completions(request(), body())
    assert model._turn_states[("episode", "root")].turns == 0


@pytest.mark.asyncio
async def test_concurrent_failures_do_not_exhaust_successful_turn_budget(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget(regular_turns=1, grace_turns=1, grace_tool_turns=0))
    attempts = 0

    async def upstream(self, req, params):
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0)
        if attempts <= 12:
            raise HTTPException(503, "unavailable")
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        results = await asyncio.gather(
            *(model.chat_completions(request(), body()) for _ in range(20)), return_exceptions=True
        )
    statuses = [r.status_code for r in results if isinstance(r, HTTPException)]
    assert statuses.count(503) == 12
    assert statuses.count(422) == 6
    assert sum(isinstance(r, NeMoGymChatCompletion) for r in results) == 2
    assert attempts == 14
    assert model._turn_states[("episode", "root")].turns == 2


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


@pytest.mark.parametrize("overrides", [{"num_workers": 4}, {"use_completions_api": True}])
def test_real_gateway_configuration_rejects_unsupported_budget_modes(overrides):
    from pydantic import ValidationError

    from responses_api_models.vllm_model.visual_eval import VisualEvalModelConfig

    with pytest.raises(ValidationError, match="Enforced eval turn budgets require"):
        VisualEvalModelConfig(
            host="localhost",
            port=8081,
            name="policy",
            entrypoint="visual_eval.py",
            base_url="http://localhost/v1",
            api_key="unused",
            model="policy",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            turn_budget=EvalTurnBudget(),
            **overrides,
        )


@pytest.mark.asyncio
async def test_multiple_completions_are_rejected_before_engine(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    upstream = __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock()
    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"), pytest.raises(HTTPException) as error:
        await model.chat_completions(request(), body().model_copy(update={"n": 2}))
    assert error.value.status_code == 400
    upstream.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", [VisualEvalModel, HostedVisualEvalModel])
async def test_child_sessions_have_independent_turn_and_grace_budgets(monkeypatch, model_class):
    model = gateway(monkeypatch, EvalTurnBudget(regular_turns=1, grace_turns=2, grace_tool_turns=1), model_class)
    seen = []

    async def upstream(self, req, params):
        seen.append((req.headers["x-opencode-session-id"], params.tool_choice))
        return completion()

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        await model.chat_completions(request("root"), body())
        await model.chat_completions(request("child"), body())
        child_grace = await model.chat_completions(request("child"), body())
        await model.chat_completions(request("child"), body())
        with pytest.raises(HTTPException) as error:
            await model.chat_completions(request("child"), body())
        assert error.value.status_code == 422
        assert model._turn_states[("episode", "root")].turns == 1
        assert model._turn_states[("episode", "root")].grace_tool_calls == 0
        root_grace = await model.chat_completions(request("root"), body())
    assert seen == [("root", "auto"), ("child", "auto"), ("child", "auto"), ("child", "none"), ("root", "auto")]
    assert len(child_grace.choices[0].message.tool_calls) == len(root_grace.choices[0].message.tool_calls) == 1
    # A reused OpenCode session ID in another rollout cannot share counters.
    with rollout_context("another-episode"):
        await model.chat_completions(request("child"), body())
    assert model._turn_states[("another-episode", "child")].turns == 1


@pytest.mark.asyncio
async def test_missing_session_identity_fails_before_upstream(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget())
    with rollout_context("episode"), pytest.raises(HTTPException) as error:
        await model.chat_completions(request(None), body())
    assert error.value.status_code == 400
    assert not model._turn_states


@pytest.mark.asyncio
@pytest.mark.parametrize("model_class", [VisualEvalModel, HostedVisualEvalModel])
async def test_idle_state_expires_without_another_request(monkeypatch, model_class):
    model = gateway(monkeypatch, EvalTurnBudget(), model_class)
    model.config.turn_state_idle_timeout_seconds = 0.01

    async def upstream(self, req, params):
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        await model.chat_completions(request(), body())
    assert model._turn_states[("episode", "root")].turns == 1
    await asyncio.sleep(0.03)
    assert not model._turn_states


@pytest.mark.asyncio
async def test_active_and_queued_requests_cannot_expire_or_split_counters(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget(regular_turns=1, grace_turns=1, grace_tool_turns=0))
    model.config.turn_state_idle_timeout_seconds = 0.01
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0

    async def upstream(self, req, params):
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        first = asyncio.create_task(model.chat_completions(request(), body()))
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(model.chat_completions(request(), body()))
        cancelled = asyncio.create_task(model.chat_completions(request(), body()))
        await asyncio.sleep(0.03)
        state = model._turn_states[("episode", "root")]
        assert state.users == 3 and state.expiry is None
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert state.users == 2 and state.expiry is None
        finish.set()
        await asyncio.gather(first, second)
        assert state.users == 0 and state.turns == 2 and calls == 2
        with pytest.raises(HTTPException) as error:
            await model.chat_completions(request(), body())
        assert error.value.status_code == 422
    await asyncio.sleep(0.03)
    assert not model._turn_states


@pytest.mark.asyncio
async def test_capacity_rejects_new_sessions_without_resetting_existing_budget(monkeypatch):
    model = gateway(monkeypatch, EvalTurnBudget(regular_turns=1, grace_turns=1, grace_tool_turns=0))
    model.config.turn_state_max_sessions = 1
    model.config.turn_state_idle_timeout_seconds = 0.01
    calls = 0

    async def upstream(self, req, params):
        nonlocal calls
        calls += 1
        return completion(0)

    monkeypatch.setattr(VLLMModel, "chat_completions", upstream)
    with rollout_context("episode"):
        await model.chat_completions(request(), body())
        await model.chat_completions(request(), body())
        with pytest.raises(HTTPException) as error:
            await model.chat_completions(request("child"), body())
        assert error.value.status_code == 503
        with pytest.raises(HTTPException) as error:
            await model.chat_completions(request(), body())
        assert error.value.status_code == 422
        assert calls == 2
        await asyncio.sleep(0.03)
        await model.chat_completions(request("child"), body())
    assert calls == 3


@pytest.mark.asyncio
async def test_shutdown_cancels_expiry_callbacks_and_preserves_parent_lifespan(monkeypatch):
    from contextlib import asynccontextmanager

    from fastapi import FastAPI

    events = []

    @asynccontextmanager
    async def lifespan(app):
        events.append("start")
        yield {"parent": True}
        events.append("stop")

    model = gateway(monkeypatch, EvalTurnBudget())
    monkeypatch.setattr(VLLMModel, "setup_webserver", lambda self: FastAPI(lifespan=lifespan))
    app = model.setup_webserver()
    key = ("episode", "root")
    async with app.router.lifespan_context(app) as lifespan_state:
        assert lifespan_state == {"parent": True}
        state = model._acquire_turn_state(key)
        model._release_turn_state(key, state)
        handle = state.expiry
        assert not handle.cancelled()
    assert handle.cancelled()
    assert not model._turn_states
    assert events == ["start", "stop"]
