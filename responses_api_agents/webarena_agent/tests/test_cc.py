# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context management (nemo_gym.agents.context_management) in the WebArena/CUA-Gym web agent."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient, rollout_path_prefix
from resources_servers.webarena.schemas import (
    WebArenaAgentRunRequest,
    WebArenaAgentVerifyResponse,
    WebArenaNeMoGymResponse,
)
from responses_api_agents.webarena_agent.app import (
    TerminationReason,
    WebAgent,
    WebAgentConfig,
    WebArenaTerminationMetadata,
)
from responses_api_agents.webarena_agent.tests.test_app import (
    FakeResponse,
    _call,
    _env_obs,
    _message,
    _model_response,
    _task,
)


OWNER = "webarena-test_g0"
MODEL_PATH = f"{rollout_path_prefix(OWNER, token_capture=True)}/v1/responses"
MEASURE_PATH = f"{rollout_path_prefix(OWNER, token_capture=True)}/context/measure"
GUARDS = {"max_total_tokens": 32768, "reserved_generation_tokens": 2048}


def _cc_config(guards=None, **context_history_overrides):
    context_history = {
        "enabled": True,
        **({"guards": guards} if guards else {}),
        "policy": {
            "type": "recency",
            "config": {"images": {"enabled": True, "protect_initial_context": True, "keep_last_groups": 1}},
        },
        "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 1},
        "max_response_retries": 0,
    }
    context_history.update(context_history_overrides)
    return WebAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="web-agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="webarena"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        max_steps=3,
        include_initial_request_input=False,
        context_history=context_history,
    )


def _cc_agent(*, token_capture=True, parent_resolution="source_history", guards=None, **overrides):
    server = MagicMock(
        spec=ServerClient,
        global_config_dict={"token_id_capture": {"enabled": token_capture, "parent_resolution": parent_resolution}},
    )
    return WebAgent(config=_cc_config(guards=guards, **overrides), server_client=server)


def _request():
    return WebArenaAgentRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "do the task"}]},
            "task_payload": _task(),
            "task_idx": 7,
        }
    )


def _wire(agent, *, model_responses, step_responses, finalize=None, measure=None):
    """Dispatch fake HTTP hops by path; record every call for assertions.

    A model response given as an exception instance is raised instead of served.
    """
    calls: list[dict] = []
    models = list(model_responses)
    steps = list(step_responses)
    measures = list(measure or [])

    async def post(**kwargs):
        calls.append(deepcopy({k: v for k, v in kwargs.items() if k != "json"}) | {"json": kwargs.get("json")})
        path = kwargs["url_path"]
        if path == "/seed_session":
            return FakeResponse({"env_id": "env-cc", "obs": [_env_obs(1, "SEED")], "tools": []}, {"resource": "seed"})
        if path == MEASURE_PATH:
            return FakeResponse({"prompt_token_count": measures.pop(0)})
        if path == MODEL_PATH:
            served = models.pop(0)
            if isinstance(served, BaseException):
                raise served
            return FakeResponse(served, {"model": "m"})
        if path == "/step":
            return FakeResponse(steps.pop(0), {"resource": "step"})
        if path == "/finalize":
            body = kwargs["json"]
            return FakeResponse(
                {**(finalize or {}), "env_id": body["env_id"], "finalize_request_id": body["finalize_request_id"]}
            )
        if path == "/close":
            return FakeResponse({"success": True})
        raise AssertionError(f"unexpected hop {path}")

    agent.server_client.post = AsyncMock(side_effect=post)
    return calls


def _paths(calls):
    return [c["url_path"] for c in calls]


def _two_action_episode():
    model_responses = [
        _model_response("response-1", [_call("call-1")]),
        _model_response("response-2", [_call("call-2", "terminate")], input_tokens=3, output_tokens=4),
    ]
    step_responses = [
        {
            "obs": [
                {"type": "function_call_output", "call_id": "call-1", "output": "executed"},
                _env_obs(2, "LATER"),
            ],
            "done": False,
        },
        {
            "obs": [{"type": "function_call_output", "call_id": "call-2", "output": "terminated"}],
            "done": True,
            "termination_status": "success",
        },
    ]
    return model_responses, step_responses


def _function_call_ids(result):
    return [
        item["call_id"] for item in result.model_dump(mode="json")["output"] if item.get("type") == "function_call"
    ]


def test_config_rejects_legacy_screenshot_pruning_with_context_history():
    with pytest.raises(ValueError, match="max_recent_screenshots"):
        WebAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="web-agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="webarena"),
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            max_recent_screenshots=2,
            context_history={"enabled": True},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_capture, parent_resolution, owner",
    [
        (False, "source_history", OWNER),
        (True, "source_history", None),
        (True, "source_history", ""),
        (True, "source_history", "bad/owner"),
        (True, "fingerprint", OWNER),
    ],
)
async def test_cc_identity_is_validated_before_any_environment_mutation(token_capture, parent_resolution, owner):
    agent = _cc_agent(token_capture=token_capture, parent_resolution=parent_resolution)
    calls = _wire(agent, model_responses=[], step_responses=[])
    with pytest.raises(ValueError):
        await agent._rollout(_request(), {}, logical_rollout_id=owner)
    assert calls == []


@pytest.mark.asyncio
async def test_cc_rollout_uses_one_owner_capture_route_and_returns_the_semantic_conversation():
    agent = _cc_agent()
    model_responses, step_responses = _two_action_episode()
    calls = _wire(agent, model_responses=model_responses, step_responses=step_responses)

    result, cookies = await agent._rollout(_request(), {"incoming": "cookie"}, logical_rollout_id=OWNER)

    assert type(result) is WebArenaNeMoGymResponse
    assert result.env_id == "env-cc" and result.group_id == "7"
    assert result.id == "response-2"
    assert "context_compaction_result" not in result.model_dump(mode="json")
    # The returned output is the ordinary conversation, in which the trainer finds
    # every accepted action; compaction only changes what the model was shown.
    assert _function_call_ids(result) == ["call-1", "call-2"]
    dumped = json.dumps(result.model_dump(mode="json")["output"])
    assert "LATER" in dumped and "executed" in dumped and "terminated" in dumped
    assert result.usage.input_tokens == 4 and result.usage.output_tokens == 6
    assert cookies["resource"] == "step"

    # Token capture resolves parents from source history: every call uses the owner route.
    model_paths = [p for p in _paths(calls) if p.endswith("/v1/responses")]
    assert model_paths == [MODEL_PATH, MODEL_PATH]
    # Recency policy (keep_last_groups=1): the second request still carries the
    # latest screenshot but the seed screenshot has aged out of the model view.
    second_request = [c["json"] for c in calls if c["url_path"] == MODEL_PATH][1]
    second_dump = json.dumps(second_request.model_dump(mode="json"))
    assert "LATER" in second_dump and "SEED" not in second_dump
    assert calls[-1]["url_path"] == "/close"


@pytest.mark.asyncio
async def test_cc_run_strips_raw_images_from_the_trainer_payload():
    agent = _cc_agent()
    model_responses, step_responses = _two_action_episode()
    calls = _wire(
        agent,
        model_responses=model_responses,
        step_responses=step_responses,
        finalize={"reward": 1.0, "evaluator_details": {"k": "v"}, "termination_status": "success", "answer": "ok"},
    )
    request = SimpleNamespace(cookies={"incoming": "cookie"}, path_params={"rollout_id": OWNER})

    result = await agent.run(request, _request())

    assert isinstance(result, WebArenaAgentVerifyResponse)
    assert result.reward == 1.0 and result.answer == "ok"
    assert "context_compaction_result" not in result.model_dump(mode="json")
    assert result.response.id == "response-2"
    response_dump = result.response.model_dump(mode="json")
    assert "input_image" not in json.dumps(response_dump["output"])
    assert "input_image" not in json.dumps(response_dump["seed_obs"])
    assert [i["call_id"] for i in response_dump["output"] if i.get("type") == "function_call"] == ["call-1", "call-2"]
    paths = _paths(calls)
    assert "/finalize" in paths and "/close" not in paths, "run() finalizes instead of closing on success"


@pytest.mark.asyncio
async def test_cc_no_tool_call_ends_the_rollout_on_that_action():
    agent = _cc_agent()
    model_responses = [_model_response("response-empty", [_message("use a tool")])]
    calls = _wire(agent, model_responses=model_responses, step_responses=[])

    result, _ = await agent._rollout(_request(), {}, logical_rollout_id=OWNER)

    assert result.id == "response-empty"
    assert "use a tool" in json.dumps(result.model_dump(mode="json")["output"])
    assert json.loads(result.metadata["webarena_termination"])["reason"] == TerminationReason.NO_TOOL_CALL.value
    assert "/step" not in _paths(calls)


@pytest.mark.asyncio
async def test_cc_max_steps_returns_every_accepted_action():
    agent = _cc_agent()
    model_responses = [_model_response(f"response-{i}", [_call(f"call-{i}")]) for i in range(1, 4)]
    step_responses = [
        {"obs": [{"type": "function_call_output", "call_id": f"call-{i}", "output": "ok"}, _env_obs(i + 1, f"S{i}")]}
        for i in range(1, 4)
    ]
    _wire(agent, model_responses=model_responses, step_responses=step_responses)

    result, _ = await agent._rollout(_request(), {}, logical_rollout_id=OWNER)

    assert json.loads(result.metadata["webarena_termination"])["reason"] == TerminationReason.MAX_STEPS.value
    assert result.id == "response-3"
    assert _function_call_ids(result) == ["call-1", "call-2", "call-3"]


@pytest.mark.asyncio
async def test_cc_guard_measures_each_view_on_the_owner_route():
    agent = _cc_agent(guards=GUARDS)
    model_responses, step_responses = _two_action_episode()
    calls = _wire(agent, model_responses=model_responses, step_responses=step_responses, measure=[1234, 2345])

    result, _ = await agent._rollout(_request(), {}, logical_rollout_id=OWNER)

    assert json.loads(result.metadata["webarena_termination"])["reason"] == TerminationReason.ENV_DONE.value
    assert [p for p in _paths(calls) if p in (MEASURE_PATH, MODEL_PATH)] == [
        MEASURE_PATH,
        MODEL_PATH,
        MEASURE_PATH,
        MODEL_PATH,
    ]


@pytest.mark.asyncio
async def test_cc_guard_rejection_ends_the_rollout_as_context_limit():
    agent = _cc_agent(guards=GUARDS)
    model_responses, step_responses = _two_action_episode()
    calls = _wire(agent, model_responses=model_responses, step_responses=step_responses, measure=[1234, 40000])

    result, _ = await agent._rollout(_request(), {}, logical_rollout_id=OWNER)

    assert json.loads(result.metadata["webarena_termination"])["reason"] == TerminationReason.CONTEXT_LIMIT.value
    assert result.id == "response-1"
    assert _function_call_ids(result) == ["call-1"]
    assert sum(1 for p in _paths(calls) if p == MODEL_PATH) == 1


@pytest.mark.asyncio
async def test_cc_model_failure_before_any_action_is_terminal():
    agent = _cc_agent()
    calls = _wire(agent, model_responses=[ConnectionError("lost acknowledgement")], step_responses=[])

    with pytest.raises(Exception):
        await agent._rollout(_request(), {}, logical_rollout_id=OWNER)
    # A failed model call is never retried on the same logical rollout (client is closed).
    assert sum(1 for p in _paths(calls) if p == MODEL_PATH) == 1
    assert calls[-1]["url_path"] == "/close"


@pytest.mark.asyncio
async def test_cc_terminal_rollout_failure_is_reported_as_422_not_500():
    """A lost model call must surface as a per-prompt (4xx) failure: NeMo-RL's
    SingleController classifies a generic 500 from /run as a transport failure and
    aborts the run (jobs 1970855 / 1971056)."""
    from fastapi import HTTPException

    agent = _cc_agent()
    _wire(agent, model_responses=[ConnectionError("server disconnected")], step_responses=[])
    request = SimpleNamespace(cookies={}, path_params={"rollout_id": OWNER})
    with pytest.raises(HTTPException) as info:
        await agent.run(request, _request())
    assert info.value.status_code == 422
    assert "Model call failed" in str(info.value.detail)


@pytest.mark.asyncio
async def test_cc_model_failure_after_an_accepted_action_returns_a_masked_rollout():
    """A lost model call after an accepted action keeps that action in the returned
    conversation; the model_error reason masks the rollout instead of failing the prompt."""
    agent = _cc_agent()
    model_responses = [_model_response("response-1", [_call("call-1")]), ConnectionError("server disconnected")]
    step_responses = [
        {"obs": [{"type": "function_call_output", "call_id": "call-1", "output": "executed"}, _env_obs(2, "LATER")]}
    ]
    calls = _wire(agent, model_responses=model_responses, step_responses=step_responses)

    result, _ = await agent._rollout(_request(), {}, logical_rollout_id=OWNER)

    assert result.id == "response-1"
    assert _function_call_ids(result) == ["call-1"]
    termination = json.loads(result.metadata["webarena_termination"])
    assert termination["reason"] == TerminationReason.MODEL_ERROR.value
    assert sum(1 for p in _paths(calls) if p == MODEL_PATH) == 2
    assert calls[-1]["url_path"] == "/close"

    merged = {"reward": 1.0}
    WebAgent._apply_termination_outcome(merged, WebArenaTerminationMetadata.model_validate(termination))
    assert merged["instance_config"] == {"mask_sample": True}
