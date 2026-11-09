# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientConnectorError, ClientResponseError
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest, OutcomeAction, agent_outcome_policy
from nemo_gym.client import ServerClient
from nemo_gym.config import ResourcesServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.langgraph_agent.orchestrator_agent import OrchestratorAgent, OrchestratorAgentConfig
from responses_api_agents.langgraph_agent.parallel_thinking_agent import (
    ParallelThinkingAgent,
    ParallelThinkingAgentConfig,
)
from responses_api_agents.langgraph_agent.reflection_agent import ReflectionAgent, ReflectionAgentConfig
from responses_api_agents.langgraph_agent.rewoo_agent import ReWOOAgent, ReWOOAgentConfig
from responses_api_agents.langgraph_agent.tests.test_app import _mock_model_response, _usage


FLAVORS = [
    (ReflectionAgent, ReflectionAgentConfig, ["draft", "critique", "<answer>42</answer>"], {}),
    (OrchestratorAgent, OrchestratorAgentConfig, ["1. Compute", "42", "<answer>42</answer>"], {}),
    (
        ParallelThinkingAgent,
        ParallelThinkingAgentConfig,
        ["42", "42", "<answer>42</answer>"],
        {"num_parallel_paths": 2},
    ),
    (ReWOOAgent, ReWOOAgentConfig, ["Plan: Compute. #E1 = LLM[6 * 7]", "42", "<answer>42</answer>"], {}),
]


def _seed() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="graph-session",
        episode_id=EpisodeId(rollout_id="graph-rollout", attempt=1),
        task_id=TaskId(taskset="test", task_id="multiply"),
    )


def _agent(agent_class, config_class, options):
    return agent_class(
        config=config_class(
            name="graph",
            host="127.0.0.1",
            port=8000,
            entrypoint="app.py",
            num_workers=1,
            model_server={"type": "responses_api_models", "name": "model"},
            **options,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.mark.parametrize("agent_class,config_class,replies,options", FLAVORS)
@pytest.mark.parametrize("training", [False, True])
def test_graph_session_replays_and_preserves_capture(agent_class, config_class, replies, options, training):
    agent = _agent(agent_class, config_class, options)
    agent.server_client.post.side_effect = [_mock_model_response(text, _usage(2, 3, 1)) for text in replies]
    seed = _seed()
    path = (
        f"/ng-rollout/{seed.episode_id.capture_key}"
        + ("/training-token-capture" if training else "")
        + "/v1/responses"
    )
    body = {"input": [{"role": "user", "content": "What is 6 * 7?"}]}
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
        assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
        assert client.post("/ng-rollout/wrong/v1/responses", json=body).status_code == 409
        assert client.post("/v1/responses", json=body).status_code == 409
        result = client.post(path, json=body)
        assert result.status_code == 200, result.text
        assert result.json()["output"][-1]["content"][0]["text"] == "<answer>42</answer>"
        assert result.json()["usage"]["total_tokens"] == len(replies) * 5
        assert client.post(path, json=body).json() == result.json()
        assert client.post(path, json={"input": "different"}).status_code == 409
        assert client.post("/v1/responses", json=body).status_code == 409
        assert agent.server_client.post.call_count == len(replies)
        for call in agent.server_client.post.call_args_list:
            assert call.kwargs["url_path"] == path
            assert not dict(call.kwargs.get("cookies") or {})
        close_body = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
        closed = client.post("/v1/agent_sessions/close", json=close_body)
        assert closed.status_code == 200, closed.text
        assert closed.json()["resources_cookies"] is None
        assert closed.json()["agent_observations"]["gaps"][0]["code"] == "observation_capture_failed"
        assert client.post("/v1/agent_sessions/close", json=close_body).json() == closed.json()
        assert client.post(path, json=body).status_code == 409
        assert agent._session_records[seed.agent_session_id].state is None


@pytest.mark.parametrize("kind", ["direct_http", "mcp", "sandbox"])
@pytest.mark.parametrize("configured", [False, True])
def test_graph_rejects_unsupported_grants(kind, configured):
    agent = _agent(ReflectionAgent, ReflectionAgentConfig, {})
    seed = _seed().model_dump(mode="json")
    if kind == "sandbox":
        seed["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {}},
            "workdir": "/workspace",
        }
    else:
        grant = {"kind": kind, "name": "required-tools", "required": True}
        grant.update(
            {"base_url": "http://resources:8000"}
            if kind == "direct_http"
            else {
                "connection": {"transport": "streamable_http", "url": "http://resources:8000/mcp"},
            }
        )
        if configured:
            agent.config = type(agent.config).model_validate(agent.config.model_dump() | {"tool_accesses": [grant]})
        else:
            seed["tool_accesses"] = [grant]
    detail = check_agent_seed_rejected(agent.setup_webserver(), AgentSeedSessionRequest.model_validate(seed))
    assert "does not support" in detail
    assert not agent._session_records
    agent.server_client.post.assert_not_called()


def test_graph_optional_grants_and_failed_activation_close():
    agent = _agent(ReflectionAgent, ReflectionAgentConfig, {})
    seed = _seed().model_dump(mode="json")
    seed["tool_accesses"] = [
        {"kind": "direct_http", "name": "optional", "required": False, "base_url": "http://resources:8000"}
    ]
    agent.server_client.post.side_effect = RuntimeError("model unavailable")
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed).status_code == 200
        assert (
            client.post(
                f"/ng-rollout/{_seed().episode_id.capture_key}/v1/responses", json={"input": "question"}
            ).status_code
            == 500
        )
        assert (
            client.post(
                f"/ng-rollout/{_seed().episode_id.capture_key}/v1/responses", json={"input": "question"}
            ).status_code
            == 500
        )
        assert agent.server_client.post.call_count == 1
        close = client.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": seed["agent_session_id"], "episode_id": seed["episode_id"]},
        )
        assert close.status_code == 200
        assert close.json()["agent_observations"]["gaps"]


@pytest.mark.parametrize("agent_class,config_class,replies,options", FLAVORS)
def test_missing_one_call_usage_keeps_aggregate_unknown(agent_class, config_class, replies, options):
    agent = _agent(agent_class, config_class, options)
    agent.server_client.post.side_effect = [
        _mock_model_response(text, None if index == 0 else _usage(2, 3, 1)) for index, text in enumerate(replies)
    ]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/responses", json={"input": "What is 6 * 7?"})
    assert response.status_code == 200, response.text
    assert response.json().get("usage") is None


@pytest.mark.parametrize("agent_class,config_class,replies,options", FLAVORS)
def test_legacy_run_preserves_verifier_fields_and_resource_cookies(agent_class, config_class, replies, options):
    import json

    agent = _agent(agent_class, config_class, options)
    agent.config.resources_server = ResourcesServerRef(type="resources_servers", name="resources")
    agent.config = config_class.model_validate(agent.config.model_dump())
    agent.server_client.global_config_dict = {"observability_enabled": True}
    seeded = _mock_model_response()
    seeded.cookies = {"resources": "seed-cookie"}
    generated = _mock_model_response()
    generated.cookies = {"model": "model-cookie"}
    verified = _mock_model_response()
    verification = {
        "responses_create_params": {"input": "question"},
        "response": generated.json.return_value,
        "reward": 0.0,
        "question": "question",
        "expected_answer": "42",
        "_ng_failure_class": "judge_failure",
    }
    verified.json.return_value = verification
    verified.read.return_value = json.dumps(verification).encode()
    agent.server_client.post.side_effect = [seeded, generated, verified]
    row = {
        "responses_create_params": {"input": "question"},
        "question": "question",
        "expected_answer": "42",
        "metadata": {"source_dataset": "test"},
        "_ng_rollout_id": "legacy-rollout",
    }
    with TestClient(agent.setup_webserver()) as client:
        result = client.post("/run", json=row)
    assert result.status_code == 200, result.text
    assert result.json()["_ng_failure_class"] == "judge_failure"
    seed_call, response_call, verify_call = [call.kwargs for call in agent.server_client.post.call_args_list]
    assert seed_call["json"]["expected_answer"] == "42"
    assert response_call["url_path"] == "/ng-rollout/legacy-rollout/v1/responses"
    assert not response_call.get("cookies")
    assert verify_call["cookies"] == {"resources": "seed-cookie"}
    assert verify_call["json"]["metadata"] == row["metadata"]
    assert verify_call["json"]["question"] == "question"


def _request(session=None):
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": f"/ng-rollout/{EpisodeId(rollout_id='rollout', attempt=1).capture_key}/v1/responses",
            "path_params": {"rollout_id": EpisodeId(rollout_id="rollout", attempt=1).capture_key},
            "headers": [],
            "session": session or {},
        }
    )


async def _seed_agent():
    agent = _agent(ReflectionAgent, ReflectionAgentConfig, {})
    seed = AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="test", task_id="task"),
    )
    request = _request()
    await agent.seed_agent_session(request, seed)
    return agent, seed, request


async def test_stateless_activation_survives_disconnected_waiter_and_replays():
    agent, seed, request = await _seed_agent()
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = 0

    async def operation():
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return NeMoGymResponse(
            id="result",
            created_at=0,
            model="test",
            output=[],
            object="response",
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    first = asyncio.create_task(agent._run_graph_session(request, body, operation=operation))
    await asyncio.wait_for(entered.wait(), 2)
    retry = asyncio.create_task(agent._run_graph_session(request, body, operation=operation))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    result = await asyncio.wait_for(retry, 2)
    result.id = "mutated-copy"
    replay = await agent._run_graph_session(request, body, operation=operation)
    assert replay.id == "result"
    assert calls == 1
    with pytest.raises(HTTPException) as conflict:
        await agent._run_graph_session(request, body.model_copy(update={"input": "changed"}), operation=operation)
    assert conflict.value.status_code == 409
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )


async def test_close_drains_inflight_activation_and_fences_stale_or_unknown_cookie():
    agent, seed, request = await _seed_agent()
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def operation():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    body = NeMoGymResponseCreateParamsNonStreaming(input="question")
    pending = asyncio.create_task(agent._run_graph_session(request, body, operation=operation))
    await asyncio.wait_for(entered.wait(), 2)
    close = await asyncio.wait_for(
        agent.close_agent_session(
            request,
            AgentCloseSessionRequest(
                agent_session_id=seed.agent_session_id,
                episode_id=seed.episode_id,
            ),
        ),
        2,
    )
    assert stopped.is_set()
    assert close.agent_observations.gaps
    assert close.resources_cookies is None
    with pytest.raises(asyncio.CancelledError):
        await pending
    for marker in (seed.agent_session_id, "unknown"):
        with pytest.raises(HTTPException) as stale:
            await agent._run_graph_session(_request({"agent_session_id": marker}), body, operation=operation)
        assert stale.value.status_code == 409


async def test_parallel_failure_cancels_and_drains_siblings():
    agent = _agent(
        ParallelThinkingAgent,
        ParallelThinkingAgentConfig,
        {"num_parallel_paths": 2, "perspective_prompts": ["fail", "wait"]},
    )
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def model(*args, **kwargs):
        prompt = kwargs["json"].input[0].content
        if prompt.startswith("fail"):
            await entered.wait()
            raise RuntimeError("model failed")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    agent.server_client.post.side_effect = model
    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        await agent._invoke_graph(
            NeMoGymResponseCreateParamsNonStreaming(input="question"), model_url_path="/v1/responses"
        )
    assert stopped.is_set()


@pytest.mark.parametrize("agent_class,config_class,replies,options", FLAVORS)
def test_in_process_calls_preserve_unset_sampling_fields(agent_class, config_class, replies, options):
    agent = _agent(agent_class, config_class, options)
    agent.server_client.post.side_effect = [_mock_model_response(text) for text in replies]
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/responses", json={"input": "What is 6 * 7?"})
    assert response.status_code == 200, response.text
    assert "harness_execution" not in (response.json().get("metadata") or {})
    assert agent.server_client.post.call_count == len(replies)
    for call in agent.server_client.post.call_args_list:
        assert set(call.kwargs["json"].model_dump(exclude_unset=True)) == {"input"}


def _http_error(status: int, code: str | None = None) -> ClientResponseError:
    info = SimpleNamespace(url="http://model", method="POST", headers={}, real_url="http://model")
    error = ClientResponseError(info, (), status=status, message="model failed")
    if code is not None:
        # The model server's OpenAI error envelope, which Gym's raise_for_status keeps on the error.
        envelope = {"message": "maximum context length", "type": "invalid_request_error", "param": None, "code": code}
        error.response_content = json.dumps({"error": envelope}).encode()
    return error


def _activate(agent, answers) -> NeMoGymResponse:
    """One session activation whose model answers ``answers`` in order (a reply text or an exception)."""
    agent.server_client.post.side_effect = [
        answer if isinstance(answer, BaseException) else _mock_model_response(answer) for answer in answers
    ]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200
        response = client.post(f"/ng-rollout/{_seed().episode_id.capture_key}/v1/responses", json={"input": "6 * 7?"})
    assert response.status_code == 200, response.text
    return NeMoGymResponse.model_validate(response.json())


@pytest.mark.parametrize(
    "answers,outcome,action,replies",
    [
        # main answered 500 for every model error, which lost the status and was retried.
        ([_http_error(401)], {"failure_kind": "model_request_rejected"}, OutcomeAction.FAIL, 0),
        (
            ["draft", _http_error(400)],
            {"failure_kind": "model_request_error", "status_code": 400},
            OutcomeAction.VERIFY,
            1,
        ),
        (
            ["draft", _http_error(503)],
            {"failure_kind": "model_request_error", "status_code": 503},
            OutcomeAction.RETRY,
            1,
        ),
        (
            ["draft", ClientConnectorError(SimpleNamespace(ssl=None, host="model", port=1), OSError("refused"))],
            {"failure_kind": "model_transport_error"},
            OutcomeAction.RETRY,
            1,
        ),
    ],
)
def test_a_model_error_ends_the_graph_with_a_classified_failed_response(answers, outcome, action, replies):
    response = _activate(_agent(ReflectionAgent, ReflectionAgentConfig, {}), answers)
    assert response.status == "failed"
    assert json.loads(response.metadata["ng_agent_outcome"]) == outcome
    assert agent_outcome_policy(response) == action
    # The steps that finished keep their outputs.
    assert len([item for item in response.output if item.type == "message"]) == replies
    assert response.error.message.startswith("Model call failed")


@pytest.mark.parametrize(
    "answers,status,outcome",
    [
        # A context overflow is graded as incomplete, before or after the first reply (decision 1).
        ([_http_error(400, "context_length_exceeded")], "incomplete", None),
        (["draft", _http_error(400, "context_length_exceeded")], "incomplete", None),
        # A 400 with another code stays a model request error.
        (
            ["draft", _http_error(400, "invalid_request")],
            "failed",
            {"failure_kind": "model_request_error", "status_code": 400},
        ),
    ],
)
def test_a_propagated_context_overflow_is_a_graded_incomplete_response(answers, status, outcome):
    response = _activate(_agent(ReflectionAgent, ReflectionAgentConfig, {}), answers)
    assert response.status == status
    assert json.loads(response.metadata.get("ng_agent_outcome", "null")) == outcome
    assert agent_outcome_policy(response) == OutcomeAction.VERIFY
    if status == "incomplete":
        assert response.metadata["ng_agent_stop_reason"] == "context_limit"
        assert response.error is None


def test_parallel_paths_that_all_fail_on_the_model_report_its_error():
    agent = _agent(ParallelThinkingAgent, ParallelThinkingAgentConfig, {"num_parallel_paths": 2})
    response = _activate(agent, [_http_error(429), _http_error(429)])
    assert json.loads(response.metadata["ng_agent_outcome"]) == {
        "failure_kind": "model_request_error",
        "status_code": 429,
    }


@pytest.mark.parametrize("agent_class,config_class,replies,options", FLAVORS)
def test_a_finished_graph_is_a_completed_response(agent_class, config_class, replies, options):
    response = _activate(_agent(agent_class, config_class, options), replies)
    assert response.status == "completed"
    assert response.metadata["ng_agent_stop_reason"] == "completed"
    assert "ng_agent_outcome" not in response.metadata
