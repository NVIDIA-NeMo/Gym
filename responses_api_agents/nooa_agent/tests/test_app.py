# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, TrajectoryRecord
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunResult
from responses_api_agents.nooa_agent.tests.test_config import agent_config
from responses_api_agents.nooa_agent.tests.test_gym_llm import model_response


def request(*, session_id: str | None = None, capture: bool = False, rollout_id: str = "rollout-a2") -> Request:
    prefix = "/training-token-capture" if capture else ""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [],
            "path": f"/ng-rollout/{rollout_id}{prefix}/v1/responses",
            "path_params": {"rollout_id": rollout_id},
            "session": {"agent_session_id": session_id} if session_id else {},
        }
    )


def seed(**overrides) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest.model_validate(
        {
            "agent_session_id": "session",
            "episode_id": {"rollout_id": "rollout", "attempt": 2},
            "task_id": {"taskset": "tests", "task_id": "task"},
            **overrides,
        }
    )


def result(*, reason: str | None = None) -> NOOARunResult:
    return NOOARunResult(
        episode=AgentEpisode(
            response=NeMoGymResponse.model_validate(model_response()),
            observations=AgentObservationBundle(source="nooa"),
        ),
        return_value="answer",
        model_cookies={"session": "model-cookie"},
        resource_cookies={"session": "resource-cookie"},
        termination_reason=reason,
        trajectory=TrajectoryRecord(task_id="task", rollout_id="rollout-a2"),
    )


def agent() -> NOOAAgent:
    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    instance.runner = SimpleNamespace(run=AsyncMock(return_value=result()))
    return instance


async def open_session(instance: NOOAAgent, **overrides) -> Request:
    req = request()
    await instance.seed_agent_session(req, seed(**overrides))
    return req


async def close(instance: NOOAAgent):
    return await instance.close_agent_session(
        request(session_id="session"),
        AgentCloseSessionRequest(
            agent_session_id="session",
            episode_id=seed().episode_id,
        ),
    )


async def test_native_activation_and_close_preserve_identity_cookies_and_evidence() -> None:
    instance = agent()
    req = await open_session(
        instance,
        tool_accesses=[
            {
                "kind": "direct_http",
                "name": "tools",
                "required": True,
                "base_url": "http://resources:8000",
                "cookies": {"session": "seed-cookie"},
            }
        ],
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    response = await instance.responses(req, body)
    run_request = instance.runner.run.call_args.args[0]
    assert run_request.task_id == "task"
    assert run_request.rollout_id == "rollout-a2"
    assert run_request.model_url_path == "/ng-rollout/rollout-a2/v1/responses"
    assert run_request.model_cookies == {}
    assert run_request.resource_cookies == {"session": "seed-cookie"}
    assert str(run_request.tool_access.base_url) == "http://resources:8000/"
    assert response.output[-1].content[0].text == "answer"
    receipt = await close(instance)
    assert receipt.agent_observations.source == "nooa"
    assert receipt.resources_cookies == {"session": "resource-cookie"}
    assert await close(instance) == receipt
    instance.server_client.post.assert_not_called()


async def test_retries_share_one_execution_and_waiter_cancellation_does_not_stop_it() -> None:
    instance = agent()
    req = await open_session(instance)
    started, finish = asyncio.Event(), asyncio.Event()

    async def run(_):
        assert current_rollout_id() == "rollout-a2"
        started.set()
        await finish.wait()
        return result()

    instance.runner.run.side_effect = run
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    waiter = asyncio.create_task(instance.responses(req, body))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    state = instance._require_agent_session("session")
    assert not state.execution.done()
    finish.set()
    first = await instance.responses(req, body)
    assert await instance.responses(req, body) == first
    assert instance.runner.run.call_count == 1
    with pytest.raises(HTTPException, match="another activation"):
        await instance.responses(req, body.model_copy(update={"temperature": 0.2}))
    with pytest.raises(HTTPException, match="another activation"):
        await instance.responses(request(session_id="session", capture=True), body)
    await close(instance)


async def test_close_cancels_execution_and_keeps_partial_evidence() -> None:
    instance = agent()
    req = await open_session(instance)
    started = asyncio.Event()

    async def run(_):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.nooa_result = result()
            raise

    instance.runner.run.side_effect = run
    waiter = asyncio.create_task(instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await started.wait()
    receipt = await close(instance)
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert "cancelled" in {gap.code for gap in receipt.agent_observations.gaps}
    assert receipt.agent_observations.source == "nooa"
    with pytest.raises(HTTPException):
        await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))


@pytest.mark.parametrize("transient", [False, True])
async def test_failed_activation_replays_error_and_returns_evidence_on_close(transient: bool) -> None:
    instance = agent()
    req = await open_session(instance)
    cause = aiohttp.ClientConnectionError("connection failed") if transient else ValueError("adapter defect")
    failure = NOOARunFailure(cause, result())
    failure.__cause__ = cause
    instance.runner.run.side_effect = failure
    for _ in range(2):
        with pytest.raises(HTTPException if transient else NOOARunFailure) as caught:
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        if transient:
            assert caught.value.status_code == 503
    assert instance.runner.run.call_count == 1
    receipt = await close(instance)
    assert "infrastructure_error" in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize("reason", ["policy_budget_exceeded", "invalid_policy_output"])
async def test_policy_termination_remains_a_gradable_response(reason: str) -> None:
    instance = agent()
    instance.runner.run.return_value = result(reason=reason)
    req = await open_session(instance)
    response = await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert response.status == "failed"
    assert reason in response.error.message
    assert reason in {gap.code for gap in (await close(instance)).agent_observations.gaps}


async def test_sessions_and_required_grants_are_validated_before_execution() -> None:
    instance = agent()
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    with pytest.raises(HTTPException, match="requires an agent session"):
        await instance.responses(request(), body)
    with pytest.raises(HTTPException, match="direct HTTP"):
        await open_session(
            instance,
            tool_accesses=[
                {
                    "kind": "mcp",
                    "name": "tools",
                    "required": True,
                    "connection": {"transport": "streamable_http", "url": "http://resources/mcp"},
                }
            ],
        )
    req = await open_session(instance)
    with pytest.raises(HTTPException, match="rollout route"):
        await instance.responses(request(session_id="session", rollout_id="other"), body)
    with pytest.raises(HTTPException, match="direct HTTP grant"):
        await instance.responses(req, body.model_copy(update={"tools": [{"type": "function", "name": "tool"}]}))
    instance.runner.run.assert_not_called()
    await close(instance)


def test_legacy_run_route_is_removed() -> None:
    paths = agent().setup_webserver().openapi()["paths"]
    assert "/run" not in paths
    assert "/v1/agent_sessions" in paths
    assert "/v1/agent_sessions/close" in paths
    assert "/v1/agent_sessions/finish" in paths


async def test_no_tool_grant_preserves_resources_cookie_jar() -> None:
    instance = agent()
    req = await open_session(instance)
    await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await close(instance)
    # SWE-bench Pro has no resource tools; only Resources knows its session cookie.
    assert receipt.resources_cookies is None
