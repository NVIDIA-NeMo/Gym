# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from nemo_gym.base_responses_api_agent import (
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
)
from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    current_rollout_id,
)
from nemo_gym.trajectory import AgentEpisode, AgentObservationBundle, TrajectoryRecord
from responses_api_agents.nooa_agent import app as app_module
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunFailure, NOOARunResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
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


def result(*, reason: NOOATerminationReason | None = None) -> NOOARunResult:
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


def agent(monkeypatch: pytest.MonkeyPatch) -> tuple[NOOAAgent, AsyncMock]:
    run = AsyncMock(return_value=result())
    monkeypatch.setattr(InProcessNOOARunner, "run", run)
    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    return instance, run


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


async def test_several_workers_seed_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Session routing sends a session's later requests to the worker that seeded it."""
    monkeypatch.setattr(InProcessNOOARunner, "run", AsyncMock(return_value=result()))
    instance = NOOAAgent(config=agent_config(num_workers=2), server_client=MagicMock(spec=ServerClient))
    req = await open_session(instance)

    assert isinstance(instance._require_agent_session("session").runner, InProcessNOOARunner)
    assert req.session[AGENT_SESSION_COOKIE_KEY] == "session"


async def test_embedded_runners_are_created_per_session_at_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock(wraps=InProcessNOOARunner)
    monkeypatch.setattr(app_module, "InProcessNOOARunner", constructor)
    instance = NOOAAgent(config=agent_config(), server_client=MagicMock(spec=ServerClient))
    constructor.assert_not_called()
    assert not hasattr(instance, "runner")

    first_request = await open_session(instance)
    first = instance._require_agent_session("session")
    constructor.assert_called_once()
    assert isinstance(first.runner, InProcessNOOARunner)
    await instance.seed_agent_session(first_request, seed())
    constructor.assert_called_once()
    assert instance._require_agent_session("session") is first

    second_request = await open_session(instance, agent_session_id="second")
    second = instance._require_agent_session("second")
    assert constructor.call_count == 2
    assert isinstance(second.runner, InProcessNOOARunner)
    assert first.runner is not second.runner
    await close(instance)
    assert not second.closing
    await instance.close_agent_session(
        second_request, AgentCloseSessionRequest(agent_session_id="second", episode_id=seed().episode_id)
    )


async def test_agent_session_activation_and_close_preserve_identity_cookies_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, runner_run = agent(monkeypatch)
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
    body = NeMoGymResponseCreateParamsNonStreaming(input="task", metadata={"agent_timeout_sec": "45"})
    response = await instance.responses(req, body)
    run_request = runner_run.call_args.args[0]
    assert run_request.responses_create_params.metadata == {"agent_timeout_sec": "45"}
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


async def test_retries_share_one_execution_and_waiter_cancellation_does_not_stop_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance, runner_run = agent(monkeypatch)
    req = await open_session(instance)
    started, finish = asyncio.Event(), asyncio.Event()

    async def run(_):
        assert current_rollout_id() == "rollout-a2"
        started.set()
        await finish.wait()
        return result()

    runner_run.side_effect = run
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
    assert runner_run.call_count == 1
    with pytest.raises(HTTPException, match="another activation"):
        await instance.responses(req, body.model_copy(update={"temperature": 0.2}))
    with pytest.raises(HTTPException, match="another activation"):
        await instance.responses(request(session_id="session", capture=True), body)
    await close(instance)


async def test_close_cancels_execution_and_keeps_partial_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    req = await open_session(instance)
    started = asyncio.Event()

    async def run(_):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            error.nooa_result = result()
            raise

    runner_run.side_effect = run
    waiter = asyncio.create_task(instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await started.wait()
    receipt = await close(instance)
    with pytest.raises(HTTPException) as interrupted:
        await waiter
    assert interrupted.value.status_code == 409
    assert NOOATerminationReason.CANCELLED in {gap.code for gap in receipt.agent_observations.gaps}
    assert receipt.agent_observations.source == "nooa"
    with pytest.raises(HTTPException):
        await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))


@pytest.mark.parametrize("transient", [False, True])
async def test_failed_activation_replays_error_and_returns_evidence_on_close(
    monkeypatch: pytest.MonkeyPatch, transient: bool
) -> None:
    instance, runner_run = agent(monkeypatch)
    req = await open_session(instance)
    cause = aiohttp.ClientConnectionError("connection failed") if transient else ValueError("adapter defect")
    failure = NOOARunFailure(cause, result())
    failure.__cause__ = cause
    runner_run.side_effect = failure
    for _ in range(2):
        with pytest.raises(HTTPException if transient else NOOARunFailure) as caught:
            await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
        if transient:
            assert caught.value.status_code == 503
    assert runner_run.call_count == 1
    receipt = await close(instance)
    assert NOOATerminationReason.INFRASTRUCTURE_ERROR in {gap.code for gap in receipt.agent_observations.gaps}


@pytest.mark.parametrize(
    "reason",
    [
        NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
        NOOATerminationReason.MAX_ITERATIONS_EXCEEDED,
        NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
        NOOATerminationReason.INVALID_POLICY_OUTPUT,
    ],
)
async def test_policy_termination_remains_a_gradable_response(
    monkeypatch: pytest.MonkeyPatch, reason: NOOATerminationReason
) -> None:
    instance, runner_run = agent(monkeypatch)
    partial = result(reason=reason)
    partial.return_value = None
    partial.episode.response.output = [
        NeMoGymResponseOutputMessage(
            id="partial-answer",
            content=[NeMoGymResponseOutputText(text="Work completed before the limit", annotations=[])],
        )
    ]
    runner_run.return_value = partial
    req = await open_session(instance)
    response = await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    if reason is NOOATerminationReason.INVALID_POLICY_OUTPUT:
        assert response.status == "failed"
        assert reason in response.error.message
    else:
        assert response.status == "incomplete"
        assert response.error is None
    assert response.output == partial.episode.response.output
    assert "_ng_failure_class" not in response.model_dump(mode="json")
    assert reason in {gap.code for gap in (await close(instance)).agent_observations.gaps}
    runner_run.assert_awaited_once()
    instance.server_client.post.assert_not_called()


async def test_sessions_and_required_grants_are_validated_before_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
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
    runner_run.assert_not_called()
    await close(instance)


def test_legacy_run_route_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, _ = agent(monkeypatch)
    paths = instance.setup_webserver().openapi()["paths"]
    assert "/run" not in paths
    assert "/v1/agent_sessions" in paths
    assert "/v1/agent_sessions/close" in paths
    assert "/v1/agent_sessions/finish" not in paths


async def test_no_tool_grant_preserves_resources_cookie_jar(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, runner_run = agent(monkeypatch)
    req = await open_session(instance)
    await instance.responses(req, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    receipt = await close(instance)
    # SWE-bench Pro has no resource tools; only Resources knows its session cookie.
    assert receipt.resources_cookies is None
