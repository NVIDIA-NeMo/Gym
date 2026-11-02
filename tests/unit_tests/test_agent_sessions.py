# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agent_activation_types import AgentContinuationCapabilities
from nemo_gym.agent_utils.sandbox_session import ExpiryUnsupportedError
from nemo_gym.base_responses_api_agent import (
    ROLLOUT_STOP_METADATA_KEY,
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    AgentSessionCapabilitiesResponse,
    AgentSessionState,
    RolloutStopRequest,
    RolloutStopResponse,
    SimpleResponsesAPIAgent,
)
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord, TrajectoryTurn


class _Agent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError


def _seed_request(*, tool_accesses=None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task_id=TaskId(taskset="test", task_id="task"),
        tool_accesses=tool_accesses or [],
    )


def test_agent_session_rejects_duplicate_tool_names() -> None:
    duplicate = [
        MCPToolAccess(
            name="tools",
            required=True,
            connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp"),
        ),
        DirectHTTPToolAccess(
            name="tools",
            required=False,
            base_url="http://resources:8000",
        ),
    ]

    with pytest.raises(ValidationError, match="names must be unique"):
        _seed_request(tool_accesses=duplicate)


def test_episode_tool_access_overrides_configured_access_by_name() -> None:
    configured = DirectHTTPToolAccess(
        name="memory",
        required=True,
        base_url="http://configured:8000",
    )
    episode = MCPToolAccess(
        name="memory",
        required=True,
        connection=MCPStreamableHTTPConnection(url="http://episode:8000/mcp"),
    )
    agent = _Agent.model_construct(
        config=MagicMock(tool_accesses=[configured]),
        server_client=MagicMock(),
    )

    assert agent.effective_tool_accesses(_seed_request(tool_accesses=[episode])) == [episode]


def test_base_agent_exposes_unimplemented_session_routes() -> None:
    agent = _Agent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    request = MagicMock()
    # A configuration that routes episodes to an agent without sessions fails each episode once, not on retry.
    detail = check_agent_seed_rejected(agent.setup_webserver(), _seed_request())
    assert detail == "This agent does not implement episode sessions"
    assert agent._session_records == {}
    with pytest.raises(NotImplementedError, match="does not implement"):
        asyncio.run(
            agent.close_agent_session(
                request,
                AgentCloseSessionRequest(
                    agent_session_id="agent-session",
                    episode_id=EpisodeId(rollout_id="rollout", attempt=2),
                ),
            )
        )

    paths = agent.setup_webserver().openapi()["paths"]
    assert "/v1/agent_sessions" in paths
    assert "/v1/agent_sessions/close" in paths
    assert "/v1/agent_sessions/capabilities" in paths


def test_the_capabilities_route_declares_continuation_without_seeding() -> None:
    """An environment server asks before provisioning: a multi-step run that must continue one harness session
    is refused up front for an adapter that answers without ``continuation``, as Harbor refuses
    ``resume_trajectory`` for an agent without ``SUPPORTS_RESUME`` before any environment spend."""

    class _Continues(_Agent):
        def supports_task_tool_context(self) -> bool:
            return True

        def _agent_continuation_capabilities(self):
            return AgentContinuationCapabilities(observations=["trajectory", "timing"])

    for agent_cls, expected in (
        (_Agent, AgentSessionCapabilitiesResponse()),
        (
            _Continues,
            AgentSessionCapabilitiesResponse(
                supports_task_tool_context=True,
                continuation=AgentContinuationCapabilities(observations=["trajectory", "timing"]),
            ),
        ),
    ):
        agent = agent_cls.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
        reply = TestClient(agent.setup_webserver()).get("/v1/agent_sessions/capabilities")
        assert reply.status_code == 200
        assert AgentSessionCapabilitiesResponse.model_validate(reply.json()) == expected
        # Asking seeds nothing.
        assert agent._session_records == {}


class _GrantCheckingAgent(_Agent):
    """Serves sessions without tool grants and refuses a seed that requires one."""

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        if body.tool_accesses:
            raise AgentSeedRejectedError("this agent cannot use tool grants")
        return AgentSessionState(request=body)


class _PlainValueErrorAgent(_Agent):
    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        raise ValueError("this agent cannot use tool grants")


_REQUIRED_GRANT = [
    MCPToolAccess(
        name="resources",
        required=True,
        connection=MCPStreamableHTTPConnection(url="http://resources:8000/mcp"),
    )
]


def test_a_rejected_seed_is_still_a_value_error() -> None:
    rejected = AgentSeedRejectedError("unsupported")

    assert isinstance(rejected, ValueError)
    assert rejected.status_code == 422
    assert rejected.detail == "unsupported"


def test_an_unsupported_seed_fails_with_a_final_422_and_keeps_no_session() -> None:
    agent = _GrantCheckingAgent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    app = agent.setup_webserver()

    detail = check_agent_seed_rejected(app, _seed_request(tool_accesses=_REQUIRED_GRANT))

    assert detail == "this agent cannot use tool grants"
    assert agent._session_records == {}
    # The refusal bound nothing, so the same ID can still be seeded with a request the agent supports.
    assert TestClient(app).post("/v1/agent_sessions", json=_seed_request().model_dump(mode="json")).status_code == 200


@pytest.mark.parametrize(
    ("agent_class", "detail"),
    [
        (_Agent, "This agent does not implement episode sessions"),
        (_GrantCheckingAgent, "this agent cannot use tool grants"),
    ],
)
def test_a_refused_seed_logs_its_reason_once(caplog, agent_class, detail) -> None:
    """FastAPI answers the 422 inside the router, so the exception middleware never logs the refusal."""
    agent = agent_class.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())

    with caplog.at_level(logging.WARNING, logger="nemo_gym.base_responses_api_agent"):
        check_agent_seed_rejected(agent.setup_webserver(), _seed_request(tool_accesses=_REQUIRED_GRANT))

    [record] = [record for record in caplog.records if record.name == "nemo_gym.base_responses_api_agent"]
    assert record.levelno == logging.WARNING
    assert record.getMessage() == f"Refused agent session seed agent-session: {detail}"


def test_the_check_catches_a_seed_refusal_that_would_be_retried() -> None:
    agent = _PlainValueErrorAgent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())

    with pytest.raises(AssertionError, match="final 422 from AgentSeedRejectedError, got 500"):
        check_agent_seed_rejected(agent.setup_webserver(), _seed_request(tool_accesses=_REQUIRED_GRANT))


def test_the_check_requires_a_422_with_a_string_detail() -> None:
    # An app without the session route answers 404, which is final but is not a refusal.
    with pytest.raises(AssertionError, match="final 422 from AgentSeedRejectedError, got 404"):
        check_agent_seed_rejected(FastAPI(), _seed_request())

    # A request validation error is also 422, but its detail is a list of errors, not the agent's reason.
    invalid = FastAPI()

    @invalid.post("/v1/agent_sessions")
    async def seed() -> JSONResponse:
        return JSONResponse({"detail": [{"msg": "field required"}]}, status_code=422)

    with pytest.raises(AssertionError, match="must carry a string detail"):
        check_agent_seed_rejected(invalid, _seed_request())


def test_close_response_carries_the_harness_trajectory() -> None:
    """The close returns the harness's own turns next to its observations; an older harness's reply still validates."""
    trajectory = TrajectoryRecord(
        task_id="task",
        rollout_id="rollout",
        turns=[
            TrajectoryTurn(
                invocation_id="root", task_id="task", rollout_id="rollout", turn_no=1, timestamp=1.0, step_count=2
            )
        ],
    )
    close = AgentCloseSessionResponse(agent_session_id="agent-session", trajectory=trajectory)
    wire = close.model_dump(mode="json")
    assert wire["trajectory"]["turns"][0]["step_count"] == 2
    assert AgentCloseSessionResponse.model_validate(wire) == close
    assert AgentCloseSessionResponse.model_validate({"agent_session_id": "agent-session"}).trajectory is None


def test_seed_response_carries_the_harness_task_tool_context_declaration() -> None:
    """The environment server refuses a task with MCP servers or skills to a harness that answers False here."""

    class _Seeded(_Agent):
        async def _seed_agent_session_state(self, body):
            return AgentSessionState(request=body)

    class _ReadsTaskTools(_Seeded):
        def supports_task_tool_context(self) -> bool:
            return True

    for agent_cls, declared in ((_Seeded, False), (_ReadsTaskTools, True)):
        agent = agent_cls.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
        response = asyncio.run(agent.seed_agent_session(MagicMock(session={}), _seed_request()))
        assert response == AgentSeedSessionResponse(
            agent_session_id="agent-session", supports_task_tool_context=declared
        )
    # The field has a default so an older harness's reply still validates, as not supporting it.
    assert AgentSeedSessionResponse.model_validate({"agent_session_id": "x"}).supports_task_tool_context is False


def test_agent_session_seed_carries_the_tasks_agent_timeout() -> None:
    """The environment server hands the task's `[agent].timeout_sec` to the harness in the seed."""
    seed = _seed_request()
    assert seed.agent_timeout_sec is None and "agent_timeout_sec" in seed.model_dump(mode="json")
    timed = AgentSeedSessionRequest.model_validate({**seed.model_dump(mode="json"), "agent_timeout_sec": 900.0})
    assert timed.agent_timeout_sec == 900.0
    with pytest.raises(ValidationError):
        AgentSeedSessionRequest.model_validate({**seed.model_dump(mode="json"), "agent_timeout_sec": 0})


class _StoppableAgent(_Agent):
    """Hands each session one fake sandbox session; seeding waits for ``installed``."""

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        await self._installed.wait()
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    def _rollout_sandbox_session(self, state: AgentSessionState):
        return self._sandbox_session


def _stoppable_agent(expired: list[bool]) -> _StoppableAgent:
    agent = _StoppableAgent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    agent._installed = asyncio.Event()
    agent._sandbox_session = MagicMock(expire=AsyncMock(side_effect=lambda *, timeout, reason: expired.pop(0)))
    return agent


def test_rollout_stop_answers_501_for_an_agent_without_sandbox_sessions() -> None:
    agent = _Agent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    response = TestClient(agent.setup_webserver()).post("/rollouts/rollout-a2/stop")
    assert response.status_code == 501


def test_rollout_stop_expires_the_running_harness_once() -> None:
    async def scenario():
        agent = _stoppable_agent([True, False])
        agent._installed.set()
        await agent.seed_agent_session(MagicMock(session={}), _seed_request())
        with pytest.raises(HTTPException) as unknown:
            await agent.stop_rollout("other-rollout")
        assert unknown.value.status_code == 404
        first = await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="budget"))
        second = await agent.stop_rollout("rollout-a2")
        return agent, first, second

    agent, first, second = asyncio.run(scenario())
    assert first == RolloutStopResponse(rollout_id="rollout-a2", stopped=True)
    assert second == RolloutStopResponse(rollout_id="rollout-a2", stopped=False)
    assert [call.kwargs["reason"] for call in agent._sandbox_session.expire.await_args_list] == ["budget", None]


def test_rollout_stop_answers_501_when_the_sandbox_cannot_stop_a_running_harness() -> None:
    async def scenario():
        agent = _stoppable_agent([])
        unsupported = ExpiryUnsupportedError("the test sandbox's provider cannot stop a running harness")
        agent._sandbox_session.expire = AsyncMock(side_effect=unsupported)
        agent._installed.set()
        await agent.seed_agent_session(MagicMock(session={}), _seed_request())
        with pytest.raises(HTTPException) as refused:
            await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="budget"))
        return refused.value

    refused = asyncio.run(scenario())
    assert refused.status_code == 501
    assert refused.detail == (
        "Cannot stop rollout rollout-a2: the test sandbox's provider cannot stop a running harness"
    )


def test_rollout_stop_during_install_expires_the_harness_at_launch() -> None:
    async def scenario():
        agent = _stoppable_agent([True])
        seeding = asyncio.create_task(agent.seed_agent_session(MagicMock(session={}), _seed_request()))
        await asyncio.sleep(0)
        first = await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="budget"))
        second = await agent.stop_rollout("rollout-a2", RolloutStopRequest(reason="operator"))
        agent._sandbox_session.expire.assert_not_awaited()
        agent._installed.set()
        await seeding
        return agent, first, second

    agent, first, second = asyncio.run(scenario())
    assert (first.stopped, second.stopped) == (True, False)
    # The first stop's reason is the one the harness's launch carries.
    agent._sandbox_session.expire.assert_awaited_once()
    assert agent._sandbox_session.expire.await_args.kwargs["reason"] == "budget"


def test_rollout_stop_after_close_is_unknown() -> None:
    async def scenario():
        agent = _stoppable_agent([])
        agent._installed.set()
        await agent.seed_agent_session(MagicMock(session={}), _seed_request())
        await agent.close_agent_session(
            MagicMock(session={}),
            AgentCloseSessionRequest(
                agent_session_id="agent-session", episode_id=EpisodeId(rollout_id="rollout", attempt=2)
            ),
        )
        with pytest.raises(HTTPException) as closed:
            await agent.stop_rollout("rollout-a2")
        return closed.value.status_code

    assert asyncio.run(scenario()) == 404


@pytest.mark.parametrize(
    "status, expired, marked",
    [("incomplete", True, True), ("incomplete", False, False), ("failed", True, False), ("completed", True, False)],
)
def test_rollout_stop_marks_only_the_timeout_shaped_response_it_ended(status: str, expired: bool, marked: bool):
    agent = _Agent.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    response = NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        status=status,
        metadata={"existing": "retained"},
    )
    reported = agent.report_rollout_stop(response, MagicMock(expired=expired, expire_reason="budget"))
    if marked:
        # A stop is not a failure: the marker replaces any failure kind, and the status stays ``incomplete``.
        assert reported.metadata == {"existing": "retained", ROLLOUT_STOP_METADATA_KEY: '{"reason":"budget"}'}
        assert reported.model_dump(exclude={"metadata"}) == response.model_dump(exclude={"metadata"})
    else:
        assert reported is response
