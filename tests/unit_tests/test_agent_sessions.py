# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.agents import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentContinuationCapabilities,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
    AgentSessionCapabilitiesResponse,
    AgentSessionState,
    SimpleResponsesAPIAgent,
)
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.trajectory import TrajectoryRecord, TrajectoryTurn


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

    with caplog.at_level(logging.WARNING, logger="nemo_gym.agents.base_responses_api_agent"):
        check_agent_seed_rejected(agent.setup_webserver(), _seed_request(tool_accesses=_REQUIRED_GRANT))

    [record] = [record for record in caplog.records if record.name == "nemo_gym.agents.base_responses_api_agent"]
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
