# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    SimpleResponsesAPIAgent,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess


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
