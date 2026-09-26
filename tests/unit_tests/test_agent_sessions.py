# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    SimpleResponsesAPIAgent,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess


class _Agent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError


def _seed_request(*, tool_accesses=None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
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
    with pytest.raises(NotImplementedError, match="does not implement"):
        asyncio.run(agent.seed_agent_session(request, _seed_request()))
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
