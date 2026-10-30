# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    BaseResponsesAPIAgentConfig,
    StatelessResponsesAPIAgent,
)
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.server_utils import ServerClient


class _Agent(StatelessResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError


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
    agent = _Agent(
        config=BaseResponsesAPIAgentConfig(name="stateless", host="127.0.0.1", port=8000, entrypoint="app.py"),
        server_client=MagicMock(spec=ServerClient),
    )
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
    first = asyncio.create_task(agent._run_stateless_agent_session(request, body, operation=operation))
    await asyncio.wait_for(entered.wait(), 2)
    retry = asyncio.create_task(agent._run_stateless_agent_session(request, body, operation=operation))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    finish.set()
    result = await asyncio.wait_for(retry, 2)
    result.id = "mutated-copy"
    replay = await agent._run_stateless_agent_session(request, body, operation=operation)
    assert replay.id == "result"
    assert calls == 1
    with pytest.raises(HTTPException) as conflict:
        await agent._run_stateless_agent_session(
            request, body.model_copy(update={"input": "changed"}), operation=operation
        )
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
    pending = asyncio.create_task(agent._run_stateless_agent_session(request, body, operation=operation))
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
            await agent._run_stateless_agent_session(_request({"agent_session_id": marker}), body, operation=operation)
        assert stale.value.status_code == 409
