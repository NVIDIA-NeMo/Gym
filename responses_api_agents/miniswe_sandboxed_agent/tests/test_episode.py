# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.miniswe_sandboxed_agent.app import MiniSWESandboxedConfig, empty_response
from responses_api_agents.miniswe_sandboxed_agent.episode import MiniSWEEpisodeAgent
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome
from responses_api_agents.miniswe_sandboxed_agent.models import AgentExecutionResult


def fixture(tmp_path):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"borrowed": {"local": {}}}
    agent = MiniSWEEpisodeAgent(
        config=MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="agent",
            entrypoint="episode.py",
            model_server={"type": "responses_api_models", "name": "model"},
            artifacts_dir=tmp_path,
            shutdown_timeout_sec=0.1,
        ),
        server_client=client,
    )
    request = SimpleNamespace(
        session={SESSION_ID_KEY: "owner"}, cookies={"session": "cookie"}, path_params={"rollout_id": "rollout-a2"}
    )
    seed = AgentSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task_id=TaskId(taskset="unrelated-benchmark", task_id="problem-17"),
        sandbox_access={
            "connection": {"provider_config_ref": "borrowed", "descriptor": {"sandbox_id": "test"}},
            "workdir": "/problem",
        },
    )
    return agent, request, seed


async def test_native_agent_deduplicates_activation_and_never_calls_resources(tmp_path, monkeypatch):
    agent, request, seed = fixture(tmp_path)
    params = NeMoGymResponseCreateParamsNonStreaming(input="Fix the public project")
    finished = asyncio.Event()

    async def execute(execution_seed, received_params, **kwargs):
        assert execution_seed.instruction == params.input
        assert execution_seed.workdir == "/problem"
        assert execution_seed.task_id == "problem-17"
        assert kwargs["quiesce"] is True
        await finished.wait()
        return AgentExecutionResult(
            responses_create_params=received_params,
            response=empty_response(received_params, "model"),
            agent_started=True,
            termination=HarnessOutcome(reason="nonzero_exit", detail="LimitsExceeded"),
        )

    monkeypatch.setattr(MiniSWEEpisodeAgent, "execute", AsyncMock(side_effect=execute))
    seeded = await agent.seed_agent_session(request, seed)
    assert await agent.seed_agent_session(request, seed) == seeded
    first = asyncio.create_task(agent.responses(request, params))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    retry = asyncio.create_task(agent.responses(request, params))
    finished.set()
    response = await retry
    agent.execute.assert_awaited_once()
    assert response.status == "incomplete"
    assert response.metadata["termination_detail"] == "LimitsExceeded"
    close = AgentCloseSessionRequest(agent_session_id=seeded.agent_session_id, episode_id=seed.episode_id)
    assert await agent.close_agent_session(request, close) == await agent.close_agent_session(request, close)
    agent.server_client.post.assert_not_called()
    with pytest.raises(HTTPException, match="closed"):
        await agent.responses(request, params)


async def test_native_agent_rejects_identity_changes_and_foreign_cookie(tmp_path):
    agent, request, seed = fixture(tmp_path)
    seeded = await agent.seed_agent_session(request, seed)
    changed = seed.model_copy(deep=True)
    changed.task_id = TaskId(taskset="other", task_id="task")
    with pytest.raises(HTTPException, match="bound"):
        await agent.seed_agent_session(request, changed)
    with pytest.raises(HTTPException, match="Close identity"):
        await agent.close_agent_session(
            request,
            AgentCloseSessionRequest(
                agent_session_id=seeded.agent_session_id, episode_id=EpisodeId(rollout_id="other")
            ),
        )
    request.path_params["rollout_id"] = "rollout"
    with pytest.raises(HTTPException, match="Rollout route"):
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    request.session[SESSION_ID_KEY] = "foreign-owner"
    with pytest.raises(HTTPException, match="Unknown"):
        await agent.close_agent_session(
            request, AgentCloseSessionRequest(agent_session_id=seeded.agent_session_id, episode_id=seed.episode_id)
        )


async def test_native_close_cancels_and_joins_activation(tmp_path, monkeypatch):
    agent, request, seed = fixture(tmp_path)
    started, stopped = asyncio.Event(), asyncio.Event()

    async def execute(execution_seed, params, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            return AgentExecutionResult(
                responses_create_params=params,
                response=empty_response(params, "model"),
                termination=HarnessOutcome(reason="cancelled"),
            )

    monkeypatch.setattr(MiniSWEEpisodeAgent, "execute", AsyncMock(side_effect=execute))
    seeded = await agent.seed_agent_session(request, seed)
    pending = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await started.wait()
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seeded.agent_session_id, episode_id=seed.episode_id)
    )
    assert stopped.is_set()
    assert (await pending).metadata["termination_reason"] == "cancelled"


async def test_retried_close_does_not_interrupt_worker_cleanup(tmp_path, monkeypatch):
    agent, request, seed = fixture(tmp_path)
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(execution_seed, params, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            return AgentExecutionResult(
                responses_create_params=params,
                response=empty_response(params, "model"),
                termination=HarnessOutcome(reason="cancelled"),
            )

    monkeypatch.setattr(MiniSWEEpisodeAgent, "execute", AsyncMock(side_effect=execute))
    seeded = await agent.seed_agent_session(request, seed)
    pending = asyncio.create_task(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="task")))
    await started.wait()
    body = AgentCloseSessionRequest(agent_session_id=seeded.agent_session_id, episode_id=seed.episode_id)
    first_close = asyncio.create_task(agent.close_agent_session(request, body))
    await cleaning.wait()
    retry_close = asyncio.create_task(agent.close_agent_session(request, body))
    await asyncio.sleep(0)
    release.set()
    assert (await pending).metadata["termination_reason"] == "cancelled"
    assert await first_close == await retry_close
