# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.episode import (
    TerminalBench4EpisodeConfig,
    TerminalBench4EpisodeResourcesServer,
)
from responses_api_agents.miniswe_sandboxed_agent.app import empty_response
from responses_api_agents.miniswe_sandboxed_agent.tests.test_server import fixture  # noqa: F401


def native(f, monkeypatch):
    server = TerminalBench4EpisodeResourcesServer(
        config=TerminalBench4EpisodeConfig(**f.server.config.model_dump(), sandbox_provider_ref="sandbox"),
        server_client=f.server.server_client,
    )
    server._loader = f.server._loader
    factory = lifecycle.Environment

    def environment(*args, **kwargs):
        env = factory(*args, **kwargs)
        env.pool = "default"
        env.agent_workdir = AsyncMock(return_value="/task")
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    body = ResourcesSeedSessionRequest(
        episode_id=EpisodeId(rollout_id="episode", attempt=2),
        task_id=TaskId(taskset="tb4", task_id=f.body.task_name),
        task_data=f.body.model_dump(),
    )
    f.server = server
    return server, body


async def test_native_tb4_seed_verify_and_idempotent_close(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    seed = await server.seed_session(f.request, body)
    assert seed.agent_context.instruction == "Solve task"
    assert seed.agent_context.timeout_sec == 28800
    assert seed.agent_context.user == "task-user"
    assert seed.sandbox_access.workdir == "/task"
    assert seed.sandbox_access.connection.provider_config_ref == "sandbox"
    assert await server.seed_session(f.request, body) == seed
    params = f.body.responses_create_params
    response = empty_response(params, "a-different-harness")
    response.status = "incomplete"
    response.metadata = {"termination_reason": "timeout", "agent_started": "true"}
    verification = ResponsesResourcesVerifyRequest(
        episode_id=body.episode_id,
        task_id=body.task_id,
        verification_input={"responses_create_params": params, "response": response},
    )
    result = await server.verify(f.request, verification)
    assert result.reward == 0.75 and result.evaluation_completed
    assert result.termination.reason == "timeout"
    assert await server.verify(f.request, verification) == result
    f.grade.assert_awaited_once()
    close = ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=body.episode_id)
    assert await server.close_session(f.request, close) == await server.close_session(f.request, close)
    assert all(env.closed for env in f.envs)


async def test_native_tb4_close_without_agent_skips_grading(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    seed = await server.seed_session(f.request, body)
    with pytest.raises(HTTPException, match="Episode identity"):
        await server.close_session(
            f.request,
            ResourcesCloseSessionRequest(
                resources_session_id=seed.resources_session_id, episode_id=EpisodeId(rollout_id="wrong")
            ),
        )
    await server.close_session(
        f.request,
        ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=body.episode_id),
    )
    assert all(env.closed for env in f.envs)
    f.grade.assert_not_awaited()
    params = f.body.responses_create_params
    with pytest.raises(HTTPException, match="closed without verification"):
        await server.verify(
            f.request,
            ResponsesResourcesVerifyRequest(
                episode_id=body.episode_id,
                task_id=body.task_id,
                verification_input={"responses_create_params": params, "response": empty_response(params, "model")},
            ),
        )


async def test_native_tb4_bad_identity_never_allocates(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    body.task_id = TaskId(taskset="tb4", task_id="wrong-task")
    with pytest.raises(HTTPException, match="task_name"):
        await server.seed_session(f.request, body)
    assert not f.envs


async def test_native_tb4_handoff_failure_rolls_back_provisioning(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    factory = lifecycle.Environment

    def environment(*args, **kwargs):
        env = factory(*args, **kwargs)
        env.agent_workdir = AsyncMock(side_effect=RuntimeError("Unable to resolve workdir"))
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    with pytest.raises(RuntimeError, match="workdir"):
        await server.seed_session(f.request, body)
    assert all(env.closed for env in f.envs)
    f.grade.assert_not_awaited()
