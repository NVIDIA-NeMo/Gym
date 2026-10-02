# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.episode import (
    TerminalBench4EpisodeConfig,
    TerminalBench4EpisodeResourcesServer,
    TerminalBench4NativeVerifyRequest,
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
        resources_session_id="test-native-resource",
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
    verification = TerminalBench4NativeVerifyRequest(**(f.body.model_dump() | {"response": response}))
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
            TerminalBench4NativeVerifyRequest(**(f.body.model_dump() | {"response": empty_response(params, "model")})),
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


async def test_native_tb4_close_recovers_without_seed_response_cookie(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    await server.seed_session(f.request, body)
    f.request.session.clear()
    await server.close_session(
        f.request,
        ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id),
    )
    assert all(env.closed for env in f.envs)
    f.grade.assert_not_awaited()


async def test_native_tb4_close_before_seed_prevents_allocation(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    await server.close_session(f.request, close)
    with pytest.raises(HTTPException, match="closed"):
        await server.seed_session(f.request, body)
    assert not f.envs


async def test_native_tb4_failed_destruction_does_not_acknowledge_close_and_can_retry(fixture, monkeypatch):
    f = fixture
    server, body = native(f, monkeypatch)
    await server.seed_session(f.request, body)
    env = f.envs[0]
    stop = env.stop
    env.stop = AsyncMock(side_effect=RuntimeError("provider deletion unavailable"))
    close = ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id)
    with pytest.raises(HTTPException, match="cleanup is incomplete"):
        await server.close_session(f.request, close)
    assert not env.closed
    env.stop = stop
    assert (await server.close_session(f.request, close)).resources_session_id == body.resources_session_id
    assert all(env.closed for env in f.envs)
    f.grade.assert_not_awaited()


@pytest.mark.parametrize("audit", [False, True])
def test_native_tb4_http_close_routes_to_owner_cleanup(fixture, monkeypatch, audit):
    f = fixture
    server, body = native(f, monkeypatch)
    if audit:
        from benchmarks.terminal_bench_4.swapping_smoke import audited_classes
        from responses_api_agents.miniswe_sandboxed_agent.episode import MiniSWEEpisodeAgent

        resource_cls, _ = audited_classes(type(server), MiniSWEEpisodeAgent, server.config.artifacts_dir)
        audited = resource_cls(config=server.config, server_client=server.server_client)
        audited._loader = server._loader
        server = audited
    app = server.setup_webserver()
    assert len([route for route in app.routes if getattr(route, "path", None) == "/close_session"]) == 1
    with TestClient(app) as client:
        seed = client.post("/seed_session", json=body.model_dump(mode="json"))
        assert seed.status_code == 200, seed.text
        client.cookies.clear()
        close = client.post(
            "/close_session",
            json={"resources_session_id": body.resources_session_id, "episode_id": body.episode_id.model_dump()},
        )
        assert close.status_code == 200, close.text
        assert all(env.closed for env in f.envs)
        f.grade.assert_not_awaited()
        if audit:
            import json

            events = json.loads((server.config.artifacts_dir / "lifecycle.json").read_text())
            assert events[-1]["event"] == "resources_close_complete"
