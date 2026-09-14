# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxHandle
from nemo_gym.sandbox.agent import empty_response
from nemo_gym.sandbox.handoff import AgentTermination, SandboxedVerifyRequest, SessionRequest
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_4 import app as module
from resources_servers.terminal_bench_4.app import (
    TerminalBench4Config,
    TerminalBench4ResourcesServer,
    TerminalBench4SeedRequest,
)
from resources_servers.terminal_bench_4.runtime import ExternalAgent


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    server = TerminalBench4ResourcesServer(
        config=TerminalBench4Config(
            host="localhost",
            port=1,
            name="tb4",
            entrypoint="app.py",
            environment={},
            artifacts_dir=tmp_path,
            agent_max_timeout_sec=2,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    task = next(iter(server._tasks.values()))
    body = TerminalBench4SeedRequest(
        task_name="terminal-bench/" + task["name"],
        task_ref=task["ref"],
        dataset_ref=server._manifest["ref"],
        rollout_id="rollout-1",
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"})
    env = SimpleNamespace(
        main_connection=AsyncMock(return_value={"provider": "gpu", "sandbox_id": "box", "workdir": "/task"}),
        quiesce_agent=AsyncMock(),
    )
    adapter = ExternalAgent(logs_dir=tmp_path)
    trial = SimpleNamespace(
        agent=adapter,
        task=SimpleNamespace(
            has_steps=False,
            instruction="Solve task",
            config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=28800, user="task-user")),
        ),
    )

    async def run():
        await adapter.setup(env)
        await adapter.run("Solve task", env, None)
        return SimpleNamespace(model_dump=lambda **_: {"verifier_result": {"rewards": {"reward": 0.75}}})

    trial.run = AsyncMock(side_effect=run)
    create = AsyncMock(return_value=trial)
    monkeypatch.setattr(module.Trial, "create", create)
    return server, request, body, trial, env, create


def verify_body(session_id, reason="completed"):
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    return SandboxedVerifyRequest(
        session_id=session_id,
        responses_create_params=params,
        response=empty_response(params, "model"),
        termination=AgentTermination(reason=reason),
    )


async def test_pins_duplicates_handoff_verification_and_restart(fixture):
    server, request, body, trial, env, create = fixture
    first, duplicate = await asyncio.gather(server.seed_session(request, body), server.seed_session(request, body))
    assert first == duplicate
    assert first.sandbox.provider == "gpu"
    assert first.sandbox.workdir == "/task"
    assert first.user == "task-user"
    assert first.agent_timeout_sec == 2
    assert first.setup_timeout_sec == 360
    assert create.await_count == 1
    assert create.await_args.args[0].task.path is None
    assert create.await_args.args[0].task.ref == body.task_ref
    assert not trial.agent.episode.running.is_set()
    budget = await server.start_session(request, SessionRequest(session_id=first.session_id))
    assert 0 < budget["agent_timeout_sec"] <= 2
    retry_budget = await server.start_session(request, SessionRequest(session_id=first.session_id))
    assert 0 < retry_budget["agent_timeout_sec"] <= budget["agent_timeout_sec"]
    verified, retry = await asyncio.gather(
        server.verify(request, verify_body(first.session_id)),
        server.verify(request, verify_body(first.session_id)),
    )
    assert verified == retry
    assert verified.reward == 0.75
    assert verified.evaluation_completed
    assert verified.infrastructure_error is None
    env.quiesce_agent.assert_awaited_once_with(first.session_id)
    assert trial.run.await_count == 1
    restarted = TerminalBench4ResourcesServer(config=server.config, server_client=MagicMock(spec=ServerClient))
    assert await restarted.verify(request, verify_body(first.session_id, "timeout")) == verified
    with pytest.raises(HTTPException) as exc:
        await restarted.seed_session(request, body)
    assert exc.value.status_code == 409


@pytest.mark.parametrize(
    "field,value", [("task_name", "../../etc/passwd"), ("task_ref", "latest"), ("dataset_ref", "wrong")]
)
async def test_untrusted_pins_rejected_before_allocation(fixture, field, value):
    server, request, body, _, _, create = fixture
    with pytest.raises(HTTPException) as exc:
        await server.seed_session(request, body.model_copy(update={field: value}))
    assert exc.value.status_code == 422
    create.assert_not_awaited()


async def test_session_isolation_and_cancel_during_setup(fixture):
    server, request, body, trial, _, _ = fixture
    seed = await server.seed_session(request, body)
    stranger = SimpleNamespace(session={SESSION_ID_KEY: "stranger"})
    with pytest.raises(HTTPException) as exc:
        await server.verify(stranger, verify_body(seed.session_id))
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        await server.verify(request, verify_body(seed.session_id))
    assert exc.value.status_code == 409
    await server.cancel_session(request, SessionRequest(session_id=seed.session_id))
    assert server._sessions[seed.session_id].task.cancelled()
    assert not trial.agent.episode.running.is_set()


async def test_disconnected_verify_does_not_cancel_owner(fixture):
    server, request, body, trial, _, _ = fixture
    seed = await server.seed_session(request, body)
    await server.start_session(request, SessionRequest(session_id=seed.session_id))
    task = asyncio.create_task(server.verify(request, verify_body(seed.session_id)))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    result = await server.verify(request, verify_body(seed.session_id))
    assert result.evaluation_completed
    assert trial.run.await_count == 1


@pytest.mark.parametrize("reward", [0, 1, None])
async def test_official_grades_and_missing_reward_are_distinct(fixture, reward):
    server, request, body, trial, _, _ = fixture
    seed = await server.seed_session(request, body)
    await server.start_session(request, SessionRequest(session_id=seed.session_id))
    session = server._sessions[seed.session_id]
    session.episode.finished.set()
    session.episode.termination = AgentTermination(reason="completed")
    await session.task
    session.result = {"verifier_result": {"rewards": {} if reward is None else {"reward": reward}}}
    result = await server.verify(request, verify_body(seed.session_id, "timeout"))
    assert result.evaluation_completed == (reward is not None)
    assert result.reward == (reward or 0)
    assert bool(result.model_dump().get("_ng_failure_class")) == (reward is None)


@pytest.mark.parametrize("operation", ["release", "stop"])
async def test_borrowed_release_cannot_destroy_owner(operation):
    provider = MagicMock()
    provider.close = AsyncMock()
    provider.aclose = AsyncMock()
    sandbox = AsyncSandbox(provider, owns_sandbox=False)
    sandbox._handle = SandboxHandle(sandbox_id="borrowed", provider_name="test", raw=None)
    sandbox._stopped = False
    await getattr(sandbox, operation)()
    await sandbox.stop()
    provider.close.assert_not_awaited()
    provider.aclose.assert_awaited_once()


async def test_owner_stop_still_destroys_remote():
    provider = MagicMock(close=AsyncMock(), aclose=AsyncMock())
    sandbox = AsyncSandbox(provider)
    sandbox._handle = SandboxHandle(sandbox_id="owned", provider_name="test", raw=None)
    sandbox._stopped = False
    await sandbox.stop()
    provider.close.assert_awaited_once()


async def test_seed_retry_before_cookie_response_reuses_episode(fixture):
    server, request, body, _, _, create = fixture
    body = body.model_copy(update={"client_session_id": "stable-agent-session"})
    first = await server.seed_session(request, body)
    retried_request = SimpleNamespace(session={SESSION_ID_KEY: "new-resources-cookie"})
    retry = await server.seed_session(retried_request, body)
    assert first == retry
    create.assert_awaited_once()
    await server.cancel_session(retried_request, SessionRequest(session_id=retry.session_id))
    assert await server.cancel_session(retried_request, SessionRequest(session_id=retry.session_id)) == {
        "session_id": retry.session_id,
        "phase": "closed",
    }


async def test_authoritative_timeout_keeps_worker_artifact_references(fixture):
    server, request, body, _, _, _ = fixture
    seed = await server.seed_session(request, body)
    await server.start_session(request, SessionRequest(session_id=seed.session_id))
    session = server._sessions[seed.session_id]
    session.episode.termination = AgentTermination(reason="timeout", detail="Resources deadline reached")
    submitted = verify_body(seed.session_id)
    submitted.termination.artifacts = ["worker/trajectory.json", "worker/trajectory.json"]
    result = await server.verify(request, submitted)
    assert result.termination.reason == "timeout"
    assert result.termination.detail == "Resources deadline reached"
    assert result.termination.artifacts == ["worker/trajectory.json"]


async def test_another_worker_cannot_execute_the_same_active_rollout(fixture):
    server, request, body, _, _, create = fixture
    body = body.model_copy(update={"execution_id": "first-worker"})
    seed = await server.seed_session(request, body)
    assert await server.seed_session(request, body) == seed
    with pytest.raises(HTTPException) as exc:
        await server.seed_session(request, body.model_copy(update={"execution_id": "another-worker"}))
    assert exc.value.status_code == 409
    create.assert_awaited_once()
    await server.cancel_session(request, SessionRequest(session_id=seed.session_id))


async def test_shutdown_cancels_owner_and_closed_episode_cannot_restart(fixture):
    server, request, body, _, _, _ = fixture
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        seed = await server.seed_session(request, body)
    session = server._sessions[seed.session_id]
    assert session.task.cancelled()
    restarted = TerminalBench4ResourcesServer(config=server.config, server_client=MagicMock(spec=ServerClient))
    for instance in [server, restarted]:
        with pytest.raises(HTTPException) as exc:
            await instance.start_session(request, SessionRequest(session_id=seed.session_id))
        assert exc.value.status_code == 409
        result = await instance.verify(request, verify_body(seed.session_id))
        assert not result.evaluation_completed
        assert result.termination.reason == "cancelled"
        assert result.infrastructure_error


@pytest.mark.parametrize("failure", ["create", "steps"])
async def test_partial_preparation_failure_is_recorded(fixture, failure):
    server, request, body, trial, _, create = fixture
    if failure == "create":
        create.side_effect = RuntimeError("provisioning failed")
    else:
        trial.task.has_steps = True
    with pytest.raises(HTTPException) as exc:
        await server.seed_session(request, body)
    assert exc.value.status_code == 409
    assert next(iter(server._sessions.values())).episode.phase == "closed"
    assert next(iter(server._sessions.values())).result["exception_info"]
    trial.run.assert_not_awaited()


async def test_rollout_conflict_and_interrupted_restart_rejected(fixture):
    server, request, body, _, _, _ = fixture
    seed = await server.seed_session(request, body)
    second = list(server._tasks.values())[1]
    with pytest.raises(HTTPException) as exc:
        await server.seed_session(
            request,
            body.model_copy(update={"task_name": "terminal-bench/" + second["name"], "task_ref": second["ref"]}),
        )
    assert exc.value.status_code == 409
    restarted = TerminalBench4ResourcesServer(config=server.config, server_client=MagicMock(spec=ServerClient))
    with pytest.raises(HTTPException) as exc:
        restarted._session(request, seed.session_id)
    assert exc.value.status_code == 409
    stranger = SimpleNamespace(session={SESSION_ID_KEY: "stranger"})
    with pytest.raises(HTTPException) as exc:
        restarted._session(stranger, seed.session_id)
    assert exc.value.status_code == 404
    for missing in ["../../secret", "tb4-" + "0" * 32]:
        with pytest.raises(HTTPException) as exc:
            restarted._session(request, missing)
        assert exc.value.status_code == 404
    await server.cancel_session(request, SessionRequest(session_id=seed.session_id))


@pytest.mark.parametrize("reason", ["cancelled", "timeout", "nonzero_exit", "infrastructure_error"])
async def test_running_termination_survives_verification(fixture, reason):
    server, request, body, _, env, _ = fixture
    seed = await server.seed_session(request, body)
    await server.start_session(request, SessionRequest(session_id=seed.session_id))
    if reason == "cancelled":
        await server.cancel_session(request, SessionRequest(session_id=seed.session_id))
    else:
        await server.verify(request, verify_body(seed.session_id, reason))
    session = server._sessions[seed.session_id]
    # Simulate Harbor's official grade after its graded-agent exception path.
    session.result = {"verifier_result": {"rewards": {"reward": 1}}}
    session.verified_response = None
    result = await server.verify(request, verify_body(seed.session_id, reason))
    assert result.reward == 1
    assert result.evaluation_completed
    assert result.termination.reason == reason
    assert bool(result.infrastructure_error) == (reason == "infrastructure_error")
    env.quiesce_agent.assert_awaited_once()
