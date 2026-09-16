# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxHandle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.agent import empty_response
from resources_servers.terminal_bench_4.app import (
    TerminalBench4Config,
    TerminalBench4ResourcesServer,
    TerminalBench4SeedRequest,
)
from resources_servers.terminal_bench_4.handoff import AgentTermination, SandboxedVerifyRequest, SessionRequest
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_environment import environment_config


@pytest.fixture
async def fixture(tmp_path, monkeypatch):
    server = TerminalBench4ResourcesServer(
        config=TerminalBench4Config(
            host="localhost",
            port=1,
            name="tb4",
            entrypoint="app.py",
            environment=environment_config(sandbox_provider={"local": {}}),
            artifacts_dir=tmp_path,
            agent_max_timeout_sec=2,
            shutdown_timeout_sec=0.01,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    pin = next(iter(server._tasks.values()))
    body = TerminalBench4SeedRequest(
        task_name="terminal-bench/" + pin["name"],
        task_ref=pin["ref"],
        dataset_ref=server._manifest["ref"],
        rollout_id="rollout",
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"})
    config = TaskSettings.model_validate(
        {
            "environment": {"docker_image": "agent"},
            "agent": {"timeout_sec": 28800, "user": "task-user"},
            "verifier": {"environment": {"docker_image": "verifier"}},
        }
    )
    task = SimpleNamespace(config=config, instruction="Solve task")
    server._loader.load = AsyncMock(return_value=task)
    envs = []
    events = []

    def create(task, config, session_id, directory, verifier=False):
        name = "verifier" if verifier else "agent"
        env = SimpleNamespace(
            task=task,
            session_id=session_id,
            closed=False,
            resources=[],
            cleanup_errors=[],
            shared_logs=None,
            efs_logs_fallback=None,
            build_spec=lambda: None,
            resource_identities=lambda: [],
            main=MagicMock(),
            main_connection=AsyncMock(return_value={"provider": "gpu", "sandbox_id": "box", "workdir": "/task"}),
            healthcheck=AsyncMock(),
            quiesce_agent=AsyncMock(side_effect=lambda _: events.append("quiesce")),
        )

        async def start():
            events.append(name + "_start")

        async def stop():
            events.append(name + "_stop")
            env.closed = True

        env.start, env.stop = AsyncMock(side_effect=start), AsyncMock(side_effect=stop)
        envs.append(env)
        return env

    monkeypatch.setattr(lifecycle, "Environment", create)
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock())
    monkeypatch.setattr(lifecycle, "collect", AsyncMock(side_effect=lambda *a: events.append("collect")))
    monkeypatch.setattr(lifecycle, "restore", AsyncMock(side_effect=lambda *a: events.append("restore")))
    grade = AsyncMock(return_value={"rewards": {"reward": 0.75}})
    monkeypatch.setattr(lifecycle, "run_verifier", grade)
    yield SimpleNamespace(server=server, request=request, body=body, envs=envs, grade=grade, events=events)
    await lifecycle.shutdown(list(server._sessions.values()), 0.01)


@pytest.mark.parametrize("failure", [None, "start", "prepare", "cleanup", "unsupported"])
async def test_efs_session_owns_helper_until_workloads_are_stopped(fixture, monkeypatch, failure):
    f = fixture
    f.server.config.environment.efs_logs_host_path = "/mnt/efs/data/shared"
    logs = SimpleNamespace(
        session_id="logs",
        closed=False,
        resources=[],
        cleanup_errors=[],
        resource_identities=lambda: [{"efs_subpath": "owned"}],
        restored_archive="/logs/snapshot.tar.gz",
    )

    async def initialize():
        f.events.append("logs_start")
        if failure == "start":
            raise RuntimeError("logs start failed")
        if failure == "unsupported":
            raise RuntimeError("VOLUME::HOST_PATH_NOT_ALLOWED /mnt/efs/data/shared")

    async def prepare():
        assert f.envs[0].closed and not f.envs[1].closed
        f.events.append("logs_prepare")
        if failure == "prepare":
            raise RuntimeError("snapshot unavailable; use host transfer")

    async def close(*, remove_data=True):
        if failure != "unsupported":
            assert all(env.closed for env in f.envs)
        assert remove_data
        f.events.append("logs_stop")
        if failure == "cleanup":
            raise RuntimeError("EFS cleanup failed")
        logs.closed = True

    logs.start = AsyncMock(side_effect=initialize)
    logs.prepare_verifier = AsyncMock(side_effect=prepare)
    logs.stop = AsyncMock(side_effect=close)
    monkeypatch.setattr(lifecycle, "SharedLogs", lambda env: logs)
    if failure == "start":
        with pytest.raises(HTTPException):
            await seed(f)
        f.grade.assert_not_awaited()
    else:
        session_id = await start(f)
        response = await f.server.verify(f.request, verify_body(session_id))
        assert response.evaluation_completed and response.reward == 0.75
        assert response.infrastructure_error is None
        if failure == "unsupported":
            assert all(env.shared_logs is None for env in f.envs)
        if failure == "cleanup":
            session = f.server._sessions[session_id]
            assert session.result["exception_info"]["exception_message"] == "EFS cleanup failed"
        assert f.events.index("logs_start") < f.events.index("agent_start")
        assert f.events.index("agent_stop") < f.events.index("logs_prepare") < f.events.index("verifier_start")
    assert f.events[-1] == "logs_stop"
    assert logs.stop.await_count == (2 if failure == "unsupported" else 1)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions


def verify_body(session_id, reason="completed"):
    params = NeMoGymResponseCreateParamsNonStreaming(input=[])
    return SandboxedVerifyRequest(
        session_id=session_id,
        responses_create_params=params,
        response=empty_response(params, "model"),
        termination=AgentTermination(reason=reason),
    )


async def seed(f):
    return await f.server.seed_session(f.request, f.body)


async def start(f):
    result = await seed(f)
    await f.server.start_session(f.request, SessionRequest(session_id=result.session_id))
    return result.session_id


def restart(f):
    return TerminalBench4ResourcesServer(config=f.server.config, server_client=MagicMock(spec=ServerClient))


async def test_pins_duplicates_handoff_verification_and_restart(fixture):
    f = fixture
    first, duplicate = await asyncio.gather(seed(f), seed(f))
    assert first == duplicate
    assert first.sandbox.provider == "gpu" and first.sandbox.workdir == "/task"
    assert first.user == "task-user" and first.agent_timeout_sec == 2 and first.setup_timeout_sec == 360
    f.server._loader.load.assert_awaited_once_with(f.body.task_name, f.body.task_ref)
    f.envs[0].main_connection.assert_awaited_once()
    directory = f.server._sessions[first.session_id].directory
    assert all((directory / name).is_dir() for name in ("agent", "verifier", "artifacts/logs/artifacts"))
    budgets = [await f.server.start_session(f.request, SessionRequest(session_id=first.session_id)) for _ in range(2)]
    assert 0 < budgets[1]["agent_timeout_sec"] <= budgets[0]["agent_timeout_sec"] <= 2
    verified, retry = await asyncio.gather(
        *(f.server.verify(f.request, verify_body(first.session_id)) for _ in range(2))
    )
    assert verified == retry and verified.reward == 0.75 and verified.evaluation_completed
    assert verified.infrastructure_error is None
    assert "provenance" not in verified.model_dump()
    assert f.events == [
        "agent_start",
        "quiesce",
        "collect",
        "agent_stop",
        "verifier_start",
        "restore",
        "verifier_stop",
    ]
    f.grade.assert_awaited_once()
    assert await restart(f).verify(f.request, verify_body(first.session_id, "timeout")) == verified
    with pytest.raises(HTTPException):
        await restart(f).seed_session(f.request, f.body)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions


@pytest.mark.parametrize(
    "field,value", [("task_name", "../../secret"), ("task_ref", "latest"), ("dataset_ref", "wrong")]
)
async def test_untrusted_pins_rejected_before_allocation(fixture, field, value):
    f = fixture
    with pytest.raises(HTTPException) as exc:
        await f.server.seed_session(f.request, f.body.model_copy(update={field: value}))
    assert exc.value.status_code == 422
    f.server._loader.load.assert_not_awaited()


async def test_cookie_isolation_and_worker_conflict(fixture):
    f = fixture
    f.body = f.body.model_copy(update={"client_session_id": "stable", "execution_id": "worker1"})
    first = await seed(f)
    new_request = SimpleNamespace(session={SESSION_ID_KEY: "another-cookie"})
    assert await f.server.seed_session(new_request, f.body) == first
    with pytest.raises(HTTPException) as exc:
        await f.server.seed_session(new_request, f.body.model_copy(update={"execution_id": "worker2"}))
    assert exc.value.status_code == 409
    with pytest.raises(HTTPException) as exc:
        await f.server.verify(SimpleNamespace(session={SESSION_ID_KEY: "stranger"}), verify_body(first.session_id))
    assert exc.value.status_code == 404
    f.server._loader.load.assert_awaited_once()


async def test_setup_cancel_does_not_grade_and_premature_verify_rejected(fixture):
    f = fixture
    session_id = (await seed(f)).session_id
    with pytest.raises(HTTPException) as exc:
        await f.server.verify(f.request, verify_body(session_id))
    assert exc.value.status_code == 409
    for _ in range(2):
        assert await f.server.cancel_session(f.request, SessionRequest(session_id=session_id)) == {
            "session_id": session_id,
            "phase": "closed",
        }
    f.grade.assert_not_awaited()
    assert f.envs[0].closed
    result = await restart(f).verify(f.request, verify_body(session_id))
    assert not result.evaluation_completed and result.termination.reason == "cancelled"


async def test_start_rejected_after_setup_cancellation_wins(fixture):
    f = fixture
    session_id = (await seed(f)).session_id
    session = f.server._sessions[session_id]
    # Request cancellation without yielding to the cleanup task: start must
    # honor the accepted cancellation even while the phase is still ready.
    finish = await lifecycle.request_finish(session, AgentTermination(reason="cancelled"))
    with pytest.raises(HTTPException) as exc:
        await lifecycle.start_session(session)
    assert exc.value.status_code == 409
    await finish
    assert session.termination.reason == "cancelled"
    assert "agent_execution" not in session.result
    f.grade.assert_not_awaited()


@pytest.mark.parametrize("phase", ["seed", "verify"])
async def test_http_disconnect_does_not_cancel_resource_work(fixture, phase):
    f = fixture
    gate = asyncio.Event()
    if phase == "seed":
        task = f.server._loader.load.return_value

        async def load(*args):
            await gate.wait()
            return task

        f.server._loader.load.side_effect = load
        call = asyncio.create_task(seed(f))
    else:
        session_id = await start(f)

        async def grade(*args):
            await gate.wait()
            return {"rewards": {"reward": 1}}

        f.grade.side_effect = grade
        call = asyncio.create_task(f.server.verify(f.request, verify_body(session_id)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    call.cancel()
    await asyncio.gather(call, return_exceptions=True)
    gate.set()
    if phase == "seed":
        assert (await seed(f)).instruction == "Solve task"
    else:
        assert (await f.server.verify(f.request, verify_body(session_id))).reward == 1
        f.grade.assert_awaited_once()


@pytest.mark.parametrize("reason", ["completed", "cancelled", "timeout", "nonzero_exit", "infrastructure_error"])
@pytest.mark.parametrize("reward", [0, 1, None])
async def test_official_grades_and_agent_failure(fixture, reason, reward):
    f = fixture
    f.grade.return_value = {"rewards": {} if reward is None else {"reward": reward}}
    session_id = await start(f)
    result = await f.server.verify(f.request, verify_body(session_id, reason))
    assert result.reward == (reward or 0)
    assert result.evaluation_completed == (reward is not None)
    assert result.termination.reason == reason
    assert bool(result.infrastructure_error) == (reward is None or reason == "infrastructure_error")


async def test_deadline_grades_without_worker_and_late_first_verify_replays(fixture):
    f = fixture
    f.server.config.agent_max_timeout_sec = 0.01
    session_id = await start(f)
    await asyncio.sleep(0.04)
    session = f.server._sessions[session_id]
    await session.finalization
    assert session.phase == "closed" and session.verify_body is None
    f.grade.assert_awaited_once()
    saved = json.loads(f.server._state_path(session.identity).read_text())
    body = verify_body(session_id)
    body.termination.artifacts = ["worker/trajectory.json"]
    response = await restart(f).verify(f.request, body)
    assert response.termination.reason == "timeout"
    assert response.termination.artifacts == body.termination.artifacts
    assert response.evaluation_completed
    assert await restart(f).verify(f.request, verify_body(session_id, "cancelled")) == response
    replayed = json.loads(f.server._state_path(session.identity).read_text())
    for field in ("resources", "deadlines", "diagnostics"):
        assert replayed[field] == saved[field]


async def test_expired_deadline_wins_even_when_watchdog_delayed(fixture):
    f = fixture
    session_id = await start(f)
    session = f.server._sessions[session_id]
    session.watchdog.cancel()
    session.deadline = 0
    result = await f.server.verify(f.request, verify_body(session_id))
    assert result.termination.reason == "timeout"
    f.grade.assert_awaited_once()


@pytest.mark.parametrize("first", ["verify", "cancel"])
async def test_finish_race_first_winner_and_exact_response(fixture, first):
    f = fixture
    session_id = await start(f)
    calls = [
        f.server.verify(f.request, verify_body(session_id)),
        f.server.cancel_session(f.request, SessionRequest(session_id=session_id)),
    ]
    if first == "cancel":
        calls.reverse()
    await asyncio.gather(*calls)
    result = await f.server.verify(f.request, verify_body(session_id, "infrastructure_error"))
    assert result.termination.reason == ("completed" if first == "verify" else "cancelled")
    f.grade.assert_awaited_once()


async def test_setup_budget_excludes_queue_and_provisioning_includes_descriptor(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(lifecycle, "SETUP_TIMEOUT_SEC", 0.03)
    await f.server._slots.acquire()
    f.server._slots = asyncio.Semaphore(0)
    call = asyncio.create_task(seed(f))
    await asyncio.sleep(0.04)
    f.server._slots.release()
    result = await call
    session = f.server._sessions[result.session_id]
    assert session.phase == "ready"
    await asyncio.sleep(0.06)
    await session.finalization
    assert session.phase == "closed"
    assert session.result["exception_info"]["exception_type"] == "AgentSetupTimeoutError"
    f.grade.assert_not_awaited()


@pytest.mark.parametrize(
    "failure", ["load", "start", "healthcheck", "descriptor", "quiesce", "collect", "verifier", "cleanup"]
)
async def test_failures_release_slots_and_attempt_remaining_cleanup(fixture, monkeypatch, failure):
    f = fixture
    if failure == "load":
        f.server._loader.load.side_effect = RuntimeError("load failed")
    else:
        original = lifecycle.Environment

        def create(*args, **kwargs):
            env = original(*args, **kwargs)
            method = {
                "start": "start",
                "healthcheck": "healthcheck",
                "descriptor": "main_connection",
                "quiesce": "quiesce_agent",
                "cleanup": "stop",
            }.get(failure)
            if method and not kwargs.get("verifier"):
                getattr(env, method).side_effect = RuntimeError(failure + " failed")
            return env

        monkeypatch.setattr(lifecycle, "Environment", create)
    if failure in {"load", "start", "healthcheck", "descriptor"}:
        with pytest.raises(HTTPException):
            await seed(f)
    else:
        session_id = await start(f)
        if failure == "collect":
            monkeypatch.setattr(lifecycle, "collect", AsyncMock(side_effect=RuntimeError("collection failed")))
        if failure == "verifier":
            f.grade.side_effect = RuntimeError("verification failed")
        result = await f.server.verify(f.request, verify_body(session_id))
        assert result.evaluation_completed == (failure == "cleanup")
    session = next(iter(f.server._sessions.values()))
    assert session.phase == "closed" and not session.owns_slot
    if f.envs:
        f.envs[-1].stop.assert_awaited()
    if failure == "quiesce":
        f.grade.assert_not_awaited()


async def test_shutdown_and_interrupted_restart(fixture):
    f = fixture
    app = f.server.setup_webserver()
    async with app.router.lifespan_context(app):
        result = await seed(f)
        with pytest.raises(HTTPException) as exc:
            restart(f)._session(f.request, result.session_id)
        assert exc.value.status_code == 409
    assert f.envs[0].closed
    with pytest.raises(HTTPException) as exc:
        await seed(f)
    assert exc.value.status_code == 503


async def test_legacy_closed_result_without_response(fixture):
    f = fixture
    session_id = await start(f)
    result = await f.server.verify(f.request, verify_body(session_id))
    session = f.server._sessions[session_id]
    path = f.server._state_path(session.identity)
    state = json.loads(path.read_text())
    state.pop("record_version")
    state.pop("verify_body")
    state["verified_response"] = None
    state["result"].pop("runtime")
    state["result"].pop("runtime_version")
    path.write_text(json.dumps(state))
    replay = await restart(f).verify(f.request, verify_body(session_id))
    assert replay.reward == result.reward and replay.evaluation_completed
    assert "provenance" not in replay.model_dump()


@pytest.mark.parametrize("owned,operation", [(False, "release"), (False, "stop"), (True, "stop")])
async def test_borrowed_handles_do_not_destroy_owned_collection(owned, operation):
    provider = MagicMock(close=AsyncMock(), aclose=AsyncMock())
    sandbox = AsyncSandbox(provider, owns_sandbox=owned)
    sandbox._handle = SandboxHandle(sandbox_id="box", provider_name="test", raw=None)
    sandbox._stopped = False
    await getattr(sandbox, operation)()
    await sandbox.stop()
    assert provider.close.await_count == int(owned)
    provider.aclose.assert_awaited_once()


async def test_descriptor_timeout_is_owned_and_cleans_partial_preparation(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(lifecycle, "SETUP_TIMEOUT_SEC", 0.01)
    original = lifecycle.Environment

    def create(*args, **kwargs):
        env = original(*args, **kwargs)

        async def blocked():
            await asyncio.Event().wait()

        env.main_connection.side_effect = blocked
        return env

    monkeypatch.setattr(lifecycle, "Environment", create)
    with pytest.raises(HTTPException):
        await seed(f)
    session = next(iter(f.server._sessions.values()))
    await session.finalization
    assert f.envs[0].closed and not session.owns_slot
    assert session.result["exception_info"]["exception_type"] == "AgentSetupTimeoutError"
    assert session.result["agent_setup"]["finished_at"]
    f.grade.assert_not_awaited()


async def test_delayed_start_cannot_extend_setup_or_execution(fixture):
    f = fixture
    session_id = (await seed(f)).session_id
    session = f.server._sessions[session_id]
    session.watchdog.cancel()
    session.setup_deadline = 0
    with pytest.raises(HTTPException):
        await f.server.start_session(f.request, SessionRequest(session_id=session_id))
    assert session.phase == "closed"
    f.body = f.body.model_copy(update={"rollout_id": "second"})
    session_id = await start(f)
    session = f.server._sessions[session_id]
    session.watchdog.cancel()
    session.deadline = 0
    with pytest.raises(HTTPException):
        await f.server.start_session(f.request, SessionRequest(session_id=session_id))
    assert session.termination.reason == "timeout"
    f.grade.assert_awaited_once()


async def test_shutdown_interrupts_verification_and_preserves_cleanup(fixture):
    f = fixture
    session_id = await start(f)
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    f.grade.side_effect = blocked
    verify = asyncio.create_task(f.server.verify(f.request, verify_body(session_id)))
    await entered.wait()
    await lifecycle.shutdown(list(f.server._sessions.values()), 0.01)
    result = await verify
    assert not result.evaluation_completed and result.infrastructure_error == "CancelledError"
    assert all(env.closed for env in f.envs)
    assert not f.server._sessions[session_id].owns_slot


async def test_build_timeout_and_missing_agent_logs_are_diagnostic(fixture, monkeypatch):
    f = fixture
    f.server._loader.load.return_value.config.environment.build_timeout_sec = 0.01
    original = lifecycle.Environment

    def create(*args, **kwargs):
        env = original(*args, **kwargs)

        async def blocked():
            await asyncio.Event().wait()

        env.start.side_effect = blocked
        return env

    monkeypatch.setattr(lifecycle, "Environment", create)
    with pytest.raises(HTTPException):
        await seed(f)
    session = next(iter(f.server._sessions.values()))
    assert session.result["exception_info"]["exception_type"] == "EnvironmentStartTimeoutError"
    monkeypatch.setattr(lifecycle, "Environment", original)
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock(side_effect=RuntimeError("logs unavailable")))
    f.body = f.body.model_copy(update={"rollout_id": "second"})
    session_id = await start(f)
    response = await f.server.verify(f.request, verify_body(session_id))
    assert response.evaluation_completed
    assert any(d.get("operation") == "agent_logs" for d in f.server._sessions[session_id].diagnostics)
