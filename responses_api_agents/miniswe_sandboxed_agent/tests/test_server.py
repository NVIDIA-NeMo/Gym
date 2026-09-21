# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.rollout_collection import _trajectory_identity
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.app import (
    TerminalBench4Config,
    TerminalBench4ResourcesServer,
    TerminalBench4RunRequest,
)
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_environment import environment_config
from responses_api_agents.miniswe_sandboxed_agent import app as module
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome


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
            shutdown_timeout_sec=0.01,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    agent = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=2,
            name="agent",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "tb4"},
            model_server={"type": "responses_api_models", "name": "model"},
            agent_max_timeout_sec=2,
            shutdown_timeout_sec=0.01,
            artifacts_dir=tmp_path / "agent",
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    pin = next(iter(server._tasks.values()))
    body = TerminalBench4RunRequest(
        task_name="terminal-bench/" + pin["name"],
        task_ref=pin["ref"],
        dataset_ref=server._manifest["ref"],
        rollout_id="rollout",
        responses_create_params={"input": []},
    )
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"}, cookies={"session": "incoming"})
    task = SimpleNamespace(
        config=TaskSettings.model_validate(
            {
                "environment": {"docker_image": "agent"},
                "agent": {"timeout_sec": 28800, "user": "task-user"},
                "verifier": {"environment": {"docker_image": "verifier"}},
            }
        ),
        instruction="Solve task",
    )
    server._loader.load = AsyncMock(return_value=task)
    envs, events, harnesses = [], [], []

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
            provider_config={"local": {}},
            main=SimpleNamespace(
                serialize=AsyncMock(return_value={"sandbox_id": session_id}),
                exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="/task\n")),
            ),
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

    def harness(**kwargs):
        async def execute(budget):
            events.append("execute")
            assert 0 < budget <= 2
            response = module.empty_response(kwargs["params"], "model")
            return response, HarnessOutcome(reason="completed"), {"harness_version": "test"}

        instance = SimpleNamespace(
            **kwargs,
            setup=AsyncMock(side_effect=lambda: events.append("setup")),
            execute=AsyncMock(side_effect=execute),
        )
        harnesses.append(instance)
        return instance

    async def connect(descriptor, **kwargs):
        assert callable(kwargs["provider"].aclose)
        return next(e.main for e in envs if e.session_id == descriptor["sandbox_id"])

    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(side_effect=connect))

    async def post(*, url_path, json, cookies, **kwargs):
        # Serialize at each boundary: the agent cannot access the resource's in-memory Session.
        resource_request = SimpleNamespace(
            session={
                SESSION_ID_KEY: "owner",
                "tb4_client_session_id": json.get("client_session_id", cookies.get("owner", "owner")),
            },
            cookies=cookies,
        )
        if url_path == "/seed_session":
            value = await server.seed_session(resource_request, TerminalBench4RunRequest.model_validate(json))
        elif url_path == "/verify":
            value = await server.verify(resource_request, module.SandboxedVerifyRequest.model_validate(json))
        else:
            raise AssertionError(url_path)
        return SimpleNamespace(
            value=value.model_dump(mode="json"),
            cookies={"session": "resource", "owner": resource_request.session["tb4_client_session_id"]},
        )

    agent.server_client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(side_effect=lambda r: r.value))
    monkeypatch.setattr(lifecycle, "Environment", create)
    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock())
    monkeypatch.setattr(lifecycle, "collect", AsyncMock(side_effect=lambda *a: events.append("collect")))
    monkeypatch.setattr(lifecycle, "restore", AsyncMock(side_effect=lambda *a: events.append("restore")))

    async def grade(*args):
        events.append("grade")
        return {"rewards": {"reward": 0.75}}

    grader = AsyncMock(side_effect=grade)
    monkeypatch.setattr(lifecycle, "run_verifier", grader)
    yield SimpleNamespace(
        server=server,
        agent=agent,
        request=request,
        body=body,
        envs=envs,
        grade=grader,
        events=events,
        harnesses=harnesses,
    )
    await agent.shutdown()
    await lifecycle.shutdown(list(server._sessions.values()), 0.01)


async def test_agent_owns_loop_and_replays_exact_result(fixture):
    f = fixture
    result, retry = await asyncio.gather(f.agent.run(f.request, f.body), f.agent.run(f.request, f.body))
    assert result == retry
    assert result.reward == 0.75 and result.evaluation_completed
    assert result.harness_version == "test"
    assert len(f.harnesses) == 1
    assert f.harnesses[0].sandbox is f.envs[0].main
    assert f.harnesses[0].context.workdir == "/task"
    assert f.harnesses[0].context.user == "task-user"
    assert f.body.responses_create_params.input == []
    assert f.events == [
        "agent_start",
        "setup",
        "execute",
        "quiesce",
        "collect",
        "agent_stop",
        "verifier_start",
        "restore",
        "grade",
        "verifier_stop",
    ]
    assert all(env.closed for env in f.envs)
    f.server._loader.load.assert_awaited_once()
    f.server.server_client.post.assert_not_called()
    session = f.server._sessions[result.session_id]
    assert (await f.server.verify(f.request, session.verify_body)).model_dump(mode="json") == result.model_dump(
        mode="json"
    )
    f.server._sessions.clear()
    f.server._by_identity.clear()
    f.agent._runs.clear()
    f.agent._runs.clear()
    assert await f.agent.run(f.request, f.body) == result
    assert len(f.harnesses) == 1


@pytest.mark.parametrize("field", ["task_name", "task_ref", "dataset_ref"])
async def test_bad_pins_rejected_before_allocation(fixture, field):
    f = fixture
    with pytest.raises(HTTPException) as err:
        await f.agent.run(f.request, f.body.model_copy(update={field: "untrusted"}))
    assert err.value.status_code == 422
    assert not f.envs


async def test_owner_isolation_and_conflicting_retry(fixture):
    f = fixture
    result = await f.agent.run(f.request, f.body)
    with pytest.raises(HTTPException) as err:
        await f.agent.run(f.request, f.body.model_copy(update={"artifact_directory": "different"}))
    assert err.value.status_code == 409
    stranger = SimpleNamespace(session={SESSION_ID_KEY: "stranger"}, cookies={})
    with pytest.raises(HTTPException) as err:
        f.server._session(stranger, result.session_id)
    assert err.value.status_code == 404
    other = await f.agent.run(stranger, f.body)
    assert other.session_id != result.session_id


@pytest.mark.parametrize("stage", ["provision", "setup", "execute", "grade"])
async def test_disconnected_http_caller_does_not_interrupt_episode(fixture, monkeypatch, stage):
    f = fixture
    entered, release = asyncio.Event(), asyncio.Event()

    async def block():
        entered.set()
        await release.wait()

    original_env, original_harness = lifecycle.Environment, module.MiniSWEHarness

    def environment(*a, **kw):
        env = original_env(*a, **kw)
        if stage == "provision" and not kw.get("verifier"):
            env.start.side_effect = block
        return env

    def harness(**kw):
        h = original_harness(**kw)
        if stage == "setup":
            h.setup.side_effect = block
        elif stage == "execute":
            execute = h.execute.side_effect

            async def run(budget):
                await block()
                return await execute(budget)

            h.execute.side_effect = run
        return h

    monkeypatch.setattr(lifecycle, "Environment", environment)
    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    if stage == "grade":

        async def grade(*args):
            await block()
            return {"rewards": {"reward": 0}}

        f.grade.side_effect = grade
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not next(iter(f.agent._runs.values()))[1].done()
    release.set()
    result = await f.agent.run(f.request, f.body)
    assert result.evaluation_completed and all(e.closed for e in f.envs)
    assert len(f.harnesses) == 1


@pytest.mark.parametrize("stage", ["provision", "workdir", "setup", "execute", "collect", "restore", "grade"])
async def test_failures_cleanup_and_do_not_grade_failed_setup(fixture, monkeypatch, stage):
    f = fixture
    original_env, original_harness = lifecycle.Environment, module.MiniSWEHarness

    def environment(*a, **kw):
        env = original_env(*a, **kw)
        if not kw.get("verifier"):
            if stage == "provision":
                env.start.side_effect = RuntimeError(stage)
            if stage == "workdir":
                env.main.exec.side_effect = RuntimeError(stage)
        return env

    def harness(**kw):
        h = original_harness(**kw)
        if stage in ("setup", "execute"):
            getattr(h, stage).side_effect = RuntimeError(stage)
        return h

    monkeypatch.setattr(lifecycle, "Environment", environment)
    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    if stage in ("collect", "restore"):
        getattr(lifecycle, stage).side_effect = RuntimeError(stage)
    if stage == "grade":
        f.grade.side_effect = RuntimeError(stage)
    result = await f.agent.run(f.request, f.body)
    assert result.infrastructure_error
    assert all(env.closed for env in f.envs)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions
    assert f.grade.await_count == int(stage in ("execute", "grade"))


@pytest.mark.parametrize("reason", ["completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"])
async def test_agent_outcomes_still_collect_and_grade(fixture, monkeypatch, reason):
    f = fixture
    original = module.MiniSWEHarness

    def harness(**kw):
        h = original(**kw)
        h.execute.side_effect = None
        h.execute.return_value = module.empty_response(kw["params"], "model"), HarnessOutcome(reason=reason), {}
        return h

    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    result = await f.agent.run(f.request, f.body)
    assert result.termination.reason == reason
    assert result.reward == 0.75 and result.evaluation_completed
    assert bool(result.infrastructure_error) == (reason == "infrastructure_error")
    assert all(env.closed for env in f.envs)


@pytest.mark.parametrize("stage", ["setup", "execute", "grade"])
async def test_shutdown_cancels_worker_before_collection_and_cleans_up(fixture, monkeypatch, stage):
    f = fixture
    entered = asyncio.Event()

    async def block(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            f.events.append("worker_joined")

    original = module.MiniSWEHarness

    def harness(**kw):
        h = original(**kw)
        if stage != "grade":
            getattr(h, stage).side_effect = block
        return h

    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    if stage == "grade":
        f.grade.side_effect = block
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    await entered.wait()
    await f.agent.shutdown()
    await lifecycle.shutdown(list(f.server._sessions.values()), 0.01)
    result = (await asyncio.gather(caller, return_exceptions=True))[0]
    if isinstance(result, asyncio.CancelledError):
        result = next(iter(f.server._sessions.values())).verified_response
    assert all(env.closed for env in f.envs)
    if stage == "setup":
        assert not result.evaluation_completed
        f.grade.assert_not_awaited()
    elif stage == "execute":
        assert f.events.index("worker_joined") < f.events.index("collect")
        assert result.evaluation_completed
    else:
        assert not result.evaluation_completed


async def test_setup_budget_covers_workdir_and_never_grades(fixture, monkeypatch):
    f = fixture
    f.agent.config.setup_timeout_sec = 0.01
    original = lifecycle.Environment

    async def block(*args, **kwargs):
        await asyncio.Event().wait()

    def environment(*a, **kw):
        env = original(*a, **kw)
        env.main.exec.side_effect = block
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    result = await f.agent.run(f.request, f.body)
    assert result.termination.reason == "timeout"
    assert result.infrastructure_error == "AgentSetupTimeoutError"
    f.grade.assert_not_awaited()
    assert all(env.closed for env in f.envs)


async def test_live_model_callback_preserves_cookies_and_capture_route(fixture, monkeypatch):
    f = fixture
    original = module.MiniSWEHarness
    model_response = module.empty_response(f.body.responses_create_params, "model")
    post = f.agent.server_client.post.side_effect

    async def model_post(**kwargs):
        if kwargs["url_path"].endswith("/v1/responses"):
            return SimpleNamespace(value=model_response.model_dump())
        return await post(**kwargs)

    f.agent.server_client.post.side_effect = model_post
    monkeypatch.setattr(
        module.MiniSWESandboxedAgent, "rollout_id_from_run", lambda self, body: body.capture_rollout_id
    )
    monkeypatch.setattr(module.MiniSWESandboxedAgent, "_token_id_capture_enabled", lambda self: True)

    def harness(**kw):
        h = original(**kw)

        async def execute(budget):
            response = await kw["query"]({"input": "task"})
            return response, HarnessOutcome(reason="completed"), {}

        h.execute.side_effect = execute
        return h

    monkeypatch.setattr(module, "MiniSWEHarness", harness)
    body = f.body.model_copy(
        update={"capture_rollout_id": "rollout-1", "capture_model_calls": True, "capture_token_ids": True}
    )
    result = await f.agent.run(f.request, body)
    call = next(
        c.kwargs for c in f.agent.server_client.post.await_args_list if c.kwargs["url_path"].endswith("/v1/responses")
    )
    assert call["url_path"] == "/ng-rollout/rollout-1/training-token-capture/v1/responses"
    assert call["cookies"] == {"session": "resource", "owner": "owner"}
    assert call["headers"] == {"x-session-id": result.session_id}
    assert result.response == model_response
    f.server._sessions.clear()
    f.server._by_identity.clear()
    f.agent._runs.clear()
    assert await f.agent.run(f.request, body) == result


async def test_active_record_cannot_resume_after_process_restart(fixture):
    f = fixture
    result = await f.agent.run(f.request, f.body)
    session = f.server._sessions[result.session_id]
    path = f.server._state_path(session.identity)
    state = json.loads(path.read_text())
    state["phase"] = "agent_running"
    path.write_text(json.dumps(state))
    f.server._sessions.clear()
    f.server._by_identity.clear()
    f.agent._runs.clear()
    with pytest.raises(HTTPException, match="cannot resume"):
        await f.agent.run(f.request, f.body)


@pytest.mark.parametrize("failure", [None, "start", "prepare", "cleanup", "unsupported", "workload_cleanup"])
async def test_shared_logs_stay_owned_until_workload_cleanup(fixture, monkeypatch, failure):
    f = fixture
    f.server.config.environment.efs_logs_host_path = "/mnt/efs/data/shared"
    logs = SimpleNamespace(
        session_id="logs",
        closed=False,
        resources=[],
        cleanup_errors=[],
        resource_identities=lambda: [{"efs_subpath": "owned"}],
        restored_archive="/logs/snapshot",
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
            raise RuntimeError("snapshot unavailable")

    async def close(*, remove_data=True):
        f.events.append("logs_stop")
        assert remove_data == (failure != "workload_cleanup")
        if failure == "cleanup":
            raise RuntimeError("EFS cleanup failed")
        logs.closed = True

    logs.start = AsyncMock(side_effect=initialize)
    logs.prepare_verifier = AsyncMock(side_effect=prepare)
    logs.stop = AsyncMock(side_effect=close)
    monkeypatch.setattr(lifecycle, "SharedLogs", lambda env: logs)
    original = lifecycle.Environment

    def environment(*a, **kw):
        env = original(*a, **kw)
        if failure == "workload_cleanup" and not kw.get("verifier"):
            env.stop.side_effect = RuntimeError("delete failed")
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    result = await f.agent.run(f.request, f.body)
    if failure == "start":
        assert not result.evaluation_completed
        f.grade.assert_not_awaited()
    else:
        assert result.evaluation_completed
        if failure == "unsupported":
            assert all(env.shared_logs is None for env in f.envs)
        if failure != "workload_cleanup":
            assert f.events.index("agent_stop") < f.events.index("logs_prepare") < f.events.index("verifier_start")
    assert f.events[-1] == "logs_stop"
    assert f.server._slots._value == f.server.config.max_concurrent_sessions


@pytest.mark.parametrize("role", ["agent", "verifier"])
async def test_mount_fallback_is_recorded_for_either_role(fixture, monkeypatch, role):
    f = fixture
    original = lifecycle.Environment

    def environment(*a, **kw):
        env = original(*a, **kw)
        if kw.get("verifier", False) == (role == "verifier"):
            env.efs_logs_fallback = "unsupported host mount"
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    result = await f.agent.run(f.request, f.body)
    assert result.evaluation_completed
    session = f.server._sessions[result.session_id]
    assert {"operation": "efs_logs_fallback", "role": role, "error": "unsupported host mount"} in session.diagnostics


async def test_build_timeout_releases_slot_without_grading(fixture, monkeypatch):
    f = fixture
    f.server._loader.load.return_value.config.environment.build_timeout_sec = 0.01
    original = lifecycle.Environment

    async def block(*args, **kwargs):
        await asyncio.Event().wait()

    def environment(*a, **kw):
        env = original(*a, **kw)
        env.start.side_effect = block
        return env

    monkeypatch.setattr(lifecycle, "Environment", environment)
    result = await f.agent.run(f.request, f.body)
    assert result.infrastructure_error == "EnvironmentStartTimeoutError"
    assert all(env.closed for env in f.envs)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions
    f.grade.assert_not_awaited()


async def test_optional_agent_logs_do_not_discard_official_grade(fixture, monkeypatch):
    monkeypatch.setattr(lifecycle, "download_dir", AsyncMock(side_effect=FileNotFoundError("no agent logs")))
    f = fixture
    result = await f.agent.run(f.request, f.body)
    assert result.reward == 0.75
    assert any(d.get("operation") == "agent_logs" for d in f.server._sessions[result.session_id].diagnostics)


async def test_resource_routes_only_seed_and_verify(fixture):
    paths = {route.path for route in fixture.server.setup_webserver().routes}
    assert {"/seed_session", "/verify"} <= paths
    assert "/run" not in paths
    assert "/cancel_session" not in paths


@pytest.mark.parametrize("fault", ["owner", "version", "identity", "no_response"])
async def test_invalid_restart_records_cannot_be_replayed(fixture, fault):
    f = fixture
    result = await f.agent.run(f.request, f.body)
    session = f.server._sessions[result.session_id]
    path = f.server._state_path(session.identity)
    state = json.loads(path.read_text())
    if fault == "owner":
        state["owner"] = "another-owner"
    elif fault == "version":
        state["record_version"] = 1
    elif fault == "identity":
        (f.server.config.artifacts_dir / (session.session_id + ".state")).write_text("../bad")
    else:
        state["verified_response"] = None
    path.write_text(json.dumps(state))
    f.server._sessions.clear()
    f.server._by_identity.clear()
    f.agent._runs.clear()
    with pytest.raises(HTTPException):
        await f.agent.run(f.request, f.body)
    assert len(f.harnesses) == 1


async def test_task_identity_uses_name_instead_of_collector_index(fixture):
    f = fixture
    f.body = f.body.model_copy(update={"_ng_task_index": 25, "_ng_rollout_index": 0})
    result = await f.agent.run(f.request, f.body)
    expected = f.body.task_name
    assert expected.startswith("terminal-bench/")
    assert result.task_id == expected
    assert f.harnesses[0].context.task_id == expected
    # The response remains usable by the collector and by offline smoke tooling.
    record = result.model_dump(mode="json") | {"_ng_task_index": 25, "_ng_rollout_index": 0}
    assert _trajectory_identity(record)[0] == expected
    assert record["_ng_task_index"] == 25


@pytest.mark.parametrize("global_config", [{}, {"observability_enabled": False}, {"observability_enabled": True}])
async def test_harness_observability_follows_global_opt_in(fixture, global_config):
    f = fixture
    f.agent.server_client.global_config_dict = global_config
    await f.agent.run(f.request, f.body)
    assert f.harnesses[0].observability_enabled is global_config.get("observability_enabled", False)


async def test_seed_and_verify_retries_share_resource_work(fixture):
    f = fixture
    first, retry = await asyncio.gather(
        f.server.seed_session(f.request, f.body), f.server.seed_session(f.request, f.body)
    )
    assert first == retry
    assert first.instruction == "Solve task"
    assert first.user == "task-user"
    assert first.agent_timeout_sec == 28800
    assert first.sandbox_descriptor == {"sandbox_id": first.session_id}
    assert first.sandbox_provider == {"local": {}}
    assert f.events == ["agent_start"]
    assert not f.harnesses
    verify = module.SandboxedVerifyRequest(
        session_id=first.session_id,
        responses_create_params=f.body.responses_create_params,
        response=module.empty_response(f.body.responses_create_params, "model"),
        termination={"reason": "completed"},
        agent_started=True,
    )
    result, retry = await asyncio.gather(f.server.verify(f.request, verify), f.server.verify(f.request, verify))
    assert result == retry
    assert result.reward == 0.75
    f.grade.assert_awaited_once()
    with pytest.raises(HTTPException) as exc:
        await f.server.verify(f.request, verify.model_copy(update={"agent_started": False}))
    assert exc.value.status_code == 409
    assert all(e.closed for e in f.envs)


async def test_resource_shutdown_cleans_seeded_session_without_agent(fixture):
    f = fixture
    await f.server.seed_session(f.request, f.body)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions - 1
    await lifecycle.shutdown(list(f.server._sessions.values()), 0.01)
    assert all(e.closed for e in f.envs)
    assert f.server._slots._value == f.server.config.max_concurrent_sessions
    f.grade.assert_not_awaited()


async def test_reconnect_failure_still_requests_cleanup(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(module.AsyncSandbox, "connect", AsyncMock(side_effect=RuntimeError("reconnect failed")))
    result = await f.agent.run(f.request, f.body)
    assert result.termination.reason == "infrastructure_error"
    assert "reconnect failed" in result.termination.detail
    assert not result.evaluation_completed
    assert not f.harnesses
    assert all(e.closed for e in f.envs)
    f.grade.assert_not_awaited()
    assert [call.kwargs["url_path"] for call in f.agent.server_client.post.await_args_list] == [
        "/seed_session",
        "/verify",
    ]


async def test_abandoned_seed_expires_and_releases_slot(fixture):
    f = fixture
    assert f.server.config.seeded_session_timeout_sec == 10 * 60 * 60
    f.server.config.seeded_session_timeout_sec = 0.05
    f.server._slots = asyncio.Semaphore(1)
    seed = await f.server.seed_session(f.request, f.body)
    session = f.server._sessions[seed.session_id]
    deadline = session.agent_deadline
    retry = await f.server.seed_session(f.request, f.body)
    assert retry == seed and session.agent_deadline == deadline
    assert f.server._slots.locked()
    async with asyncio.timeout(1):
        await session.expiry_task
        await session.finalization
        await f.server._slots.acquire()
    f.server._slots.release()
    assert session.phase == "closed"
    assert session.termination.reason == "timeout"
    assert all(e.closed for e in f.envs)
    assert f.events.index("quiesce") < f.events.index("agent_stop")
    assert all(e.stop.await_count == 1 for e in f.envs)
    assert session.result["exception_info"]["exception_type"] == "SeededSessionExpired"
    state = json.loads(f.server._state_path(session.identity).read_text())
    assert state["deadlines"]["agent_expired_at"]
    assert state["deadlines"]["agent_expires_at"]
    f.grade.assert_not_awaited()
    for restarted in (False, True):
        if restarted:
            f.server._sessions.clear()
            f.server._by_identity.clear()
        with pytest.raises(HTTPException) as exc:
            await f.server.seed_session(f.request, f.body)
        assert exc.value.status_code == 410


async def test_verify_takes_over_before_seed_deadline(fixture):
    f = fixture
    f.server.config.seeded_session_timeout_sec = 0.1
    seed = await f.server.seed_session(f.request, f.body)
    session = f.server._sessions[seed.session_id]
    deadline = session.agent_deadline
    entered, release = asyncio.Event(), asyncio.Event()

    async def grade(*args):
        entered.set()
        await release.wait()
        return {"rewards": {"reward": 0.75}}

    f.grade.side_effect = grade
    body = module.SandboxedVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=f.body.responses_create_params,
        response=module.empty_response(f.body.responses_create_params, "model"),
        termination={"reason": "completed"},
        agent_started=True,
    )
    caller = asyncio.create_task(f.server.verify(f.request, body))
    try:
        async with asyncio.timeout(1):
            await entered.wait()
        await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()) + 0.01)
        assert session.expiry_task.cancelled()
        assert session.agent_deadline is None
        assert "agent_expired_at" not in session.deadlines
        assert session.deadlines["verification_started_at"]
        assert not f.envs[1].closed
        assert session.termination.reason == "completed"
    finally:
        release.set()
    result = await caller
    assert result.reward == 0.75 and result.evaluation_completed
    assert all(e.closed for e in f.envs)


async def test_late_verify_cannot_race_expiry_cleanup(fixture):
    f = fixture
    seed = await f.server.seed_session(f.request, f.body)
    session = f.server._sessions[seed.session_id]
    # Simulate /verify winning scheduling ahead of an overdue timer callback.
    session.agent_deadline = asyncio.get_running_loop().time() - 1
    body = module.SandboxedVerifyRequest(
        session_id=seed.session_id,
        responses_create_params=f.body.responses_create_params,
        response=module.empty_response(f.body.responses_create_params, "model"),
        termination={"reason": "completed"},
        agent_started=True,
    )
    with pytest.raises(HTTPException) as exc:
        await f.server.verify(f.request, body)
    assert exc.value.status_code == 410
    await session.finalization
    with pytest.raises(HTTPException) as exc:
        await f.server.verify(f.request, body)
    assert exc.value.status_code == 410
    assert all(e.stop.await_count == 1 for e in f.envs)
    f.grade.assert_not_awaited()


@pytest.mark.parametrize("shutdown_timeout", [0, 0.02])
@pytest.mark.parametrize("blocked_stage", ["preparation", "response"])
async def test_shutdown_is_bounded_when_seed_response_never_arrives(fixture, shutdown_timeout, blocked_stage):
    f = fixture
    f.agent.config.shutdown_timeout_sec = shutdown_timeout
    f.server.config.seeded_session_timeout_sec = 0.1
    entered, transport_cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    post = f.agent.server_client.post.side_effect
    if blocked_stage == "preparation":
        task = f.server._loader.load.return_value

        async def blocked_load(*args):
            entered.set()
            await release.wait()
            return task

        f.server._loader.load.side_effect = blocked_load

    async def lose_seed_response(**kwargs):
        try:
            response = await post(**kwargs)
            if kwargs["url_path"] == "/seed_session" and blocked_stage == "response":
                entered.set()
                await asyncio.Event().wait()
            return response
        except asyncio.CancelledError:
            transport_cancelled.set()
            raise

    f.agent.server_client.post.side_effect = lose_seed_response
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    async with asyncio.timeout(1):
        await entered.wait()
        await f.agent.shutdown()
        await transport_cancelled.wait()
        with pytest.raises(asyncio.CancelledError):
            await caller
    assert not f.harnesses
    assert [c.kwargs["url_path"] for c in f.agent.server_client.post.await_args_list] == ["/seed_session"]
    session = next(iter(f.server._sessions.values()))
    release.set()
    async with asyncio.timeout(1):
        await session.execution
        await session.expiry_task
        await session.finalization
    assert session.phase == "closed" and not session.owns_slot
    assert all(e.closed for e in f.envs)
    f.grade.assert_not_awaited()


async def test_shutdown_finishes_seeding_and_requests_cleanup_within_budget(fixture):
    f = fixture
    f.agent.config.shutdown_timeout_sec = 0.5
    entered, release = asyncio.Event(), asyncio.Event()
    post = f.agent.server_client.post.side_effect

    async def delay_seed(**kwargs):
        response = await post(**kwargs)
        if kwargs["url_path"] == "/seed_session":
            entered.set()
            await release.wait()
        return response

    f.agent.server_client.post.side_effect = delay_seed
    caller = asyncio.create_task(f.agent.run(f.request, f.body))
    async with asyncio.timeout(1):
        await entered.wait()
        shutdown = asyncio.create_task(f.agent.shutdown())
        await asyncio.sleep(0)
        release.set()
        await shutdown
        result = await caller
    assert result.termination.reason == "cancelled"
    assert not f.harnesses
    assert all(e.closed for e in f.envs)
    assert [c.kwargs["url_path"] for c in f.agent.server_client.post.await_args_list] == ["/seed_session", "/verify"]
    session = next(iter(f.server._sessions.values()))
    assert session.expiry_task.cancelled()
    f.grade.assert_not_awaited()
