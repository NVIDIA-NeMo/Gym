# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

import resources_servers.deepswe_external1.app as module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.deepswe.app import AgentSandboxSession, VerifierResult
from resources_servers.deepswe.validate_golden import _empty_response
from resources_servers.deepswe_external1.app import (
    DeepsweExternal1ResourcesServer,
    DeepsweExternal1ResourcesServerConfig,
    DeepsweExternal1SeedSessionRequest,
    DeepsweExternal1VerifyRequest,
)
from resources_servers.deepswe_external1.prepare_examples import task_row
from resources_servers.deepswe_external1.task_store import PreparedTask


def make_server(task: PreparedTask, *, mode: str = "golden") -> DeepsweExternal1ResourcesServer:
    config = DeepsweExternal1ResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="test",
        tasks_dir=task.task_dir.parent,
        expected_task_count=1,
        is_verifying_golden_patch=mode == "golden",
        is_verifying_null_patch=mode == "null",
        sandbox_provider="sandbox",
        sandbox_config={},
        logs_dir=task.task_dir.parent.parent / "logs",
    )
    return DeepsweExternal1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def body(task: PreparedTask) -> DeepsweExternal1VerifyRequest:
    return DeepsweExternal1VerifyRequest(**task_row(task), response=_empty_response())


def request() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "session"}})


def sandbox(sandbox_id: str, events: list[str]) -> AsyncMock:
    box = AsyncMock()
    box.serialize.return_value = {"sandbox_id": sandbox_id}

    async def stop() -> None:
        events.append("stop-" + sandbox_id)

    box.stop.side_effect = stop
    return box


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["golden", "null", "agent"])
async def test_entire_lifecycle_uses_distinct_sandboxes(task: PreparedTask, mode: str) -> None:
    server = make_server(task, mode=mode)
    events = []
    agent, verifier = sandbox("A", events), sandbox("B", events)

    async def create(_task: PreparedTask, *, phase: str) -> AsyncMock:
        events.append("create-" + phase)
        return agent if phase == "agent" else verifier

    server._create_sandbox = AsyncMock(side_effect=create)

    async def golden(*args) -> None:
        events.append("golden-A")

    server._execute_golden = AsyncMock(side_effect=golden)

    async def collect(*args) -> bytes:
        events.append("collect-A")
        return b"" if mode == "null" else b"committed patch"

    server._collect_model_patch = AsyncMock(side_effect=collect)

    async def grade(*args) -> VerifierResult:
        events.append("grade-B")
        assert args[2] == (b"" if mode == "null" else b"committed patch")
        return VerifierResult(evaluation_completed=True, reward=float(mode != "null"))

    server._run_verifier = AsyncMock(side_effect=grade)
    if mode == "agent":
        server._agent_sessions["session"] = AgentSandboxSession(
            task_id=task.definition.task_id,
            image=task.definition.image,
            sandbox=agent,
            sandbox_handle="A",
            sandbox_descriptor={"sandbox_id": "A"},
        )
    result = await server.verify(request(), body(task))
    assert result.evaluation_completed and not result.mask_sample
    assert result.agent_sandbox_id == "A" and result.verifier_sandbox_id == "B"
    assert result.validation_mode == mode and result.cleanup_errors == []
    assert events.index("collect-A") < events.index("stop-A") < events.index("create-verifier")
    assert events[-2:] == ["grade-B", "stop-B"]
    assert server._execute_golden.await_count == (mode == "golden")
    assert not server._agent_sessions
    assert Path(result.log_dir, "result.json").is_file()
    assert Path(result.log_dir, "model.patch").read_bytes() == (b"" if mode == "null" else b"committed patch")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["golden", "collection", "empty", "verifier_setup", "verifier", "same_id", "agent_cleanup"]
)
async def test_failures_are_masked_and_owned_sandboxes_are_released(task: PreparedTask, failure: str) -> None:
    server = make_server(task)
    events = []
    agent = sandbox("A", events)
    verifier = sandbox("A" if failure == "same_id" else "B", events)
    server._create_sandbox = AsyncMock(
        side_effect=[agent, RuntimeError("setup") if failure == "verifier_setup" else verifier]
    )
    server._execute_golden = AsyncMock(side_effect=RuntimeError("solution") if failure == "golden" else None)
    server._collect_model_patch = AsyncMock(
        return_value=b"" if failure == "empty" else b"patch",
        side_effect=RuntimeError("capture") if failure == "collection" else None,
    )
    server._run_verifier = AsyncMock(side_effect=RuntimeError("grader"))
    if failure == "agent_cleanup":
        agent.stop.side_effect = RuntimeError("delete unavailable")
    result = await server.verify(request(), body(task))
    assert not result.evaluation_completed and result.mask_sample and result.reward == 0
    assert result.failure_kind == "verifier_error" and result.verifier_error
    assert result.failure_stage
    agent.stop.assert_awaited_once()
    if failure in {"verifier", "same_id"}:
        verifier.stop.assert_awaited_once()
    else:
        verifier.stop.assert_not_awaited()
    if failure == "agent_cleanup":
        assert result.cleanup_errors == ["agent"]
        assert server._create_sandbox.await_count == 1


@pytest.mark.asyncio
async def test_invalid_task_closes_seeded_session(task: PreparedTask) -> None:
    server = make_server(task, mode="agent")
    agent = sandbox("A", [])
    server._agent_sessions["session"] = AgentSandboxSession(
        task.definition.task_id, task.definition.image, agent, "A", {}
    )
    changed = body(task).model_copy(update={"task_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        await server.verify(request(), changed)
    agent.stop.assert_awaited_once()
    assert not server._agent_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["absent", "task", "image", "handle"])
async def test_agent_requires_matching_seeded_session(task: PreparedTask, mismatch: str) -> None:
    server = make_server(task, mode="agent")
    agent = sandbox("A", [])
    if mismatch != "absent":
        server._agent_sessions["session"] = AgentSandboxSession(
            "wrong" if mismatch == "task" else task.definition.task_id,
            "wrong" if mismatch == "image" else task.definition.image,
            agent,
            "A",
            {},
        )
    data = body(task)
    if mismatch == "handle":
        data.sandbox_handle = "wrong"
    result = await server.verify(request(), data)
    assert result.mask_sample and not result.evaluation_completed
    assert agent.stop.await_count == (mismatch != "absent")


@pytest.mark.asyncio
async def test_shutdown_releases_unfinished_sessions(task: PreparedTask) -> None:
    server = make_server(task, mode="agent")
    agent = sandbox("A", [])
    server._agent_sessions["session"] = AgentSandboxSession(
        task.definition.task_id, task.definition.image, agent, "A", {}
    )
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        agent.stop.assert_not_awaited()
    agent.stop.assert_awaited_once()
    assert not server._agent_sessions


def test_network_and_validation_modes(task: PreparedTask) -> None:
    server = make_server(task)
    assert server._provider_options(phase="agent")["network_policy"] == {"defaultAction": "deny", "egress": []}
    assert server._provider_options(phase="verifier")["network_policy"] == {"defaultAction": "deny", "egress": []}
    server.config.enforce_verifier_no_network = False
    assert "network_policy" not in server._provider_options(phase="verifier")
    settings = server.config.model_dump() | {"is_verifying_null_patch": True}
    with pytest.raises(ValueError, match="mutually exclusive"):
        DeepsweExternal1ResourcesServerConfig(**settings)


@pytest.mark.asyncio
async def test_null_mode_cannot_seed(task: PreparedTask) -> None:
    server = make_server(task, mode="null")
    with pytest.raises(RuntimeError, match="null-validation"):
        await server.seed_session(request(), DeepsweExternal1SeedSessionRequest(**task_row(task)))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["agent", "verifier"])
@pytest.mark.parametrize("setup_fails", [False, True])
async def test_sandbox_spec_and_native_setup(
    task: PreparedTask, monkeypatch: pytest.MonkeyPatch, phase: str, setup_fails: bool
) -> None:
    server = make_server(task)
    server.config.task_cpu_multiplier = 2
    server.config.task_memory_multiplier = 1.5
    server.config.sandbox_config = {"env": {"EXPLICIT": "yes"}, "ttl_s": 3600, "ready_timeout_s": 60}
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=int(setup_fails), stderr="setup diagnostic")
    specs = []

    async def start(spec, setup) -> None:
        specs.append(spec)
        await setup(box)

    box.start_with_setup.side_effect = start
    constructor = MagicMock(return_value=box)
    monkeypatch.setattr(module, "AsyncSandbox", constructor)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda name, config: {"provider": name})
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda name, config: {"owner": "test"})
    if setup_fails:
        with pytest.raises(RuntimeError, match=f"{phase} image setup failed"):
            await server._create_sandbox(task, phase=phase)
    else:
        assert await server._create_sandbox(task, phase=phase) is box
    constructor.assert_called_once_with({"provider": "sandbox"})
    spec = specs[0]
    assert spec.image == (task.definition.image if phase == "agent" else task.definition.verifier_image)
    assert spec.workdir == "/app" and spec.files == {}
    assert spec.resources.cpu == 2 and spec.resources.memory_mib == 1536
    assert spec.ttl_s == 3600 and spec.ready_timeout_s == 60
    assert spec.env == {"EXPLICIT": "yes"}
    assert spec.metadata == {
        "owner": "test",
        "task": task.definition.task_id,
        "phase": phase,
        "nemo_gym_agent": "test",
    }
    command = box.exec.await_args.args[0]
    assert "git rev-parse --show-toplevel" in command and task.definition.base_commit in command
    assert ("user.email" in command) == (phase == "agent")
    assert ("command -v python3" in command) == (phase == "verifier")
    assert box.exec.await_args.kwargs == {"timeout_s": 60}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "mkdir", "solution"])
async def test_golden_executes_original_script_and_records_exit(
    task: PreparedTask, tmp_path: Path, failure: str | None
) -> None:
    server = make_server(task)
    box = AsyncMock()
    box.exec.side_effect = [
        SimpleNamespace(return_code=int(failure == "mkdir"), stdout="", stderr=""),
        SimpleNamespace(return_code=2 if failure == "solution" else 0, stdout="solution output", stderr="diagnostic"),
    ]
    logs = tmp_path / "golden"
    if failure:
        with pytest.raises(RuntimeError, match="solution directory" if failure == "mkdir" else "exit code 2"):
            await server._execute_golden(box, task, logs)
    else:
        await server._execute_golden(box, task, logs)
    if failure == "mkdir":
        box.upload.assert_not_awaited()
        assert not logs.exists()
    else:
        assert box.upload.await_count == 2
        assert [call.args[1] for call in box.upload.await_args_list] == [
            "/solution/solve.sh",
            "/solution/solution.patch",
        ]
        box.exec.assert_awaited_with(
            "bash /solution/solve.sh", cwd="/app", timeout_s=task.definition.solution_timeout_sec
        )
        assert (logs / "golden.log").read_text() == "solution outputdiagnostic"


@pytest.mark.asyncio
async def test_seed_session_preserves_descriptor(task: PreparedTask) -> None:
    server = make_server(task, mode="agent")
    box = sandbox("A", [])
    box.serialize.return_value = {"sandbox_id": "A", "workdir": "/app"}
    server._create_sandbox = AsyncMock(return_value=box)
    seeded = await server.seed_session(request(), DeepsweExternal1SeedSessionRequest(**task_row(task)))
    assert seeded.sandbox_descriptor == {"sandbox_id": "A", "workdir": "/app"}
    assert server._agent_sessions["session"].sandbox is box


@pytest.mark.asyncio
async def test_verifier_cleanup_error_does_not_replace_completed_grade(task: PreparedTask) -> None:
    server = make_server(task)
    agent, verifier = sandbox("A", []), sandbox("B", [])
    verifier.stop.side_effect = RuntimeError("delete unavailable")
    server._create_sandbox = AsyncMock(side_effect=[agent, verifier])
    server._execute_golden = AsyncMock()
    server._collect_model_patch = AsyncMock(return_value=b"patch")
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1))
    result = await server.verify(request(), body(task))
    assert result.evaluation_completed and result.reward == 1 and not result.mask_sample
    assert result.cleanup_errors == ["verifier"]
