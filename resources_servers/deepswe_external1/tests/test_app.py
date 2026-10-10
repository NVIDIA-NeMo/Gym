# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

import resources_servers.deepswe_external1.app as module
from nemo_gym.reward_profile import select_measured
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.deepswe.app import AgentSandboxSession, VerifierResult
from resources_servers.deepswe.validate_golden import _empty_response
from resources_servers.deepswe_external1.app import (
    DeepsweExternal1ResourcesServer,
    DeepsweExternal1ResourcesServerConfig,
    DeepsweExternal1SeedSessionRequest,
    DeepsweExternal1VerifyRequest,
)
from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.prepare_examples import task_row


def make_server(task: InlineTask, *, mode: str = "golden") -> DeepsweExternal1ResourcesServer:
    config = DeepsweExternal1ResourcesServerConfig(
        host="127.0.0.1",
        port=8000,
        entrypoint="app.py",
        name="test",
        is_verifying_golden_patch=mode == "golden",
        is_verifying_null_patch=mode == "null",
        sandbox_provider="sandbox",
        sandbox_config={},
        logs_dir=Path.cwd() / "logs",
    )
    return DeepsweExternal1ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def body(task: InlineTask) -> DeepsweExternal1VerifyRequest:
    return DeepsweExternal1VerifyRequest(**task_row(task.data, "original instruction\n"), response=_empty_response())


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
@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for collection")
@pytest.mark.parametrize("failure", ["git", "head", "provider", "exec_error", "missing_patch", "collect_error"])
async def test_collection_distinguishes_invalid_submission_from_infrastructure(
    task: InlineTask, tmp_path: Path, failure: str
) -> None:
    repo, artifacts = tmp_path / "repo", tmp_path / "artifacts"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        check=True,
    )
    base = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    task.data.base_commit = base
    task = InlineTask(task.data)
    if failure == "git":
        shutil.rmtree(repo / ".git")
    elif failure == "head":
        (repo / ".git/HEAD").write_text("ref: refs/heads/missing\n")

    integrity_results: list[SandboxExecResult] = []

    async def execute(command: str, *, timeout_s: float) -> SandboxExecResult:
        if failure == "provider":
            raise ConnectionError("sandbox transport unavailable")
        if failure == "exec_error":
            return SandboxExecResult("", "provider unavailable", 125, error_type="ProviderError")
        if failure == "collect_error" and command == task.config.verifier.collect[0].command:
            return SandboxExecResult("", "simulated collect failure", 1)
        checking_integrity = command.startswith('test "$(readlink -f "$(git -C /app rev-parse --show-toplevel)")"')
        command = command.replace("/app", shlex.quote(str(repo))).replace(
            "/logs/artifacts", shlex.quote(str(artifacts))
        )
        completed = subprocess.run(
            ["sh", "-c", command],
            env=os.environ | {"HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
        result = SandboxExecResult(completed.stdout, completed.stderr, completed.returncode)
        if checking_integrity:
            integrity_results.append(result)
        return result

    async def download(remote: str, local: Path) -> None:
        if failure == "missing_patch":
            raise FileNotFoundError(remote)
        shutil.copyfile(artifacts / Path(remote).name, local)

    agent = sandbox("A", [])
    agent.exec.side_effect = execute
    agent.download.side_effect = download
    server = make_server(task, mode="agent")
    server._agent_sessions["session"] = AgentSandboxSession(task.data.task_id, task.data.image, agent, "A", {})
    server._create_sandbox = AsyncMock(side_effect=AssertionError("No verifier for failed collection"))
    result = await server.verify(request(), body(task))

    invalid = failure in {"git", "head"}
    assert result.reward == 0 and not result.evaluation_completed
    assert result.mask_sample is not invalid
    assert result.failure_kind == ("deepswe_external1:invalid_submission" if invalid else "verifier_error")
    assert result.failure_stage == "patch_collection"
    _, measured, masked, _ = select_measured([{}], [result.model_dump()])
    assert [row["reward"] for row in measured] == ([0] if invalid else [])
    assert len(masked) == (0 if invalid else 1)
    if failure == "collect_error":
        assert len(integrity_results) == 1 and integrity_results[0].return_code == 0
        assert not integrity_results[0].error_type
        assert (
            result.failure_reason == "RuntimeError: DeepSWE collect hook exited with code 1: simulated collect failure"
        )
    agent.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("clear", [False, True])
async def test_log_retention_preserves_response_but_can_remove_all_attempt_files(
    task: InlineTask, failed: bool, clear: bool
) -> None:
    server = make_server(task, mode="null")
    server.config = type(server.config).model_validate(server.config.model_dump() | {"clear_verifier_logs": clear})
    server._create_sandbox = AsyncMock(side_effect=[sandbox("A", []), sandbox("B", [])])
    server._collect_model_patch = AsyncMock(return_value=b"")

    async def grade(box, inline_task, patch, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "run.log").write_text("verifier output")
        if failed:
            raise RuntimeError("provider failed while grading")
        return VerifierResult(evaluation_completed=True, reward=1)

    server._run_verifier = AsyncMock(side_effect=grade)
    result = await server.verify(request(), body(task))
    assert result.reward == (0 if failed else 1)
    assert result.mask_sample is failed
    if clear:
        assert result.log_dir == ""
        assert not list((server.config.logs_dir / task.data.task_id).iterdir())
    else:
        assert Path(result.log_dir, "result.json").is_file()
        assert Path(result.log_dir, "run.log").read_text() == "verifier output"


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for setup")
@pytest.mark.parametrize("damaged_head", [False, True])
async def test_seed_rejects_preexisting_invalid_head(
    task: InlineTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damaged_head: bool
) -> None:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        check=True,
    )
    task.data.base_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if damaged_head:
        (repo / ".git/HEAD").write_text("ref: refs/heads/missing\n")
    box = sandbox("A", [])

    async def execute(command: str, *, timeout_s: float) -> SandboxExecResult:
        completed = subprocess.run(
            ["sh", "-c", command.replace("/app", shlex.quote(str(repo)))],
            env=os.environ | {"HOME": str(tmp_path)},
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return SandboxExecResult(completed.stdout, completed.stderr, completed.returncode)

    async def start(spec, setup) -> None:
        await setup(box)

    box.exec.side_effect = execute
    box.start_with_setup.side_effect = start
    monkeypatch.setattr(module, "AsyncSandbox", MagicMock(return_value=box))
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda name, config: {})
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda name, config: {})
    server = make_server(task, mode="agent")
    seed = DeepsweExternal1SeedSessionRequest(**task_row(task.data, "original instruction\n"))
    if damaged_head:
        with pytest.raises(RuntimeError, match="agent image setup failed"):
            await server.seed_session(request(), seed)
        assert not server._agent_sessions
    else:
        await server.seed_session(request(), seed)
        assert "session" in server._agent_sessions


@pytest.mark.asyncio
async def test_log_cleanup_failure_is_reported_without_losing_completed_grade(
    task: InlineTask, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = make_server(task, mode="null")
    server.config.clear_verifier_logs = True
    server._create_sandbox = AsyncMock(side_effect=[sandbox("A", []), sandbox("B", [])])
    server._collect_model_patch = AsyncMock(return_value=b"candidate")
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1))

    def cannot_remove(path, **kwargs):
        raise PermissionError("attempt directory is not writable")

    monkeypatch.setattr(module, "rmtree", cannot_remove)
    result = await server.verify(request(), body(task))
    assert result.evaluation_completed and result.reward == 1 and not result.mask_sample
    assert result.cleanup_errors == ["verifier_logs"]
    assert Path(result.log_dir, "model.patch").read_bytes() == b"candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["golden", "null", "agent"])
@pytest.mark.parametrize("include_patch", [None, False])
async def test_entire_lifecycle_uses_distinct_sandboxes(
    task: InlineTask, mode: str, include_patch: bool | None
) -> None:
    server = make_server(task, mode=mode)
    if include_patch is not None:
        server.config.include_model_patch_in_response = include_patch
    events = []
    agent, verifier = sandbox("A", events), sandbox("B", events)

    async def create(_task: InlineTask, *, phase: str, files=None) -> AsyncMock:
        events.append("create-" + phase)
        if phase == "verifier":
            assert files == task.data.files.verification_files(b"" if mode == "null" else b"committed patch")
            assert not any(path.startswith("/solution/") for path in files)
        else:
            assert files == (task.data.files.solution_files() if mode == "golden" else {})
            assert not any(path.startswith("/tests/") for path in files)
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
            task_id=task.data.task_id,
            image=task.data.image,
            sandbox=agent,
            sandbox_handle="A",
            sandbox_descriptor={"sandbox_id": "A"},
        )
    result = await server.verify(request(), body(task))
    assert result.evaluation_completed and not result.mask_sample
    assert result.agent_sandbox_id == "A" and result.verifier_sandbox_id == "B"
    assert result.validation_mode == mode and result.cleanup_errors == []
    assert "files" not in result.model_dump() and "task_fingerprint" not in result.model_dump()
    assert events.index("collect-A") < events.index("stop-A") < events.index("create-verifier")
    assert events[-2:] == ["grade-B", "stop-B"]
    assert server._execute_golden.await_count == (mode == "golden")
    assert not server._agent_sessions
    assert Path(result.log_dir, "result.json").is_file()
    expected_patch = b"" if mode == "null" else b"committed patch"
    assert Path(result.log_dir, "model.patch").read_bytes() == expected_patch
    assert result.model_patch == (None if include_patch is False else expected_patch.decode())


@pytest.mark.asyncio
async def test_verification_does_not_impose_an_eight_request_limit(task: InlineTask) -> None:
    server = make_server(task, mode="null")
    all_started = asyncio.Event()
    release = asyncio.Event()
    started = 0
    sandboxes = []

    async def create(_task: InlineTask, *, phase: str, files=None) -> AsyncMock:
        box = sandbox(f"{phase}-{len(sandboxes)}", [])
        sandboxes.append(box)
        return box

    async def grade(*args) -> VerifierResult:
        nonlocal started
        started += 1
        if started == 9:
            all_started.set()
        await release.wait()
        return VerifierResult(evaluation_completed=True, reward=0)

    server._create_sandbox = AsyncMock(side_effect=create)
    server._collect_model_patch = AsyncMock(return_value=b"")
    server._run_verifier = AsyncMock(side_effect=grade)
    requests = [asyncio.create_task(server.verify(request(), body(task))) for _ in range(9)]
    try:
        await asyncio.wait_for(all_started.wait(), timeout=5)
        assert started == 9
        assert not any(pending.done() for pending in requests)
    finally:
        release.set()
        results = await asyncio.gather(*requests)
    assert all(result.evaluation_completed and result.reward == 0 for result in results)
    assert len({result.log_dir for result in results}) == 9
    assert len(sandboxes) == 18
    for box in sandboxes:
        box.stop.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["golden", "collection", "empty", "verifier_setup", "verifier", "same_id", "agent_cleanup"]
)
async def test_failures_are_masked_and_owned_sandboxes_are_released(task: InlineTask, failure: str) -> None:
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
@pytest.mark.parametrize("mismatch", ["absent", "task", "image", "handle"])
async def test_agent_requires_matching_seeded_session(task: InlineTask, mismatch: str) -> None:
    server = make_server(task, mode="agent")
    agent = sandbox("A", [])
    if mismatch != "absent":
        server._agent_sessions["session"] = AgentSandboxSession(
            "wrong" if mismatch == "task" else task.data.task_id,
            "wrong" if mismatch == "image" else task.data.image,
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
async def test_shutdown_releases_unfinished_sessions(task: InlineTask) -> None:
    server = make_server(task, mode="agent")
    agent = sandbox("A", [])
    server._agent_sessions["session"] = AgentSandboxSession(task.data.task_id, task.data.image, agent, "A", {})
    app = server.setup_webserver()
    async with app.router.lifespan_context(app):
        agent.stop.assert_not_awaited()
    agent.stop.assert_awaited_once()
    assert not server._agent_sessions


def test_network_and_validation_modes(task: InlineTask) -> None:
    server = make_server(task)
    assert server._provider_options(phase="agent")["network_policy"] == {"defaultAction": "deny", "egress": []}
    assert "network_policy" not in server._provider_options(phase="verifier")
    server.config.enforce_verifier_no_network = True
    assert server._provider_options(phase="verifier")["network_policy"] == {"defaultAction": "deny", "egress": []}
    settings = server.config.model_dump() | {"is_verifying_null_patch": True}
    with pytest.raises(ValueError, match="mutually exclusive"):
        DeepsweExternal1ResourcesServerConfig(**settings)


@pytest.mark.asyncio
async def test_null_mode_cannot_seed(task: InlineTask) -> None:
    server = make_server(task, mode="null")
    with pytest.raises(RuntimeError, match="golden/null-validation"):
        await server.seed_session(
            request(), DeepsweExternal1SeedSessionRequest(**task_row(task.data, "original instruction\n"))
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["agent", "verifier"])
@pytest.mark.parametrize("setup_fails", [False, True])
async def test_sandbox_spec_and_native_setup(
    task: InlineTask, monkeypatch: pytest.MonkeyPatch, phase: str, setup_fails: bool
) -> None:
    server = make_server(task)
    server.config.task_cpu_multiplier = 2
    server.config.task_memory_multiplier = 1.5
    server.config.sandbox_config = {"env": {"EXPLICIT": "yes"}, "ttl_s": 3600, "ready_timeout_s": 60}
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=int(setup_fails), stdout="", stderr="setup diagnostic")
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
            await server._create_sandbox(task, phase=phase, files={"/tests/file": "test data"})
    else:
        assert await server._create_sandbox(task, phase=phase, files={"/tests/file": "test data"}) is box
    constructor.assert_called_once_with({"provider": "sandbox"})
    spec = specs[0]
    assert spec.image == (task.data.image if phase == "agent" else task.data.verifier_image)
    assert spec.workdir == "/app" and spec.files == {"/tests/file": "test data"}
    assert spec.resources.cpu == 2 and spec.resources.memory_mib == 1536
    assert spec.ttl_s == 3600 and spec.ready_timeout_s == 60
    assert spec.env == {"EXPLICIT": "yes"}
    if phase == "agent":
        assert spec.provider_options["network_policy"] == {"defaultAction": "deny", "egress": []}
    else:
        assert "network_policy" not in spec.provider_options
    assert spec.metadata == {
        "owner": "test",
        "task": task.data.task_id,
        "phase": phase,
        "nemo_gym_agent": "test",
    }
    command = box.exec.await_args_list[0].args[0]
    assert "git rev-parse --show-toplevel" in command and task.data.base_commit in command
    assert ("user.email" in command) == (phase == "agent")
    assert "command -v python3" not in command
    assert box.exec.await_args_list[0].kwargs == {"timeout_s": 60}
    if phase == "verifier" and not setup_fails:
        assert box.exec.await_count == 2
        bootstrap = box.exec.await_args_list[1]
        assert bootstrap.args[0] == "ALLOW_PYTHON_INSTALL=1\n" + module.VERIFIER_PYTHON_SETUP
        assert bootstrap.kwargs == {"timeout_s": 300}
    else:
        assert box.exec.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("exit_code", [0, 1, 124])
async def test_verifier_python_setup_policy_and_failures(task: InlineTask, offline: bool, exit_code: int) -> None:
    server = make_server(task)
    server.config.enforce_verifier_no_network = offline
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=exit_code, stdout="package output\n", stderr="diagnostic")
    if exit_code:
        with pytest.raises(RuntimeError, match=rf"Verifier Python setup failed \(exit {exit_code}\): package output"):
            await server._ensure_verifier_python(box)
    else:
        await server._ensure_verifier_python(box)
    box.exec.assert_awaited_once_with(
        f"ALLOW_PYTHON_INSTALL={int(not offline)}\n" + module.VERIFIER_PYTHON_SETUP,
        timeout_s=300,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 2])
async def test_golden_executes_original_script_and_records_exit(
    task: InlineTask, tmp_path: Path, exit_code: int
) -> None:
    server = make_server(task)
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=exit_code, stdout="solution output", stderr="diagnostic")
    logs = tmp_path / "golden"
    if exit_code:
        with pytest.raises(RuntimeError, match="exit code 2"):
            await server._execute_golden(box, task, logs)
    else:
        await server._execute_golden(box, task, logs)
    box.upload.assert_not_awaited()
    box.exec.assert_awaited_once_with("bash /solution/solve.sh", cwd="/app", timeout_s=task.data.solution_timeout_sec)
    assert (logs / "golden.log").read_text() == "solution outputdiagnostic"


@pytest.mark.asyncio
async def test_seed_session_preserves_descriptor(task: InlineTask) -> None:
    server = make_server(task, mode="agent")
    box = sandbox("A", [])
    box.serialize.return_value = {"sandbox_id": "A", "workdir": "/app"}
    server._create_sandbox = AsyncMock(return_value=box)
    seeded = await server.seed_session(
        request(), DeepsweExternal1SeedSessionRequest(**task_row(task.data, "original instruction\n"))
    )
    assert seeded.sandbox_descriptor == {"sandbox_id": "A", "workdir": "/app"}
    assert server._agent_sessions["session"].sandbox is box


@pytest.mark.asyncio
async def test_verifier_cleanup_error_does_not_replace_completed_grade(task: InlineTask) -> None:
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


@pytest.mark.asyncio
async def test_stage_prepares_inline_files_without_local_uploads(task: InlineTask) -> None:
    server = make_server(task)
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=0, stderr="")
    await server._stage_verifier(box, task, b"candidate")
    assert box.exec.await_count == 1
    assert box.exec.await_args.kwargs == {"timeout_s": 60}
    box.upload.assert_not_awaited()
    box.exec.return_value.return_code = 1
    with pytest.raises(RuntimeError, match="prepare DeepSWE verifier"):
        await server._stage_verifier(box, task, b"candidate")


def test_server_has_no_filesystem_task_store_or_store_configuration(task: InlineTask) -> None:
    server = make_server(task)
    assert not hasattr(server, "_task_store")
    assert "tasks_dir" not in type(server.config).model_fields
    assert "expected_task_count" not in type(server.config).model_fields


@pytest.mark.parametrize("workdir", ["/app", "/workspace/repo"])
async def test_stage_applies_candidate_patch_when_the_grader_does_not(task: InlineTask, workdir: str) -> None:
    data = task.data.model_copy(update={"grader_applies_model_patch": False, "workdir": workdir})
    inline = InlineTask(data)
    server = make_server(inline)
    ok = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    box = sandbox("B", [])
    box.exec.side_effect = [ok, ok]
    await server._stage_verifier(box, inline, b"candidate")
    assert box.exec.await_count == 2
    apply_command = box.exec.await_args_list[1].args[0]
    assert apply_command.startswith(f"cd {workdir} && python3 -I - {data.base_commit}")
    assert 'git", "apply"' in apply_command and "/logs/artifacts/model.patch" in apply_command
    assert 'git", "checkout", base' in apply_command  # touched files are reset to the base commit first
    assert box.exec.await_args_list[1].kwargs == {"timeout_s": 300}

    # Nothing to apply for a null patch; the default (v1.0 grader applies it) never adds the step.
    box.exec.reset_mock()
    box.exec.side_effect = [ok]
    await server._stage_verifier(box, inline, b"")
    assert box.exec.await_count == 1
    box.exec.reset_mock()
    box.exec.side_effect = [ok]
    await server._stage_verifier(box, task, b"candidate")
    assert box.exec.await_count == 1

    # A patch that does not apply is a graded outcome, not an infrastructure error.
    box.exec.reset_mock()
    box.exec.side_effect = [
        ok,
        SimpleNamespace(return_code=1, stdout="", stderr="error: patch failed", error_type=None),
    ]
    await server._stage_verifier(box, inline, b"candidate")
    assert box.exec.await_count == 2
