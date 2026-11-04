# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resources tool boundary, cleanup ownership, and unchanged shell output."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pocketshell import run

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from resources_servers.browsecomp_advanced_harness import app as harness
from resources_servers.browsecomp_advanced_harness import sandbox as shell_module
from resources_servers.browsecomp_advanced_harness.app import BashCommandRequest
from resources_servers.browsecomp_advanced_harness.shell_output import format_shell_output
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import (
    _close,
    _seed,
    _server,
    _verify_body,
)


def _request(sid: str = "resources-session") -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: sid}})


@pytest.fixture
def sandboxed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    server = _server(tmp_path, workspace="per_session", sandbox_provider="sandbox", allow_host_execution=False)
    monkeypatch.setattr(harness, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"docker": {}})
    monkeypatch.setattr(harness, "resolve_provider_metadata", lambda *_: {"owner": "default", "pool": "shared"})
    monkeypatch.setattr(harness, "create_provider", lambda *_: object())
    containers = []

    def create(_provider):
        sandbox = AsyncMock()
        containers.append(sandbox)
        return sandbox

    async def install(state, *, timeout):
        state.ready = True

    monkeypatch.setattr(harness, "AsyncSandbox", create)
    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "install", install)
    return server, containers


@pytest.fixture
def shell_install(monkeypatch: pytest.MonkeyPatch):
    sandbox = AsyncMock(spec=harness.AsyncSandbox)
    sandbox.provider_name = "enroot"
    runtime = SimpleNamespace(
        python="/private/runtime/python/bin/python3",
        directory="/private/runtime",
        is_installed=AsyncMock(return_value=False),
        install=AsyncMock(),
    )
    monkeypatch.setattr(shell_module, "_RUNTIME", runtime)
    readlink = MagicMock(return_value="pid:[100]")
    monkeypatch.setattr(shell_module, "os", SimpleNamespace(readlink=readlink))
    session = SimpleNamespace(sandbox=sandbox, session_dir="/private/session", workdir="/workspace")
    return shell_module.BrowsecompShellSandbox(session=session), sandbox, runtime, readlink


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "return_code", "error_type"),
    [
        ("pid:[100]\npid:[100]\n", 0, None),
        ("pid:[200]\npid:[100]\n", 0, None),
        ("pid:[200]\npid:[200]\n", 1, None),
        ("pid:[200]\npid:[200]\n", 0, "timeout"),
        (None, 0, None),
        ("unavailable\nunavailable\n", 0, None),
    ],
)
async def test_enroot_install_refuses_unproven_pid_isolation(shell_install, stdout, return_code, error_type) -> None:
    state, sandbox, runtime, readlink = shell_install
    sandbox.exec.return_value = SimpleNamespace(stdout=stdout, return_code=return_code, error_type=error_type)
    with pytest.raises(RuntimeError, match="older Enroot builds can ignore ENROOT_UNSHARE_PID"):
        await state.install(timeout=20)
    runtime.install.assert_awaited_once_with(sandbox, timeout_s=20)
    readlink.assert_called_once_with("/proc/self/ns/pid")
    assert not state.ready
    sandbox.exec.assert_awaited_once()  # Failed isolation prevents workspace activation.
    command = sandbox.exec.await_args.args[0]
    assert f"{runtime.python} -I -c " in command
    assert "/proc/self/ns/pid" in command and "/proc/1/ns/pid" in command


@pytest.mark.asyncio
async def test_enroot_install_requires_readable_host_pid_namespace(shell_install) -> None:
    state, sandbox, _, readlink = shell_install
    readlink.side_effect = OSError("no proc mount")
    with pytest.raises(RuntimeError, match="Cannot read the host PID namespace"):
        await state.install(timeout=20)
    assert not state.ready
    sandbox.exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_enroot_install_accepts_distinct_namespace_with_matching_proc_mount(shell_install) -> None:
    state, sandbox, _, _ = shell_install
    sandbox.exec.side_effect = [
        SimpleNamespace(stdout="pid:[200]\npid:[200]\n", return_code=0, error_type=None),
        SimpleNamespace(return_code=0, error_type=None),
    ]
    await state.install(timeout=20)
    assert state.ready
    assert sandbox.exec.await_count == 2


@pytest.mark.asyncio
async def test_other_providers_do_not_probe_host_pid_namespace(shell_install) -> None:
    state, sandbox, _, readlink = shell_install
    sandbox.provider_name = "docker"
    sandbox.exec.return_value = SimpleNamespace(return_code=0, error_type=None)
    await state.install(timeout=20)
    assert state.ready
    readlink.assert_not_called()
    sandbox.exec.assert_awaited_once()
    assert "ns/pid" not in sandbox.exec.await_args.args[0]


@pytest.mark.asyncio
async def test_pid_guard_failure_keeps_failed_stop_retryable(
    shell_install, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, sandbox, _, _ = shell_install
    server = _server(tmp_path, workspace="per_session", sandbox_provider="sandbox", allow_host_execution=False)
    monkeypatch.setattr(harness, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"enroot": {}})
    monkeypatch.setattr(harness, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(harness, "create_provider", lambda *_: object())
    monkeypatch.setattr(harness, "AsyncSandbox", lambda *_: sandbox)
    sandbox.exec.return_value = SimpleNamespace(stdout="pid:[100]\npid:[100]\n", return_code=0, error_type=None)
    sandbox.stop.side_effect = [RuntimeError("stop unavailable"), None]
    with pytest.raises(RuntimeError, match="namespace probe did not confirm isolation"):
        await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    assert "resources-session" in server._shell_sessions
    assert not server._shell_sessions["resources-session"].ready
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert sandbox.stop.await_count == 2
    assert not server._shell_sessions and not server._session_workspaces


def test_missing_provider_refuses_seed_without_creating_workspace(tmp_path: Path) -> None:
    server = _server(tmp_path, workspace="per_session", allow_host_execution=False)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "sandbox_provider" in response.json()["detail"]
    assert not server._typed_sessions and not server._shell_sessions
    assert list(tmp_path.iterdir()) == []


def test_local_provider_requires_explicit_development_opt_in(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, containers = sandboxed
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"local": {}})
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "does not isolate" in response.json()["detail"]
    assert not containers


@pytest.mark.asyncio
async def test_seed_is_idempotent_and_close_stops_each_owned_sandbox(sandboxed) -> None:
    server, containers = sandboxed
    for sid in ("resources-session", "resources-session", "another-session"):
        await server.seed_session(_request(sid), ResourcesSeedSessionRequest.model_validate(_seed(sid)))
    assert len(containers) == 2
    assert len({state.session.session_dir for state in server._shell_sessions.values()}) == 2
    for sid in ("resources-session", "another-session"):
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close(sid)))
    for container in containers:
        container.start.assert_awaited_once()
        container.stop.assert_awaited_once()
    assert not server._shell_sessions and not server._session_workspaces


@pytest.mark.asyncio
async def test_startup_delivers_only_shell_source_and_merges_provider_metadata(sandboxed) -> None:
    server, containers = sandboxed
    server.config.sandbox_config["metadata"] = {"owner": "configured", "resources_session_id": "override"}
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    spec = containers[0].start.await_args.args[0]
    assert spec.metadata == {
        "owner": "configured",
        "pool": "shared",
        "resources_session_id": "resources-session",
        "nemo_gym_resources": server.config.name,
    }
    assert spec.files == shell_module.shell_source_files()
    assert any(path.endswith("/pocketshell/__init__.py") for path in spec.files)
    assert any(path.endswith("/shell_runner.py") for path in spec.files)
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))


@pytest.mark.asyncio
async def test_lifespan_stops_remaining_sandbox_on_exception(sandboxed) -> None:
    server, containers = sandboxed
    app = server.setup_webserver()
    with pytest.raises(RuntimeError, match="server failed"):
        async with app.router.lifespan_context(app):
            await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
            raise RuntimeError("server failed")
    containers[0].stop.assert_awaited_once()
    assert not server._shell_sessions and not server._session_workspaces


@pytest.mark.asyncio
async def test_bash_uses_sandbox_and_retains_guard_output(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, _ = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    shell = server._shell_sessions["resources-session"]
    shell.execute = AsyncMock(return_value="page text\n[exit_code=0]")
    host = MagicMock(side_effect=AssertionError("model command reached the host"))
    monkeypatch.setattr(harness, "pocketshell_run", host)
    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat pages/0001.txt"))
    assert output.results_string == "page text\n[exit_code=0]"
    assert shell.execute.await_args.kwargs["command"] == "cat pages/0001.txt"
    assert shell.execute.await_args.kwargs["workspace"] == Path(server.config.workspace_root) / "resources-session"
    blocked = await server.bash_command(_request(), BashCommandRequest(keystrokes="rm -rf pages"))
    assert blocked.results_string == "[blocked: command 'rm' is blocked]\n[exit_code=-3]"
    shell.execute.assert_awaited_once()
    host.assert_not_called()
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))


@pytest.mark.asyncio
async def test_setup_failure_retains_failed_cleanup_for_retry(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, containers = sandboxed

    async def fail_install(state, *, timeout):
        state.session.sandbox.stop.side_effect = [RuntimeError("stop unavailable"), None]
        raise RuntimeError("install unavailable")

    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "install", fail_install)
    with pytest.raises(RuntimeError, match="install unavailable"):
        await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    assert "resources-session" in server._shell_sessions
    assert "resources-session" in server._session_workspaces
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert containers[0].stop.await_count == 2
    assert not server._shell_sessions and not server._session_workspaces


@pytest.mark.asyncio
async def test_verify_does_not_remove_workspace_until_sandbox_stop_succeeds(sandboxed) -> None:
    server, containers = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    workspace = server._session_workspaces["resources-session"].workspace
    containers[0].stop.side_effect = [RuntimeError("stop unavailable"), None]
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    with pytest.raises(RuntimeError, match="stop unavailable"):
        await server.verify(_request(), body)
    assert workspace.exists() and "resources-session" in server._shell_sessions
    assert "resources-session" in server._session_id_to_metrics
    output = await server.verify(_request(), body)
    assert output.reward == 1
    assert not workspace.exists() and not server._shell_sessions


@pytest.mark.asyncio
async def test_close_waits_for_cancelled_tool_cleanup(sandboxed) -> None:
    server, containers = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    started, cleanup_started, cleanup_allowed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def execute(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await cleanup_allowed.wait()

    server._shell_sessions["resources-session"].execute = execute
    tool = asyncio.create_task(server.bash_command(_request(), BashCommandRequest(keystrokes="ls")))
    await asyncio.wait_for(started.wait(), 1)
    tool.cancel()
    await asyncio.wait_for(cleanup_started.wait(), 1)
    close = asyncio.create_task(server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close())))
    await asyncio.sleep(0)
    containers[0].stop.assert_not_awaited()
    assert "resources-session" in server._session_workspaces
    cleanup_allowed.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(tool, 1)
    await asyncio.wait_for(close, 1)
    containers[0].stop.assert_awaited_once()


@pytest.mark.parametrize(
    "command",
    [
        "cat pages/a.txt",
        "grep -P 'Alpha(?= Beta)' pages/a.txt",
        "cat pages/missing.txt",
        "cat /etc/passwd",
        "cat pages/big.txt",
    ],
)
def test_real_runner_preserves_pocketshell_output(tmp_path: Path, command: str) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "a.txt").write_text("Alpha Beta\nsecond line\n")
    (pages / "big.txt").write_text("x" * 100_000)
    input_file, output_file = tmp_path / "input.json", tmp_path / "output.txt"
    input_file.write_text(json.dumps({"command": command, "duration": 5, "workspace": str(tmp_path)}))
    runner = Path(shell_module.__file__).with_name("shell_runner.py")
    subprocess.run(
        [sys.executable, "-E", "-s", str(runner), str(input_file), str(output_file)], check=True, timeout=15
    )
    expected = run(command, str(tmp_path), 5)
    assert output_file.read_text() == format_shell_output(expected.stdout, expected.stderr, expected.exit_code)
    assert len(output_file.read_text()) < 25_000


@pytest.mark.asyncio
async def test_controller_transfers_only_pages_and_manifest_and_never_interpolates_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "pages").mkdir(parents=True)
    (workspace / "pages/a.txt").write_text("page")
    (workspace / "manifest.tsv").write_text("manifest")
    (workspace / "private-key").write_text("not a retrieved page")
    sandbox = AsyncMock()
    sandbox.exec.return_value = SimpleNamespace(return_code=0, error_type=None)
    session = SimpleNamespace(sandbox=sandbox, session_dir="/private/session", workdir="/workspace")
    directories, commands, payloads = [], [], []

    async def begin(*, session_dir, timeout):
        directories.append(session_dir)
        session.session_dir = session_dir

    async def execute(*, stage_activation, collect, **kwargs):
        commands.append(await stage_activation())
        return "result"

    async def upload(_sandbox, *, path, text):
        payloads.append(json.loads(text))

    session.begin_activation, session.execute = begin, execute
    monkeypatch.setattr(shell_module, "upload_text", upload)
    state = shell_module.BrowsecompShellSandbox(session=session, ready=True)
    command = 'cat pages/a.txt; echo "$(not-shell-source)"'
    for _ in range(2):
        await state.execute(command=command, duration=5, workspace=workspace, close_timeout=10)
    assert directories[0] != directories[1]
    assert all(command not in " ".join(call.argv) for call in commands)
    assert [payload["command"] for payload in payloads] == [command, command]
    sources = [call.args[0] for call in sandbox.upload.await_args_list]
    assert sources.count(workspace / "pages/a.txt") == 1
    assert sources.count(workspace / "manifest.tsv") == 2
    assert workspace / "private-key" not in sources

    (workspace / "pages/escape.txt").symlink_to(workspace / "private-key")
    with pytest.raises(ValueError, match="not a regular file"):
        await state.execute(command="ls", duration=5, workspace=workspace, close_timeout=10)
