# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resources tool boundary, cleanup ownership, and unchanged shell output."""

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pocketshell import run
from pydantic import ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.sandbox import create_provider
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.verification import JudgeError
from resources_servers.browsecomp_advanced_harness import app as harness
from resources_servers.browsecomp_advanced_harness import sandbox as shell_module
from resources_servers.browsecomp_advanced_harness.app import BashCommandRequest, JudgeEvaluation
from resources_servers.browsecomp_advanced_harness.shell_output import format_shell_output
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import (
    _close,
    _seed,
    _server,
    _verify_body,
)


def _request(sid: str = "resources-session") -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: sid}})


async def test_lifecycle_logging_preserves_failed_close_for_retry(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    session = SimpleNamespace(
        sandbox=SimpleNamespace(provider_name="docker", sandbox_id="tool-sandbox"),
        close=AsyncMock(side_effect=[RuntimeError("stop failed"), None, None]),
    )
    shell = shell_module.BrowsecompShellSandbox(
        session=session, resources_session_id="episode-tools", bash_calls=3, uploaded_pages={"one", "two"}
    )
    with pytest.raises(RuntimeError, match="stop failed"):
        await shell.close(timeout=1)
    await shell.close(timeout=1)
    await shell.close(timeout=1)
    lines = [record.message for record in caplog.records if "[sandbox_stop]" in record.message]
    assert len(lines) == 2
    assert "status=failed:RuntimeError" in lines[0] and "status=ok" in lines[1]
    assert all("provider=docker id=tool-sandbox" in line and "bash_calls=3 pages=2" in line for line in lines)


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
    ("stdout", "return_code", "error_type", "final"),
    [
        ("pid:[100]\npid:[100]\n", 0, None, True),
        ("pid:[200]\npid:[100]\n", 0, None, True),
        ("pid:[200]\npid:[200]\n", 1, None, False),
        ("pid:[200]\npid:[200]\n", 0, "timeout", False),
        (None, 0, None, False),
        ("unavailable\nunavailable\n", 0, None, False),
    ],
)
async def test_enroot_install_refuses_unproven_pid_isolation(
    shell_install, stdout, return_code, error_type, final
) -> None:
    state, sandbox, runtime, readlink = shell_install
    sandbox.exec.return_value = SimpleNamespace(stdout=stdout, return_code=return_code, error_type=error_type)
    with pytest.raises(RuntimeError, match="older Enroot builds can ignore ENROOT_UNSHARE_PID") as refused:
        await state.install(timeout=20)
    # Only a finished probe that names the host's namespace, or a mismatched /proc, is final; the rest may retry.
    assert isinstance(refused.value, shell_module.SandboxIsolationError) is final
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


def test_a_sandbox_in_the_host_pid_namespace_refuses_the_seed_for_good(
    shell_install, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every sandbox from this provider shares the namespace alike, so a retryable 500 would only repeat the probe.
    _, sandbox, _, _ = shell_install
    server = _server(tmp_path, workspace="per_session", sandbox_provider="sandbox", allow_host_execution=False)
    monkeypatch.setattr(harness, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"enroot": {}})
    monkeypatch.setattr(harness, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(harness, "create_provider", lambda *_: object())
    monkeypatch.setattr(harness, "AsyncSandbox", lambda *_: sandbox)
    sandbox.exec.return_value = SimpleNamespace(stdout="pid:[100]\npid:[100]\n", return_code=0, error_type=None)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
        assert response.status_code == 422
        assert "namespace probe did not confirm isolation" in response.json()["detail"]
        sandbox.stop.assert_awaited_once()
        assert not server._shell_sessions
        assert client.post("/close_session", json=_close()).status_code == 200
    assert not server._session_workspaces


def test_an_unknown_provider_refuses_the_seed_for_good(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, containers = sandboxed
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"no-such-provider": {}})
    monkeypatch.setattr(harness, "create_provider", create_provider)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "Unknown sandbox provider 'no-such-provider'" in response.json()["detail"]
    assert not containers and not server._shell_sessions


def test_sandbox_timeouts_must_be_finite(tmp_path: Path) -> None:
    for field in ("sandbox_install_timeout_seconds", "sandbox_close_timeout_seconds"):
        with pytest.raises(ValidationError, match="finite number"):
            _server(tmp_path, **{field: float("inf")})


def test_missing_provider_refuses_seed_without_creating_workspace(tmp_path: Path) -> None:
    server = _server(tmp_path, workspace="per_session", allow_host_execution=False)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "sandbox_provider" in response.json()["detail"]
    assert not server._episode_sessions and not server._shell_sessions
    assert list(tmp_path.iterdir()) == []


def test_local_provider_requires_explicit_development_opt_in(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, containers = sandboxed
    monkeypatch.setattr(harness, "resolve_provider_config", lambda *_: {"local": {}})
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "does not isolate" in response.json()["detail"]
    assert not containers


def _missing_provider_block(*_) -> dict:
    raise ValueError("Sandbox provider config 'sandbox' was not found")


@pytest.mark.parametrize(
    ("sandbox_config", "resolve"),
    [({"ttl": 7200}, None), ({}, _missing_provider_block)],
    ids=["unknown sandbox_config key", "missing provider block"],
)
def test_invalid_sandbox_configuration_refuses_the_seed_for_good(
    sandboxed, monkeypatch: pytest.MonkeyPatch, sandbox_config, resolve
) -> None:
    # Every seed fails the same way, so a retryable 500 would only repeat the failure for each episode.
    server, containers = sandboxed
    server.config.sandbox_config.update(sandbox_config)
    if resolve is not None:
        monkeypatch.setattr(harness, "resolve_provider_config", resolve)
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/seed_session", json=_seed())
    assert response.status_code == 422
    assert "sandbox configuration is invalid" in response.json()["detail"]
    assert not containers and not server._episode_sessions


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
@pytest.mark.parametrize(
    ("error", "expected", "masked"),
    [
        (
            shell_module.ShellCommandTimedOut("supervisor deadline"),
            "[command timed out after 10s]\n[exit_code=-1]",
            False,
        ),
        (
            RuntimeError("exec failed: provider detail"),
            "[bash error: the sandbox could not run the command]\n[exit_code=-2]",
            True,
        ),
        # A provider timeout (docker cp, the exec, a read) is not the command's deadline.
        (
            TimeoutError("docker command timed out after 60s"),
            "[bash error: the sandbox could not run the command]\n[exit_code=-2]",
            True,
        ),
    ],
    ids=["supervisor deadline", "provider error", "provider timeout"],
)
async def test_sandbox_failures_answer_with_the_host_error_envelope(sandboxed, error, expected, masked) -> None:
    server, _ = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    server._shell_sessions["resources-session"].execute = AsyncMock(side_effect=error)
    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat pages/0001.txt"))
    assert output.results_string == expected
    # A slow command is the model's doing; a sandbox that cannot run one is not, so only that sample is masked.
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    verified = await server.verify(_request(), body)
    assert verified.reward == 1
    assert verified.mask_sample is masked
    assert verified.failure_kind == ("provider_unavailable" if masked else None)
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert not server._shell_failures


@pytest.mark.asyncio
async def test_run_seed_starts_the_sandbox_with_the_first_shell_command(
    sandboxed, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A /run agent can skip /verify, so only a run that used the shell may hold a sandbox.
    server, containers = sandboxed
    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "execute", AsyncMock(return_value="text\n[exit_code=0]"))
    await server.seed_session(_request(), harness.BaseSeedSessionRequest())
    assert not containers and "resources-session" in server._session_workspaces
    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat pages/0001.txt"))
    assert output.results_string == "text\n[exit_code=0]"
    containers[0].start.assert_awaited_once()
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    assert (await server.verify(_request(), body)).reward == 1
    containers[0].stop.assert_awaited_once()
    assert not server._shell_sessions and not server._session_workspaces


@pytest.mark.asyncio
async def test_a_run_whose_sandbox_cannot_start_gets_the_envelope_and_a_masked_sample(
    sandboxed, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, containers = sandboxed

    async def fail_install(state, *, timeout):
        raise RuntimeError("docker daemon unavailable: provider detail")

    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "install", fail_install)
    await server.seed_session(_request(), harness.BaseSeedSessionRequest())
    blocked = await server.bash_command(_request(), BashCommandRequest(keystrokes="rm -rf pages"))
    assert blocked.results_string == "[blocked: command 'rm' is blocked]\n[exit_code=-3]"
    assert not containers  # A blocked command needs no sandbox.

    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat pages/0001.txt"))
    assert output.results_string == "[bash error: the sandbox could not run the command]\n[exit_code=-2]"
    containers[0].stop.assert_awaited_once()
    assert not server._shell_sessions

    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    verified = await server.verify(_request(), body)
    assert (verified.reward, verified.mask_sample, verified.failure_kind) == (1, True, "provider_unavailable")
    assert "provider detail" not in verified.failure_reason
    assert not server._shell_failures and not server._session_workspaces


@pytest.mark.asyncio
async def test_commands_after_a_failed_start_do_not_start_another_sandbox(
    sandboxed, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, containers = sandboxed
    install = AsyncMock(side_effect=RuntimeError("docker daemon unavailable"))
    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "install", install)
    await server.seed_session(_request(), harness.BaseSeedSessionRequest())
    for _ in range(3):
        output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat pages/0001.txt"))
        assert output.results_string == "[bash error: the sandbox could not run the command]\n[exit_code=-2]"
    assert len(containers) == 1 and install.await_count == 1
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    assert (await server.verify(_request(), body)).mask_sample is True


@pytest.mark.asyncio
async def test_a_run_whose_start_and_first_stop_fail_is_masked_and_stopped_by_verify(
    sandboxed, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, containers = sandboxed

    async def fail_install(state, *, timeout):
        state.session.sandbox.stop.side_effect = [RuntimeError("stop unavailable"), None]
        raise RuntimeError("install unavailable")

    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "install", fail_install)
    await server.seed_session(_request(), harness.BaseSeedSessionRequest())
    for _ in range(2):
        output = await server.bash_command(_request(), BashCommandRequest(keystrokes="ls"))
        assert output.results_string == "[bash error: the sandbox could not run the command]\n[exit_code=-2]"
    # The failed stop retains the sandbox, so the second command neither starts another nor reaches the host.
    assert "resources-session" in server._shell_sessions and len(containers) == 1
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    verified = await server.verify(_request(), body)
    assert (verified.mask_sample, verified.failure_kind) == (True, "provider_unavailable")
    assert containers[0].stop.await_count == 2
    assert not server._shell_sessions and not server._session_workspaces


@pytest.mark.asyncio
async def test_concurrent_commands_on_a_run_seed_start_one_sandbox(sandboxed, monkeypatch: pytest.MonkeyPatch) -> None:
    server, containers = sandboxed
    entered, release = asyncio.Event(), asyncio.Event()
    running = peak = 0

    async def execute(state, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        entered.set()
        await release.wait()
        running -= 1
        return f"{kwargs['command']}\n[exit_code=0]"

    monkeypatch.setattr(shell_module.BrowsecompShellSandbox, "execute", execute)
    await server.seed_session(_request(), harness.BaseSeedSessionRequest())
    calls = [
        asyncio.create_task(server.bash_command(_request(), BashCommandRequest(keystrokes=f"cat pages/{n}.txt")))
        for n in range(3)
    ]
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.sleep(0.01)
    release.set()
    outputs = await asyncio.wait_for(asyncio.gather(*calls), 1)
    assert sorted(output.results_string for output in outputs) == [
        f"cat pages/{n}.txt\n[exit_code=0]" for n in range(3)
    ]
    # The session's lock serializes the lazy start and the commands.
    assert len(containers) == 1 and peak == 1
    containers[0].start.assert_awaited_once()
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    await server.verify(_request(), body)
    containers[0].stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_provider_upload_timeout_is_a_masked_sandbox_failure(sandboxed) -> None:
    # Through the real SandboxSession: docker cp timed out before launch, so no supervisor deadline passed.
    server, containers = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    containers[0].exec.return_value = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    containers[0].upload.side_effect = TimeoutError("docker command timed out after 60s: docker cp")
    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat manifest.tsv"))
    assert output.results_string == "[bash error: the sandbox could not run the command]\n[exit_code=-2]"
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    verified = await server.verify(_request(), body)
    assert (verified.mask_sample, verified.failure_kind) == (True, "provider_unavailable")
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))


@pytest.mark.asyncio
async def test_a_provider_exec_timeout_without_a_timed_out_receipt_is_a_masked_sandbox_failure(sandboxed) -> None:
    server, _ = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    shell = server._shell_sessions["resources-session"]
    shell.session.begin_activation = AsyncMock()
    shell.session.execute = AsyncMock(side_effect=TimeoutError("supervisor exceeded its execution deadline"))
    shell.session.cleanup = {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": False}
    output = await server.bash_command(_request(), BashCommandRequest(keystrokes="cat manifest.tsv"))
    assert output.results_string == "[bash error: the sandbox could not run the command]\n[exit_code=-2]"
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    assert (await server.verify(_request(), body)).mask_sample is True
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))


@pytest.mark.asyncio
@pytest.mark.parametrize("shell_failed", [True, False], ids=["masked", "unmasked"])
async def test_a_judge_failure_raises_only_for_an_unmasked_sample(sandboxed, shell_failed: bool) -> None:
    # A re-sent verify, or one retried after the judge error, must read the same session state as the first.
    server, _ = sandboxed
    server.config.use_judge = True
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    if shell_failed:
        server._shell_sessions["resources-session"].execute = AsyncMock(side_effect=RuntimeError("exec failed"))
        await server.bash_command(_request(), BashCommandRequest(keystrokes="ls"))
    judged = JudgeEvaluation(reasoning="", extracted_final_answer="Paris", reward=1.0)
    server._verify_answer_with_judge = AsyncMock(side_effect=[(judged, JudgeError("judge down")), (judged, None)])
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    if shell_failed:
        first = await server.verify(_request(), body)
        assert (first.mask_sample, first.failure_kind) == (True, "provider_unavailable")
    else:
        with pytest.raises(JudgeError, match="judge down"):
            await server.verify(_request(), body)
    retried = await server.verify(_request(), body)
    assert (retried.reward, retried.mask_sample) == (1, shell_failed)
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert not server._shell_failures and "resources-session" not in server._session_id_to_metrics


@pytest.mark.asyncio
@pytest.mark.parametrize(("timed_out", "raised"), [(True, shell_module.ShellCommandTimedOut), (False, RuntimeError)])
async def test_a_command_the_supervisor_ends_at_its_deadline_reads_as_a_timeout(timed_out, raised) -> None:
    # The supervisor kills the runner and exits 0 with a timed_out receipt, so only the missing output fails.
    session = SimpleNamespace(
        session_dir="/private/session/calls/first",
        workdir="/workspace",
        begin_activation=AsyncMock(),
        execute=AsyncMock(side_effect=RuntimeError("output.txt is missing")),
        cleanup={"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": timed_out},
    )
    state = shell_module.BrowsecompShellSandbox(session=session)
    state.ready = True
    with pytest.raises(raised):
        await state.execute(command="grep -r x pages", duration=10, workspace=Path("/unused"), close_timeout=5)


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
async def test_episode_verify_keeps_the_session_and_close_retries_a_failed_stop(sandboxed) -> None:
    # The Environment Server closes every session it seeds, so the close, not verify, stops the sandbox.
    server, containers = sandboxed
    await server.seed_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()))
    workspace = server._session_workspaces["resources-session"].workspace
    containers[0].stop.side_effect = [RuntimeError("stop unavailable"), None]
    body = harness.TavilySearchVerifyRequest.model_validate(_verify_body("Answer: Paris Confidence: 90%"))
    assert (await server.verify(_request(), body)).reward == 1
    containers[0].stop.assert_not_awaited()
    assert workspace.exists() and "resources-session" in server._shell_sessions
    assert "resources-session" in server._session_id_to_metrics
    with pytest.raises(RuntimeError, match="stop unavailable"):
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert workspace.exists() and "resources-session" in server._shell_sessions
    assert server._fences.get("resources-session") is None
    await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close()))
    assert containers[0].stop.await_count == 2
    assert not workspace.exists() and not server._shell_sessions
    assert "resources-session" not in server._session_id_to_metrics
    assert server._fences.get("resources-session") is not None


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
    # Long enough for a close that ignored the lock to finish.
    await asyncio.sleep(0.05)
    assert not close.done()
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


def test_real_runner_keeps_the_envelope_for_a_lone_surrogate(tmp_path: Path) -> None:
    # A tool argument JSON-escaped as "\ud83d" (half an emoji) reaches echo as a lone surrogate.
    command = "echo a\ud83db"
    input_file, output_file = tmp_path / "input.json", tmp_path / "output.txt"
    input_file.write_text(json.dumps({"command": command, "duration": 5, "workspace": str(tmp_path)}))
    runner = Path(shell_module.__file__).with_name("shell_runner.py")
    subprocess.run(
        [sys.executable, "-E", "-s", str(runner), str(input_file), str(output_file)], check=True, timeout=15
    )
    expected = run(command, str(tmp_path), 5)
    assert "\ud83d" in expected.stdout
    rendered = format_shell_output(expected.stdout, expected.stderr, expected.exit_code).replace("\ud83d", "?")
    assert output_file.read_text(encoding="utf-8") == rendered
    assert rendered.endswith("[exit_code=0]")


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
    session = SimpleNamespace(sandbox=sandbox, session_dir="/private/session", workdir="/workspace", cleanup=None)
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
