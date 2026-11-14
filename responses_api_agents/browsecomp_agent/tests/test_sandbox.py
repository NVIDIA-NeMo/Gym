# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The process/HTTP boundary, source bundle and sandbox session lifecycle."""

import asyncio
import json
import socket
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from nemo_gym.agents import UnsupportedSandboxPlatformError, check_agent_seed_rejected
from nemo_gym.core import NeMoGymResponse, NgFailureEvent
from responses_api_agents.browsecomp_agent.sandbox import (
    BrowsecompSandboxSession,
    SandboxIsolationError,
    _write_sources,
)
from responses_api_agents.browsecomp_agent.tests.test_agent_sessions import (
    EPISODE,
    GRANT,
    QUESTION,
    ROUTE,
    _agent,
    _message,
    _model_response,
    _seed,
    _tool_call,
)


@pytest.mark.parametrize(
    "stdout",
    ["pid:[100]\npid:[100]\n", "pid:[200]\npid:[100]\n"],
    ids=["the host's namespace", "a /proc mount of another namespace"],
)
async def test_enroot_rejects_missing_process_isolation(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    from responses_api_agents.browsecomp_agent.sandbox import SandboxIsolationError

    session = MagicMock()
    session.sandbox.provider_name = "enroot"
    session.sandbox.exec = AsyncMock(return_value=MagicMock(stdout=stdout, return_code=0, error_type=None))
    state = BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.sandbox.os.readlink", lambda path: "pid:[100]")
    with pytest.raises(SandboxIsolationError, match="isolated PID namespace"):
        await state._check_process_isolation(python="/private/python")


@pytest.mark.parametrize(
    ("stdout", "return_code", "error_type", "message"),
    [
        ("", None, "timeout", "could not finish its process isolation probe: timeout"),
        ("", 137, None, "could not finish its process isolation probe: exit code 137"),
        ("", 0, None, "did not name a PID namespace"),
        ("not a namespace\nnot a namespace\n", 0, None, "did not name a PID namespace"),
    ],
    ids=["timeout", "killed", "no output", "no namespace"],
)
async def test_an_unfinished_isolation_probe_is_retryable_not_a_refusal(
    monkeypatch: pytest.MonkeyPatch, stdout: str, return_code: int | None, error_type: str | None, message: str
) -> None:
    from responses_api_agents.browsecomp_agent.sandbox import SandboxIsolationError

    session = MagicMock()
    session.sandbox.provider_name = "enroot"
    session.sandbox.exec = AsyncMock(
        return_value=MagicMock(stdout=stdout, return_code=return_code, error_type=error_type)
    )
    state = BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.sandbox.os.readlink", lambda path: "pid:[100]")
    with pytest.raises(RuntimeError, match=message) as error:
        await state._check_process_isolation(python="/private/python")
    # SandboxIsolationError refuses the seed for good (422); a probe that gave no answer may pass on a retry.
    assert not isinstance(error.value, SandboxIsolationError)


async def test_enroot_accepts_a_distinct_process_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    session = MagicMock()
    session.sandbox.provider_name = "enroot"
    session.sandbox.exec = AsyncMock(
        return_value=MagicMock(stdout="pid:[200]\npid:[200]\n", return_code=0, error_type=None)
    )
    state = BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.sandbox.os.readlink", lambda path: "pid:[100]")
    await state._check_process_isolation(python="/private/python")
    assert session.sandbox.exec.await_args.kwargs["timeout_s"] == 30


async def test_process_probe_does_not_run_on_other_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_host_proc(path: str) -> str:
        raise AssertionError("Other providers must not inspect host /proc")

    monkeypatch.setattr("responses_api_agents.browsecomp_agent.sandbox.os.readlink", no_host_proc)
    session = MagicMock()
    session.sandbox.provider_name = "docker"
    session.sandbox.exec = AsyncMock()
    state = BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})
    await state._check_process_isolation(python="/private/python")
    session.sandbox.exec.assert_not_awaited()


async def test_isolation_failure_prevents_source_and_input_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    from responses_api_agents.browsecomp_agent import sandbox

    session = MagicMock()
    session.sandbox.upload = AsyncMock()
    state = BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})
    runtime = MagicMock(install=AsyncMock())
    monkeypatch.setattr(sandbox, "_runtime", lambda: runtime)
    monkeypatch.setattr(state, "_check_process_isolation", AsyncMock(side_effect=RuntimeError("isolation failed")))
    with pytest.raises(RuntimeError, match="isolation failed"):
        await state.install_runtime(timeout=60)
    runtime.install.assert_awaited_once()
    session.sandbox.upload.assert_not_awaited()


def test_host_execution_is_an_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([], monkeypatch)
    reply = client.post("/v1/responses", json=QUESTION)
    assert reply.status_code == 422
    assert "host execution is disabled" in reply.text


def test_seed_requires_a_sandbox_and_retains_no_failed_session(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([], monkeypatch, sandbox_provider=None)
    reply = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
    assert reply.status_code == 422
    assert "sandbox_access or sandbox_provider" in reply.text


def test_local_provider_is_not_accepted_as_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([], monkeypatch)
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.resolve_provider_config", lambda *args: {"local": {}}
    )
    reply = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
    assert reply.status_code == 422
    assert "not isolated" in reply.text


def test_retries_replay_and_conflicting_activations_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client, model, _ = _agent([_model_response(_message("Paris"))], monkeypatch)
    client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
    first = client.post(ROUTE, json=QUESTION)
    second = client.post(ROUTE, json=QUESTION)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert model.post.await_count == 1
    assert client.post(ROUTE, json={"input": "Different request"}).status_code == 409
    assert client.post("/v1/responses", json=QUESTION).status_code == 409
    assert client.post("/ng-rollout/wrong/v1/responses", json=QUESTION).status_code == 409


def test_no_tool_grant_is_valid_for_a_tool_free_pairing(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([_model_response(_message("42"))], monkeypatch)
    assert client.post("/v1/agent_sessions", json=_seed(tool_accesses=[]).model_dump(mode="json")).status_code == 200
    assert client.post(ROUTE, json=QUESTION).status_code == 200


async def test_close_retains_cookies_and_an_explicit_evidence_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([], monkeypatch)
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    state = await agent._seed_agent_session_state(_seed())
    state.session.artifacts = {"resources_cookies": {"harness": "final-cookie"}}
    result = await agent._close_agent_session_state(state)
    state.session.close.assert_awaited_once()
    assert result.resources_cookies == {"harness": "final-cookie"}
    assert result.agent_observations.gaps[0].code == "native_turn_evidence_unavailable"


def test_source_bundle_contains_only_runtime_code(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    _write_sources(archive)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
    assert "responses_api_agents/browsecomp_agent/app.py" in names
    assert "nemo_gym/core/responses_api_types.py" in names
    assert all(Path(name).suffix in {".py", ".sh"} for name in names)
    assert not any(part in {".venv", "tests", "__pycache__"} for name in names for part in Path(name).parts)


async def test_runner_process_uses_scoped_model_and_tool_urls(tmp_path: Path) -> None:
    """Run the actual shim/loop in another interpreter, driving a real HTTP model and tool stub."""
    calls = []

    async def model(request: web.Request) -> web.Response:
        calls.append(await request.json())
        return web.json_response(
            _model_response(_tool_call("ls")) if len(calls) == 1 else _model_response(_message("Paris"))
        )

    async def tool(request: web.Request) -> web.Response:
        assert request.headers["x-grant"] == "1"
        assert request.cookies["harness"] == "seed-cookie"
        assert await request.json() == {"keystrokes": "ls"}
        response = web.json_response({"results_string": "manifest.tsv\npages"})
        response.set_cookie("harness", "after-tool")
        return response

    app = web.Application()
    app.router.add_post(ROUTE, model)
    app.router.add_post("/tools/bash_command", tool)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    archive = tmp_path / "source.zip"
    _write_sources(archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(tmp_path / "source")
    payload = {
        "config": {
            "name": "browsecomp",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "nudge_steps": False,
        },
        "seed": _seed().model_dump(mode="json"),
        "tool_access": GRANT.model_copy(update={"base_url": f"{origin}/tools"}).model_dump(mode="json"),
        "body": QUESTION,
        "model_url": f"{origin}/ng-rollout/{EPISODE.capture_key}/v1",
    }
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            str(tmp_path / "source/responses_api_agents/browsecomp_agent/sandbox_runner.py"),
            str(input_path),
            str(output_path),
            cwd=tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        assert process.returncode == 0, (stdout + stderr).decode(errors="replace")
        result = json.loads(output_path.read_text())
        assert "error" not in result, result
        assert result["runtime"]["pid"] == process.pid
        assert result["resources_cookies"] == {"harness": "after-tool"}
        assert result["response"]["output"][1]["output"] == "manifest.tsv\npages"
        assert len(calls) == 2
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await runner.cleanup()


@pytest.mark.parametrize("borrowed", [False, True])
async def test_seed_selects_ownership_and_propagates_provider_metadata(
    monkeypatch: pytest.MonkeyPatch, borrowed: bool
) -> None:
    from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
    from responses_api_agents.browsecomp_agent import app

    client, _, _ = _agent([], monkeypatch, sandbox_config={"workdir": "/sandbox", "metadata": {"priority": "task"}})
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *args: {"priority": "provider", "owner": "test"})
    access = (
        SandboxAccess(
            connection=DirectSandboxConnection(provider_config_ref="sandbox", descriptor={"id": "borrowed"}),
            workdir="/task",
        )
        if borrowed
        else None
    )
    state = await agent._seed_agent_session_state(_seed(sandbox_access=access))
    assert state.session.owns_sandbox is not borrowed
    assert state.session.workdir == ("/task" if borrowed else "/sandbox")
    if borrowed:
        app.AsyncSandbox.connect.assert_awaited_once()
        app.AsyncSandbox.return_value.start.assert_not_awaited()
    else:
        spec = app.AsyncSandbox.return_value.start.await_args.args[0]
        assert spec.metadata == {"priority": "task", "owner": "test", "agent_session_id": "agent-session"}
        assert spec.ttl_s > agent.config.sandbox_runner_timeout_seconds


async def test_failed_install_retains_cleanup_handle_if_release_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_gym.agents import AgentSessionSetupError

    client, _, _ = _agent([], monkeypatch)
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    monkeypatch.setattr(
        BrowsecompSandboxSession, "install_runtime", AsyncMock(side_effect=RuntimeError("install failed"))
    )
    close = AsyncMock(side_effect=[RuntimeError("provider unavailable"), None])
    monkeypatch.setattr(BrowsecompSandboxSession, "close", close)
    with pytest.raises(AgentSessionSetupError) as error:
        await agent._seed_agent_session_state(_seed())
    state = error.value.state
    assert state.session.owns_sandbox
    assert "install failed" in str(error.value)
    await agent._close_agent_session_state(state)
    assert close.await_count == 2


async def test_close_stops_sandbox_before_cancelling_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import AsyncMock

    client, _, _ = _agent([], monkeypatch)
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    state = await agent._seed_agent_session_state(_seed())
    state.task = asyncio.create_task(asyncio.Event().wait())
    await asyncio.sleep(0)

    async def release(*, timeout: float) -> None:
        assert not state.task.cancelled()

    state.session.close = AsyncMock(side_effect=release)
    await state.close(timeout=1)
    assert state.task.cancelled()
    state.session.close.assert_awaited_once()


async def test_start_upload_failure_retains_owned_sandbox_for_cleanup_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from nemo_gym.agents import AgentSessionSetupError, SandboxSession
    from nemo_gym.sandbox import AsyncSandbox
    from responses_api_agents.browsecomp_agent import app

    # Restore the real sandbox/close after the transport-focused test factory patches them.
    real_close = SandboxSession.close
    client, _, _ = _agent([], monkeypatch, sandbox_config={"workdir": "/sandbox", "files": {"/task.txt": "task"}})
    agent = next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")
    provider = MagicMock()
    handle = MagicMock()
    provider.create = AsyncMock(return_value=handle)
    provider.upload_file = AsyncMock(side_effect=RuntimeError("file delivery failed"))
    provider.close = AsyncMock(side_effect=[RuntimeError("stop failed"), RuntimeError("stop failed again"), None])
    provider.aclose = AsyncMock()
    monkeypatch.setattr(app, "create_provider", lambda *args: provider)
    monkeypatch.setattr(app, "AsyncSandbox", AsyncSandbox)
    monkeypatch.setattr(SandboxSession, "close", real_close)
    with pytest.raises(AgentSessionSetupError) as error:
        await agent._seed_agent_session_state(_seed())
    assert provider.close.await_count == 2
    assert error.value.state.session.sandbox._handle is handle
    await agent._close_agent_session_state(error.value.state)
    assert provider.close.await_count == 3
    provider.aclose.assert_awaited()


async def test_an_unreachable_endpoint_fails_the_activation_before_the_loop(tmp_path: Path) -> None:
    """The runner reports an endpoint it cannot connect to instead of spinning until its deadline."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    archive = tmp_path / "source.zip"
    _write_sources(archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(tmp_path / "source")
    payload = {
        "config": {
            "name": "browsecomp",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "rollout_timeout_s": 600,
        },
        "seed": _seed(tool_accesses=[]).model_dump(mode="json"),
        "tool_access": None,
        "body": QUESTION,
        "model_url": f"http://127.0.0.1:{closed_port}/ng-rollout/{EPISODE.capture_key}/v1",
    }
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    runner = str(tmp_path / "source/responses_api_agents/browsecomp_agent/sandbox_runner.py")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", runner, str(input_path), str(output_path), cwd=tmp_path
    )
    try:
        assert await asyncio.wait_for(process.wait(), timeout=60) == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    receipt = json.loads(output_path.read_text())
    assert receipt["error"].startswith(f"ConnectionError: The sandbox cannot connect to 127.0.0.1:{closed_port}")
    assert "response" not in receipt


def test_a_tool_free_session_answers_a_tool_call_without_a_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, harness = _agent([_model_response(_tool_call("ls")), _model_response(_message("42"))], monkeypatch)
    assert client.post("/v1/agent_sessions", json=_seed(tool_accesses=[]).model_dump(mode="json")).status_code == 200
    reply = client.post(ROUTE, json=QUESTION)
    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][1]["output"] == "Tool 'bash_command' is not available in this session."
    assert harness.calls == []


def test_retries_count_from_the_first_attempts_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from responses_api_agents.browsecomp_agent.app import BrowsecompAgent

    client, _, _ = _agent([], monkeypatch, max_run_retries=3, rollout_timeout_s=60)
    starts = []

    async def empty_attempt(self, request, response, body, *, session=None, started_at=None):
        starts.append(started_at)
        return NeMoGymResponse.model_validate(_model_response(_message("")))

    monkeypatch.setattr(BrowsecompAgent, "_responses_impl", empty_attempt)
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200
    assert client.post(ROUTE, json=QUESTION).status_code == 200
    assert len(starts) == 3 and starts[0] is not None and len(set(starts)) == 1


def test_a_loopback_model_url_refuses_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from responses_api_agents.browsecomp_agent.app import BrowsecompAgent

    client, _, _ = _agent([], monkeypatch)
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.resolve_provider_config", lambda *args: {"e2b": {}})
    monkeypatch.setattr(BrowsecompAgent, "resolve_model_base_url", lambda *args: "http://127.0.0.1:9001/v1")
    reply = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
    assert reply.status_code == 422
    assert "loopback" in reply.text


def test_a_loopback_token_count_url_refuses_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    from responses_api_agents.browsecomp_agent.app import BrowsecompAgent

    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.get_first_server_config_dict",
        lambda *args: {"base_url": "http://10.0.0.5:9001/v1", "api_key": "unused"},
    )
    client, server_client, _ = _agent([], monkeypatch, save_model_call_using_vllm_tokenize_endpoint=True)
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.resolve_provider_config", lambda *args: {"e2b": {}})
    monkeypatch.setattr(BrowsecompAgent, "resolve_model_base_url", lambda *args: "http://10.0.0.5:9001/v1")
    server_client._resolve_base_url.return_value = "http://127.0.0.1:8080"
    reply = client.post("/v1/agent_sessions", json=_seed(tool_accesses=[]).model_dump(mode="json"))
    assert reply.status_code == 422
    assert "127.0.0.1:8080/v1/browsecomp/token_count" in reply.text


@pytest.mark.parametrize(("provider", "status"), [("e2b", 422), ("docker", 200)])
def test_a_loopback_tool_grant_url_refuses_the_seed_except_under_docker(
    monkeypatch: pytest.MonkeyPatch, provider: str, status: int
) -> None:
    # A Docker container on the host network reaches the host's loopback, so the shared check only warns there.
    from responses_api_agents.browsecomp_agent.app import BrowsecompAgent

    client, _, _ = _agent([], monkeypatch)
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.resolve_provider_config", lambda *args: {provider: {}}
    )
    monkeypatch.setattr(BrowsecompAgent, "resolve_model_base_url", lambda *args: "http://10.0.0.5:9001/v1")
    grant = GRANT.model_copy(update={"base_url": "http://127.0.0.1:9000"})
    reply = client.post("/v1/agent_sessions", json=_seed(tool_accesses=[grant]).model_dump(mode="json"))
    assert reply.status_code == status, reply.text
    if status == 422:
        assert "127.0.0.1:9000" in reply.text


def _unknown_provider(*args):
    raise ValueError("Unknown sandbox provider 'dokcer'")


@pytest.mark.parametrize(
    ("config", "create_provider"),
    [({"sandbox_config": {"ttl": 7200, "workdir": "/sandbox"}}, None), ({}, _unknown_provider)],
    ids=["unknown sandbox_config key", "unknown provider"],
)
def test_an_unusable_sandbox_configuration_refuses_the_seed(
    monkeypatch: pytest.MonkeyPatch, config: dict, create_provider
) -> None:
    # Every seed fails the same way, so a retryable 500 would only repeat the failure for each episode.
    client, _, _ = _agent([], monkeypatch, **config)
    if create_provider is not None:
        monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.create_provider", create_provider)
    reply = client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json"))
    assert reply.status_code == 422
    assert "sandbox provider is not usable" in reply.text


async def test_the_loop_deadline_counts_from_the_runners_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The supervisor counts from launch, so the endpoint checks must spend the loop's budget, not its grace."""
    from responses_api_agents.browsecomp_agent import sandbox_runner

    calls = []

    async def check_reachable(urls):
        calls.append(("check", time.monotonic()))

    async def run_session_loop(self, request, response, body, state, *, started_at=None):
        calls.append(("loop", started_at))
        return NeMoGymResponse.model_validate(_model_response(_message("Paris")))

    monkeypatch.setattr(sandbox_runner, "check_reachable", check_reachable)
    monkeypatch.setattr(sandbox_runner.SandboxBrowsecompAgent, "_run_session_loop", run_session_loop)
    monkeypatch.setattr(sandbox_runner, "get_global_aiohttp_client", lambda: MagicMock(close=AsyncMock()))
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    payload = {
        "config": {
            "name": "browsecomp",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "rollout_timeout_s": 600,
        },
        "seed": _seed(tool_accesses=[]).model_dump(mode="json"),
        "tool_access": None,
        "body": QUESTION,
        "model_url": f"http://model:9001/ng-rollout/{EPISODE.capture_key}/v1",
    }
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    await sandbox_runner.run(input_path, output_path)
    assert [name for name, _ in calls] == ["check", "loop"]
    assert calls[1][1] is not None and calls[1][1] <= calls[0][1]
    assert "response" in json.loads(output_path.read_text())


@pytest.mark.parametrize("lost", ["model", "tool", "token count"])
async def test_an_endpoint_lost_mid_episode_fails_the_activation_instead_of_timing_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: str
) -> None:
    """The loop's calls stop resending a refused connection, so an outage leaves an error receipt (retryable)
    instead of spinning until the rollout deadline and scoring as a timed-out zero."""
    from responses_api_agents.browsecomp_agent import app, sandbox_runner

    # Production rides out about 90 s of outage; two resends keep this test fast.
    monkeypatch.setattr(app, "_SANDBOX_LOOP_CONNECTION_RETRIES", 2)
    monkeypatch.setattr(sandbox_runner, "_SANDBOX_LOOP_CONNECTION_RETRIES", 2)
    runners: dict[str, web.AppRunner] = {}
    origins: dict[str, str] = {}

    async def model(request: web.Request) -> web.Response:
        if lost == "tool":
            await runners["tool"].cleanup()
        return web.json_response(_model_response(_tool_call("ls")))

    async def tool(request: web.Request) -> web.Response:
        if lost != "tool":
            await runners[lost].cleanup()
        return web.json_response({"results_string": "pages"})

    async def token_count(request: web.Request) -> web.Response:
        return web.json_response({"count": 1})

    for name, path, handler in (
        ("model", ROUTE, model),
        ("tool", "/tools/bash_command", tool),
        ("token count", "/token_count", token_count),
    ):
        application = web.Application()
        application.router.add_post(path, handler)
        runners[name] = web.AppRunner(application)
        await runners[name].setup()
        site = web.TCPSite(runners[name], "127.0.0.1", 0)
        await site.start()
        origins[name] = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", lambda *args: None)
    monkeypatch.setattr(sandbox_runner, "get_global_aiohttp_client", lambda: MagicMock(close=AsyncMock()))
    monkeypatch.setenv("NEMO_GYM_CONFIG_DICT", "{}")
    payload = {
        "config": {
            "name": "browsecomp",
            "host": "0.0.0.0",
            "port": 1,
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy"},
            "nudge_steps": False,
            "rollout_timeout_s": 60,
            "save_model_call_using_vllm_tokenize_endpoint": True,
        },
        "seed": _seed().model_dump(mode="json"),
        "tool_access": GRANT.model_copy(update={"base_url": f"{origins['tool']}/tools"}).model_dump(mode="json"),
        "body": QUESTION,
        "model_url": f"{origins['model']}/ng-rollout/{EPISODE.capture_key}/v1",
        "token_count_url": f"{origins['token count']}/token_count",
    }
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(payload))
    try:
        await asyncio.wait_for(sandbox_runner.run(input_path, output_path), timeout=30)
    finally:
        for runner in runners.values():
            await runner.cleanup()
    receipt = json.loads(output_path.read_text())
    assert "response" not in receipt
    assert receipt["error"].startswith(("ClientConnectorError", "ServerDisconnectedError")), receipt["error"]


def test_the_sandbox_activation_clamps_its_deadline_and_fails_on_an_error_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_gym.agents import SandboxSession
    from responses_api_agents.browsecomp_agent.app import BrowsecompAgent

    activation = BrowsecompAgent._sandbox_responses  # _agent() runs the loop on the host instead.
    payloads, timeouts = [], []
    receipts = [
        {"response": _model_response(_message("Paris")), "resources_cookies": {"harness": "next-cookie"}},
        {"error": "ConnectionError: The sandbox cannot connect to model:9001", "resources_cookies": {}},
    ]

    async def stage_activation(self, payload):
        payloads.append(payload)

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        timeouts.append(timeout)
        return receipts.pop(0)

    def activate():
        client, _, _ = _agent([], monkeypatch, sandbox_runner_timeout_seconds=900)
        monkeypatch.setattr(BrowsecompAgent, "_sandbox_responses", activation)
        monkeypatch.setattr(BrowsecompSandboxSession, "stage_activation", stage_activation)
        monkeypatch.setattr(SandboxSession, "execute", execute)
        assert client.post("/v1/agent_sessions", json=_seed(agent_timeout_sec=120).model_dump(mode="json")).is_success
        return client.post(ROUTE, json=QUESTION)

    reply = activate()
    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][0]["content"][0]["text"] == "Paris"
    # The task's own limit wins over the runner's, and the supervisor gets its grace on top.
    assert payloads[0]["config"]["rollout_timeout_s"] == 120 and timeouts == [150]
    assert payloads[0]["token_count_url"] is None
    assert payloads[0]["seed"]["tool_accesses"] == [] and payloads[0]["seed"]["sandbox_access"] is None

    with pytest.raises(NgFailureEvent) as failed:  # The served app answers it with 500.
        activate()
    assert "BrowseComp sandbox failed: ConnectionError" in str(failed.value.__cause__)


@pytest.mark.parametrize(
    "error",
    [
        SandboxIsolationError("BrowseComp requires an isolated PID namespace"),
        UnsupportedSandboxPlatformError("No sandbox Python runtime is available for Darwin on arm64"),
    ],
    ids=["no process isolation", "unsupported platform"],
)
def test_a_sandbox_no_retry_can_fix_refuses_the_seed_after_stopping_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    from nemo_gym.agents import SandboxSession

    client, _, _ = _agent([], monkeypatch)
    monkeypatch.setattr(BrowsecompSandboxSession, "install_runtime", AsyncMock(side_effect=error))
    assert str(error) in check_agent_seed_rejected(client.app, _seed())
    SandboxSession.close.assert_awaited_once()


def _receipt_session(downloaded: bytes | Exception, log: str = "runner log tail") -> BrowsecompSandboxSession:
    async def download(remote_path: str, local_path: Path) -> None:
        if isinstance(downloaded, Exception):
            raise downloaded
        Path(local_path).write_bytes(downloaded)

    session = MagicMock(session_dir="/private/session")
    session.sandbox.download = download
    session.read_output_log = AsyncMock(return_value=log)
    return BrowsecompSandboxSession(request=_seed(), session=session, tool_access=None, resources_cookies={})


@pytest.mark.parametrize(
    ("downloaded", "raised", "message"),
    [
        (FileNotFoundError("output.json"), RuntimeError, "did not produce a receipt: runner log tail"),
        (b'{"response": {"id": "resp", "out', RuntimeError, "did not produce a receipt: runner log tail"),
        (b"\xff\xfe not json", RuntimeError, "did not produce a receipt: runner log tail"),
        (b'["not", "an", "object"]', TypeError, "must be an object"),
    ],
    ids=["missing", "truncated", "garbled", "not an object"],
)
async def test_an_unusable_receipt_fails_the_collection_with_the_runner_log(downloaded, raised, message) -> None:
    with pytest.raises(raised, match=message):
        await _receipt_session(downloaded).collect()


async def test_a_well_formed_receipt_is_returned_as_is() -> None:
    receipt = {"response": {"id": "resp"}, "resources_cookies": {"harness": "c"}}
    assert await _receipt_session(json.dumps(receipt).encode()).collect() == receipt


async def test_a_missing_receipt_reports_only_the_end_of_a_long_runner_log() -> None:
    log = "early step\n" * 1000 + "Traceback: the runner's last words"
    with pytest.raises(RuntimeError) as failed:
        await _receipt_session(FileNotFoundError("output.json"), log=log).collect()
    assert str(failed.value) == f"BrowseComp runner did not produce a receipt: {log[-4000:]}"
    assert str(failed.value).endswith("the runner's last words")


@pytest.mark.parametrize("field", ["sandbox_install_timeout_seconds", "sandbox_runner_timeout_seconds"])
def test_sandbox_timeouts_must_be_finite(field: str) -> None:
    from pydantic import ValidationError

    from responses_api_agents.browsecomp_agent.app import BrowsecompAgentConfig

    with pytest.raises(ValidationError, match="finite number"):
        BrowsecompAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="browsecomp_agent",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            **{field: float("inf")},
        )
