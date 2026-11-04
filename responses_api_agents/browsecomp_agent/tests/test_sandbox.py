# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The process/HTTP boundary, source bundle and sandbox session lifecycle."""

import asyncio
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from responses_api_agents.browsecomp_agent.sandbox import BrowsecompSandboxSession, _write_sources
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
