# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants in Claude Code sandbox sessions: seeding, the session's --mcp-config, and required servers."""

import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.agents import HARNESS_RUNNER_FILE, HARNESS_RUNNER_SCRIPT, check_agent_seed_rejected
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.claude_code_agent import app
from responses_api_agents.claude_code_agent import sandbox as adapter
from responses_api_agents.claude_code_agent.parsing import parse_stream_json
from responses_api_agents.claude_code_agent.tests.test_sandbox import agent, result, seed


TOKEN_HEADER = {"X-NeMo-Gym-Session-Token": "episode-token"}


def mcp_grant(name="magic", *, required=True, url="http://mcp-sidecar:8000/mcp", headers=None):
    return {
        "kind": "mcp",
        "name": name,
        "required": required,
        "connection": {"url": url, "headers": headers or {}},
    }


def direct_http_grant(name="resources", *, required=True):
    return {"kind": "direct_http", "name": name, "required": required, "base_url": "http://resources:8080"}


def init_event(**statuses):
    return {
        "type": "system",
        "subtype": "init",
        "tools": ["Bash"],
        "mcp_servers": [{"name": name, "status": status, "source": "dynamic"} for name, status in statuses.items()],
    }


ANSWER = {"type": "assistant", "message": {"id": "served", "content": [{"type": "text", "text": "pineapple"}]}}
SUCCESS = {"type": "result", "subtype": "success", "is_error": False, "usage": {"input_tokens": 3, "output_tokens": 1}}


def events(*items):
    return "\n".join(json.dumps(item) for item in items) + "\n"


def owned_seed_mocks(monkeypatch, provider_name="opensandbox"):
    remote = SimpleNamespace(exec=AsyncMock(return_value=result()), upload=AsyncMock(), start=AsyncMock())
    constructor = MagicMock(return_value=remote)
    constructor.connect = AsyncMock(return_value=remote)
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    provider = SimpleNamespace(name=provider_name, aclose=AsyncMock())
    monkeypatch.setattr(app, "create_provider", lambda config: provider)
    monkeypatch.setattr(app, "install_sandbox_runtime", AsyncMock())
    return remote, constructor, provider


# Seeding


def test_a_required_direct_http_grant_is_refused_with_a_final_422(monkeypatch):
    connect = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    instance = agent()
    body = seed(tool_accesses=[direct_http_grant(), mcp_grant()])
    detail = check_agent_seed_rejected(instance.setup_webserver(), body)
    assert (
        detail == "Claude Code in the sandbox supports only MCP tool grants; required grants it cannot use: resources"
    )
    assert not instance._session_records
    connect.assert_not_awaited()


@pytest.mark.parametrize(
    ("static_names", "grant_names", "colliding"),
    [
        (["magic"], ["magic", "other"], "magic"),
        # Claude Code names both servers' tools mcp__my_server__<tool>, and one server's tools hide the other's.
        (["my server"], ["my.server"], "my.server"),
        ([], ["my.server", "my_server", "other"], "my.server, my_server"),
    ],
    ids=["static_name", "static_tool_names", "grant_tool_names"],
)
@pytest.mark.parametrize("required", [True, False])
def test_grants_whose_tool_names_collide_are_refused(
    monkeypatch, tmp_path, static_names, grant_names, colliding, required
):
    static = tmp_path / "mcp.json"
    static.write_text(
        json.dumps({"mcpServers": {name: {"type": "stdio", "command": "server"} for name in static_names}})
    )
    connect = AsyncMock()
    monkeypatch.setattr(app.AsyncSandbox, "connect", connect)
    instance = agent(mcp_config=str(static))
    body = seed(tool_accesses=[mcp_grant(name, required=required) for name in grant_names])
    detail = check_agent_seed_rejected(instance.setup_webserver(), body)
    assert detail == (
        "MCP tool grants collide with another grant or an mcp_config server in Claude Code's tool names "
        "(mcp__<server>__<tool>, where characters other than letters, digits, '_' and '-' become '_'): " + colliding
    )
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_required_and_optional_mcp_grants_are_accepted(monkeypatch):
    _, constructor, _ = owned_seed_mocks(monkeypatch)
    body = seed(tool_accesses=[mcp_grant(), mcp_grant("extra", required=False), direct_http_grant(required=False)])
    state = await agent(sandbox_provider="sandbox")._seed_agent_session_state(body)
    constructor.connect.assert_awaited_once()
    assert state.request == body


@pytest.mark.asyncio
async def test_model_and_tools_opens_the_host_of_each_mcp_grant(monkeypatch):
    remote, _, _ = owned_seed_mocks(monkeypatch)
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app"},
        anthropic_base_url="http://model.example:9000",
        network_access="model_and_tools",
    )
    body = seed(
        borrowed=False,
        tool_accesses=[
            mcp_grant(url="http://tools.example:8000/mcp"),
            mcp_grant("extra", required=False, url="https://extra.example/mcp"),
            # A direct HTTP grant is not written into Claude Code's configuration, so it opens no host.
            direct_http_grant(required=False),
        ],
    )
    await instance._seed_agent_session_state(body)
    assert remote.start.await_args.args[0].provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [
            {"action": "allow", "target": "extra.example"},
            {"action": "allow", "target": "model.example"},
            {"action": "allow", "target": "tools.example"},
        ],
    }


def test_model_only_refuses_a_required_mcp_grant_before_the_sandbox_exists(monkeypatch):
    _, constructor, provider = owned_seed_mocks(monkeypatch)
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app"},
        anthropic_base_url="http://model.example:9000",
        network_access="model_only",
    )
    detail = check_agent_seed_rejected(
        instance.setup_webserver(), seed(borrowed=False, tool_accesses=[mcp_grant(url="http://tools.example/mcp")])
    )
    assert "model_only blocks the hosts of the required tool grants (http://tools.example/mcp)" in detail
    constructor.assert_not_called()
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_only_keeps_an_optional_mcp_grant_out_of_the_allowlist(monkeypatch):
    remote, _, _ = owned_seed_mocks(monkeypatch)
    instance = agent(
        sandbox_provider="sandbox",
        sandbox_config={"workdir": "/app"},
        anthropic_base_url="http://model.example:9000",
        network_access="model_only",
    )
    body = seed(borrowed=False, tool_accesses=[mcp_grant(required=False, url="http://tools.example/mcp")])
    await instance._seed_agent_session_state(body)
    assert remote.start.await_args.args[0].provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.example"}],
    }


# Activation


def activation_state(tool_accesses, *, timed_out=False, return_code=0):
    return adapter.ClaudeCodeSandboxSession(
        request=seed(tool_accesses=tool_accesses),
        runtime="/runtime",
        session=SimpleNamespace(
            session_dir="/tmp/session",
            workdir="/app",
            artifacts=None,
            cleanup={"timed_out": timed_out, "return_code": return_code, "cleanup_confirmed": True},
        ),
    )


def mock_execute(monkeypatch, raw):
    """Stand in for the sandbox run, which leaves ``raw`` as the CLI's stdout; returns the mock."""
    execute = AsyncMock(return_value=adapter.ClaudeCodeArtifacts(raw))
    monkeypatch.setattr(adapter.ClaudeCodeSandboxSession, "execute", execute)
    return execute


async def respond(instance, state):
    body = NeMoGymResponseCreateParamsNonStreaming(input="What is the magic word?")
    return await instance._sandbox_response(state, body, prompt="What is the magic word?", system=None)


def gap_codes(state):
    return [(gap.code, gap.detail) for gap in state.observations.gaps]


@pytest.mark.asyncio
async def test_grants_and_the_static_config_form_the_session_mcp_config(monkeypatch, tmp_path):
    static = tmp_path / "mcp.json"
    static.write_text(json.dumps({"mcpServers": {"static": {"type": "stdio", "command": "static-server"}}}))
    instance = agent(mcp_config=str(static))
    state = activation_state(
        [mcp_grant(headers=TOKEN_HEADER), mcp_grant("extra", required=False, url="https://extra.example/mcp")]
    )
    execute = mock_execute(monkeypatch, events(init_event(magic="connected", extra="connected"), ANSWER, SUCCESS))
    response = await respond(instance, state)
    payload = execute.await_args.args[0]
    assert payload["mcp_config"] == {
        "mcpServers": {
            "static": {"type": "stdio", "command": "static-server"},
            "magic": {"type": "http", "url": "http://mcp-sidecar:8000/mcp", "headers": TOKEN_HEADER},
            "extra": {"type": "http", "url": "https://extra.example/mcp"},
        }
    }
    command = payload["command"]
    assert command[0] == "/runtime/claude"
    # Explicit, so it applies alongside the default --bare.
    assert "--bare" in command
    assert command[command.index("--mcp-config") + 1] == "/tmp/session/mcp.json"
    assert payload["required_mcp_servers"] == ["magic"]
    assert response.status == "completed"
    assert response.output[-1].content[0].text == "pineapple"
    # The cached static config is not changed by a session's grants.
    assert set(instance._get_static_mcp_config()["mcpServers"]) == {"static"}


@pytest.mark.asyncio
async def test_a_session_without_grants_or_static_config_passes_no_mcp_config(monkeypatch):
    state = activation_state([])
    execute = mock_execute(monkeypatch, events(init_event(), ANSWER, SUCCESS))
    await respond(agent(), state)
    payload = execute.await_args.args[0]
    assert payload["mcp_config"] is None
    assert payload["required_mcp_servers"] == []
    assert "--mcp-config" not in payload["command"]
    assert not [code for code, _ in gap_codes(state) if code.startswith("mcp_")]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "pending", "needs-auth"])
async def test_a_required_server_that_did_not_connect_fails_the_turn(monkeypatch, status):
    state = activation_state([mcp_grant(headers=TOKEN_HEADER)], return_code=143)
    # The model answered without the tool: that answer must not be graded.
    mock_execute(monkeypatch, events(init_event(magic=status), ANSWER, SUCCESS))
    with pytest.raises(HTTPException) as error:
        await respond(agent(), state)
    message = f"Required MCP servers did not connect before Claude Code's first model request: magic ({status})"
    assert error.value.status_code == 502
    assert error.value.detail == message
    assert ("agent_startup_failed", message) in gap_codes(state)
    sandbox = next(record for record in state.observations.records if record.kind == "sandbox")
    assert sandbox.outcome == "failed"
    assert sandbox.error_type == "required_mcp_server_unavailable"
    assert "episode-token" not in json.dumps(state.observations.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_a_required_server_missing_from_the_init_event_fails_the_turn(monkeypatch):
    state = activation_state([mcp_grant()])
    mock_execute(monkeypatch, events(init_event(), ANSWER, SUCCESS))
    with pytest.raises(HTTPException, match=r"magic \(not listed\)"):
        await respond(agent(), state)


@pytest.mark.asyncio
async def test_a_required_server_without_an_init_event_fails_the_turn(monkeypatch):
    state = activation_state([mcp_grant()], return_code=1)
    mock_execute(monkeypatch, events(ANSWER))
    with pytest.raises(HTTPException) as error:
        await respond(agent(), state)
    assert error.value.status_code == 502
    assert error.value.detail == (
        "Claude Code ended failed (process_exit_1) without an init event, so these required MCP servers were never "
        "confirmed: magic"
    )
    assert "mcp_status_unavailable" in [code for code, _ in gap_codes(state)]


@pytest.mark.asyncio
async def test_an_optional_server_that_did_not_connect_is_a_gap(monkeypatch):
    state = activation_state([mcp_grant(), mcp_grant("extra", required=False)])
    mock_execute(monkeypatch, events(init_event(magic="connected", extra="failed"), ANSWER, SUCCESS))
    response = await respond(agent(), state)
    assert response.status == "completed"
    assert ("mcp_server_skipped", "extra: failed") in gap_codes(state)
    assert "agent_startup_failed" not in [code for code, _ in gap_codes(state)]


@pytest.mark.asyncio
async def test_a_timeout_keeps_its_cause(monkeypatch):
    # The run hit its time limit before Claude Code wrote the init event: it is graded as a timeout.
    state = activation_state([mcp_grant()], timed_out=True, return_code=None)
    mock_execute(monkeypatch, events())
    response = await respond(agent(), state)
    assert response.status == "incomplete"
    assert "mcp_status_unavailable" in [code for code, _ in gap_codes(state)]
    assert "agent_startup_failed" not in [code for code, _ in gap_codes(state)]


def test_parsing_keeps_the_statuses_of_the_first_init_event():
    raw = events(
        {"type": "system", "subtype": "ui_invalidate"},
        init_event(magic="connected", extra="failed"),
        init_event(magic="failed"),
        ANSWER,
        SUCCESS,
    )
    _, metadata = parse_stream_json(raw)
    assert metadata["mcp_servers"] == {"magic": "connected", "extra": "failed"}
    _, metadata = parse_stream_json(events(ANSWER, SUCCESS))
    assert "mcp_servers" not in metadata


# The init-event check inside the sandbox, run with the shared bash runner as the sandbox runs it


def staged_command(tmp_path: Path, code: str, required: list[str]) -> list[str]:
    async def upload(source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    shutil.copyfile(HARNESS_RUNNER_SCRIPT, tmp_path / HARNESS_RUNNER_FILE)
    child = tmp_path / "child.py"
    child.write_text(code)
    state = adapter.ClaudeCodeSandboxSession(
        request=seed(),
        runtime="/runtime",
        session=SimpleNamespace(session_dir=str(tmp_path), sandbox=SimpleNamespace(upload=upload)),
    )
    payload = {
        "command": [sys.executable, str(child)],
        "cwd": str(tmp_path),
        "env": {},
        "settings": {},
        "mcp_config": None,
        "required_mcp_servers": required,
        "skills_path": None,
    }
    return asyncio.run(state.stage_activation(payload)).argv


def stub_cli(*chunks: str, then: str = "time.sleep(30)") -> str:
    """A CLI that writes ``chunks`` to stdout 0.2 s apart, then runs ``then``."""
    return (
        "import os, sys, time\n"
        + "".join(f"os.write(1, {chunk.encode()!r})\ntime.sleep(0.2)\n" for chunk in chunks)
        + 'os.write(1, b\'{"type":"after_init"}\\n\')\n'
        + then
        + "\n"
    )


def compact(event: dict) -> str:
    return json.dumps(event, separators=(",", ":")) + "\n"


def run_staged(command: list[str]) -> tuple[subprocess.CompletedProcess, float]:
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, timeout=20)
    return completed, time.monotonic() - started


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


@needs_bash
def test_the_cli_runs_under_the_check_only_with_required_grants(tmp_path):
    command = staged_command(tmp_path, "", ["magic", "my.server"])
    params = (tmp_path / "params.sh").read_text()
    assert command == ["bash", f"{tmp_path}/{HARNESS_RUNNER_FILE}", f"{tmp_path}/params.sh"]
    assert (tmp_path / "require_mcp_servers.sh").read_bytes() == adapter.REQUIRE_MCP_SERVERS_SCRIPT.read_bytes()
    assert f"command=(bash {tmp_path}/require_mcp_servers.sh {tmp_path}/events.jsonl magic my.server -- " in params
    other = tmp_path / "other"
    other.mkdir()
    staged_command(other, "", [])
    assert "require_mcp_servers.sh" not in (other / "params.sh").read_text()
    assert not (other / "require_mcp_servers.sh").exists()


@needs_bash
@pytest.mark.parametrize("pieces", [1, 2], ids=["whole_line", "line_in_two_writes"])
def test_the_check_stops_the_cli_when_a_required_server_did_not_connect(tmp_path, pieces):
    line = compact(init_event(magic="connected", search="failed"))
    chunks = [line] if pieces == 1 else [line[:40], line[40:]]
    completed, elapsed = run_staged(staged_command(tmp_path, stub_cli(*chunks), ["magic", "search"]))
    assert completed.returncode != 0
    assert elapsed < 10
    raw = (tmp_path / "events.jsonl").read_text()
    _, metadata = parse_stream_json(raw)
    assert metadata["mcp_servers"] == {"magic": "connected", "search": "failed"}
    assert "Required MCP server search is failed; stopping Claude Code" in (tmp_path / "stderr.log").read_text()


@needs_bash
@pytest.mark.parametrize(
    "init",
    [
        init_event(magic="connected"),
        # A name with a dot matches only itself, not another server whose name has any other character there.
        init_event(my_server="failed", **{"my.server": "connected"}),
    ],
    ids=["connected", "dotted_name"],
)
def test_the_check_lets_a_run_with_its_required_servers_finish(tmp_path, init):
    required = [server["name"] for server in init["mcp_servers"] if server["status"] == "connected"]
    completed, _ = run_staged(staged_command(tmp_path, stub_cli(compact(init), then="sys.exit(7)"), required))
    assert "require_mcp_servers.sh" in (tmp_path / "params.sh").read_text()
    assert completed.returncode == 7, completed.stderr
    assert '{"type":"after_init"}' in (tmp_path / "events.jsonl").read_text()
    assert "stopping Claude Code" not in (tmp_path / "stderr.log").read_text()


@needs_bash
def test_the_check_stops_nothing_on_an_init_event_it_does_not_recognize(tmp_path):
    # Spaced JSON is not the CLI's format; the agent server still checks the statuses after the run.
    spaced = json.dumps(init_event(magic="failed")) + "\n"
    completed, _ = run_staged(staged_command(tmp_path, stub_cli(spaced, then="sys.exit(0)"), ["magic"]))
    assert "require_mcp_servers.sh" in (tmp_path / "params.sh").read_text()
    assert completed.returncode == 0, completed.stderr
    _, metadata = parse_stream_json((tmp_path / "events.jsonl").read_text())
    assert metadata["mcp_servers"] == {"magic": "failed"}
