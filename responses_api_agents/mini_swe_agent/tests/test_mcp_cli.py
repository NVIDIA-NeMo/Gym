# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task-local MCP CLI that mini-SWE's bash tool calls.

The CLI runs with the sandbox runtime's MCP 2 SDK. Tests that start real MCP servers need that runtime's interpreter in
``MINI_SWE_RUNTIME_PYTHON`` (see test_sandbox_runner.py).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.mini_swe_agent import mcp_cli


RUNTIME_PYTHON = os.environ.get("MINI_SWE_RUNTIME_PYTHON")
needs_runtime = pytest.mark.skipif(
    not RUNTIME_PYTHON, reason="set MINI_SWE_RUNTIME_PYTHON to a mini-SWE 2.4.6 runtime"
)


class FakeSession:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def short_dir():
    """A directory short enough for a Unix socket path on every platform (macOS caps it at 104 bytes)."""
    path = Path(tempfile.mkdtemp(prefix="msw", dir="/tmp"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def calls_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setattr(mcp_cli, "CALLS_FILE", path)
    return path


async def test_a_call_is_recorded_with_the_bash_call_that_made_it(calls_file) -> None:
    result = SimpleNamespace(
        is_error=False, model_dump=lambda **kwargs: {"content": [{"type": "text", "text": "sunny"}], "isError": False}
    )
    session = FakeSession(result=result)
    message = {
        "operation": "call",
        "server": "task_tools",
        "tool": "get_weather",
        "arguments": {"city": "SF"},
        "tool_call_id": "call_3",
    }

    answer = await mcp_cli.dispatch({"task_tools": {"session": session}}, message)

    assert answer == {"content": [{"type": "text", "text": "sunny"}], "isError": False}
    assert session.calls == [("get_weather", {"city": "SF"})]
    (record,) = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert (record["tool_call_id"], record["server"], record["tool"], record["is_error"]) == (
        "call_3",
        "task_tools",
        "get_weather",
        False,
    )
    assert json.loads(record["output"]) == answer and record["duration_ms"] >= 0


async def test_a_failed_or_unknown_server_call_is_recorded_as_an_error(calls_file) -> None:
    message = {"operation": "call", "server": "gone", "tool": "t", "arguments": {}, "tool_call_id": None}

    with pytest.raises(KeyError, match="not available: ConnectError"):
        await mcp_cli.dispatch({"gone": {"error": "ConnectError"}}, message)
    session = FakeSession(error=TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await mcp_cli.dispatch({"s": {"session": session}}, {**message, "server": "s"})

    records = [json.loads(line) for line in calls_file.read_text().splitlines()]
    assert [(r["server"], r["is_error"], r["output"]) for r in records] == [
        ("gone", True, "KeyError: \"MCP server 'gone' is not available: ConnectError\""),
        ("s", True, "TimeoutError: slow"),
    ]


async def test_list_reports_each_servers_tools_or_why_it_did_not_connect(monkeypatch) -> None:
    async def list_tools(session):
        return [{"name": "get_weather"}]

    monkeypatch.setattr(mcp_cli, "list_tools", list_tools)

    connections = {"ok": {"session": object()}, "down": {"error": "ConnectError: refused"}}
    listing = await mcp_cli.dispatch(connections, {"operation": "list"})

    assert listing == {"ok": {"tools": [{"name": "get_weather"}]}, "down": {"error": "ConnectError: refused"}}


async def test_a_server_whose_listing_fails_is_reported_without_failing_the_others(monkeypatch) -> None:
    broken = object()

    async def list_tools(session):
        if session is broken:
            raise ExceptionGroup("listing", [ConnectionResetError("peer closed")])
        return [{"name": "get_weather"}]

    monkeypatch.setattr(mcp_cli, "list_tools", list_tools)

    listing = await mcp_cli.dispatch(
        {"ok": {"session": object()}, "flaky": {"session": broken}}, {"operation": "list"}
    )

    # The runner then applies the required-server policy to it, as to a server that did not connect.
    assert listing == {
        "ok": {"tools": [{"name": "get_weather"}]},
        "flaky": {"error": "ConnectionResetError: peer closed"},
    }


def test_a_call_without_json_arguments_shows_the_usage(capsys) -> None:
    assert mcp_cli.main(["mcp_cli.py", "call", "task_tools"]) == 2
    assert "call SERVER TOOL 'JSON_ARGUMENTS'" in capsys.readouterr().err


def test_a_call_without_a_running_server_fails_with_a_message(short_dir, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mcp_cli, "SOCKET_FILE", short_dir / "mcp.sock")

    assert mcp_cli.main(["mcp_cli.py", "call", "task_tools", "get_weather", "{}"]) == 1
    assert "FileNotFoundError" in capsys.readouterr().err


SERVER = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("weather")


@server.tool()
def get_weather(city: str) -> str:
    \"\"\"Today's weather.\"\"\"
    return f"sunny in {city}"


server.run("stdio")
"""


@needs_runtime
def test_the_cli_keeps_real_sessions_open_across_calls(short_dir) -> None:
    tmp_path = short_dir
    shutil.copy(mcp_cli.__file__, tmp_path / "mcp_cli.py")
    (tmp_path / "server.py").write_text(SERVER)
    servers = [
        {"name": "weather", "transport": "stdio", "command": RUNTIME_PYTHON, "args": [str(tmp_path / "server.py")]},
        # Nothing listens here: the CLI reports it and keeps serving the others.
        {"name": "down", "transport": "streamable_http", "url": "http://127.0.0.1:9/mcp", "headers": {"A": "t"}},
    ]
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers))
    cli = [RUNTIME_PYTHON, "-E", "-s", str(tmp_path / "mcp_cli.py")]
    serve = subprocess.Popen([*cli, "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "mcp.sock").exists():
            assert serve.poll() is None, serve.stdout.read().decode()
            assert time.monotonic() < deadline
            time.sleep(0.1)
        listed = subprocess.run([*cli, "list"], capture_output=True, text=True, timeout=60, check=True)
        listing = json.loads(listed.stdout)
        assert [tool["name"] for tool in listing["weather"]["tools"]] == ["get_weather"]
        assert "inputSchema" in listing["weather"]["tools"][0]
        assert "error" in listing["down"]

        environment = {**os.environ, mcp_cli.TOOL_CALL_ID_VARIABLE: "call_9"}
        for _ in range(2):
            called = subprocess.run(
                [*cli, "call", "weather", "get_weather", '{"city": "SF"}'],
                capture_output=True,
                text=True,
                timeout=60,
                env=environment,
                check=True,
            )
        assert json.loads(called.stdout)["content"][0]["text"] == "sunny in SF"
    finally:
        serve.terminate()
        serve.wait(timeout=10)
    records = [json.loads(line) for line in (tmp_path / "mcp_calls.jsonl").read_text().splitlines()]
    assert [(r["tool_call_id"], r["tool"], r["is_error"]) for r in records] == [("call_9", "get_weather", False)] * 2


AUTHENTICATED_SERVER = """
import asyncio, json, socket, sys

import uvicorn
from mcp.server.mcpserver import MCPServer

server = MCPServer("weather")


@server.tool()
def get_weather(city: str) -> str:
    \"\"\"Today's weather.\"\"\"
    return f"sunny in {city}"


def guarded(app):
    async def wrapped(scope, receive, send):
        if scope["type"] == "http" and dict(scope["headers"]).get(b"authorization") != b"Bearer t":
            await send({"type": "http.response.start", "status": 401, "headers": []})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await app(scope, receive, send)

    return wrapped


async def main(port_file):
    ports, serving = {}, []
    for name, app in {"streamable_http": server.streamable_http_app(), "sse": server.sse_app()}.items():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        ports[name] = sock.getsockname()[1]
        serving.append(uvicorn.Server(uvicorn.Config(guarded(app), log_level="warning")).serve(sockets=[sock]))
    open(port_file, "w").write(json.dumps(ports))
    await asyncio.gather(*serving)


asyncio.run(main(sys.argv[1]))
"""


@needs_runtime
def test_the_grant_headers_authenticate_discovery_and_calls_over_both_http_transports(short_dir) -> None:
    tmp_path = short_dir
    shutil.copy(mcp_cli.__file__, tmp_path / "mcp_cli.py")
    (tmp_path / "server.py").write_text(AUTHENTICATED_SERVER)
    host = subprocess.Popen([RUNTIME_PYTHON, str(tmp_path / "server.py"), str(tmp_path / "ports.json")])
    cli = [RUNTIME_PYTHON, "-E", "-s", str(tmp_path / "mcp_cli.py")]
    serve = None
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "ports.json").exists():
            assert host.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
        ports = json.loads((tmp_path / "ports.json").read_text())
        token = {"Authorization": "Bearer t"}
        http_url = f"http://127.0.0.1:{ports['streamable_http']}/mcp"
        servers = [
            {"name": "http", "transport": "streamable_http", "url": http_url, "headers": token},
            # Harbor's default transport for a task's own server.
            {"name": "sse", "transport": "sse", "url": f"http://127.0.0.1:{ports['sse']}/sse", "headers": token},
            {"name": "no_token", "transport": "streamable_http", "url": http_url},
        ]
        (tmp_path / "mcp_servers.json").write_text(json.dumps(servers))
        serve = subprocess.Popen([*cli, "serve"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while not (tmp_path / "mcp.sock").exists():
            assert serve.poll() is None, serve.stdout.read().decode()
            assert time.monotonic() < deadline
            time.sleep(0.1)

        listed = subprocess.run([*cli, "list"], capture_output=True, text=True, timeout=60, check=True)
        listing = json.loads(listed.stdout)

        assert [tool["name"] for tool in listing["http"]["tools"]] == ["get_weather"]
        assert [tool["name"] for tool in listing["sse"]["tools"]] == ["get_weather"]
        # Without the grant's headers the server refuses the session, and the CLI reports it instead of its tools.
        assert set(listing["no_token"]) == {"error"}
        for name in ("http", "sse"):
            called = subprocess.run(
                [*cli, "call", name, "get_weather", '{"city": "SF"}'],
                capture_output=True,
                text=True,
                timeout=60,
                env={**os.environ, mcp_cli.TOOL_CALL_ID_VARIABLE: f"call_{name}"},
                check=True,
            )
            assert json.loads(called.stdout)["content"][0]["text"] == "sunny in SF"
    finally:
        for process in (serve, host):
            if process is not None:
                process.terminate()
                process.wait(timeout=10)
    records = [json.loads(line) for line in (tmp_path / "mcp_calls.jsonl").read_text().splitlines()]
    assert [(r["tool_call_id"], r["server"], r["is_error"]) for r in records] == [
        ("call_http", "http", False),
        ("call_sse", "sse", False),
    ]


PID_SERVER = SERVER.replace(
    'server.run("stdio")', 'open(sys.argv[1], "w").write(str(os.getpid()))\nserver.run("stdio")'
)


@needs_runtime
def test_sigterm_stops_the_stdio_servers_the_cli_started(short_dir) -> None:
    tmp_path = short_dir
    shutil.copy(mcp_cli.__file__, tmp_path / "mcp_cli.py")
    (tmp_path / "server.py").write_text("import os, sys\n" + PID_SERVER)
    pid_file = tmp_path / "server.pid"
    servers = [
        {
            "name": "weather",
            "transport": "stdio",
            "command": RUNTIME_PYTHON,
            "args": [str(tmp_path / "server.py"), str(pid_file)],
        }
    ]
    (tmp_path / "mcp_servers.json").write_text(json.dumps(servers))
    serve = subprocess.Popen([RUNTIME_PYTHON, "-E", "-s", str(tmp_path / "mcp_cli.py"), "serve"])
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "mcp.sock").exists() or not pid_file.exists():
            assert serve.poll() is None and time.monotonic() < deadline
            time.sleep(0.1)
        server_pid = int(pid_file.read_text())
        serve.terminate()
        assert serve.wait(timeout=10) == 0
    finally:
        if serve.poll() is None:
            serve.kill()
    # The SDK started the server in a session of its own; SIGTERM to the CLI closes its session, which stops it.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    raise AssertionError(f"stdio MCP server {server_pid} outlived the CLI")
