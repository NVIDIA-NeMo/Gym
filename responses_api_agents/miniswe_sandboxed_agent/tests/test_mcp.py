# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import shutil
import site
import subprocess
import sys
import sysconfig
import time
import venv
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.miniswe_sandboxed_agent import mcp_client as module


COUNTER_SERVER = """from mcp.server.fastmcp import FastMCP
mcp = FastMCP("counter")
count = 0
@mcp.tool()
def increment() -> int:
    global count
    count += 1
    return count
mcp.run(transport="stdio")
"""


@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
async def test_transport_connections_use_declared_endpoints(monkeypatch, transport):
    calls = []

    @asynccontextmanager
    async def connection(*args, **kwargs):
        calls.append((args, kwargs))
        yield ("read", "write", "unused")

    client = SimpleNamespace(initialize=AsyncMock())

    @asynccontextmanager
    async def session(read, write):
        assert (read, write) == ("read", "write")
        yield client

    for method in ["stdio_client", "sse_client", "streamablehttp_client"]:
        monkeypatch.setattr(module, method, connection)
    monkeypatch.setattr(module, "ClientSession", session)
    server = {
        "name": "browser",
        "transport": transport,
        "command": "tool",
        "args": ["serve"],
        "url": "http://sidecar/mcp",
    }
    async with AsyncExitStack() as stack:
        assert await module.connect(server, stack) is client
    client.initialize.assert_awaited_once()
    if transport == "stdio":
        assert calls[0][0][0].command == "tool"
    else:
        assert calls[0][0][0] == server["url"]
        assert calls[0][1]["httpx_client_factory"] is module.http_client


async def test_mcp_session_preserves_tool_state_across_cli_calls(monkeypatch):
    state = []

    async def call_tool(tool, args):
        if tool == "fail":
            raise ValueError("tool failed")
        state.append(args["value"])
        return SimpleNamespace(model_dump=lambda **_: {"values": list(state)})

    client = SimpleNamespace(
        call_tool=call_tool,
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(tools=[SimpleNamespace(model_dump=lambda **_: {"name": "one"})], nextCursor="next"),
                SimpleNamespace(tools=[SimpleNamespace(model_dump=lambda **_: {"name": "two"})], nextCursor=None),
            ]
        ),
    )
    connect = AsyncMock(return_value=client)
    monkeypatch.setattr(module, "connect", connect)
    with TemporaryDirectory(dir="/tmp", prefix="tb4-mcp-") as directory:
        socket = str(Path(directory) / "mcp.sock")
        daemon = asyncio.create_task(module.serve([{"name": "browser"}], socket))
        try:
            async with asyncio.timeout(5):
                while not Path(socket).exists():
                    if daemon.done():
                        await daemon
                    await asyncio.sleep(0.01)
            assert await module.request(socket, {"operation": "list"}) == {
                "browser": [{"name": "one"}, {"name": "two"}]
            }
            for value in [1, 2]:
                result = await module.request(
                    socket, {"operation": "call", "server": "browser", "tool": "save", "arguments": {"value": value}}
                )
            assert result == {"values": [1, 2]}
            connect.assert_awaited_once()
            with pytest.raises(RuntimeError, match="tool failed"):
                await module.request(
                    socket, {"operation": "call", "server": "browser", "tool": "fail", "arguments": {}}
                )
        finally:
            daemon.cancel()
            await asyncio.gather(daemon, return_exceptions=True)


@pytest.mark.parametrize(
    "argv", [["serve"], ["list"], ["call", "browser", "navigate", '{"url":"http://app"}']], ids=lambda argv: argv[0]
)
async def test_cli_dispatches_declared_tool_arguments(tmp_path, monkeypatch, capsys, argv):
    operation = argv[0]
    monkeypatch.setattr(module, "__file__", str(tmp_path / "client.py"))
    servers = [{"name": "browser", "transport": "sse", "url": "http://sidecar/mcp"}]
    (tmp_path / "servers.json").write_text(json.dumps(servers))
    serve = AsyncMock()
    request = AsyncMock(return_value={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr(module, "serve", serve)
    monkeypatch.setattr(module, "request", request)
    await module.main(module.parse_args(argv))
    if operation == "serve":
        serve.assert_awaited_once_with(servers, str(tmp_path / "server.sock"))
        request.assert_not_awaited()
    else:
        assert json.loads(capsys.readouterr().out)["content"][0]["text"] == "ok"
        expected = (
            {"operation": "list"}
            if operation == "list"
            else {"operation": "call", "server": "browser", "tool": "navigate", "arguments": {"url": "http://app"}}
        )
        request.assert_awaited_once_with(str(tmp_path / "server.sock"), expected)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "required: command"),
        (["call", "counter"], "required: TOOL, JSON_ARGUMENTS"),
        (["call", "counter", "increment"], "required: JSON_ARGUMENTS"),
        (["call", "counter", "increment", "{name: World}"], "not valid JSON"),
        # An unquoted object reaches the client split by the shell.
        (["call", "counter", "increment", '{"name":', '"World"}'], "not valid JSON"),
        (["call", "counter", "increment", '["World"]'], "must be a JSON object"),
        (["increment"], "invalid choice"),
        (["list", "counter"], "unrecognized arguments"),
    ],
)
def test_malformed_cli_call_prints_usage_and_exits_2(capsys, argv, message):
    with pytest.raises(SystemExit) as raised:
        module.parse_args(argv)
    assert raised.value.code == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("usage: ")
    assert message in stderr


async def test_http_transport_uses_aiohttp():
    async with module.http_client(timeout=10) as client:
        assert client.follow_redirects
        assert type(client).__name__ == "HttpxAiohttpClient"


async def test_real_mcp_stdio_server_retains_state(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(COUNTER_SERVER)
    with TemporaryDirectory(dir="/tmp", prefix="tb4-mcp-real-") as directory:
        socket = str(Path(directory) / "mcp.sock")
        daemon = asyncio.create_task(
            module.serve(
                [{"name": "counter", "transport": "stdio", "command": sys.executable, "args": [str(script)]}], socket
            )
        )
        try:
            async with asyncio.timeout(15):
                while not Path(socket).exists():
                    if daemon.done():
                        await daemon
                    await asyncio.sleep(0.02)
                schemas = await module.request(socket, {"operation": "list"})
                assert schemas["counter"][0]["name"] == "increment"
                for expected in [1, 2]:
                    result = await module.request(
                        socket, {"operation": "call", "server": "counter", "tool": "increment", "arguments": {}}
                    )
                    assert not result["isError"]
                    assert result["content"][0]["text"] == str(expected)
        finally:
            daemon.cancel()
            await asyncio.gather(daemon, return_exceptions=True)


def test_client_run_by_another_python_reruns_under_its_venv():
    """A model may run the client with whichever `python` it finds. The client reruns under the venv it sits in, and
    `exec` keeps the PID, so stopping the PID that started `serve` stops the server."""
    with TemporaryDirectory(dir="/tmp", prefix="mcp-venv-") as directory:
        root = Path(directory)
        # A symlinked venv python, as `uv venv` makes, that reaches this test's MCP packages without installing them.
        venv.EnvBuilder(symlinks=True).create(root)
        packages = sysconfig.get_path("purelib", "venv", vars={"base": directory, "platbase": directory})
        Path(packages, "test-packages.pth").write_text("\n".join(site.getsitepackages()) + "\n")
        shutil.copy(module.__file__, root / "client.py")
        (root / "server.py").write_text(COUNTER_SERVER)
        servers = [
            {"name": "counter", "transport": "stdio", "command": sys.executable, "args": [str(root / "server.py")]}
        ]
        (root / "servers.json").write_text(json.dumps(servers))
        # Without site-packages or PYTHON* variables, this interpreter cannot import the MCP packages itself.
        other = [sys.executable, "-I", "-S", str(root / "client.py")]

        def cli(*argv):
            return subprocess.run([*other, *argv], capture_output=True, text=True, timeout=60)

        with open(root / "serve.log", "w") as log:
            daemon = subprocess.Popen([*other, "serve"], stdout=log, stderr=log, start_new_session=True)
        try:
            for _ in range(300):
                assert daemon.poll() is None, (root / "serve.log").read_text()
                if (root / "server.sock").exists():
                    break
                time.sleep(0.1)
            listed = cli("list")
            assert listed.returncode == 0, listed.stderr
            assert json.loads(listed.stdout)["counter"][0]["name"] == "increment"
            for expected in ["1", "2"]:
                called = cli("call", "counter", "increment", "{}")
                assert called.returncode == 0, called.stderr
                assert json.loads(called.stdout)["content"][0]["text"] == expected
            malformed = cli("call", "counter", "increment", "{bad")
            assert malformed.returncode == 2
            assert "not valid JSON" in malformed.stderr and "Traceback" not in malformed.stderr
        finally:
            daemon.terminate()
            daemon.wait(10)
        assert cli("list").returncode != 0
