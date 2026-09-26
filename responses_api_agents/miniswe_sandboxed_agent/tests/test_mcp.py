# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.miniswe_sandboxed_agent import mcp_client as module
from responses_api_agents.miniswe_sandboxed_agent.harness import responses_input
from responses_api_agents.miniswe_sandboxed_agent.sandbox_runner import GymModel


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


def test_mcp_images_are_multimodal_model_inputs():
    model = GymModel({}, {"model_history": [], "tool_history": []}, lambda: None)
    messages = model.format_observation_messages(
        {"extra": {"actions": [{"command": "screenshot", "tool_call_id": "call_image"}]}},
        [{"output": "Screenshot", "returncode": 0, "images": ["data:image/png;base64,AA=="]}],
    )
    assert messages[0]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
    assert responses_input(messages) == [
        {"type": "function_call_output", "call_id": "call_image", "output": messages[0]["content"]}
    ]


@pytest.mark.parametrize("operation", ["serve", "list", "call"])
async def test_cli_dispatches_declared_tool_arguments(tmp_path, monkeypatch, capsys, operation):
    import json

    monkeypatch.setattr(module, "__file__", str(tmp_path / "client.py"))
    servers = [{"name": "browser", "transport": "sse", "url": "http://sidecar/mcp"}]
    (tmp_path / "servers.json").write_text(json.dumps(servers))
    monkeypatch.setattr(module.sys, "argv", ["client.py", operation, "browser", "navigate", '{"url":"http://app"}'])
    serve = AsyncMock()
    request = AsyncMock(return_value={"content": [{"type": "text", "text": "ok"}]})
    monkeypatch.setattr(module, "serve", serve)
    monkeypatch.setattr(module, "request", request)
    await module.main()
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


async def test_http_transport_uses_aiohttp():
    async with module.http_client(timeout=10) as client:
        assert client.follow_redirects
        assert type(client).__name__ == "HttpxAiohttpClient"


async def test_real_mcp_stdio_server_retains_state(tmp_path):
    import sys

    script = tmp_path / "server.py"
    script.write_text("""from mcp.server.fastmcp import FastMCP
mcp = FastMCP("counter")
count = 0
@mcp.tool()
def increment() -> int:
    global count
    count += 1
    return count
mcp.run(transport="stdio")
""")
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
