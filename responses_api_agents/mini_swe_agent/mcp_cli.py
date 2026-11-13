# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-local MCP CLI that mini-SWE's bash tool calls; it runs in the task sandbox with the harness runtime.

``serve`` connects to every server in ``mcp_servers.json`` beside this file and keeps the sessions open behind a
Unix socket, so state such as a browser page survives between bash calls. A server that does not connect is reported
by ``list`` with its error; the sandbox runner decides whether the run can start. ``list`` prints each server's tools
and ``call SERVER TOOL JSON_ARGUMENTS`` prints a tool's result. Every call is appended to ``mcp_calls.jsonl`` with the
bash tool call that made it, so the agent server can report it. No model credentials pass through here.
"""

import asyncio
import json
import os
import signal
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path


SERVERS_FILE = Path(__file__).with_name("mcp_servers.json")
SOCKET_FILE = Path(__file__).with_name("mcp.sock")
CALLS_FILE = Path(__file__).with_name("mcp_calls.jsonl")
# mini-SWE's local environment sets it for each bash command; see sandbox_runner.py.
TOOL_CALL_ID_VARIABLE = "NEMO_GYM_MINI_SWE_TOOL_CALL_ID"
# Tool results and listings can be large (screenshots, page snapshots).
MESSAGE_LIMIT = 64 * 1024 * 1024
# Servers connect concurrently, each within this budget; the runner waits for all of them (see sandbox_runner.py).
CONNECT_TIMEOUT_S = 30


async def connect(server: dict, stack: AsyncExitStack):
    import httpx2
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    transport = server.get("transport", "streamable_http")
    if transport == "stdio":
        streams = stdio_client(
            StdioServerParameters(command=server["command"], args=server.get("args", []), env=server.get("env"))
        )
    elif transport == "sse":
        streams = sse_client(server["url"], headers=server.get("headers"))
    else:
        # The SDK's default timeouts; the grant's headers carry the episode's tool access.
        client = httpx2.AsyncClient(headers=server.get("headers"), timeout=httpx2.Timeout(30, read=300))
        await stack.enter_async_context(client)
        streams = streamable_http_client(server["url"], http_client=client)
    read, write = await stack.enter_async_context(streams)
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def list_tools(session) -> list[dict]:
    from mcp.types import PaginatedRequestParams

    tools, cursor = [], None
    while True:
        page = await session.list_tools(params=None if cursor is None else PaginatedRequestParams(cursor=cursor))
        tools.extend(tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in page.tools)
        cursor = page.next_cursor
        if cursor is None:
            return tools


def record_call(message: dict, started_at: float, started: float, *, output: str, is_error: bool) -> None:
    record = {
        "tool_call_id": message.get("tool_call_id") or None,
        "server": message["server"],
        "tool": message["tool"],
        "arguments": message["arguments"],
        "started_at": started_at,
        "completed_at": time.time(),
        "duration_ms": (time.monotonic() - started) * 1000,
        "is_error": is_error,
        "output": output,
    }
    with CALLS_FILE.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def describe(error: BaseException) -> str:
    """The first underlying error, also when a transport's task group wrapped it."""
    while isinstance(error, BaseExceptionGroup) and error.exceptions:
        error = error.exceptions[0]
    return f"{type(error).__name__}: {error}"


async def hold(server: dict, connection: dict, ready: asyncio.Event) -> None:
    """Connect to one server and keep its session open; ``connection`` gets the session, or the error that ended it.

    Each server has its own task, which enters and exits every context the SDK opens: a transport that fails cancels
    the task's scope, and the error surfaces only once those contexts exit.
    """
    try:
        async with AsyncExitStack() as stack:
            async with asyncio.timeout(CONNECT_TIMEOUT_S):
                connection["session"] = await connect(server, stack)
            ready.set()
            await asyncio.Event().wait()
    except BaseException as error:
        connection.setdefault("error", describe(error))
    finally:
        ready.set()


async def dispatch(connections: dict[str, dict], message: dict) -> object:
    if message["operation"] == "list":
        listing = {}
        for name, connection in connections.items():
            if "error" in connection:
                listing[name] = {"error": connection["error"]}
                continue
            # A connected server whose listing fails is reported like one that did not connect; the runner decides.
            try:
                listing[name] = {"tools": await list_tools(connection["session"])}
            except Exception as error:
                listing[name] = {"error": describe(error)}
        return listing
    started_at, started = time.time(), time.monotonic()
    try:
        connection = connections.get(message["server"], {"error": "not granted"})
        if "error" in connection:
            raise KeyError(f"MCP server {message['server']!r} is not available: {connection['error']}")
        result = await connection["session"].call_tool(message["tool"], message["arguments"])
        output = json.dumps(result.model_dump(mode="json", by_alias=True, exclude_none=True))
    except Exception as error:
        record_call(message, started_at, started, output=f"{type(error).__name__}: {error}", is_error=True)
        raise
    record_call(message, started_at, started, output=output, is_error=bool(getattr(result, "is_error", False)))
    return json.loads(output)


async def serve() -> None:
    # The runner stops this process with SIGTERM. Cancelling instead of dying lets each server's session close, and the
    # SDK then stops the stdio servers it started in sessions of their own, which no group signal reaches.
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    servers = json.loads(SERVERS_FILE.read_text())
    connections = {server["name"]: {} for server in servers}
    ready = [asyncio.Event() for _ in servers]
    holders = [
        asyncio.create_task(hold(server, connections[server["name"]], event))
        for server, event in zip(servers, ready, strict=True)
    ]
    await asyncio.gather(*(event.wait() for event in ready))

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            result = await dispatch(connections, json.loads(await reader.readline()))
            writer.write(json.dumps({"result": result}).encode() + b"\n")
        except Exception as error:
            writer.write(json.dumps({"error": f"{type(error).__name__}: {error}"}).encode() + b"\n")
        finally:
            await writer.drain()
            writer.close()
            await writer.wait_closed()

    # The socket appears only once every server has connected or failed, which is what the runner waits for.
    server = await asyncio.start_unix_server(handle, path=str(SOCKET_FILE), limit=MESSAGE_LIMIT)
    SOCKET_FILE.chmod(0o600)
    async with server:
        try:
            await server.serve_forever()
        finally:
            for holder in holders:
                holder.cancel()
            # Bounded below the runner's five-second wait after SIGTERM.
            await asyncio.wait(holders, timeout=4)


async def request(message: dict) -> object:
    reader, writer = await asyncio.open_unix_connection(str(SOCKET_FILE), limit=MESSAGE_LIMIT)
    try:
        writer.write(json.dumps(message).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
    finally:
        writer.close()
        await writer.wait_closed()
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


def main(argv: list[str]) -> int:
    if argv[1:2] == ["serve"]:
        try:
            asyncio.run(serve())
        except asyncio.CancelledError:
            pass  # Stopped by SIGTERM, after its sessions closed.
        return 0
    if argv[1:2] == ["list"]:
        message = {"operation": "list"}
    elif argv[1:2] == ["call"] and len(argv) == 5:
        message = {
            "operation": "call",
            "server": argv[2],
            "tool": argv[3],
            "arguments": json.loads(argv[4]),
            "tool_call_id": os.environ.get(TOOL_CALL_ID_VARIABLE),
        }
    else:
        print(f"usage: {argv[0]} list | call SERVER TOOL 'JSON_ARGUMENTS'", file=sys.stderr)
        return 2
    try:
        result = asyncio.run(request(message))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
