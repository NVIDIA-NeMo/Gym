# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small task-local CLI for mini-SWE's bash harness; no provider credentials."""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from httpx_aiohttp import HttpxAiohttpClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client


def http_client(**kwargs):
    return HttpxAiohttpClient(follow_redirects=True, **kwargs)


async def connect(server, stack):
    if server["transport"] == "stdio":
        transport = stdio_client(StdioServerParameters(command=server["command"], args=server.get("args", [])))
    elif server["transport"] == "sse":
        transport = sse_client(server["url"], httpx_client_factory=http_client)
    else:
        transport = streamablehttp_client(server["url"], httpx_client_factory=http_client)
    streams = await stack.enter_async_context(transport)
    client = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
    await client.initialize()
    return client


async def list_tools(client):
    tools = []
    cursor = None
    while True:
        page = await client.list_tools(cursor=cursor)
        tools.extend(tool.model_dump(mode="json") for tool in page.tools)
        cursor = page.nextCursor
        if cursor is None:
            return tools


async def dispatch(clients, message):
    if message["operation"] == "list":
        return {name: await list_tools(client) for name, client in clients.items()}
    return (await clients[message["server"]].call_tool(message["tool"], message["arguments"])).model_dump(mode="json")


async def serve(servers, socket_path):
    # Keep MCP sessions alive across CLI calls (e.g. browser page state).
    async with AsyncExitStack() as stack:
        clients = {server["name"]: await connect(server, stack) for server in servers}

        async def handle(reader, writer):
            try:
                message = json.loads(await reader.readline())
                result = await dispatch(clients, message)
                writer.write(json.dumps({"result": result}).encode() + b"\n")
            except Exception as exc:
                writer.write(json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode() + b"\n")
            finally:
                await writer.drain()
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_unix_server(handle, path=socket_path, limit=16 * 1024 * 1024)
        Path(socket_path).chmod(0o600)
        async with server:
            await server.serve_forever()


async def request(socket_path, message):
    reader, writer = await asyncio.open_unix_connection(socket_path, limit=16 * 1024 * 1024)
    try:
        writer.write(json.dumps(message).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    servers = json.loads(Path(__file__).with_name("servers.json").read_text())
    socket_path = str(Path(__file__).with_name("server.sock"))
    if sys.argv[1] == "serve":
        await serve(servers, socket_path)
        return
    if sys.argv[1] == "list":
        message = {"operation": "list"}
    else:
        message = {
            "operation": "call",
            "server": sys.argv[2],
            "tool": sys.argv[3],
            "arguments": json.loads(sys.argv[4]),
        }
    result = await request(socket_path, message)
    print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
