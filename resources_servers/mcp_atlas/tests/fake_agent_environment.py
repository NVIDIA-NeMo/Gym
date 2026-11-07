# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the upstream agent-environment HTTP API (``/health``, ``/list-tools``, ``/call-tool``)."""

from __future__ import annotations

import socket
from typing import Any

from aiohttp import web


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeAgentEnvironment:
    """Serves ``tools``; ``results`` maps a tool name to ``(status, body)`` for ``/call-tool``."""

    def __init__(self, tools: list[str], results: dict[str, tuple[int, Any]]) -> None:
        self.tools = tools
        self.results = results
        self.calls: list[dict[str, Any]] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def _health(self, _: web.Request) -> web.Response:
        return web.json_response({"status": "health_and_client_connection_ok"})

    async def _list_tools(self, _: web.Request) -> web.Response:
        return web.json_response([{"name": name, "description": "", "inputSchema": {}} for name in self.tools])

    async def _call_tool(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.calls.append(payload)
        status, body = self.results.get(payload["tool_name"], (200, [{"type": "text", "text": "ok"}]))
        return web.json_response(body, status=status)

    async def start(self, port: int | None = None) -> str:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/list-tools", self._list_tools)
        app.router.add_post("/call-tool", self._call_tool)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self.port = port or free_port()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
