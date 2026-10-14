# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded rate-limit retries for Codex providers, which disable native 429 retries."""

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import web

from nemo_gym.server_utils import request


@asynccontextmanager
async def rate_limit_proxy(base_url: str, invocation_id: str, *, max_retries: int = 2) -> AsyncIterator[str]:
    """Forward Responses requests, retaining every attempt at the Gym capture endpoint."""

    async def forward(incoming: web.Request) -> web.StreamResponse:
        body = await incoming.read()
        headers = {
            k: v
            for k, v in incoming.headers.items()
            if k.lower() not in {"host", "content-length", "connection", "x-session-id"}
        }
        headers["x-session-id"] = invocation_id
        for attempt in range(max_retries + 1):
            upstream = await request("POST", f"{base_url}/responses", data=body, headers=headers)
            try:
                if upstream.status == 429 and attempt < max_retries:
                    await upstream.read()
                    try:
                        delay = float(upstream.headers.get("Retry-After", 2**attempt))
                    except ValueError:
                        delay = 2**attempt
                    await asyncio.sleep(min(max(delay, 0), 30))
                    continue
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        k: v
                        for k, v in upstream.headers.items()
                        if k.lower() not in {"content-length", "transfer-encoding", "connection", "content-encoding"}
                    },
                )
                await response.prepare(incoming)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
            finally:
                upstream.release()
        raise AssertionError("retry loop exhausted without returning a response")

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_post("/responses", forward)
    runner = web.AppRunner(app, shutdown_timeout=1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        await runner.setup()
        try:
            await web.SockSite(runner, sock).start()
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            await runner.cleanup()
