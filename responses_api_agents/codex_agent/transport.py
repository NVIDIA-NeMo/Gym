# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded model-request retries for Codex, whose own retries are off behind this forwarder.

Codex 0.144.4 never retries a 429, and resends a 401, 404 or 409 like a 5xx without reading Retry-After. This
forwarder resends only 408, 425, 429, 5xx and refused or dropped connections, at most twice, waiting at most 10 s
before each resend.
"""

import asyncio
import math
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp import ClientConnectionError, web

from nemo_gym.client import request
from nemo_gym.core import StreamFailureReportReader
from nemo_gym.server_runtime import is_retryable_dependency_status


MAX_RETRY_DELAY_S = 10.0


def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait before a resend: a numeric Retry-After within [0, 10], else 2**attempt."""
    try:
        delay = float(retry_after) if retry_after is not None else math.nan
    except ValueError:
        delay = math.nan
    return min(max(delay, 0.0), MAX_RETRY_DELAY_S) if math.isfinite(delay) else float(2**attempt)


@asynccontextmanager
async def model_retry_proxy(base_url: str, invocation_id: str, *, max_retries: int = 2) -> AsyncIterator[str]:
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
            last = attempt == max_retries
            try:
                # One connection attempt each: this loop owns the count and the wait.
                upstream = await request("POST", f"{base_url}/responses", data=body, headers=headers, _max_num_tries=1)
            except ClientConnectionError as error:
                # A request that timed out has used its time; one that never connected or was dropped is resent.
                if last or isinstance(error, asyncio.TimeoutError):
                    return web.Response(status=502, reason="Upstream model connection failed")
                await asyncio.sleep(retry_delay(None, attempt))
                continue
            try:
                if is_retryable_dependency_status(upstream.status) and not last:
                    await upstream.read()
                    await asyncio.sleep(retry_delay(upstream.headers.get("Retry-After"), attempt))
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
                # The model's report rides the stream's terminal event when its headers went out first.
                reader = StreamFailureReportReader()
                try:
                    async for chunk in upstream.content.iter_any():
                        reader.feed(chunk)
                        await response.write(chunk)
                    await response.write_eof()
                finally:
                    # A disconnect mid-stream still keeps what the stream reported so far.
                    reader.observe()
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
