# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A bounded streaming admission service for one shared policy endpoint.

The permit covers the entire upstream response. This service never replays an inference request;
the caller retains its existing recovery policy. GET requests do not consume inference permits.
"""

import argparse
import asyncio

import aiohttp
from aiohttp import web
from yarl import URL

from nemo_gym.client.http_client import (
    GlobalAIOHTTPAsyncClientConfig,
    close_global_aiohttp_client,
    set_global_aiohttp_client,
)
from nemo_gym.client.http_client import request as http_request


_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}


def _headers(headers: dict) -> dict[str, str]:
    excluded = _HOP_HEADERS | {value.strip().lower() for value in headers.get("Connection", "").split(",")}
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


def create_app(*, upstream: str, max_inflight: int, max_queued: int) -> web.Application:
    """Create a relay with one process-wide admission budget and a bounded waiting queue."""
    if max_inflight <= 0 or max_queued < 0:
        raise ValueError("max_inflight must be positive and max_queued nonnegative")
    endpoint = URL(upstream)
    if (
        endpoint.scheme not in {"http", "https"}
        or endpoint.host is None
        or endpoint.user is not None
        or endpoint.query_string
        or endpoint.fragment
        or endpoint.path != "/"
    ):
        raise ValueError("upstream must be an HTTP(S) origin without credentials, a path or query")
    permits = asyncio.Semaphore(max_inflight)
    outstanding = 0

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"max_inflight": max_inflight, "max_queued": max_queued, "outstanding": outstanding})

    async def relay(incoming: web.Request) -> web.StreamResponse:
        nonlocal outstanding
        admitted = incoming.method == "POST"
        if admitted:
            if outstanding >= max_inflight + max_queued:
                raise web.HTTPServiceUnavailable(text="Policy admission queue is full", headers={"Retry-After": "1"})
            outstanding += 1
        acquired = False
        response = None
        try:
            if admitted:
                await permits.acquire()
                acquired = True
            response = await http_request(
                incoming.method,
                URL(str(endpoint).rstrip("/") + incoming.raw_path, encoded=True),
                headers=_headers(incoming.headers),
                data=incoming.content.iter_chunked(65536) if incoming.can_read_body else None,
                allow_redirects=False,
                auto_decompress=False,
                timeout=aiohttp.ClientTimeout(total=None),
                _max_num_tries=1,
                _max_connection_retries=0,
                _resend_after_disconnect=False,
                _retry_other_errors=False,
            )
            outgoing = web.StreamResponse(status=response.status, headers=_headers(response.headers))
            await outgoing.prepare(incoming)
            async for chunk in response.content.iter_chunked(65536):
                await outgoing.write(chunk)
            await outgoing.write_eof()
            return outgoing
        finally:
            if response is not None:
                response.close()
            if acquired:
                permits.release()
            if admitted:
                outstanding -= 1

    async def cleanup(_: web.Application) -> None:
        await close_global_aiohttp_client()

    async def startup(_: web.Application) -> None:
        set_global_aiohttp_client(
            GlobalAIOHTTPAsyncClientConfig(global_aiohttp_connector_limit=0, global_aiohttp_connector_limit_per_host=0)
        )

    app = web.Application(client_max_size=0, handler_args={"auto_decompress": False})
    app.router.add_get("/health", health)
    app.router.add_route("*", "/v1/{path:.*}", relay)
    app.on_cleanup.append(cleanup)
    app.on_startup.append(startup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--max-inflight", type=int, required=True)
    parser.add_argument("--max-queued", type=int, required=True)
    args = parser.parse_args()
    web.run_app(
        create_app(upstream=args.upstream, max_inflight=args.max_inflight, max_queued=args.max_queued),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        access_log=None,
    )


if __name__ == "__main__":
    main()
