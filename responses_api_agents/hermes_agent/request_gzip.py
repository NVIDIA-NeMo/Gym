# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compress serialized OpenAI requests in the Hermes sandbox's HTTP clients."""

import asyncio
import gzip
from functools import wraps
from urllib.parse import urlsplit

import httpx
import openai


def _compress(request: httpx.Request, body: bytes) -> None:
    if len(body) < 4096 or request.headers.get("content-encoding", "identity").lower() != "identity":
        return
    compressed = gzip.compress(body, compresslevel=1, mtime=0)
    if len(compressed) < len(body):
        request._content = compressed
        request.stream = httpx.ByteStream(compressed)
        request.headers["content-encoding"] = "gzip"
        request.headers["content-length"] = str(len(compressed))
        request.headers.pop("transfer-encoding", None)


def install_request_gzip(base_url: str) -> None:
    """Attach request hooks to root, auxiliary and child clients targeting Gym."""
    base = urlsplit(base_url)
    paths = {base.path.rstrip("/") + suffix for suffix in ("/chat/completions", "/responses")}

    def matches(request: httpx.Request) -> bool:
        url = urlsplit(str(request.url))
        return (
            request.method == "POST" and (url.scheme, url.netloc) == (base.scheme, base.netloc) and url.path in paths
        )

    def compress_request(request: httpx.Request) -> None:
        if matches(request):
            _compress(request, request.read())

    async def compress_async_request(request: httpx.Request) -> None:
        if matches(request):
            body = await request.aread()
            await asyncio.to_thread(_compress, request, body)

    initialize_sync = openai.OpenAI.__init__
    initialize_async = openai.AsyncOpenAI.__init__

    @wraps(initialize_sync)
    def init_sync(client: openai.OpenAI, *args: object, **kwargs: object) -> None:
        initialize_sync(client, *args, **kwargs)
        client._client.event_hooks["request"].append(compress_request)

    @wraps(initialize_async)
    def init_async(client: openai.AsyncOpenAI, *args: object, **kwargs: object) -> None:
        initialize_async(client, *args, **kwargs)
        client._client.event_hooks["request"].append(compress_async_request)

    openai.OpenAI.__init__ = init_sync
    openai.AsyncOpenAI.__init__ = init_async
