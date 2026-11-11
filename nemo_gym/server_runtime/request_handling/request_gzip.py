# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gzip request decoding before model capture and JSON parsing."""

import asyncio
import zlib

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _decode(body: bytes) -> bytes:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    result = decoder.decompress(body)
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Expected one complete gzip member")
    return result


class GzipRequestMiddleware:
    """Decode gzip bodies once and expose identity bytes to downstream middleware."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        encodings = [value.strip().lower() for name, value in scope["headers"] if name.lower() == b"content-encoding"]
        if not encodings or encodings == [b"identity"]:
            await self.app(scope, receive, send)
            return
        if encodings != [b"gzip"]:
            await JSONResponse({"error": "Unsupported Content-Encoding"}, status_code=415)(scope, receive, send)
            return
        try:
            compressed_body = await Request(scope, receive).body()
            body = await asyncio.to_thread(_decode, compressed_body)
        except (ValueError, zlib.error):
            await JSONResponse({"error": "Invalid gzip request body"}, status_code=400)(scope, receive, send)
            return
        downstream = dict(scope)
        downstream["headers"] = [
            (name, value)
            for name, value in scope["headers"]
            if name.lower() not in {b"content-encoding", b"content-length", b"transfer-encoding"}
        ]
        downstream["headers"].append((b"content-length", str(len(body)).encode()))
        delivered = False

        async def decoded_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(downstream, decoded_receive, send)
