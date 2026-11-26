# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gzip request decoding before model capture and JSON parsing."""

import asyncio
import zlib

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _DecodedBodyTooLarge(Exception):
    pass


def _decode(body: bytes, max_decoded_bytes: int) -> bytes:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    result = decoder.decompress(body, max_decoded_bytes + 1)
    if len(result) > max_decoded_bytes:
        raise _DecodedBodyTooLarge
    if not decoder.eof or decoder.unused_data:
        raise ValueError("Expected one complete gzip member")
    return result


def _invalid_request_response(message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}},
        status_code=status_code,
    )


class GzipRequestMiddleware:
    """Decode gzip bodies once and expose identity bytes to downstream middleware."""

    def __init__(self, app: ASGIApp, *, max_decoded_bytes: int) -> None:
        self.app = app
        self.max_decoded_bytes = max_decoded_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        encodings = [value.strip().lower() for name, value in scope["headers"] if name.lower() == b"content-encoding"]
        if encodings != [b"gzip"]:
            await self.app(scope, receive, send)
            return
        compressed_body = await Request(scope, receive).body()
        try:
            body = await asyncio.to_thread(_decode, compressed_body, self.max_decoded_bytes)
        except _DecodedBodyTooLarge:
            response = _invalid_request_response(
                f"Decoded gzip request body exceeds {self.max_decoded_bytes} bytes.", status_code=413
            )
            await response(scope, receive, send)
            return
        except (ValueError, zlib.error):
            response = _invalid_request_response("Invalid gzip request body.", status_code=400)
            await response(scope, receive, send)
            return
        del compressed_body
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
