# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapter pipeline middleware for responses_api_models FastAPI apps."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
)


logger = logging.getLogger(__name__)

# Hop-by-hop headers (RFC 7230 §6.1) plus framing/encoding fields the ASGI
# layer rewrites itself. ``content-encoding`` is included because aiohttp
# auto-decodes upstream bodies — re-emitting the original encoding would
# mislead any gzip-aware client into trying to decompress plain bytes.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
        "server",
    }
)

_SESSION_PATH_RE = re.compile(r"^/s/([a-f0-9]+)(/.*)?$")


def _override_request_body(request: Request, new_body: bytes) -> None:
    """Make downstream handlers see *new_body* instead of the original."""
    request._body = new_body  # type: ignore[attr-defined]
    if hasattr(request, "_json"):
        delattr(request, "_json")

    new_len = str(len(new_body)).encode("ascii")
    scope_headers = [(k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"content-length"]
    scope_headers.append((b"content-length", new_len))
    request.scope["headers"] = scope_headers


async def _starlette_response_to_adapter(
    starlette_resp: Response,
    ctx: InterceptorContext,
) -> AdapterResponse:
    chunks: list[bytes] = []
    body_iter = getattr(starlette_resp, "body_iterator", None)
    if body_iter is not None:
        async for chunk in body_iter:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            chunks.append(chunk)
    else:
        raw = getattr(starlette_resp, "body", b"")
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        chunks.append(raw)
    raw_body = b"".join(chunks)

    # Preserve duplicate header keys (e.g. multiple Set-Cookie) — dict
    # collapse would drop all but the last value.
    headers: list[tuple[bytes, bytes]] = list(starlette_resp.raw_headers)

    content_type = ""
    for name, value in headers:
        if name.lower() == b"content-type":
            content_type = value.decode("latin-1").lower()
            break

    body: dict[str, Any] | bytes
    if "application/json" in content_type or content_type == "":
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except (ValueError, UnicodeDecodeError):
            body = raw_body
    else:
        body = raw_body

    return AdapterResponse(
        status_code=starlette_resp.status_code,
        headers=headers,
        body=body,
        latency_ms=0.0,
        ctx=ctx,
    )


def _adapter_response_to_starlette(resp: AdapterResponse) -> Response:
    raw = resp.headers or []
    if isinstance(raw, dict):
        header_pairs: list[tuple[bytes, bytes]] = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in raw.items()]
    else:
        header_pairs = list(raw)

    fwd_pairs: list[tuple[bytes, bytes]] = [
        (name, value) for name, value in header_pairs if name.decode("latin-1").lower() not in _HOP_BY_HOP_HEADERS
    ]

    if isinstance(resp.body, bytes):
        out = Response(content=resp.body, status_code=resp.status_code)
    else:
        out = JSONResponse(content=resp.body, status_code=resp.status_code)

    # Keep Starlette's framing content-length; drop framing content-type
    # only if the adapter chain supplied one.
    fwd_keys_lower = {name.lower() for name, _ in fwd_pairs}
    framing_kept = [
        (k, v) for k, v in out.raw_headers if k.lower() == b"content-length" or k.lower() not in fwd_keys_lower
    ]
    out.raw_headers = framing_kept + fwd_pairs
    return out


def install_middleware(
    app: FastAPI,
    interceptor_specs: list[dict[str, Any]] | None,
) -> None:
    """Install the adapter pipeline as FastAPI middleware on *app*.

    ``interceptor_specs`` is a list of ``{"name": ..., "config": ...}`` dicts.
    Empty or ``None`` disables the middleware.
    """
    if not interceptor_specs:
        return

    # ``endpoint`` performs its own upstream call; inside a host server this
    # would double-forward. The proxy host mode (``start_adapter_proxy``)
    # appends it itself.
    if any(s.get("name") == "endpoint" for s in interceptor_specs):
        raise ValueError(
            "install_middleware: the 'endpoint' interceptor cannot be used in a "
            "middleware-hosted chain — the model server already forwards upstream. "
            "Drop it from `adapters` or use start_adapter_proxy() instead."
        )

    pipeline = AdapterPipeline.from_config(interceptor_specs)

    @app.middleware("http")
    async def _adapter_middleware(request: Request, call_next):  # noqa: ANN202
        if request.method != "POST":
            return await call_next(request)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
                status_code=400,
            )

        path = request.url.path
        ctx = InterceptorContext()

        m = _SESSION_PATH_RE.match(path)
        if m:
            ctx.extra["session_id"] = m.group(1)
            path = m.group(2) or "/"

        adapter_req = AdapterRequest(
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=body,
            ctx=ctx,
        )

        async def _upstream(req: AdapterRequest) -> AdapterResponse:
            new_body = json.dumps(req.body).encode("utf-8")
            _override_request_body(request, new_body)
            # Re-point the ASGI scope at the prefix-stripped path so the router
            # can match it (the /s/<hex>/ session prefix is middleware-only).
            request.scope["path"] = req.path
            request.scope["raw_path"] = req.path.encode("latin-1")
            starlette_resp = await call_next(request)
            return await _starlette_response_to_adapter(starlette_resp, req.ctx)

        try:
            resp = await pipeline.process(adapter_req, upstream_call=_upstream)
        except GracefulError as exc:
            logger.warning(
                "Adapter middleware: graceful termination for session %s: %s",
                ctx.extra.get("session_id", "?"),
                exc,
            )
            return JSONResponse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "code": "session_budget_exhausted",
                    },
                },
                status_code=429,
            )
        except Exception:
            logger.exception("Adapter pipeline error")
            return JSONResponse(
                {"error": {"message": "Internal adapter middleware error", "type": "server_error"}},
                status_code=500,
            )

        return _adapter_response_to_starlette(resp)
