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
"""FastAPI middleware that runs the Gym adapter pipeline on POST requests.

Unlike NEL's ``proxy.py`` (which boots a parallel uvicorn server that forwards
to an upstream URL), this module attaches the interceptor pipeline directly
to an existing ``responses_api_models`` FastAPI app via
``app.middleware("http")``. The model server keeps its own port and lifecycle;
the middleware only mutates request/response payloads in-flight.

Layering semantics (Phase 1.5):
    1. REQUEST interceptors run in order — they mutate the request.
    2. REQUEST_TO_RESPONSE interceptors may optionally short-circuit
       (e.g. ``caching`` on a cache hit) by returning an ``AdapterResponse``.
    3. If nothing short-circuited, the middleware calls ``call_next`` so the
       model server's own route handler does the upstream call against the
       (possibly mutated) body. This is the load-bearing wrap-not-replace
       distinction from Phase 1, which previously required ``endpoint`` in
       every chain.
    4. The route handler's Starlette ``Response`` is converted to an
       ``AdapterResponse`` and passed through RESPONSE interceptors in
       reverse order (``log_tokens``, ``response_stats``, ...).

The ``endpoint`` interceptor remains in the registry as an opt-in for chains
that want to replace the model server's upstream call (NEL-compat use).
"""

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

# Hop-by-hop headers (RFC 7230 §6.1 plus a few framing fields the ASGI
# server must rewrite itself). Stripped before returning the response so we
# don't double-set Content-Length or leak upstream Server identification.
#
# ``content-encoding`` is included here because aiohttp (used by the
# ``endpoint`` interceptor and the global request client) auto-decompresses
# gzip/deflate response bodies — by the time the body reaches the adapter
# pipeline it has already been decoded. Leaving the original
# ``Content-Encoding: gzip`` header on the re-emitted response would mislead
# any downstream consumer into trying to gunzip an already-plain body. The
# correct thing is to strip the encoding header; the decompressed body is
# served as-is. Today the in-tree model servers don't gzip their replies,
# so this is defensive; flip a gzipping reverse proxy in front of vLLM and
# this filter is what keeps clients from receiving a "bad gzip" error.
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

# Session-id prefix: clients send POST /s/<hex>/v1/... to scope a request to
# a server-side session. The middleware strips the prefix before forwarding
# to the model and records the session id on the interceptor context.
_SESSION_PATH_RE = re.compile(r"^/s/([a-f0-9]+)(/.*)?$")


def _override_request_body(request: Request, new_body: bytes) -> None:
    """Replace the request's cached body so downstream handlers see *new_body*.

    Starlette's ``BaseHTTPMiddleware._CachedRequest.wrapped_receive`` reads
    ``self._body`` when forwarding to downstream apps, so overwriting that
    attribute is sufficient to make the model server's route handler observe
    the post-REQUEST-stage payload.

    We also reset ``_json`` (Starlette's parsed-JSON cache) and rewrite the
    ``content-length`` header in the ASGI scope — a wrong content-length
    causes downstream consumers that read raw chunks to truncate to zero
    bytes, even if ``.json()`` would still work via the cache.
    """
    request._body = new_body  # type: ignore[attr-defined]
    if hasattr(request, "_json"):
        # Clear the parsed-JSON cache so the next .json() call re-parses
        # the mutated body. We can't predict whether downstream code reads
        # .body() or .json() first; both must agree.
        delattr(request, "_json")

    new_len = str(len(new_body)).encode("ascii")
    scope_headers = [(k, v) for k, v in request.scope.get("headers", []) if k.lower() != b"content-length"]
    scope_headers.append((b"content-length", new_len))
    request.scope["headers"] = scope_headers


async def _starlette_response_to_adapter(
    starlette_resp: Response,
    ctx: InterceptorContext,
) -> AdapterResponse:
    """Convert a Starlette response to an ``AdapterResponse``.

    The body is consumed via ``BaseHTTPMiddleware`` streaming and re-parsed
    as JSON when possible so RESPONSE interceptors (``log_tokens``,
    ``response_stats``, ...) see a dict the same way they would with NEL's
    httpx-based ``endpoint`` interceptor.
    """
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

    # Preserve duplicate header keys (Set-Cookie, etc.) by keeping the
    # original list-of-bytes-tuples shape. Collapsing to a dict here would
    # silently drop all but the last value for any multi-valued header —
    # which would break ``SessionMiddleware`` the moment more than one
    # middleware sets a ``Set-Cookie`` on the same response.
    headers: list[tuple[bytes, bytes]] = list(starlette_resp.raw_headers)

    # Content-type lookup for the body-parsing decision is single-valued in
    # practice, so a one-shot scan of the list is fine here.
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
    """Re-emit an ``AdapterResponse`` as a Starlette ``Response``.

    ``resp.headers`` is normally the list-of-byte-tuples shape produced by
    ``_starlette_response_to_adapter`` (which preserves duplicate header
    keys — ``Set-Cookie`` chief among them). Test code and some
    short-circuit interceptors (``caching``) construct an ``AdapterResponse``
    with a plain ``dict[str, str]``; both shapes are accepted here.

    Filtering must happen over the list, not the dict, otherwise duplicate
    ``Set-Cookie`` values would already have collapsed by the time we got
    here. We rebuild ``Response.raw_headers`` directly because Starlette's
    ``Response(headers=...)`` kwarg only accepts a mapping.
    """
    raw = resp.headers or []
    if isinstance(raw, dict):
        header_pairs: list[tuple[bytes, bytes]] = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in raw.items()]
    else:
        header_pairs = list(raw)

    # Strip hop-by-hop headers before re-emitting.
    fwd_pairs: list[tuple[bytes, bytes]] = [
        (name, value) for name, value in header_pairs if name.decode("latin-1").lower() not in _HOP_BY_HOP_HEADERS
    ]

    if isinstance(resp.body, bytes):
        out = Response(content=resp.body, status_code=resp.status_code)
    else:
        out = JSONResponse(content=resp.body, status_code=resp.status_code)

    # Keep the framing ``content-length`` Starlette computed (the body bytes
    # changed in the JSONResponse path; trusting the upstream value would
    # truncate the client read). Drop the framing ``content-type`` only if
    # the adapter chain provided one — otherwise the JSONResponse default
    # (``application/json``) sticks. All other forwarded headers append.
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

    ``interceptor_specs`` is a list of ``{"name": ..., "config": ...}`` dicts,
    using the same shape as NEL's ``start_adapter_proxy(interceptor_specs=...)``.
    If ``interceptor_specs`` is ``None`` or empty, this is a no-op — the model
    server behaves exactly as it would without middleware installed.
    """
    if not interceptor_specs:
        return

    pipeline = AdapterPipeline.from_config(interceptor_specs)

    @app.middleware("http")
    async def _adapter_middleware(request: Request, call_next):  # noqa: ANN202
        # Only POST goes through the pipeline. GET (health, etc.) passes through
        # so the model server's own routes (e.g. /health, /metrics) keep working.
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

        # Strip the optional /s/<hex>/ session prefix and stash the id on ctx.
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
            """Forward the (possibly mutated) request through the model server.

            Replays ``req.body`` by overwriting Starlette's cached request
            body, then defers to ``call_next`` so FastAPI's own routing and
            handlers run normally.
            """
            new_body = json.dumps(req.body).encode("utf-8")
            _override_request_body(request, new_body)
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
