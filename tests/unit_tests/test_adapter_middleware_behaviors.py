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
"""Middleware behavioral tests — replace NEL ``tests/test_adapters/test_proxy.py``.

NEL's ``test_proxy.py`` exercised a standalone uvicorn proxy spun up by
``start_adapter_proxy(...)``. Gym does not have a standalone proxy — the
adapter pipeline is installed as FastAPI middleware on an existing model
server's app via ``install_middleware``. The architectural shift means the
proxy lifecycle / port-allocation tests do not port.

What does port — and is exercised here against the middleware via a
FastAPI ``TestClient`` — are the **behavioral invariants** that NEL's
proxy guaranteed and that Gym's Phase-1.5 middleware also implements:

* ``/s/<hex>/...`` session-id prefix is parsed off the path and exposed
  to interceptors via ``ctx.extra["session_id"]``.
* Hop-by-hop response headers (``transfer-encoding``, etc.) are stripped.
* ``GracefulError`` raised inside an interceptor returns HTTP 429 with a
  ``session_budget_exhausted`` error code.
* Invalid JSON bodies return HTTP 400.
* The pipeline runs only on POST; other methods pass through unchanged.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware

from nemo_gym.adapters import install_middleware
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    RequestInterceptor,
    RequestToResponseInterceptor,
    get_context,
)


# ---------------------------------------------------------------------------
# Helper interceptors registered for use in these tests
# ---------------------------------------------------------------------------


class _HeaderCapture(RequestInterceptor):
    """Captures the last request the pipeline saw so tests can introspect ctx."""

    last_req: AdapterRequest | None = None

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        type(self).last_req = req
        return req


class _CapturingEcho(RequestToResponseInterceptor):
    """Captures the request *and* short-circuits with a canned response.

    Using a short-circuiting interceptor — rather than letting the request
    fall through to the FastAPI route — lets us assert on what the pipeline
    saw without depending on the route handler matching the session-prefixed
    URL. (FastAPI's router doesn't know about session prefixes; that's the
    middleware's job, and NEL's standalone proxy used to rewrite the URL
    before forwarding. Gym's middleware annotates ctx instead, and trusts
    the chain to handle routing semantics.)
    """

    last_req: AdapterRequest | None = None

    async def intercept_request(self, req: AdapterRequest) -> AdapterResponse:
        type(self).last_req = req
        return AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"id": "chatcmpl-canned", "echo_path": req.path},
            ctx=req.ctx,
        )


class _GracefulRaiser(RequestInterceptor):
    """Always raises ``GracefulError`` to simulate session-budget exhaustion."""

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        raise GracefulError("turns up")


class _BoomRaiser(RequestInterceptor):
    """Raises a generic (non-graceful) exception to exercise the middleware's
    500 arm."""

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        raise RuntimeError("kaboom")


def _install_helper_interceptors() -> None:
    """Make the helpers above resolvable via the registry."""
    mod = types.ModuleType("nemo_gym.adapters.interceptors._test_header_capture")
    mod.Interceptor = _HeaderCapture
    sys.modules[mod.__name__] = mod
    InterceptorRegistry.register("_test_header_capture", mod.__name__)

    mod2 = types.ModuleType("nemo_gym.adapters.interceptors._test_graceful_raiser")
    mod2.Interceptor = _GracefulRaiser
    sys.modules[mod2.__name__] = mod2
    InterceptorRegistry.register("_test_graceful_raiser", mod2.__name__)

    mod3 = types.ModuleType("nemo_gym.adapters.interceptors._test_capturing_echo")
    mod3.Interceptor = _CapturingEcho
    sys.modules[mod3.__name__] = mod3
    InterceptorRegistry.register("_test_capturing_echo", mod3.__name__)

    mod4 = types.ModuleType("nemo_gym.adapters.interceptors._test_boom_raiser")
    mod4.Interceptor = _BoomRaiser
    sys.modules[mod4.__name__] = mod4
    InterceptorRegistry.register("_test_boom_raiser", mod4.__name__)


# ---------------------------------------------------------------------------
# App / client factory
# ---------------------------------------------------------------------------


def _route_handler_default(body: dict) -> dict:
    """Default canned model-server response: echoes path and body."""
    return {
        "id": "chatcmpl-route",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "from route"},
            }
        ],
        "echo": body,
    }


def _build_test_app(
    interceptor_specs: list[dict[str, Any]] | None,
    *,
    extra_response_headers: dict[str, str] | None = None,
) -> FastAPI:
    """Build a FastAPI app whose ``/v1/chat/completions`` POST route returns
    a canned response, with the adapter middleware optionally installed on top.

    ``extra_response_headers`` is appended to the model-server's reply so
    tests can verify that hop-by-hop headers added by the (mock) upstream
    are stripped by the middleware before reaching the client.
    """
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        return JSONResponse(
            content=_route_handler_default(body),
            headers=extra_response_headers or {},
        )

    install_middleware(app, interceptor_specs)
    return app


@pytest.fixture(autouse=True)
def _setup_registry() -> None:
    _install_helper_interceptors()
    _HeaderCapture.last_req = None
    _CapturingEcho.last_req = None


# ---------------------------------------------------------------------------
# Session-id parsing
# ---------------------------------------------------------------------------


def test_session_id_parsed_into_ctx_extra() -> None:
    """``POST /s/<hex>/v1/...`` strips the prefix and exposes the id on ctx.

    Uses a short-circuiting interceptor (``_test_capturing_echo``) so the
    request never falls through to the FastAPI router — which doesn't know
    about session-prefixed URLs. The middleware-side behavior being verified
    is purely that the pipeline sees ``ctx.extra["session_id"]`` set to the
    expected hex string and the stripped path.
    """
    app = _build_test_app([{"name": "_test_capturing_echo"}])
    with TestClient(app) as client:
        resp = client.post(
            "/s/deadbeef1234/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
    assert resp.status_code == 200
    assert _CapturingEcho.last_req is not None
    assert _CapturingEcho.last_req.ctx.extra.get("session_id") == "deadbeef1234"
    # The path the interceptor saw is the clean one with the prefix removed.
    assert _CapturingEcho.last_req.path == "/v1/chat/completions"


def test_no_session_id_for_plain_path() -> None:
    """A plain ``/v1/...`` POST exposes no session_id on the interceptor ctx."""
    app = _build_test_app([{"name": "_test_capturing_echo"}])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
    assert resp.status_code == 200
    assert _CapturingEcho.last_req is not None
    assert "session_id" not in _CapturingEcho.last_req.ctx.extra
    assert _CapturingEcho.last_req.path == "/v1/chat/completions"


def test_session_prefixed_post_routes_to_real_route() -> None:
    """A session-prefixed ``POST /s/<hex>/v1/chat/completions`` must route to the
    real ``/v1/chat/completions`` handler — not 404.

    The middleware rewrites ``request.scope['path']`` / ``raw_path`` to the
    prefix-stripped path before ``call_next`` so the FastAPI router can match
    it. The session id is still parsed onto the request ctx, which the routed
    handler observes via the shared ``get_context()``.
    """
    captured: dict[str, Any] = {}
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        captured["session_id"] = get_context().extra.get("session_id")
        return JSONResponse(_route_handler_default(body))

    install_middleware(app, [{"name": "logging"}])

    with TestClient(app) as client:
        resp = client.post(
            "/s/deadbeef/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
    # Routed to the real handler (not a 404 from the unstripped /s/... path).
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == "chatcmpl-route"
    # The session id was parsed off the prefix and is visible to the handler.
    assert captured["session_id"] == "deadbeef"


# ---------------------------------------------------------------------------
# Hop-by-hop header filtering
# ---------------------------------------------------------------------------


def test_hop_by_hop_headers_stripped_from_response() -> None:
    """Hop-by-hop headers added by the underlying handler are stripped
    before being returned to the client."""
    app = _build_test_app(
        [{"name": "logging"}],
        extra_response_headers={
            "x-upstream-marker": "kept",
            "transfer-encoding": "chunked",  # must be stripped
            "connection": "keep-alive",  # must be stripped
        },
    )
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "test", "messages": []})
    assert resp.status_code == 200
    keys = {k.lower() for k in resp.headers}
    assert "transfer-encoding" not in keys
    assert "connection" not in keys
    # Non hop-by-hop custom headers must be preserved.
    assert resp.headers.get("x-upstream-marker") == "kept"


# ---------------------------------------------------------------------------
# GracefulError → 429
# ---------------------------------------------------------------------------


def test_graceful_error_returns_429() -> None:
    """When an interceptor raises ``GracefulError``, the middleware emits
    HTTP 429 with a ``session_budget_exhausted`` error code."""
    app = _build_test_app([{"name": "_test_graceful_raiser"}])
    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={"model": "test", "messages": []})
    assert resp.status_code == 429
    body = resp.json()
    assert body["error"]["code"] == "session_budget_exhausted"
    assert "turns up" in body["error"]["message"]


# ---------------------------------------------------------------------------
# Generic pipeline error → 500
# ---------------------------------------------------------------------------


def test_generic_pipeline_exception_returns_500() -> None:
    """A non-graceful exception raised inside the pipeline is caught by the
    middleware and surfaced as HTTP 500 (not propagated to the ASGI server)."""
    app = _build_test_app([{"name": "_test_boom_raiser"}])
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/v1/chat/completions", json={"model": "test", "messages": []})
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "server_error"


# ---------------------------------------------------------------------------
# Invalid JSON → 400
# ---------------------------------------------------------------------------


def test_invalid_json_body_returns_400() -> None:
    """A POST whose body is not valid JSON returns 400 before any
    interceptor runs (the pipeline must never see malformed input)."""
    app = _build_test_app([{"name": "_test_header_capture"}])
    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            content=b"not-json",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["type"] == "invalid_request_error"
    # The pipeline must not have run.
    assert _HeaderCapture.last_req is None


# ---------------------------------------------------------------------------
# Non-POST methods pass through (the pipeline only intercepts POST)
# ---------------------------------------------------------------------------


def test_get_request_passes_through_middleware() -> None:
    """GET requests bypass the pipeline entirely — the model server's own
    routes (e.g. /health) keep working unmodified.

    The default app doesn't define a GET route, so we expect 405 (Method
    Not Allowed) from the route handler — proving the middleware did not
    intercept and short-circuit with its own response."""
    app = _build_test_app([{"name": "_test_header_capture"}])
    with TestClient(app) as client:
        resp = client.get("/v1/chat/completions")
    assert resp.status_code == 405  # Method Not Allowed from FastAPI
    # Pipeline was not invoked for GET.
    assert _HeaderCapture.last_req is None


# ---------------------------------------------------------------------------
# Multi-valued response headers (Set-Cookie / SessionMiddleware) — the
# ecosystem-review blocker. ``SimpleResponsesAPIModel`` always has Starlette
# ``SessionMiddleware`` attached; before the fix, ``_starlette_response_to_adapter``
# rebuilt headers as a Python dict and silently collapsed duplicate
# ``Set-Cookie`` headers. This test mounts the adapter middleware on top of
# a FastAPI app that has ``SessionMiddleware`` installed plus an explicit
# route that sets an additional ``Set-Cookie`` — and asserts both cookies
# survive the round trip.
# ---------------------------------------------------------------------------


def test_set_cookie_headers_preserved_through_middleware() -> None:
    """Multiple ``Set-Cookie`` headers survive the adapter middleware.

    Pre-fix, ``_starlette_response_to_adapter`` collapsed response headers
    into a Python ``dict``, so a response carrying two ``Set-Cookie``
    headers (one from ``SessionMiddleware``, one from the route handler)
    arrived at the client with only the last value. Post-fix, headers
    flow as ``list[tuple[bytes, bytes]]`` end-to-end and duplicates are
    preserved.
    """
    app = FastAPI()
    # Match the order ``SimpleResponsesAPIModel.setup_webserver`` uses:
    # session middleware first, then the route handlers, then (after this
    # builder returns) the adapter middleware on top.
    app.add_middleware(SessionMiddleware, secret_key="test-secret-key")  # pragma: allowlist secret

    @app.post("/v1/chat/completions")
    async def _chat(request: Request, body: dict = Body(...)):
        # Mutate the session so SessionMiddleware actually emits its
        # ``Set-Cookie`` on the response path (it skips otherwise).
        request.session["test_key"] = "test_value"
        resp = JSONResponse(content={"ok": True, "echo": body})
        # Set a second cookie explicitly on top of the one SessionMiddleware
        # will inject. Two cookies on the same response is the exact case
        # the dict-collapse bug obliterated.
        resp.set_cookie("explicit_cookie", "explicit_value")
        return resp

    install_middleware(app, [{"name": "logging"}])

    with TestClient(app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": []},
        )
    assert resp.status_code == 200, f"body={resp.text!r}"
    # httpx exposes multi-valued headers via ``.get_list``; iterating the
    # underlying raw header list is the most portable cross-version check.
    raw_set_cookies = [v for k, v in resp.headers.raw if k.lower() == b"set-cookie"]
    assert len(raw_set_cookies) >= 2, (
        f"expected ≥2 Set-Cookie headers (SessionMiddleware + explicit), got: {raw_set_cookies!r}"
    )
    assert any(b"explicit_cookie=explicit_value" in v for v in raw_set_cookies), (
        f"explicit cookie missing from response: {raw_set_cookies!r}"
    )
    # SessionMiddleware's cookie is named "session" by default.
    assert any(v.lower().startswith(b"session=") for v in raw_set_cookies), (
        f"SessionMiddleware cookie missing from response: {raw_set_cookies!r}"
    )
