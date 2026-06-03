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
"""Middleware integration tests (Phase 1.5 — wrap, don't replace).

The middleware now layers on top of an existing FastAPI route handler:
REQUEST interceptors mutate the body, REQUEST_TO_RESPONSE interceptors may
short-circuit, otherwise ``call_next`` invokes the model server's own
handler, and RESPONSE interceptors observe the result. These tests verify:

* ``log_tokens`` (RESPONSE stage) observes the route handler's output when
  the chain has no short-circuiting interceptor.
* REQUEST-stage interceptors mutate the body the route handler sees.
* A short-circuiting REQUEST_TO_RESPONSE interceptor prevents the route
  handler from ever running.
* ``adapters=None`` / ``adapters=[]`` keep the middleware uninstalled.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.adapters import install_middleware
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
)


# ---------------------------------------------------------------------------
# Test fixtures: a minimal canned-response interceptor registered as a
# RequestToResponseInterceptor. Used by the short-circuit assertion test.
# ---------------------------------------------------------------------------


_CANNED_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello from middleware"},
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
}


class _CannedInterceptor(RequestToResponseInterceptor):
    """Returns ``_CANNED_BODY`` without touching the network."""

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest | AdapterResponse:
        return AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=dict(_CANNED_BODY),
            latency_ms=1.0,
            ctx=req.ctx,
        )


def _register_canned_interceptor() -> None:
    """Make the canned interceptor resolvable by ``InterceptorRegistry``."""
    module_name = "nemo_gym.adapters.interceptors._test_canned"
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        mod.Interceptor = _CannedInterceptor
        sys.modules[module_name] = mod
    InterceptorRegistry.register("_test_canned", module_name)


# A handler-visibility flag: the test route writes the request body it
# actually received into this list so tests can assert what the model
# server saw (e.g. mutated by REQUEST interceptors).
_route_seen: list[dict] = []


def _build_app() -> FastAPI:
    """Minimal FastAPI app with a single ``/v1/chat/completions`` POST route.

    The route stashes the parsed body into ``_route_seen`` and returns a
    sentinel body so the test can distinguish whether the request reached
    the underlying handler or was short-circuited by middleware.
    """
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat_completions(body: dict) -> dict:
        _route_seen.append(body)
        return {
            "id": "chatcmpl-route",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "?"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "from route"},
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
        }

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _register() -> None:
    _register_canned_interceptor()
    _route_seen.clear()


_SAMPLE_BODY = {
    "model": "test-model",
    "messages": [{"role": "user", "content": "hi"}],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_install_middleware_runs_pipeline_and_logging_emits_log(caplog) -> None:
    """A POST flows through the pipeline → route handler → RESPONSE stage.

    The model server's own route handler produces the response;
    ``logging`` (a request+response interceptor) emits records on both
    sides to prove the pipeline ran.
    """
    app = _build_app()
    install_middleware(
        app,
        [
            {"name": "logging", "config": {}},
        ],
    )

    with caplog.at_level(logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json=_SAMPLE_BODY)

    assert resp.status_code == 200, resp.text
    payload = resp.json()

    # The route handler ran (not a canned short-circuit).
    assert payload["choices"][0]["message"]["content"] == "from route"
    assert _route_seen == [_SAMPLE_BODY]

    # logging fires on both the request and response halves of the chain.
    log_records = [rec for rec in caplog.records if rec.name == "nemo_gym.adapters.interceptors.request_logging"]
    assert any("request POST /v1/chat/completions" in r.getMessage() for r in log_records)
    assert any("response status=200" in r.getMessage() for r in log_records)


def test_short_circuit_interceptor_prevents_route_handler() -> None:
    """A short-circuiting REQUEST_TO_RESPONSE interceptor skips ``call_next``.

    The route handler must never run, ``_route_seen`` stays empty, and the
    client receives the canned body.
    """
    app = _build_app()
    install_middleware(
        app,
        [
            {"name": "logging", "config": {}},
            {"name": "_test_canned", "config": {}},
        ],
    )

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_SAMPLE_BODY)

    assert resp.status_code == 200, resp.text
    assert resp.json()["choices"][0]["message"]["content"] == "hello from middleware"
    # Crucial: the underlying route handler was bypassed.
    assert _route_seen == [], "Route handler ran even though chain short-circuited"


def test_install_middleware_defaults_off_when_specs_is_none() -> None:
    """``adapters=None`` is a no-op: the route handler runs unchanged."""
    app = _build_app()
    install_middleware(app, None)

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_SAMPLE_BODY)

    assert resp.status_code == 200
    # Route handler reply (not the canned interceptor one) — proves no
    # middleware was inserted.
    assert resp.json()["choices"][0]["message"]["content"] == "from route"


def test_install_middleware_defaults_off_when_specs_is_empty() -> None:
    """``adapters=[]`` is also a no-op (mirrors the None case)."""
    app = _build_app()
    install_middleware(app, [])

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json=_SAMPLE_BODY)

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "from route"
