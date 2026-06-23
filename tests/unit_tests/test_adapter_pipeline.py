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
"""AdapterPipeline behavior tests — ported from NEL ``tests/test_adapters/test_pipeline.py``.

Mechanical port from ``nemo_evaluator.adapters.*`` to ``nemo_gym.adapters.*``
plus one additional Phase-1.5 test for the ``upstream_call`` parameter on
``AdapterPipeline.process()``.
"""

from __future__ import annotations

import pytest

from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
    get_context,
)


def _req(**body_overrides):
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    body.update(body_overrides)
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=body,
        ctx=InterceptorContext(),
    )


class AddHeaderInterceptor(RequestInterceptor):
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        req.headers["x-test"] = "1"
        return req


class FixedEndpoint(RequestToResponseInterceptor):
    def __init__(self):
        self.last_request = None

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest | AdapterResponse:
        self.last_request = req
        return AdapterResponse(
            status_code=200,
            headers={},
            body={"result": "ok"},
            ctx=req.ctx,
        )


class AppendBodyInterceptor(ResponseInterceptor):
    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        if isinstance(resp.body, dict):
            resp.body["appended"] = True
        return resp


class TrackingResponseInterceptor(ResponseInterceptor):
    def __init__(self):
        self.called = False

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        self.called = True
        return resp


class CacheHitInterceptor(RequestToResponseInterceptor):
    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest | AdapterResponse:
        return AdapterResponse(
            status_code=200,
            headers={},
            body={"cached": True},
            ctx=req.ctx,
        )


class FailingRequestInterceptor(RequestInterceptor):
    def __init__(self, *, best_effort=False):
        self.best_effort = best_effort

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        raise RuntimeError("boom")


def test_stage_order_validation():
    with pytest.raises(ValueError, match="Invalid interceptor order"):
        AdapterPipeline([AppendBodyInterceptor(), AddHeaderInterceptor()])


async def test_request_then_endpoint_then_response():
    endpoint = FixedEndpoint()
    pipeline = AdapterPipeline(
        [
            AddHeaderInterceptor(),
            endpoint,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.status_code == 200
    assert resp.body["result"] == "ok"
    assert endpoint.last_request.headers["x-test"] == "1"


async def test_short_circuit_skips_endpoint():
    """When a RequestToResponseInterceptor short-circuits, later request-side
    interceptors (like the endpoint) are skipped, but response interceptors
    still run so they can inspect the short-circuited response."""
    endpoint = FixedEndpoint()
    tracker = TrackingResponseInterceptor()
    pipeline = AdapterPipeline(
        [
            CacheHitInterceptor(),
            endpoint,
            tracker,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.body["cached"] is True
    assert endpoint.last_request is None  # endpoint was skipped
    assert tracker.called is True  # response interceptors still fire


async def test_best_effort_continues_on_error():
    pipeline = AdapterPipeline(
        [
            FailingRequestInterceptor(best_effort=True),
            FixedEndpoint(),
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.status_code == 200


async def test_non_best_effort_raises():
    pipeline = AdapterPipeline(
        [
            FailingRequestInterceptor(best_effort=False),
            FixedEndpoint(),
        ]
    )
    with pytest.raises(RuntimeError, match="boom"):
        await pipeline.process(_req())


async def test_response_interceptors_run_after_endpoint():
    """Response interceptors placed after the endpoint in the chain must run."""
    appender = AppendBodyInterceptor()
    pipeline = AdapterPipeline(
        [
            AddHeaderInterceptor(),
            FixedEndpoint(),
            appender,
        ]
    )
    resp = await pipeline.process(_req())
    assert resp.body.get("appended") is True


async def test_multiple_response_interceptors_run_in_reverse():
    """Multiple response interceptors run in reverse chain order."""
    order: list[str] = []

    class First(ResponseInterceptor):
        async def intercept_response(self, resp):
            order.append("first")
            return resp

    class Second(ResponseInterceptor):
        async def intercept_response(self, resp):
            order.append("second")
            return resp

    pipeline = AdapterPipeline(
        [
            FixedEndpoint(),
            First(),
            Second(),
        ]
    )
    await pipeline.process(_req())
    assert order == ["second", "first"]


# ---------------------------------------------------------------------------
# Phase 1.5 addition: upstream_call invoked when no interceptor short-circuits
# ---------------------------------------------------------------------------


async def test_upstream_call_invoked_when_chain_does_not_short_circuit():
    """When ``upstream_call`` is provided and the chain has no
    ``RequestToResponseInterceptor`` that short-circuits, the upstream is
    called with the (post-REQUEST-stage) request and its response flows
    through ``ResponseInterceptor`` instances in reverse order.

    This is the Phase 1.5 wrap-not-replace contract: an ``endpoint``
    interceptor is no longer required in every chain — the middleware can
    supply the upstream via ``upstream_call`` instead.
    """
    upstream_seen: list[AdapterRequest] = []
    appender = AppendBodyInterceptor()

    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        upstream_seen.append(req)
        return AdapterResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"result": "from-upstream"},
            ctx=req.ctx,
        )

    pipeline = AdapterPipeline(
        [
            AddHeaderInterceptor(),  # REQUEST stage
            appender,  # RESPONSE stage
        ]
    )
    resp = await pipeline.process(_req(), upstream_call=_upstream)

    # Upstream was invoked exactly once with the (mutated) request.
    assert len(upstream_seen) == 1
    assert upstream_seen[0].headers["x-test"] == "1"

    # Response flowed back through the RESPONSE stage.
    assert resp.status_code == 200
    assert resp.body["result"] == "from-upstream"
    assert resp.body["appended"] is True


async def test_upstream_call_skipped_when_chain_short_circuits():
    """A short-circuiting ``RequestToResponseInterceptor`` skips
    ``upstream_call`` entirely — the upstream must never be invoked."""
    upstream_seen: list[AdapterRequest] = []

    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        upstream_seen.append(req)
        return AdapterResponse(status_code=200, headers={}, body={"result": "from-upstream"}, ctx=req.ctx)

    pipeline = AdapterPipeline(
        [
            CacheHitInterceptor(),  # short-circuits
        ]
    )
    resp = await pipeline.process(_req(), upstream_call=_upstream)
    assert upstream_seen == []
    assert resp.body == {"cached": True}


# ===========================================================================
# Library-import path — explicit positive coverage that the NEL-style
# ``AdapterPipeline.process(req)`` call shape (no ``upstream_call``, no
# FastAPI middleware host) still works end-to-end when a
# ``RequestToResponseInterceptor`` short-circuits the chain. This is the
# shape an external library user would import; we pin it here so the
# Phase-1.5 ``upstream_call`` parameter never silently becomes mandatory.
# ===========================================================================


async def test_library_import_no_upstream_call_no_middleware_short_circuits():
    """A NEL-style library user instantiates ``AdapterPipeline`` directly and
    calls ``process(req)`` with no ``upstream_call`` and no FastAPI host.
    A short-circuiting interceptor must produce the response."""
    endpoint = FixedEndpoint()  # short-circuits with status 200, body={"result": "ok"}
    tracker = TrackingResponseInterceptor()
    pipeline = AdapterPipeline([endpoint, tracker])

    # Call without ``upstream_call`` — the library-import surface.
    resp = await pipeline.process(_req())

    assert isinstance(resp, AdapterResponse)
    assert resp.status_code == 200
    assert resp.body == {"result": "ok"}
    assert endpoint.last_request is not None  # endpoint was the source of the response
    assert tracker.called is True  # RESPONSE-stage interceptor still ran


async def test_library_import_request_stage_mutation_reaches_short_circuit():
    """Request-stage interceptors must mutate the request before the
    short-circuiting endpoint sees it, even when no ``upstream_call`` is
    provided. Pins the library-import contract end-to-end."""
    endpoint = FixedEndpoint()
    pipeline = AdapterPipeline([AddHeaderInterceptor(), endpoint])

    resp = await pipeline.process(_req())

    # The mutation from AddHeaderInterceptor must have landed on the request
    # that FixedEndpoint observed.
    assert endpoint.last_request is not None
    assert endpoint.last_request.headers["x-test"] == "1"
    assert resp.status_code == 200
    assert resp.body == {"result": "ok"}


# ===========================================================================
# Context sharing — ``process()`` reuses the request's own ``ctx`` so the
# global ``get_context()`` observed by interceptors sees the same ``ctx.extra``
# (e.g. a middleware-parsed ``session_id``) as ``request.ctx``.
# ===========================================================================


async def test_process_exposes_request_ctx_via_get_context():
    """After ``process()``, ``get_context()`` returns the request's own ctx —
    including any ``ctx.extra`` the caller populated (e.g. a session id)."""
    req = _req()
    req.ctx.extra["session_id"] = "sess-xyz"
    pipeline = AdapterPipeline([FixedEndpoint()])

    await pipeline.process(req)

    ctx = get_context()
    # Same object, not a fresh context — so extra (session_id etc.) is shared.
    assert ctx is req.ctx
    assert ctx.extra == req.ctx.extra
    assert ctx.extra.get("session_id") == "sess-xyz"


# ===========================================================================
# GracefulError is a control-flow signal: it must propagate even from a
# ``best_effort=True`` interceptor (which otherwise swallows exceptions),
# in both the request phase and the response phase.
# ===========================================================================


class _BestEffortGracefulRequest(RequestInterceptor):
    best_effort = True

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        raise GracefulError("budget exhausted (request)")


class _BestEffortGracefulResponse(ResponseInterceptor):
    best_effort = True

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        raise GracefulError("budget exhausted (response)")


async def test_graceful_error_propagates_from_best_effort_request_interceptor():
    """A ``best_effort=True`` request interceptor that raises ``GracefulError``
    must NOT have it swallowed — the control-flow signal propagates."""
    pipeline = AdapterPipeline([_BestEffortGracefulRequest(), FixedEndpoint()])
    with pytest.raises(GracefulError, match="request"):
        await pipeline.process(_req())


async def test_graceful_error_propagates_from_best_effort_response_interceptor():
    """A ``best_effort=True`` response interceptor that raises ``GracefulError``
    must NOT have it swallowed — the control-flow signal propagates."""
    pipeline = AdapterPipeline([FixedEndpoint(), _BestEffortGracefulResponse()])
    with pytest.raises(GracefulError, match="response"):
        await pipeline.process(_req())
