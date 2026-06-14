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
"""Unit coverage for adapter internals: context vars, response helpers, the
logging preview, middleware request/response conversion, proxy guards, and the
registry/pipeline error paths."""

from __future__ import annotations

import contextvars
import json

import pytest
from fastapi import FastAPI
from starlette.responses import Response, StreamingResponse

from nemo_gym.adapters import middleware as mw
from nemo_gym.adapters import proxy
from nemo_gym.adapters import registry as reg
from nemo_gym.adapters.interceptors.request_logging import _MAX, _trunc_preview
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    InterceptorContext,
    ResponseInterceptor,
    get_context,
)


# ── types: context + response helpers ────────────────────────────────


def test_get_context_creates_and_caches_in_fresh_context():
    # A brand-new contextvars.Context has no value, exercising the LookupError
    # branch that lazily creates and stores a context.
    captured: dict = {}

    def _run():
        captured["first"] = get_context()
        captured["second"] = get_context()

    contextvars.Context().run(_run)
    assert isinstance(captured["first"], InterceptorContext)
    assert captured["first"] is captured["second"]


def test_adapter_response_ok_property():
    assert AdapterResponse(status_code=200, headers={}, body={}).ok is True
    assert AdapterResponse(status_code=302, headers={}, body={}).ok is True
    assert AdapterResponse(status_code=404, headers={}, body={}).ok is False
    assert AdapterResponse(status_code=500, headers={}, body={}).ok is False


# ── request_logging._trunc_preview branches ──────────────────────────


def test_trunc_preview_decodes_bytes():
    assert _trunc_preview(b"hello") == "hello"


def test_trunc_preview_stringifies_other_types():
    assert _trunc_preview(12345) == "12345"


def test_trunc_preview_truncates_long_text():
    out = _trunc_preview("x" * (_MAX + 50))
    assert out.endswith("...")
    assert len(out) == _MAX + 3


# ── middleware response/request conversion helpers ───────────────────


async def test_starlette_to_adapter_streaming_str_chunks_falls_back_to_bytes():
    async def gen():
        yield "he"
        yield "llo"

    # Streaming body declared json but not valid json -> raw bytes fallback,
    # and str chunks get utf-8 encoded.
    resp = StreamingResponse(gen(), media_type="application/json")
    out = await mw._starlette_response_to_adapter(resp, InterceptorContext())
    assert out.body == b"hello"


async def test_starlette_to_adapter_plain_json_body():
    resp = Response(content=json.dumps({"a": 1}), media_type="application/json")
    out = await mw._starlette_response_to_adapter(resp, InterceptorContext())
    assert out.body == {"a": 1}


async def test_starlette_to_adapter_non_json_content_type_kept_as_bytes():
    resp = Response(content=b"raw", media_type="text/plain")
    out = await mw._starlette_response_to_adapter(resp, InterceptorContext())
    assert out.body == b"raw"


def test_adapter_response_to_starlette_bytes_body():
    out = mw._adapter_response_to_starlette(AdapterResponse(status_code=200, headers={"X-A": "1"}, body=b"raw-bytes"))
    assert isinstance(out, Response)
    assert out.body == b"raw-bytes"


def test_install_middleware_noop_when_empty():
    app = FastAPI()
    before = len(app.user_middleware)
    mw.install_middleware(app, None)
    mw.install_middleware(app, [])
    assert len(app.user_middleware) == before


def test_install_middleware_rejects_endpoint_interceptor():
    with pytest.raises(ValueError, match="endpoint"):
        mw.install_middleware(FastAPI(), [{"name": "endpoint", "config": {"upstream_url": "http://x"}}])


# ── proxy guards / helpers ───────────────────────────────────────────


def test_bind_port_returns_preferred():
    assert proxy._bind_port("127.0.0.1", 54321) == 54321


def test_bind_port_allocates_ephemeral_when_zero():
    port = proxy._bind_port("127.0.0.1", 0)
    assert isinstance(port, int) and port > 0


def test_wait_for_health_raises_on_timeout():
    # Nothing is listening; the loop should give up and raise.
    with pytest.raises(RuntimeError, match="did not become healthy"):
        proxy._wait_for_health("http://127.0.0.1:1", timeout=0.2)


def test_start_adapter_proxy_rejects_remote_host():
    with pytest.raises(ValueError, match="unsafe_allow_remote"):
        proxy.start_adapter_proxy("http://up", [], host="0.0.0.0")


def test_start_adapter_proxy_rejects_endpoint_interceptor():
    with pytest.raises(ValueError, match="endpoint"):
        proxy.start_adapter_proxy("http://up", [{"name": "endpoint"}])


# ── registry error paths ─────────────────────────────────────────────


@pytest.fixture
def _restore_extra():
    before = dict(reg._EXTRA)
    yield
    reg._EXTRA.clear()
    reg._EXTRA.update(before)


def test_resolve_class_import_error(_restore_extra):
    InterceptorRegistry.register("badmod", "nemo_gym.adapters.interceptors.does_not_exist")
    with pytest.raises(ValueError, match="Cannot import interceptor module"):
        InterceptorRegistry.resolve_class("badmod")


def test_resolve_class_missing_interceptor_attr(_restore_extra):
    # stdlib json imports fine but has no ``Interceptor`` attribute.
    InterceptorRegistry.register("nointercept", "json")
    with pytest.raises(ValueError, match="does not expose an 'Interceptor' class"):
        InterceptorRegistry.resolve_class("nointercept")


# ── pipeline error/best-effort paths ─────────────────────────────────


class _BogusStageInterceptor:
    stage = "not-a-real-stage"

    async def intercept_request(self, req):  # pragma: no cover - never called
        return req


def test_pipeline_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage"):
        AdapterPipeline([_BogusStageInterceptor()])


class _BestEffortFailingResponse(ResponseInterceptor):
    best_effort = True

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        raise RuntimeError("swallow me")


class _HardFailingResponse(ResponseInterceptor):
    best_effort = False

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        raise RuntimeError("propagate me")


def _req() -> AdapterRequest:
    return AdapterRequest(method="POST", path="/p", headers={}, body={}, ctx=InterceptorContext())


async def _upstream(req: AdapterRequest) -> AdapterResponse:
    return AdapterResponse(status_code=200, headers={}, body={"ok": True}, ctx=req.ctx)


async def test_response_interceptor_best_effort_failure_is_swallowed():
    pipe = AdapterPipeline([_BestEffortFailingResponse()])
    resp = await pipe.process(_req(), upstream_call=_upstream)
    assert resp.status_code == 200
    assert resp.body == {"ok": True}


async def test_response_interceptor_failure_propagates_when_not_best_effort():
    pipe = AdapterPipeline([_HardFailingResponse()])
    with pytest.raises(RuntimeError, match="propagate me"):
        await pipe.process(_req(), upstream_call=_upstream)
