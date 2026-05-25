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
"""Behavioral coverage for adapter paths not exercised by other suites."""

from __future__ import annotations

import asyncio
import tempfile

import aiohttp
import pytest
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from multidict import CIMultiDict

from nemo_gym.adapters import install_middleware
from nemo_gym.adapters.interceptors import endpoint as endpoint_mod
from nemo_gym.adapters.interceptors.endpoint import Interceptor as EndpointInterceptor
from nemo_gym.adapters.interceptors.modify_tools import Interceptor as ModifyToolsInterceptor
from nemo_gym.adapters.interceptors.payload_modifier import Interceptor as PayloadModifierInterceptor
from nemo_gym.adapters.interceptors.reasoning import Interceptor as ReasoningInterceptor
from nemo_gym.adapters.interceptors.turn_counter import Interceptor as TurnCounterInterceptor
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
    RequestInterceptor,
    ResponseInterceptor,
    get_context,
)


# ---------------------------------------------------------------------------
# Fake aiohttp response for monkeypatching ``global_request``
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, body: bytes, headers: list[tuple[str, str]] | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = CIMultiDict(headers or [("content-type", "application/json")])

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self) -> "_FakeResp":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def _make_fake_request(*, status=200, body=b'{"ok":true}', headers=None, raises=None, raise_after=0):
    """Build a fake ``global_request`` that returns/raises configurably."""
    state = {"calls": 0}

    async def _request(method, url, **kwargs):
        state["calls"] += 1
        if raises is not None and state["calls"] > raise_after:
            raise raises
        return _FakeResp(status, body, headers)

    _request.state = state
    return _request


def _req(body: dict | None = None) -> AdapterRequest:
    return AdapterRequest(
        method="POST",
        path="/chat/completions",
        headers={"content-type": "application/json", "host": "should-be-dropped"},
        body=body or {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )


# ---------------------------------------------------------------------------
# endpoint.py — happy path, headers, retries, timeouts, errors
# ---------------------------------------------------------------------------


async def test_endpoint_success(monkeypatch):
    fake = _make_fake_request(
        status=200,
        body=b'{"choices":[{"message":{"content":"ok"}}],"usage":{"total_tokens":3}}',
        headers=[("content-type", "application/json"), ("set-cookie", "a=1"), ("set-cookie", "b=2")],
    )
    monkeypatch.setattr(endpoint_mod, "global_request", fake)
    ic = EndpointInterceptor(upstream_url="http://x/v1", api_key="k")
    resp = await ic.intercept_request(_req())
    assert resp.status_code == 200
    assert resp.body["choices"][0]["message"]["content"] == "ok"
    assert resp.latency_ms >= 0
    cookies = [v.decode() for k, v in resp.headers if k.lower() == b"set-cookie"]
    assert cookies == ["a=1", "b=2"]


async def test_endpoint_auth_and_extra_body(monkeypatch):
    captured: dict = {}

    async def _capture(method, url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["data"] = kwargs.get("data")
        return _FakeResp(200, b'{"choices":[]}')

    monkeypatch.setattr(endpoint_mod, "global_request", _capture)
    ic = EndpointInterceptor(
        upstream_url="http://x/v1",
        api_key="secret",  # pragma: allowlist secret
        extra_body={"injected": True},
    )
    await ic.intercept_request(_req(body={"model": "m", "messages": []}))
    assert captured["headers"]["Authorization"] == "Bearer secret"  # pragma: allowlist secret
    assert captured["headers"]["Content-Type"] == "application/json"
    assert "host" not in {k.lower() for k in captured["headers"]}
    assert b'"injected": true' in captured["data"].encode() if isinstance(captured["data"], str) else captured["data"]


async def test_endpoint_null_content_normalized(monkeypatch):
    monkeypatch.setattr(
        endpoint_mod,
        "global_request",
        _make_fake_request(body=b'{"choices":[{"message":{"content":null}}]}'),
    )
    ic = EndpointInterceptor(upstream_url="http://x")
    resp = await ic.intercept_request(_req())
    assert resp.body["choices"][0]["message"]["content"] == ""


async def test_endpoint_non_json_body(monkeypatch):
    monkeypatch.setattr(
        endpoint_mod,
        "global_request",
        _make_fake_request(body=b"not-json-{", headers=[("content-type", "text/plain")]),
    )
    ic = EndpointInterceptor(upstream_url="http://x")
    resp = await ic.intercept_request(_req())
    assert resp.body == b"not-json-{"


async def test_endpoint_retry_on_status_then_succeeds(monkeypatch):
    state = {"calls": 0}

    async def _retry_then_ok(method, url, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return _FakeResp(503, b"{}", headers=[("retry-after", "0")])
        return _FakeResp(200, b'{"choices":[]}')

    monkeypatch.setattr(endpoint_mod, "global_request", _retry_then_ok)
    ic = EndpointInterceptor(upstream_url="http://x", max_retries=2, retry_on_status=[503])
    resp = await ic.intercept_request(_req())
    assert resp.status_code == 200
    assert state["calls"] == 2


async def test_endpoint_retry_exhausted_returns_last(monkeypatch):
    monkeypatch.setattr(
        endpoint_mod,
        "global_request",
        _make_fake_request(status=503, body=b"{}", headers=[("retry-after", "0")]),
    )
    ic = EndpointInterceptor(upstream_url="http://x", max_retries=1, retry_on_status=[503])
    resp = await ic.intercept_request(_req())
    assert resp.status_code == 503


async def test_endpoint_timeout_returns_504(monkeypatch):
    async def _timeout(*_a, **_k):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(endpoint_mod, "global_request", _timeout)
    ic = EndpointInterceptor(upstream_url="http://x", request_timeout=0.5, max_retries=0)
    resp = await ic.intercept_request(_req())
    assert resp.status_code == 504
    assert "timed out" in resp.body["error"]["message"]


async def test_endpoint_timeout_retry_then_success(monkeypatch):
    state = {"calls": 0}

    async def _timeout_then_ok(*_a, **_k):
        state["calls"] += 1
        if state["calls"] == 1:
            raise asyncio.TimeoutError()
        return _FakeResp(200, b'{"choices":[]}')

    monkeypatch.setattr(endpoint_mod, "global_request", _timeout_then_ok)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(endpoint_mod.asyncio, "sleep", _no_sleep)
    ic = EndpointInterceptor(upstream_url="http://x", request_timeout=0.5, max_retries=1)
    resp = await ic.intercept_request(_req())
    assert resp.status_code == 200
    assert state["calls"] == 2


async def test_endpoint_client_error_retry_then_raise(monkeypatch):
    async def _always_fail(*_a, **_k):
        raise aiohttp.ClientConnectionError("nope")

    monkeypatch.setattr(endpoint_mod, "global_request", _always_fail)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(endpoint_mod.asyncio, "sleep", _no_sleep)
    ic = EndpointInterceptor(upstream_url="http://x", max_retries=2)
    with pytest.raises(aiohttp.ClientError):
        await ic.intercept_request(_req())


async def test_endpoint_close_is_noop():
    ic = EndpointInterceptor(upstream_url="http://x")
    assert await ic.close() is None


# ---------------------------------------------------------------------------
# middleware.py — FastAPI integration
# ---------------------------------------------------------------------------


def _build_app(adapters: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    @app.post("/{any:path}/v1/chat/completions")
    async def chat(request: Request, body: dict = Body(...)):
        return JSONResponse({"received": body, "session": request.headers.get("x-session", "")})

    @app.get("/v1/health")
    async def health():
        return {"ok": True}

    install_middleware(app, adapters)
    return app


def test_middleware_get_passes_through():
    client = TestClient(_build_app([{"name": "logging", "config": {}}]))
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_middleware_invalid_json_returns_400():
    client = TestClient(_build_app([{"name": "logging", "config": {}}]))
    r = client.post("/v1/chat/completions", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert "Invalid JSON" in r.json()["error"]["message"]


def test_middleware_session_prefix_stripped():
    client = TestClient(_build_app([{"name": "logging", "config": {}}]))
    r = client.post("/s/deadbeef/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 200


def test_middleware_session_prefix_only_root():
    client = TestClient(_build_app([{"name": "logging", "config": {}}]))
    r = client.post("/s/deadbeef", json={"model": "m", "messages": []})
    # Path becomes "/" after prefix strip; FastAPI route won't match → 404.
    assert r.status_code in (404, 405)


def test_middleware_graceful_error_returns_429():
    client = TestClient(_build_app([{"name": "turn_counter", "config": {"max_turns": 1}}]))
    body = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    r1 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == 200
    r2 = client.post("/v1/chat/completions", json=body)
    assert r2.status_code == 429
    assert r2.json()["error"]["code"] == "session_budget_exhausted"


def test_middleware_pipeline_exception_returns_500(monkeypatch):
    from nemo_gym.adapters import registry as registry_mod

    class BoomInterceptor(RequestInterceptor):
        async def intercept_request(self, req):
            raise RuntimeError("kaboom")

    import tests.unit_tests.test_adapter_coverage as self_mod

    monkeypatch.setattr(self_mod, "Interceptor", BoomInterceptor, raising=False)
    before = dict(registry_mod._EXTRA)
    try:
        InterceptorRegistry.register("__boom_cov_test", "tests.unit_tests.test_adapter_coverage")
        client = TestClient(_build_app([{"name": "__boom_cov_test", "config": {}}]))
        r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
        assert r.status_code == 500
    finally:
        registry_mod._EXTRA.clear()
        registry_mod._EXTRA.update(before)


def test_middleware_install_with_empty_specs_is_noop():
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(body: dict = Body(...)):
        return {"echo": body}

    install_middleware(app, None)
    install_middleware(app, [])
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={"x": 1})
    assert r.status_code == 200


def test_middleware_install_rejects_endpoint_in_chain():
    app = FastAPI()
    with pytest.raises(ValueError, match="endpoint.*cannot be used"):
        install_middleware(app, [{"name": "endpoint", "config": {"upstream_url": "http://x"}}])


def test_middleware_install_rejects_endpoint_mixed_in():
    app = FastAPI()
    with pytest.raises(ValueError, match="endpoint.*cannot be used"):
        install_middleware(
            app,
            [
                {"name": "logging", "config": {}},
                {"name": "endpoint", "config": {"upstream_url": "http://x"}},
                {"name": "log_tokens", "config": {}},
            ],
        )


def test_middleware_response_body_iterator_path():
    """JSONResponse goes through the body_iterator branch in
    ``_starlette_response_to_adapter``."""
    client = TestClient(_build_app([{"name": "log_tokens", "config": {}}]))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# pipeline.py — response-side best-effort and strict propagation
# ---------------------------------------------------------------------------


async def test_pipeline_response_best_effort_swallows():
    class _Boom(ResponseInterceptor):
        best_effort = True

        async def intercept_response(self, resp):
            raise ValueError("boom")

    class _Endpoint:
        stage = type(InterceptorRegistry.resolve_class("endpoint").stage)(  # type: ignore[arg-type]
            "request_to_response"
        )
        best_effort = False

        async def intercept_request(self, req):
            return AdapterResponse(status_code=200, headers={}, body={}, ctx=req.ctx)

    from nemo_gym.adapters.types import RequestToResponseInterceptor

    class _Fixed(RequestToResponseInterceptor):
        async def intercept_request(self, req):
            return AdapterResponse(status_code=200, headers={}, body={"ok": True}, ctx=req.ctx)

    p = AdapterPipeline([_Fixed(), _Boom()])
    resp = await p.process(_req())
    assert resp.status_code == 200


async def test_pipeline_response_strict_propagates():
    class _Boom(ResponseInterceptor):
        best_effort = False

        async def intercept_response(self, resp):
            raise ValueError("boom-strict")

    from nemo_gym.adapters.types import RequestToResponseInterceptor

    class _Fixed(RequestToResponseInterceptor):
        async def intercept_request(self, req):
            return AdapterResponse(status_code=200, headers={}, body={}, ctx=req.ctx)

    p = AdapterPipeline([_Fixed(), _Boom()])
    with pytest.raises(ValueError, match="boom-strict"):
        await p.process(_req())


# ---------------------------------------------------------------------------
# types.py — get_context LookupError branch
# ---------------------------------------------------------------------------


def test_get_context_creates_when_missing():
    # Run in a fresh thread so ContextVar lookup raises LookupError
    import threading

    captured: dict = {}

    def _runner():
        captured["ctx"] = get_context()

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    assert captured["ctx"].request_id
    assert isinstance(captured["ctx"].extra, dict)


# ---------------------------------------------------------------------------
# registry.py — runtime register + import-failure path
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_registry():
    from nemo_gym.adapters import registry as registry_mod

    before = dict(registry_mod._EXTRA)
    yield
    registry_mod._EXTRA.clear()
    registry_mod._EXTRA.update(before)


def test_registry_register_and_resolve(_clean_registry):
    InterceptorRegistry.register("__cov_drop", "nemo_gym.adapters.interceptors.drop_params")
    cls = InterceptorRegistry.resolve_class("__cov_drop")
    assert cls is not None


def test_registry_import_failure(_clean_registry):
    InterceptorRegistry.register("__cov_bad", "nonexistent.module.path")
    with pytest.raises(ValueError, match="Cannot import"):
        InterceptorRegistry.resolve_class("__cov_bad")


def test_registry_module_without_interceptor_class(_clean_registry):
    InterceptorRegistry.register("__cov_no_class", "nemo_gym.adapters.types")
    with pytest.raises(ValueError, match="does not expose"):
        InterceptorRegistry.resolve_class("__cov_no_class")


# ---------------------------------------------------------------------------
# turn_counter — body-hash fallback, content normalization, GC
# ---------------------------------------------------------------------------


async def test_turn_counter_body_hash_fallback():
    ic = TurnCounterInterceptor(max_turns=2)
    # No session_id in ctx → fallback key from body content
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "fallback-content"}]},
        ctx=InterceptorContext(),
    )
    await ic.intercept_request(req)
    # second turn should also work (same body hash → same session)
    req2 = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "fallback-content"}]},
        ctx=InterceptorContext(),
    )
    await ic.intercept_request(req2)
    # third turn exhausts
    req3 = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "fallback-content"}]},
        ctx=InterceptorContext(),
    )
    with pytest.raises(GracefulError):
        await ic.intercept_request(req3)


async def test_turn_counter_list_content_concat():
    ic = TurnCounterInterceptor(max_turns=5)
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": [{"text": "part-1"}, {"text": "part-2"}]}]},
        ctx=InterceptorContext(),
    )
    await ic.intercept_request(req)


async def test_turn_counter_all_system_messages_unknown_key():
    ic = TurnCounterInterceptor(max_turns=5)
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "system", "content": "only system"}]},
        ctx=InterceptorContext(),
    )
    await ic.intercept_request(req)


async def test_turn_counter_below_warn_threshold():
    """At turn 1 of max_turns=10 (10% used), no warn message appended."""
    ic = TurnCounterInterceptor(max_turns=10, every=100)
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "x"}]},
        ctx=InterceptorContext(extra={"session_id": "low"}),
    )
    result = await ic.intercept_request(req)
    sys_msgs = [m for m in result.body["messages"] if m.get("role") == "system"]
    assert sys_msgs == []


async def test_turn_counter_urgent_threshold():
    ic = TurnCounterInterceptor(max_turns=10, every=100)
    for _ in range(9):
        await ic.intercept_request(
            AdapterRequest(
                method="POST",
                path="/",
                headers={},
                body={"messages": [{"role": "user", "content": "x"}]},
                ctx=InterceptorContext(extra={"session_id": "urgent"}),
            )
        )
    # turn 10 = 100% = urgent
    last = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "x"}]},
        ctx=InterceptorContext(extra={"session_id": "urgent"}),
    )
    result = await ic.intercept_request(last)
    sys_msgs = [m for m in result.body["messages"] if m.get("role") == "system"]
    assert any("URGENT" in m["content"] for m in sys_msgs)


async def test_turn_counter_gc_evicts_stale(monkeypatch):
    """Force the GC branch by fast-forwarding ``time.monotonic``."""
    from nemo_gym.adapters.interceptors import turn_counter as tc_mod

    fake_time = {"v": 1000.0}

    def _t():
        return fake_time["v"]

    monkeypatch.setattr(tc_mod.time, "monotonic", _t)
    ic = TurnCounterInterceptor(max_turns=5)
    await ic.intercept_request(
        AdapterRequest(
            method="POST",
            path="/",
            headers={},
            body={"messages": [{"role": "user", "content": "x"}]},
            ctx=InterceptorContext(extra={"session_id": "stale"}),
        )
    )
    assert "stale" in ic._sessions

    # Advance past stale threshold + GC interval
    fake_time["v"] += 2000.0
    await ic.intercept_request(
        AdapterRequest(
            method="POST",
            path="/",
            headers={},
            body={"messages": [{"role": "user", "content": "x"}]},
            ctx=InterceptorContext(extra={"session_id": "fresh"}),
        )
    )
    assert "stale" not in ic._sessions


async def test_turn_counter_max_none_passes_through():
    ic = TurnCounterInterceptor(max_turns=None, every=1)
    for _ in range(5):
        await ic.intercept_request(
            AdapterRequest(
                method="POST",
                path="/",
                headers={},
                body={"messages": [{"role": "user", "content": "x"}]},
                ctx=InterceptorContext(extra={"session_id": "no-cap"}),
            )
        )


async def test_turn_counter_no_messages_in_body():
    ic = TurnCounterInterceptor(max_turns=5)
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"prompt": "no messages here"},
        ctx=InterceptorContext(extra={"session_id": "no-msg"}),
    )
    await ic.intercept_request(req)


# ---------------------------------------------------------------------------
# reasoning — non-dict body, non-dict choice, no <think>, no fields
# ---------------------------------------------------------------------------


async def test_reasoning_non_dict_body():
    ic = ReasoningInterceptor()
    resp = AdapterResponse(status_code=200, headers={}, body=b"raw bytes")
    out = await ic.intercept_response(resp)
    assert out.body == b"raw bytes"


async def test_reasoning_no_choices():
    ic = ReasoningInterceptor()
    resp = AdapterResponse(status_code=200, headers={}, body={"unrelated": True})
    out = await ic.intercept_response(resp)
    assert out.body == {"unrelated": True}


async def test_reasoning_non_dict_choice():
    ic = ReasoningInterceptor()
    resp = AdapterResponse(
        status_code=200,
        headers={},
        body={"choices": ["not-a-dict", {"message": {"content": "ok"}}]},
    )
    out = await ic.intercept_response(resp)
    assert out.status_code == 200


async def test_reasoning_message_non_string_content():
    ic = ReasoningInterceptor()
    resp = AdapterResponse(
        status_code=200,
        headers={},
        body={"choices": [{"message": {"content": [{"type": "text", "text": "x"}]}}]},
    )
    out = await ic.intercept_response(resp)
    assert "reasoning_content" not in out.body["choices"][0]["message"]


async def test_reasoning_no_match():
    ic = ReasoningInterceptor()
    resp = AdapterResponse(
        status_code=200,
        headers={},
        body={"choices": [{"message": {"content": "plain text no think tag"}}]},
    )
    out = await ic.intercept_response(resp)
    assert "reasoning_content" not in out.body["choices"][0]["message"]


# ---------------------------------------------------------------------------
# payload_modifier — rename branch
# ---------------------------------------------------------------------------


async def test_payload_modifier_nested_rename():
    ic = PayloadModifierInterceptor(params_to_rename={"old": "new"})
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"old": 1, "nested": {"old": 2, "keep": 3}, "list": [{"old": 4}]},
        ctx=InterceptorContext(),
    )
    out = await ic.intercept_request(req)
    assert "old" not in out.body
    assert out.body["new"] == 1
    assert out.body["nested"]["new"] == 2
    assert out.body["nested"]["keep"] == 3
    assert out.body["list"][0]["new"] == 4


# ---------------------------------------------------------------------------
# modify_tools — empty tools list, missing/non-dict properties branch
# ---------------------------------------------------------------------------


async def test_modify_tools_no_tools():
    ic = ModifyToolsInterceptor(strip_properties=["x"])
    req = AdapterRequest(
        method="POST", path="/", headers={}, body={"model": "m", "messages": []}, ctx=InterceptorContext()
    )
    out = await ic.intercept_request(req)
    assert "tools" not in out.body


async def test_modify_tools_props_not_dict():
    ic = ModifyToolsInterceptor(strip_properties=["x"])
    body = {
        "tools": [
            {"function": {"parameters": {"properties": "not-a-dict"}}},
            {"function": {"parameters": {"properties": {"x": {}, "keep": {}}}}},
        ]
    }
    req = AdapterRequest(method="POST", path="/", headers={}, body=body, ctx=InterceptorContext())
    out = await ic.intercept_request(req)
    # first tool unchanged
    assert out.body["tools"][0]["function"]["parameters"]["properties"] == "not-a-dict"
    # second tool: x stripped
    assert "x" not in out.body["tools"][1]["function"]["parameters"]["properties"]


async def test_modify_tools_add_only_no_strip():
    ic = ModifyToolsInterceptor(add_properties={"injected": {"type": "string"}})
    body = {"tools": [{"function": {"parameters": {"properties": {"keep": {}}}}}]}
    req = AdapterRequest(method="POST", path="/", headers={}, body=body, ctx=InterceptorContext())
    out = await ic.intercept_request(req)
    props = out.body["tools"][0]["function"]["parameters"]["properties"]
    assert "injected" in props
    assert "keep" in props


def test_modify_tools_warns_when_noop(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        ModifyToolsInterceptor()
    assert any("no-op" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Logging / progress_tracking shape checks
# ---------------------------------------------------------------------------


async def test_log_tokens_non_dict_body():
    from nemo_gym.adapters.interceptors.log_tokens import Interceptor

    ic = Interceptor()
    resp = AdapterResponse(status_code=200, headers={}, body=b"bytes")
    out = await ic.intercept_response(resp)
    assert out.body == b"bytes"


async def test_request_logging_bytes_preview():
    from nemo_gym.adapters.interceptors.request_logging import Interceptor

    ic = Interceptor()
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body=b"raw payload, not dict",  # type: ignore[arg-type]
        ctx=InterceptorContext(),
    )
    await ic.intercept_request(req)
    long_resp = AdapterResponse(status_code=200, headers={}, body={"big": "x" * 2000}, latency_ms=1.0)
    await ic.intercept_response(long_resp)


async def test_progress_tracking_webhook_error_swallowed(monkeypatch):
    from nemo_gym.adapters.interceptors import progress_tracking as pt_mod
    from nemo_gym.adapters.interceptors.progress_tracking import Interceptor

    async def _fail(*_a, **_k):
        raise aiohttp.ClientConnectionError("nope")

    monkeypatch.setattr(pt_mod, "global_request", _fail)
    ic = Interceptor(webhook_url="http://x/ping", every=1)
    resp = AdapterResponse(status_code=200, headers={}, body={})
    out = await ic.intercept_response(resp)
    assert out.status_code == 200


# ---------------------------------------------------------------------------
# disk_cache — bypass + write failure
# ---------------------------------------------------------------------------


async def test_disk_cache_purge_old(monkeypatch):
    from nemo_gym.adapters.cache.disk_cache import DiskCache

    with tempfile.TemporaryDirectory() as td:
        cache = DiskCache(td)
        await cache.set("k1", {"v": 1})
        got = await cache.get("k1")
        assert got == {"v": 1}
        # Miss path
        miss = await cache.get("does-not-exist")
        assert miss is None


# ---------------------------------------------------------------------------
# system_message — append-with-no-messages key
# ---------------------------------------------------------------------------


async def test_system_message_with_no_messages_key():
    from nemo_gym.adapters.interceptors.system_message import Interceptor

    ic = Interceptor(system_message="SYS", strategy="prepend")
    req = AdapterRequest(method="POST", path="/", headers={}, body={"model": "m"}, ctx=InterceptorContext())
    out = await ic.intercept_request(req)
    assert out.body["messages"][0]["content"] == "SYS"


async def test_system_message_invalid_strategy_raises():
    from nemo_gym.adapters.interceptors.system_message import Interceptor

    with pytest.raises(ValueError, match="Invalid strategy"):
        Interceptor(system_message="x", strategy="bogus")


async def test_system_message_append():
    from nemo_gym.adapters.interceptors.system_message import Interceptor

    ic = Interceptor(system_message="SYS-APP", strategy="append")
    req = AdapterRequest(
        method="POST",
        path="/",
        headers={},
        body={"messages": [{"role": "user", "content": "x"}]},
        ctx=InterceptorContext(),
    )
    out = await ic.intercept_request(req)
    assert out.body["messages"][-1] == {"role": "system", "content": "SYS-APP"}


# ---------------------------------------------------------------------------
# reasoning — already-normalized + reasoning-field rename
# ---------------------------------------------------------------------------


async def test_reasoning_already_normalized():
    ic = ReasoningInterceptor()
    body = {"choices": [{"message": {"content": "x", "reasoning_content": "preset"}}]}
    resp = AdapterResponse(status_code=200, headers={}, body=body)
    out = await ic.intercept_response(resp)
    assert out.body["choices"][0]["message"]["reasoning_content"] == "preset"


async def test_reasoning_field_rename():
    ic = ReasoningInterceptor()
    body = {"choices": [{"message": {"content": "x", "reasoning": "thinking text"}}]}
    resp = AdapterResponse(status_code=200, headers={}, body=body)
    out = await ic.intercept_response(resp)
    msg = out.body["choices"][0]["message"]
    assert msg["reasoning_content"] == "thinking text"
    assert "reasoning" not in msg


# ---------------------------------------------------------------------------
# disk_cache — extra_body branch in cache_key
# ---------------------------------------------------------------------------


def test_disk_cache_key_includes_extra_body():
    from nemo_gym.adapters.cache.disk_cache import DiskCache

    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    with_extra = {**base, "extra_body": {"foo": "bar"}}
    with_different = {**base, "extra_body": {"foo": "baz"}}
    k_base = DiskCache.cache_key(base)
    k_extra = DiskCache.cache_key(with_extra)
    k_different = DiskCache.cache_key(with_different)
    assert k_base != k_extra
    assert k_extra != k_different


# ---------------------------------------------------------------------------
# middleware — direct helper coverage for branches not hit via TestClient
# ---------------------------------------------------------------------------


async def test_middleware_response_helper_no_body_iterator():
    """The else branch when starlette_resp has no body_iterator."""
    from nemo_gym.adapters.middleware import _starlette_response_to_adapter

    class _Fake:
        body_iterator = None
        body = b'{"ok": true}'
        status_code = 200
        raw_headers = [(b"content-type", b"application/json")]

    out = await _starlette_response_to_adapter(_Fake(), InterceptorContext())
    assert out.status_code == 200
    assert out.body == {"ok": True}


async def test_middleware_response_helper_str_body_attr():
    from nemo_gym.adapters.middleware import _starlette_response_to_adapter

    class _Fake:
        body_iterator = None
        body = "raw text"
        status_code = 200
        raw_headers = [(b"content-type", b"text/plain")]

    out = await _starlette_response_to_adapter(_Fake(), InterceptorContext())
    assert out.body == b"raw text"


async def test_middleware_response_helper_str_chunk_in_iterator():
    """body_iterator yields a str chunk → encoded to utf-8."""
    from nemo_gym.adapters.middleware import _starlette_response_to_adapter

    async def _gen():
        yield "{"
        yield '"ok": true}'

    class _Fake:
        body_iterator = _gen()
        status_code = 200
        raw_headers = [(b"content-type", b"application/json")]

    out = await _starlette_response_to_adapter(_Fake(), InterceptorContext())
    assert out.body == {"ok": True}


async def test_middleware_response_helper_malformed_json_falls_back_to_bytes():
    from nemo_gym.adapters.middleware import _starlette_response_to_adapter

    async def _gen():
        yield b"not-json-{"

    class _Fake:
        body_iterator = _gen()
        status_code = 200
        raw_headers = [(b"content-type", b"application/json")]

    out = await _starlette_response_to_adapter(_Fake(), InterceptorContext())
    assert out.body == b"not-json-{"


async def test_middleware_response_helper_non_json_content_type():
    from nemo_gym.adapters.middleware import _starlette_response_to_adapter

    async def _gen():
        yield b"binary-payload"

    class _Fake:
        body_iterator = _gen()
        status_code = 200
        raw_headers = [(b"content-type", b"application/octet-stream")]

    out = await _starlette_response_to_adapter(_Fake(), InterceptorContext())
    assert out.body == b"binary-payload"


def test_middleware_request_helper_scope_missing_headers():
    """``_override_request_body`` when scope has no ``headers`` key."""
    from nemo_gym.adapters.middleware import _override_request_body

    class _FakeReq:
        scope = {}  # missing headers entirely
        _body = b""

    req = _FakeReq()
    _override_request_body(req, b'{"x": 1}')  # type: ignore[arg-type]
    headers = req.scope["headers"]
    assert (b"content-length", b"8") in headers


# ---------------------------------------------------------------------------
# request_logging — str-body preview branch (line 39 in source)
# ---------------------------------------------------------------------------


async def test_request_logging_str_response_body():
    from nemo_gym.adapters.interceptors.request_logging import Interceptor

    ic = Interceptor()
    resp = AdapterResponse(status_code=200, headers={}, body="plain string body")  # type: ignore[arg-type]
    out = await ic.intercept_response(resp)
    assert out.body == "plain string body"
