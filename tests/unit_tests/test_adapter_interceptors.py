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
"""Per-interceptor behavior tests — ported from NEL ``tests/test_adapters/test_interceptors.py``
and ``test_interceptors_extended.py``.

Mechanical port from ``nemo_evaluator.adapters.*`` to ``nemo_gym.adapters.*``;
``GracefulError`` re-rooted at ``nemo_gym.adapters.types``.
"""

import pytest

from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
)


def _req(body=None, **kw):
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=body or {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )


def _resp(body=None, status_code=200):
    return AdapterResponse(
        status_code=status_code,
        headers={},
        body=body or {},
        ctx=InterceptorContext(),
    )


# ---------------------------------------------------------------------------
# Request-side interceptors
# ---------------------------------------------------------------------------


async def test_drop_params():
    ic = InterceptorRegistry.create("drop_params", {"params": ["temperature", "top_p"]})
    req = _req({"model": "test", "messages": [], "temperature": 0.7, "top_p": 0.9, "max_tokens": 100})
    result = await ic.intercept_request(req)
    assert "temperature" not in result.body
    assert "top_p" not in result.body
    assert result.body["max_tokens"] == 100


async def test_modify_tools():
    ic = InterceptorRegistry.create("modify_tools", {"strip_properties": ["x"]})
    req = _req(
        {
            "model": "test",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "string"},
                                "y": {"type": "integer"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                }
            ],
        }
    )
    result = await ic.intercept_request(req)
    props = result.body["tools"][0]["function"]["parameters"]["properties"]
    assert "x" not in props
    assert "y" in props
    assert "x" not in result.body["tools"][0]["function"]["parameters"]["required"]


async def test_system_message_prepend():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "Be helpful",
            "strategy": "prepend",
        },
    )
    req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    result = await ic.intercept_request(req)
    assert result.body["messages"][0] == {"role": "system", "content": "Be helpful"}
    assert result.body["messages"][1] == {"role": "user", "content": "hi"}


async def test_system_message_replace():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "New system",
            "strategy": "replace",
        },
    )
    req = _req(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "Old system"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    result = await ic.intercept_request(req)
    sys_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "New system"


async def test_payload_modifier_remove():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_remove": ["stream"]})
    req = _req({"model": "test", "messages": [], "stream": True})
    result = await ic.intercept_request(req)
    assert "stream" not in result.body


async def test_payload_modifier_add():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_add": {"temperature": 0.5}})
    req = _req({"model": "test", "messages": []})
    result = await ic.intercept_request(req)
    assert result.body["temperature"] == 0.5


# ---------------------------------------------------------------------------
# turn_counter — session-isolated budget enforcement
# ---------------------------------------------------------------------------


async def test_turn_counter_basic():
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 5})
    for _ in range(5):
        req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
        await ic.intercept_request(req)

    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(_req({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))


async def test_turn_counter_session_isolation():
    """Repeats of the same problem get independent turn budgets when
    the proxy injects distinct session_id values."""
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 3})
    body = {"model": "test", "messages": [{"role": "user", "content": "same prompt"}]}

    for session_id in ("aaa", "bbb"):
        for _ in range(3):
            r = _req(body)
            r.ctx.extra["session_id"] = session_id
            await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "aaa"
    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "ccc"
    await ic.intercept_request(r)


# ---------------------------------------------------------------------------
# raise_client_errors
# ---------------------------------------------------------------------------


async def test_raise_client_errors_4xx():
    ic = InterceptorRegistry.create("raise_client_errors")
    resp = _resp({"error": "bad request"}, status_code=400)
    with pytest.raises(RuntimeError, match="Upstream returned 400"):
        await ic.intercept_response(resp)


async def test_raise_client_errors_429_passes():
    ic = InterceptorRegistry.create("raise_client_errors")
    resp = _resp({"error": "rate limited"}, status_code=429)
    result = await ic.intercept_response(resp)
    assert result.status_code == 429


# ---------------------------------------------------------------------------
# reasoning normalization
# ---------------------------------------------------------------------------


async def test_reasoning_normalize():
    ic = InterceptorRegistry.create("reasoning")
    resp = _resp(
        {
            "choices": [
                {
                    "message": {
                        "content": "<think>reason</think>answer",
                    },
                }
            ],
        }
    )
    result = await ic.intercept_response(resp)
    msg = result.body["choices"][0]["message"]
    assert msg["reasoning_content"] == "reason"
    assert msg["content"] == "answer"


class TestEndpointInterceptorURLStripping:
    def _make(self, url, **kw):
        from nemo_gym.adapters.interceptors.endpoint import Interceptor

        return Interceptor(upstream_url=url, **kw)

    def test_strips_chat_completions(self):
        i = self._make("http://localhost:8000/v1/chat/completions")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_strips_completions(self):
        i = self._make("http://localhost:8000/v1/completions")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_strips_embeddings(self):
        i = self._make("http://localhost:8000/v1/embeddings")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_no_strip_when_no_suffix(self):
        i = self._make("http://localhost:8000/v1")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_trailing_slash_stripped(self):
        i = self._make("http://localhost:8000/v1/")
        assert i._upstream_url == "http://localhost:8000/v1"

    def test_full_api_url(self):
        i = self._make("https://integrate.api.nvidia.com/v1/chat/completions")
        assert i._upstream_url == "https://integrate.api.nvidia.com/v1"

    def test_bare_url_unchanged(self):
        i = self._make("http://vllm:8000")
        assert i._upstream_url == "http://vllm:8000"

    def test_api_key_stored(self):
        i = self._make("http://x", api_key="sk-123")  # pragma: allowlist secret
        assert i._api_key == "sk-123"  # pragma: allowlist secret

    def test_extra_body_merged(self):
        i = self._make("http://x", extra_body={"skip_special_tokens": False})
        assert i._extra_body == {"skip_special_tokens": False}

    def test_retry_on_status_default(self):
        i = self._make("http://x")
        assert 429 in i._retry_on_status
        assert 503 in i._retry_on_status


class TestCachingInterceptor:
    async def test_cache_miss_passes_through(self, tmp_path):
        from nemo_gym.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        ctx = InterceptorContext()
        req = AdapterRequest(
            method="POST",
            path="/chat/completions",
            headers={},
            body={"messages": [{"role": "user", "content": "hi"}]},
            ctx=ctx,
        )
        result = await i.intercept_request(req)
        assert isinstance(result, AdapterRequest)

    async def test_bypass_mode(self, tmp_path):
        from nemo_gym.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path), bypass=True)
        ctx = InterceptorContext()
        req = AdapterRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"messages": []},
            ctx=ctx,
        )
        result = await i.intercept_request(req)
        assert isinstance(result, AdapterRequest)

    async def test_cache_hit_returns_response(self, tmp_path):
        from nemo_gym.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        ctx = InterceptorContext()
        body = {"messages": [{"role": "user", "content": "cached"}]}
        req = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx)

        result1 = await i.intercept_request(req)
        assert isinstance(result1, AdapterRequest)

        resp = AdapterResponse(
            status_code=200,
            headers={},
            body={"choices": [{"message": {"content": "answer"}}]},
            ctx=ctx,
        )
        await i.intercept_response(resp)

        req2 = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=InterceptorContext())
        result2 = await i.intercept_request(req2)
        assert isinstance(result2, AdapterResponse)
        assert result2.status_code == 200

    async def test_cache_isolated_by_session(self, tmp_path):
        """Repeats with different session IDs never share cache entries."""
        from nemo_gym.adapters.interceptors.caching import Interceptor

        i = Interceptor(cache_dir=str(tmp_path))
        body = {"messages": [{"role": "user", "content": "hello"}]}

        ctx_a = InterceptorContext()
        ctx_a.extra["session_id"] = "session_aaa"
        req_a = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx_a)
        result_a = await i.intercept_request(req_a)
        assert isinstance(result_a, AdapterRequest)

        resp_a = AdapterResponse(
            status_code=200, headers={}, body={"choices": [{"message": {"content": "answer-a"}}]}, ctx=ctx_a
        )
        await i.intercept_response(resp_a)

        ctx_b = InterceptorContext()
        ctx_b.extra["session_id"] = "session_bbb"
        req_b = AdapterRequest(method="POST", path="/chat/completions", headers={}, body=body, ctx=ctx_b)
        result_b = await i.intercept_request(req_b)
        assert isinstance(result_b, AdapterRequest), "must be a cache miss, not a hit"


# ===========================================================================
# Observation-interceptor behavior tests — closes the gap the architect
# flagged: progress_tracking / response_stats / request_logging previously
# had only smoke + parity-replay coverage; this section asserts each
# observation interceptor does its actual job (right counter, right log
# field, right webhook payload). Each interceptor gets 2-3 cases.
# ===========================================================================


class TestProgressTrackingBehavior:
    async def test_counter_increments_on_each_call(self):
        from nemo_gym.adapters.interceptors.progress_tracking import Interceptor

        i = Interceptor(every=100)  # avoid log emission in the assert window
        for _ in range(5):
            await i.intercept_response(_resp())
        assert i._completed == 5

    async def test_emits_log_at_every_boundary(self, caplog):
        import logging as _logging

        from nemo_gym.adapters.interceptors.progress_tracking import Interceptor

        i = Interceptor(every=3)
        with caplog.at_level(_logging.INFO, logger="nemo_gym.adapters.interceptors.progress_tracking"):
            for _ in range(7):
                await i.intercept_response(_resp())

        # Should log at completion #3 and #6 only (every=3, count=7 → 3, 6).
        progress_logs = [r for r in caplog.records if r.message.startswith("progress completed=")]
        assert len(progress_logs) == 2
        assert "progress completed=3" in progress_logs[0].message
        assert "progress completed=6" in progress_logs[1].message

    async def test_webhook_fires_with_completed_payload(self, monkeypatch):
        from nemo_gym.adapters.interceptors import progress_tracking as pt

        captured: list[dict] = []

        class _FakeResp:
            def __init__(self, payload):
                self._payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def raise_for_status(self):
                return None

        async def _fake_global_request(method, url, **kw):
            captured.append({"method": method, "url": url, "json": kw.get("json")})
            return _FakeResp(kw.get("json"))

        monkeypatch.setattr(pt, "global_request", _fake_global_request)

        i = pt.Interceptor(webhook_url="http://hook.example/post", every=2)
        for _ in range(4):
            await i.intercept_response(_resp())

        # Webhook fires twice (at completed=2 and completed=4) with the running
        # ``completed`` counter as the only payload field.
        assert len(captured) == 2
        assert captured[0] == {"method": "POST", "url": "http://hook.example/post", "json": {"completed": 2}}
        assert captured[1] == {"method": "POST", "url": "http://hook.example/post", "json": {"completed": 4}}


class TestResponseStatsBehavior:
    async def test_token_total_accumulates_from_usage(self):
        from nemo_gym.adapters.interceptors.response_stats import Interceptor

        i = Interceptor(every=1000)
        await i.intercept_response(_resp(body={"usage": {"total_tokens": 12}}))
        await i.intercept_response(_resp(body={"usage": {"total_tokens": 30}}))
        await i.intercept_response(_resp(body={"usage": {"total_tokens": 5}}))
        assert i._n == 3
        assert i._tokens == 47

    async def test_missing_usage_field_counts_zero_tokens(self):
        from nemo_gym.adapters.interceptors.response_stats import Interceptor

        i = Interceptor(every=1000)
        await i.intercept_response(_resp(body={}))  # no usage
        await i.intercept_response(_resp(body={"usage": None}))  # null usage
        await i.intercept_response(_resp(body={"usage": {"total_tokens": 7}}))
        assert i._n == 3
        assert i._tokens == 7

    async def test_emits_aggregate_log_at_every_boundary(self, caplog):
        import logging as _logging

        from nemo_gym.adapters.interceptors.response_stats import Interceptor

        i = Interceptor(every=2)
        with caplog.at_level(_logging.INFO, logger="nemo_gym.adapters.interceptors.response_stats"):
            await i.intercept_response(_resp(body={"usage": {"total_tokens": 4}}))
            await i.intercept_response(_resp(body={"usage": {"total_tokens": 6}}))
            await i.intercept_response(_resp(body={"usage": {"total_tokens": 1}}))
            await i.intercept_response(_resp(body={"usage": {"total_tokens": 2}}))

        stats_logs = [r for r in caplog.records if r.message.startswith("response_stats requests=")]
        # every=2 with 4 calls → logs at #2 and #4
        assert len(stats_logs) == 2
        assert "requests=2" in stats_logs[0].message and "total_tokens=10" in stats_logs[0].message
        assert "requests=4" in stats_logs[1].message and "total_tokens=13" in stats_logs[1].message


class TestRequestLoggingBehavior:
    async def test_request_log_records_method_path_and_body_keys(self, caplog):
        import logging as _logging

        from nemo_gym.adapters.interceptors.request_logging import Interceptor

        i = Interceptor()
        req = AdapterRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "m", "messages": [], "temperature": 0.3},
            ctx=InterceptorContext(),
        )
        with caplog.at_level(_logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
            await i.intercept_request(req)

        request_logs = [r for r in caplog.records if r.message.startswith("request POST ")]
        assert len(request_logs) == 1
        msg = request_logs[0].message
        assert "POST" in msg
        assert "/v1/chat/completions" in msg
        # body_keys must reflect the request body's top-level keys
        assert "'model'" in msg and "'messages'" in msg and "'temperature'" in msg

    async def test_response_log_records_status_and_latency(self, caplog):
        import logging as _logging

        from nemo_gym.adapters.interceptors.request_logging import Interceptor

        i = Interceptor()
        resp = AdapterResponse(
            status_code=503,
            headers={},
            body={"error": "boom"},
            latency_ms=12.5,
            ctx=InterceptorContext(),
        )
        with caplog.at_level(_logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
            await i.intercept_response(resp)

        response_logs = [r for r in caplog.records if r.message.startswith("response status=")]
        assert len(response_logs) == 1
        msg = response_logs[0].message
        assert "status=503" in msg
        assert "latency_ms=12.50" in msg

    async def test_long_body_preview_is_truncated(self, caplog):
        import logging as _logging

        from nemo_gym.adapters.interceptors.request_logging import _MAX, Interceptor

        i = Interceptor()
        # Build a body whose JSON representation exceeds _MAX so the preview
        # exercises the truncation branch.
        long_value = "x" * (_MAX + 200)
        req = AdapterRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={"model": "m", "messages": [{"role": "user", "content": long_value}]},
            ctx=InterceptorContext(),
        )
        with caplog.at_level(_logging.INFO, logger="nemo_gym.adapters.interceptors.request_logging"):
            await i.intercept_request(req)

        request_logs = [r for r in caplog.records if r.message.startswith("request POST ")]
        assert len(request_logs) == 1
        # Truncated previews end with the "..." sentinel from request_logging._trunc_preview.
        assert "body_preview=" in request_logs[0].message
        assert request_logs[0].message.endswith("...")
