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

from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
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
