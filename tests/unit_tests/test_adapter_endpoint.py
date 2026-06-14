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
"""Tests for the ``endpoint`` request-to-response interceptor.

All upstream HTTP is faked by patching ``endpoint.global_request`` and
``endpoint.asyncio.sleep`` so the tests are fast and offline.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from nemo_gym.adapters.interceptors.endpoint import Interceptor
from nemo_gym.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


_ENDPOINT = "nemo_gym.adapters.interceptors.endpoint"


class _FakeResp:
    """Minimal stand-in for the aiohttp response used as an async ctx manager."""

    def __init__(self, status: int, headers: dict | None = None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self) -> bytes:
        return self._body


def _req(body: dict | None = None, path: str = "/v1/chat/completions") -> AdapterRequest:
    return AdapterRequest(
        method="POST",
        path=path,
        headers={"Host": "drop-me", "Content-Length": "3", "X-Keep": "1"},
        body=body or {"model": "m"},
        ctx=InterceptorContext(),
    )


def test_upstream_url_known_suffixes_stripped():
    for suffix in ("/chat/completions", "/completions", "/embeddings"):
        ic = Interceptor(upstream_url=f"https://api.example.com/v1{suffix}/")
        assert ic._upstream_url == "https://api.example.com/v1"
    # A non-API suffix is left intact.
    assert Interceptor(upstream_url="http://up/v1")._upstream_url == "http://up/v1"


def test_normalize_content_replaces_none_only():
    body = {
        "choices": [
            {"message": {"content": None}},
            {"delta": {"content": None}},
            {"message": {"content": "keep"}},
            {"message": {}},  # no content key -> untouched
        ]
    }
    Interceptor._normalize_content(body)
    assert body["choices"][0]["message"]["content"] == ""
    assert body["choices"][1]["delta"]["content"] == ""
    assert body["choices"][2]["message"]["content"] == "keep"
    assert "content" not in body["choices"][3]["message"]


async def test_intercept_request_success_merges_body_and_headers():
    ic = Interceptor(upstream_url="http://up/v1", api_key="sekret", extra_body={"stream": False})
    mock = AsyncMock(
        return_value=_FakeResp(
            200,
            {"Content-Type": "application/json"},
            {"choices": [{"message": {"content": None}}]},
        )
    )
    with patch(f"{_ENDPOINT}.global_request", new=mock):
        resp = await ic.intercept_request(_req(body={"model": "m"}))

    assert isinstance(resp, AdapterResponse) and resp.status_code == 200
    kwargs = mock.call_args.kwargs
    # extra_body merged into the forwarded body
    assert json.loads(kwargs["data"]) == {"model": "m", "stream": False}
    # api key injected, content-type defaulted, hop-by-hop headers stripped
    assert kwargs["headers"]["Authorization"] == "Bearer sekret"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert "Host" not in kwargs["headers"] and "Content-Length" not in kwargs["headers"]
    assert kwargs["headers"]["X-Keep"] == "1"
    assert kwargs["url"] == "http://up/v1/v1/chat/completions"
    # None content normalized to ""
    assert resp.body["choices"][0]["message"]["content"] == ""
    # response headers preserved (original case) as latin-1 byte tuples
    assert (b"Content-Type", b"application/json") in resp.headers


async def test_intercept_request_retries_on_status_then_succeeds():
    ic = Interceptor(upstream_url="http://up/v1", max_retries=2, retry_on_status=[503])
    seq = [_FakeResp(503, {"Retry-After": "0"}), _FakeResp(200, {}, {"ok": True})]
    with (
        patch(f"{_ENDPOINT}.global_request", new=AsyncMock(side_effect=seq)),
        patch(f"{_ENDPOINT}.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        resp = await ic.intercept_request(_req())
    assert resp.status_code == 200
    assert resp.body == {"ok": True}
    sleep.assert_awaited()  # Retry-After honored


async def test_intercept_request_non_json_response_passed_through_as_bytes():
    ic = Interceptor(upstream_url="http://up/v1")
    with patch(
        f"{_ENDPOINT}.global_request",
        new=AsyncMock(return_value=_FakeResp(200, {"Content-Type": "text/plain"}, b"not-json")),
    ):
        resp = await ic.intercept_request(_req())
    assert resp.body == b"not-json"


async def test_intercept_request_timeout_returns_504():
    ic = Interceptor(upstream_url="http://up/v1", max_retries=0)
    with patch(f"{_ENDPOINT}.global_request", new=AsyncMock(side_effect=asyncio.TimeoutError())):
        resp = await ic.intercept_request(_req())
    assert resp.status_code == 504
    assert resp.body["error"]["type"] == "timeout"


async def test_intercept_request_timeout_retries_then_504():
    ic = Interceptor(upstream_url="http://up/v1", max_retries=1)
    with (
        patch(f"{_ENDPOINT}.global_request", new=AsyncMock(side_effect=asyncio.TimeoutError())),
        patch(f"{_ENDPOINT}.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        resp = await ic.intercept_request(_req())
    assert resp.status_code == 504
    sleep.assert_awaited()


async def test_intercept_request_client_error_raises_without_retries():
    ic = Interceptor(upstream_url="http://up/v1", max_retries=0)
    with patch(f"{_ENDPOINT}.global_request", new=AsyncMock(side_effect=aiohttp.ClientError("boom"))):
        with pytest.raises(aiohttp.ClientError):
            await ic.intercept_request(_req())


async def test_intercept_request_client_error_retries_then_raises():
    ic = Interceptor(upstream_url="http://up/v1", max_retries=1)
    with (
        patch(f"{_ENDPOINT}.global_request", new=AsyncMock(side_effect=aiohttp.ClientError("boom"))),
        patch(f"{_ENDPOINT}.asyncio.sleep", new=AsyncMock()) as sleep,
    ):
        with pytest.raises(aiohttp.ClientError):
            await ic.intercept_request(_req())
    sleep.assert_awaited()


async def test_close_is_noop():
    ic = Interceptor(upstream_url="http://up/v1")
    assert await ic.close() is None
