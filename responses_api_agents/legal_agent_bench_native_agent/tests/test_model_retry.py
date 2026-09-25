# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from functools import partial
from unittest.mock import AsyncMock, MagicMock, call

import aiohttp
import pytest

from responses_api_agents.legal_agent_bench_native_agent import model_retry


@pytest.fixture
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(model_retry, "AsyncRetrying", partial(model_retry.AsyncRetrying, sleep=sleep))
    return sleep


def http_error(status, *, wrapped=False):
    error = aiohttp.ClientResponseError(
        request_info=MagicMock(real_url="http://policy/v1/responses"),
        history=(),
        status=500 if wrapped else status,
    )
    if wrapped:
        body = json.dumps({"error": {"code": str(status), "message": "provider error"}}).encode()
        error.response_content = json.dumps(f"Hit an exception in policy calling an inner server: {body}").encode()
    return error


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 422, 429, 500, 502, 503, 504])
async def test_http_retry_budget_and_backoff(status, wrapped, retry_sleep):
    error = http_error(status, wrapped=wrapped)
    request = AsyncMock(side_effect=error)
    with pytest.raises(aiohttp.ClientResponseError) as caught:
        await model_retry.retry_model_request(request, timeout_seconds=1)
    assert caught.value is error
    expected = 3 if status in {408, 429} or status >= 500 else 1
    assert request.await_count == expected
    assert retry_sleep.await_args_list == ([call(1), call(2)] if expected == 3 else [])


@pytest.mark.parametrize("wrapped", [False, True])
async def test_opt_in_404_retries_then_returns_success(wrapped, retry_sleep):
    result = object()
    request = AsyncMock(side_effect=[http_error(404, wrapped=wrapped), result])
    assert await model_retry.retry_model_request(request, timeout_seconds=1, retry_404=True) is result
    assert request.await_count == 2
    retry_sleep.assert_awaited_once_with(1)


@pytest.mark.parametrize(
    "error", [TimeoutError(), ConnectionError(), aiohttp.ClientConnectionError(), aiohttp.ClientPayloadError()]
)
async def test_transport_error_retries_then_succeeds(error, retry_sleep):
    request = AsyncMock(side_effect=[error, "ok"])
    assert await model_retry.retry_model_request(request, timeout_seconds=1) == "ok"
    assert request.await_count == 2


@pytest.mark.parametrize("error", [ValueError("bad response"), asyncio.CancelledError()])
async def test_non_transport_errors_and_cancellation_propagate(error, retry_sleep):
    request = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await model_retry.retry_model_request(request, timeout_seconds=1)
    request.assert_awaited_once()
    retry_sleep.assert_not_awaited()


async def test_timeout_applies_to_each_attempt(retry_sleep):
    cancelled = []

    async def slow_request():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    with pytest.raises(TimeoutError):
        await model_retry.retry_model_request(slow_request, timeout_seconds=0.001)
    assert len(cancelled) == 3
    assert retry_sleep.await_count == 2


@pytest.mark.parametrize("body", [b"not JSON", b"{}", b'{"error":{"code":"server_error"}}'])
async def test_unrecognized_500_body_keeps_http_status(body, retry_sleep):
    error = http_error(500)
    error.response_content = body
    request = AsyncMock(side_effect=[error, "ok"])
    assert await model_retry.retry_model_request(request, timeout_seconds=None) == "ok"
    assert request.await_count == 2
