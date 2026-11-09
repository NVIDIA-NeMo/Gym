# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every provider-response retry sequence ends by an attempt or an elapsed-time bound, internal clients included."""

import asyncio
import random
import socket
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import ClientResponseError, web

import nemo_gym.client.http_client
from nemo_gym.client import openai_client as openai_client_module
from nemo_gym.client.openai_client import NeMoGymAsyncOpenAI


RETRYABLE_STATUSES = [429, 503, 502, 504, 520]
OVERFLOW_BODY = b'{"error":{"code":"context_length_exceeded","message":"too long"}}'
BASE_URL = "http://127.0.0.1:8000/v1"


class _Runaway(BaseException):
    """Raised by the fake provider when a retry loop does not end (a BaseException, so no retry handler eats it)."""


def _answer(status: int, *, retry_after: str | None = None, body: bytes = b'{"error":"busy"}') -> SimpleNamespace:
    async def read(size: int = -1) -> bytes:
        chunk, remaining[0] = (remaining[0], b"") if size < 0 else (remaining[0][:size], remaining[0][size:])
        return chunk

    remaining = [body]
    return SimpleNamespace(
        status=status,
        ok=status < 400,
        headers={"Retry-After": retry_after} if retry_after is not None else {},
        content=SimpleNamespace(read=read),
        close=lambda: None,
    )


def _provider(monkeypatch: pytest.MonkeyPatch, answer, *, ceiling: int = 500) -> list[int]:
    """Patch the transport so call ``n`` is answered by ``answer(n)``; return the statuses it sent."""
    sent: list[int] = []

    async def request(**kwargs):
        if len(sent) >= ceiling:
            raise _Runaway(f"still retrying after {ceiling} requests")
        response = answer(len(sent) + 1)
        sent.append(response.status)
        return response

    monkeypatch.setattr(openai_client_module, "request", request)

    async def final_error(response, content=None):
        raise RuntimeError(f"final status {response.status}")

    monkeypatch.setattr(openai_client_module, "raise_for_status", final_error)
    return sent


def _sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        await asyncio.sleep(0)

    monkeypatch.setattr(openai_client_module, "sleep", sleep)
    return waits


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("status", [429, 503])
async def test_a_busy_provider_is_waited_out_until_the_time_budget_is_spent(monkeypatch, internal, status):
    sent = _provider(monkeypatch, lambda n: _answer(status, retry_after="1"))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=5)

    with pytest.raises(RuntimeError, match=f"final status {status}"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert waits == [1.0] * 5
    assert len(sent) == 6


@pytest.mark.parametrize("status", [502, 504, 520])
async def test_an_internal_client_waits_out_a_transient_5xx_on_the_same_time_budget(monkeypatch, status):
    sent = _provider(monkeypatch, lambda n: _answer(status, retry_after="1"))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=5)

    with pytest.raises(RuntimeError, match=f"final status {status}"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert waits == [1.0] * 5
    assert len(sent) == 6


@pytest.mark.parametrize("status", [502, 504, 520])
async def test_an_external_client_keeps_a_transient_5xx_on_the_attempt_count(monkeypatch, status):
    sent = _provider(monkeypatch, lambda n: _answer(status, retry_after="1"))
    _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, max_rate_limit_wait_seconds=5, max_http_attempts=3)

    with pytest.raises(RuntimeError, match=f"final status {status}"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert len(sent) == 3


@pytest.mark.parametrize("status", RETRYABLE_STATUSES)
async def test_a_zero_budget_puts_an_internal_client_on_the_attempt_count(monkeypatch, status):
    sent = _provider(monkeypatch, lambda n: _answer(status, retry_after="1"))
    _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(
        api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=0, max_http_attempts=4
    )

    with pytest.raises(RuntimeError, match=f"final status {status}"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert len(sent) == 4


@pytest.mark.parametrize("internal", [False, True])
async def test_a_changing_status_sequence_shares_one_counter_and_one_budget(monkeypatch, internal):
    sequence = [429, 503, 502, 429, 504, 503, 520, 429]
    sent = _provider(monkeypatch, lambda n: _answer(sequence[(n - 1) % len(sequence)], retry_after="1"))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=7)

    with pytest.raises(RuntimeError, match="final status"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    if internal:
        # One budget across every status: seven one-second waits, then the eighth answer is final.
        assert sent == sequence and waits == [1.0] * 7
    else:
        assert len(sent) <= len(sequence)  # an external client's 5xx answers stay on its attempt count


async def test_a_retry_after_beyond_the_remaining_budget_ends_the_request_without_sleeping(monkeypatch):
    sent = _provider(monkeypatch, lambda n: _answer(429, retry_after="3" if n < 3 else "10"))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=5)

    with pytest.raises(RuntimeError, match="final status 429"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert waits == [3.0] and len(sent) == 2  # 3 s is honored; the next 3 s would pass the 5 s bound


async def test_a_retry_after_is_capped_at_the_backoff_maximum(monkeypatch):
    sent = _provider(monkeypatch, lambda n: _answer(503, retry_after="86400"))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=100)

    with pytest.raises(RuntimeError, match="final status 503"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert waits == [60.0] and len(sent) == 2


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_a_propagated_context_overflow_is_terminal_whatever_its_status(monkeypatch, internal, status):
    sent = _provider(monkeypatch, lambda n: _answer(status, body=OVERFLOW_BODY))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=60)

    with pytest.raises(RuntimeError, match=f"final status {status}"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert sent == [status] and waits == []


@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_input_authentication_and_authorization_errors_are_not_retried(monkeypatch, internal, status):
    sent = _provider(monkeypatch, lambda n: _answer(status))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=60)

    response = await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert response.status == status and sent == [status] and waits == []


@pytest.mark.parametrize("internal", [False, True])
async def test_a_spent_quota_429_is_terminal(monkeypatch, internal):
    quota = b'{"error":{"code":"insufficient_quota"}}'
    sent = _provider(monkeypatch, lambda n: _answer(429, body=quota))
    waits = _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=60)

    with pytest.raises(Exception):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert sent == [429] and waits == []


async def test_the_error_body_read_for_a_retry_decision_is_a_bounded_prefix(monkeypatch):
    limit = openai_client_module.ERROR_BODY_PREFIX_BYTES
    body = b"x" * (3 * limit)
    seen: list[bytes] = []

    async def final_error(response, content=None):
        seen.append(content)
        raise RuntimeError(f"final status {response.status}")

    sent = _provider(monkeypatch, lambda n: _answer(503, retry_after="1", body=body))
    monkeypatch.setattr(openai_client_module, "raise_for_status", final_error)
    _sleeps(monkeypatch)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=2)

    with pytest.raises(RuntimeError, match="final status 503"):
        await client._request_with_retry(url=f"{BASE_URL}/responses")

    assert len(sent) == 3
    assert seen == [body[:limit]]


async def test_cancelling_a_waiting_request_stops_it_and_sends_nothing_more(monkeypatch):
    sent = _provider(monkeypatch, lambda n: _answer(429))
    waiting = asyncio.Event()

    async def blocked_sleep(seconds: float) -> None:
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(openai_client_module, "sleep", blocked_sleep)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=True, max_rate_limit_wait_seconds=60)

    task = asyncio.create_task(client._request_with_retry(url=f"{BASE_URL}/responses"))
    await asyncio.wait_for(waiting.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert sent == [429]


@pytest.mark.parametrize("internal", [False, True])
async def test_concurrent_callers_do_not_retry_in_lockstep_and_all_terminate(monkeypatch, internal):
    callers = 20
    sent = _provider(monkeypatch, lambda n: _answer(429), ceiling=5000)
    waits = _sleeps(monkeypatch)
    random.seed(1234)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=BASE_URL, internal=internal, max_rate_limit_wait_seconds=6)

    results = await asyncio.gather(
        *(client._request_with_retry(url=f"{BASE_URL}/responses") for _ in range(callers)), return_exceptions=True
    )

    assert all(isinstance(result, RuntimeError) for result in results)
    first_waits = waits[:callers]  # every caller sleeps once before any caller's second wait
    assert len(set(first_waits)) == callers
    assert len(sent) < 5000


@pytest.fixture
async def stub_provider(monkeypatch):
    """A real HTTP provider whose answers the test scripts; yields (base_url, hits)."""
    hits: list[int] = []
    script: dict = {"answer": lambda n: (503, b'{"error":"busy"}', {})}

    async def handler(request: web.Request) -> web.Response:
        await request.read()
        hits.append(1)
        status, body, headers = script["answer"](len(hits))
        return web.Response(status=status, body=body, headers=headers, content_type="application/json")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    await web.TCPSite(runner, "127.0.0.1", port).start()
    session = aiohttp.ClientSession()
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: session)
    try:
        yield f"http://127.0.0.1:{port}/v1", hits, script
    finally:
        await session.close()
        await runner.cleanup()


CALL = dict(model="m", messages=[{"role": "user", "content": "hi"}])


async def test_over_real_http_a_mixed_status_provider_ends_with_the_final_status_and_a_bounded_body(
    monkeypatch, stub_provider
):
    base_url, hits, script = stub_provider
    limit = openai_client_module.ERROR_BODY_PREFIX_BYTES
    statuses = [429, 503, 502, 504, 520]
    script["answer"] = lambda n: (statuses[(n - 1) % len(statuses)], b"y" * (3 * limit), {"Retry-After": "0"})
    monkeypatch.setattr(openai_client_module.random, "uniform", lambda low, high: 0.01)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=base_url, internal=True, max_rate_limit_wait_seconds=0.055)

    started = asyncio.get_running_loop().time()
    with pytest.raises(ClientResponseError) as raised:
        await asyncio.wait_for(client.create_chat_completion(**CALL), timeout=10)

    assert asyncio.get_running_loop().time() - started < 5
    assert len(hits) == 6  # five 0.01 s waits fit in 0.055 s; the sixth would pass it
    assert raised.value.status == statuses[(6 - 1) % len(statuses)]
    assert raised.value.response_content == b"y" * limit


@pytest.mark.parametrize("status", [400, 429, 500, 503])
async def test_over_real_http_a_context_overflow_reaches_the_provider_once(monkeypatch, stub_provider, status):
    base_url, hits, script = stub_provider
    script["answer"] = lambda n: (status, OVERFLOW_BODY, {})
    monkeypatch.setattr(openai_client_module.random, "uniform", lambda low, high: 0.01)
    client = NeMoGymAsyncOpenAI(api_key="k", base_url=base_url, internal=True, max_rate_limit_wait_seconds=5)

    with pytest.raises(ClientResponseError) as raised:
        await asyncio.wait_for(client.create_chat_completion(**CALL), timeout=10)

    assert len(hits) == 1 and raised.value.status == status
    assert raised.value.response_content == OVERFLOW_BODY
