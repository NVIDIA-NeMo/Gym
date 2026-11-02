# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP retries are counted when another attempt starts and reach the caller through response headers."""

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from aiohttp import ClientOSError, ServerDisconnectedError
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from multidict import CIMultiDict, CIMultiDictProxy

import nemo_gym.client.http_client
from nemo_gym.client.http_retries import (
    HTTP_RETRY_COUNT_HEADER,
    HttpRetryCountMiddleware,
    http_retry_scope,
    record_downstream_http_retries,
    record_http_retry_started,
)


@dataclass(frozen=True)
class RetryCase:
    id: str
    outcomes: list[Any]
    """What each transport attempt does in order: raise the exception, or return a response."""
    expected_retries: int
    raises: Optional[type[BaseException]] = None
    request_options: dict[str, Any] = field(default_factory=dict)


RETRY_CASES = [
    RetryCase("first attempt succeeds", ["ok"], expected_retries=0),
    RetryCase(
        "disconnect then success",
        [ServerDisconnectedError(), "ok"],
        expected_retries=1,
        request_options={"_internal": True},
    ),
    RetryCase(
        "two OS errors then success",
        [ClientOSError(), ClientOSError(), "ok"],
        expected_retries=2,
        request_options={"_internal": True},
    ),
    RetryCase("timeout then success", [asyncio.TimeoutError(), "ok"], expected_retries=1),
    RetryCase(
        "attempt cap exhausted",
        [RuntimeError("boom")] * 2,
        expected_retries=1,
        raises=RuntimeError,
        request_options={"_max_num_tries": 2},
    ),
    RetryCase(
        "single attempt allowed",
        [RuntimeError("boom")],
        expected_retries=0,
        raises=RuntimeError,
        request_options={"_max_num_tries": 1},
    ),
    RetryCase(
        "external default cap exhausted",
        [RuntimeError("boom")] * nemo_gym.client.http_client.MAX_NUM_TRIES,
        expected_retries=nemo_gym.client.http_client.MAX_NUM_TRIES - 1,
        raises=RuntimeError,
    ),
    RetryCase(
        "connection retry cap exhausted",
        [ServerDisconnectedError()] * 2,
        expected_retries=1,
        raises=ServerDisconnectedError,
        request_options={"_internal": True, "_max_connection_retries": 2},
    ),
    RetryCase(
        "no resend after a dropped connection",
        [ServerDisconnectedError()],
        expected_retries=0,
        raises=ServerDisconnectedError,
        request_options={"_resend_after_disconnect": False},
    ),
]


@pytest.mark.parametrize("case", RETRY_CASES, ids=[case.id for case in RETRY_CASES])
async def test_a_retry_counts_only_when_another_attempt_starts(
    monkeypatch: pytest.MonkeyPatch, case: RetryCase
) -> None:
    client = _install_transport(monkeypatch, case.outcomes)
    with http_retry_scope() as count:
        if case.raises is None:
            await nemo_gym.client.http_client.request("POST", "http://gym/run", **case.request_options)
        else:
            with pytest.raises(case.raises):
                await nemo_gym.client.http_client.request("POST", "http://gym/run", **case.request_options)
    assert client.request.await_count == len(case.outcomes)
    # Every attempt after the first is one retry; an exhausted budget starts no further attempt to count.
    assert count.total == case.expected_retries == len(case.outcomes) - 1


async def test_a_backoff_cancelled_before_the_next_attempt_counts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [RuntimeError("boom"), "ok"])
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with http_retry_scope() as count:
        with pytest.raises(asyncio.CancelledError):
            await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert count.total == 0


async def test_requests_outside_a_scope_still_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), _response({HTTP_RETRY_COUNT_HEADER: "2"})])
    response = await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert response.status == 200


@pytest.mark.parametrize(
    ("internal", "headers", "expected"),
    [
        (True, {HTTP_RETRY_COUNT_HEADER: "4"}, 4),
        (True, {HTTP_RETRY_COUNT_HEADER.upper(): "4"}, 4),
        (True, {HTTP_RETRY_COUNT_HEADER: "0"}, 0),
        (True, {}, 0),
        (False, {HTTP_RETRY_COUNT_HEADER: "4"}, 0),
        *[(True, {HTTP_RETRY_COUNT_HEADER: raw}, 0) for raw in ("-1", "+2", " 3", "3.0", "abc", "", "\u0663")],
    ],
    ids=[
        "internal response reports downstream retries",
        "header name is case-insensitive",
        "internal response reports zero",
        "header absent adds nothing",
        "external response is never trusted",
        *[f"malformed {raw!r} ignored" for raw in ("-1", "+2", " 3", "3.0", "abc", "", "\u0663")],
    ],
)
async def test_downstream_retry_counts_reach_the_callers_scope(
    monkeypatch: pytest.MonkeyPatch, internal: bool, headers: dict[str, str], expected: int
) -> None:
    _install_transport(monkeypatch, [_response(headers)])
    with http_retry_scope() as count:
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=internal)
    assert count.total == expected


async def test_a_response_without_headers_is_returned_and_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_transport(monkeypatch, [])
    client.request.side_effect = ["opaque response"]
    with http_retry_scope() as count:
        assert await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True) == "opaque response"
    assert count.total == 0


async def test_own_retries_and_downstream_retries_add_up(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), _response({HTTP_RETRY_COUNT_HEADER: "4"})])
    with http_retry_scope() as count:
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert count.total == 5


async def test_a_reply_whose_body_is_cut_off_still_reports_its_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    cut_off = _response({HTTP_RETRY_COUNT_HEADER: "4"})
    cut_off.ok = True
    cut_off.read = AsyncMock(side_effect=RuntimeError("body cut off"))
    complete = _response({HTTP_RETRY_COUNT_HEADER: "2"})
    complete.ok = True
    complete.read = AsyncMock()
    _install_transport(monkeypatch, [cut_off, complete])
    with http_retry_scope() as count:
        await nemo_gym.client.http_client.request(
            "POST", "http://gym/run", _internal=True, _read_body=True, _max_num_tries=3
        )
    # 4 downstream for the cut-off reply, 1 for our resend, 2 downstream for the complete one.
    assert count.total == 7


async def test_an_overlong_count_is_ignored_and_the_request_still_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_transport(monkeypatch, [_response({HTTP_RETRY_COUNT_HEADER: "9" * 4301})])
    with http_retry_scope() as count:
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert count.total == 0
    assert client.request.await_count == 1


async def test_a_bookkeeping_error_never_resends_an_accepted_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_transport(monkeypatch, [_response({HTTP_RETRY_COUNT_HEADER: "1"}), "ok"])
    monkeypatch.setattr(
        nemo_gym.client.http_client, "record_downstream_http_retries", Mock(side_effect=RuntimeError("handler"))
    )
    with http_retry_scope():
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert client.request.await_count == 1


async def test_concurrent_scopes_count_separately_and_child_tasks_share_their_scope() -> None:
    async def sample(retries: int) -> int:
        with http_retry_scope() as count:
            await asyncio.gather(*(_retry_after_yield() for _ in range(retries)))
        return count.total

    with http_retry_scope() as outer:
        assert await asyncio.gather(sample(2), sample(3), sample(0)) == [2, 3, 0]
    assert outer.total == 0


@pytest.mark.parametrize(
    ("query", "expected_status", "expected_count"),
    [
        ({}, 200, "0"),
        ({"retries": 2}, 200, "2"),
        ({"retries": 1, "downstream": "3"}, 200, "4"),
        ({"retries": 2, "fail": True}, 500, "2"),
        ({"app_header": "999"}, 200, "0"),
    ],
    ids=[
        "no retries reports zero",
        "retries in child tasks",
        "own and downstream retries",
        "error response still reports",
        "header set by the app is replaced",
    ],
)
async def test_every_response_reports_its_requests_retries(
    query: dict[str, Any], expected_status: int, expected_count: str
) -> None:
    async with _client(_retrying_app()) as client:
        response = await client.get("/work", params=query)
    assert response.status_code == expected_status
    assert response.headers.get_list(HTTP_RETRY_COUNT_HEADER) == [expected_count]


async def test_concurrent_requests_report_only_their_own_retries() -> None:
    async with _client(_retrying_app()) as client:
        responses = await asyncio.gather(*(client.get("/work", params={"retries": n}) for n in (3, 0, 1)))
    assert [response.headers[HTTP_RETRY_COUNT_HEADER] for response in responses] == ["3", "0", "1"]


async def test_non_http_scopes_pass_through_untouched() -> None:
    app, receive, send = AsyncMock(), AsyncMock(), AsyncMock()
    scope = {"type": "lifespan"}
    await HttpRetryCountMiddleware(app)(scope, receive, send)
    app.assert_awaited_once_with(scope, receive, send)


def _install_transport(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> SimpleNamespace:
    """Replace the shared aiohttp client with one whose attempts follow `outcomes`; "ok" is a plain response."""
    side_effect = [_response({}) if outcome == "ok" else outcome for outcome in outcomes]
    client = SimpleNamespace(request=AsyncMock(side_effect=side_effect))
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock())
    return client


def _response(headers: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(status=200, headers=CIMultiDictProxy(CIMultiDict(headers)))


async def _retry_after_yield() -> None:
    await asyncio.sleep(0)
    record_http_retry_started()


def _retrying_app() -> FastAPI:
    app = FastAPI()

    @app.get("/work")
    async def work(
        retries: int = 0, downstream: Optional[str] = None, fail: bool = False, app_header: Optional[str] = None
    ) -> Response:
        await asyncio.gather(*(_retry_after_yield() for _ in range(retries)))
        if downstream is not None:
            record_downstream_http_retries(_response({HTTP_RETRY_COUNT_HEADER: downstream}))
        if fail:
            return JSONResponse("failed", status_code=500)
        return Response(headers={HTTP_RETRY_COUNT_HEADER: app_header} if app_header is not None else None)

    app.add_middleware(HttpRetryCountMiddleware)
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gym")
