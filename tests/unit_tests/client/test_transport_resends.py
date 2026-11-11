# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport resends are counted when another attempt starts and reach the caller through a reply header."""

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
from nemo_gym.client.http_client import ConnectionRetryWindow
from nemo_gym.core.failure_transport import (
    RequestFailureLog,
    TransportResendHeader,
    observe_reported_transport_resends,
    observe_transport_resend,
)
from nemo_gym.server_runtime.request_handling.exception_handling import ExceptionHandlingMiddleware


RESEND_HEADER = TransportResendHeader.NAME


@dataclass(frozen=True)
class ResendCase:
    id: str
    outcomes: list[Any]
    """What each transport attempt does in order: raise the exception, or return a response."""
    expected_resends: int
    raises: Optional[type[BaseException]] = None
    request_options: dict[str, Any] = field(default_factory=dict)


RESEND_CASES = [
    ResendCase("first attempt succeeds", ["ok"], expected_resends=0),
    ResendCase(
        "disconnect then success",
        [ServerDisconnectedError(), "ok"],
        expected_resends=1,
        request_options={"_internal": True},
    ),
    ResendCase(
        "two OS errors then success",
        [ClientOSError(), ClientOSError(), "ok"],
        expected_resends=2,
        request_options={"_internal": True},
    ),
    ResendCase("timeout then success", [asyncio.TimeoutError(), "ok"], expected_resends=1),
    ResendCase(
        "attempt cap exhausted",
        [RuntimeError("boom")] * 2,
        expected_resends=1,
        raises=RuntimeError,
        request_options={"_max_num_tries": 2},
    ),
    ResendCase(
        "single attempt allowed",
        [RuntimeError("boom")],
        expected_resends=0,
        raises=RuntimeError,
        request_options={"_max_num_tries": 1},
    ),
    ResendCase(
        "external default cap exhausted",
        [RuntimeError("boom")] * nemo_gym.client.http_client.MAX_NUM_TRIES,
        expected_resends=nemo_gym.client.http_client.MAX_NUM_TRIES - 1,
        raises=RuntimeError,
    ),
    ResendCase(
        "connection retry cap exhausted",
        [ServerDisconnectedError()] * 2,
        expected_resends=1,
        raises=ServerDisconnectedError,
        request_options={"_internal": True, "_max_connection_retries": 2},
    ),
    ResendCase(
        "no resend after a dropped connection",
        [ServerDisconnectedError()],
        expected_resends=0,
        raises=ServerDisconnectedError,
        request_options={"_resend_after_disconnect": False},
    ),
]


@pytest.mark.parametrize("case", RESEND_CASES, ids=[case.id for case in RESEND_CASES])
async def test_a_resend_counts_only_when_another_attempt_starts(
    monkeypatch: pytest.MonkeyPatch, case: ResendCase
) -> None:
    client = _install_transport(monkeypatch, case.outcomes)
    with RequestFailureLog.capture() as log:
        if case.raises is None:
            await nemo_gym.client.http_client.request("POST", "http://gym/run", **case.request_options)
        else:
            with pytest.raises(case.raises):
                await nemo_gym.client.http_client.request("POST", "http://gym/run", **case.request_options)
    assert client.request.await_count == len(case.outcomes)
    # Every attempt after the first is one resend; an exhausted budget starts no further attempt to count.
    assert log.transport_resends == case.expected_resends == len(case.outcomes) - 1


async def test_a_connection_window_that_expires_during_the_backoff_counts_no_resend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expired window is the loop's usual exit; counting above it would add a resend that never starts."""
    client = _install_transport(monkeypatch, [ClientOSError(), "ok"])
    clock = [100.0]
    monkeypatch.setattr(nemo_gym.client.http_client.time, "monotonic", lambda: clock[0])

    async def backoff(seconds: float) -> None:
        clock[0] += 5.0

    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", backoff)
    with RequestFailureLog.capture() as log:
        with pytest.raises(ClientOSError):
            await nemo_gym.client.http_client.request(
                "POST", "http://gym/run", _internal=True, _connection_retry_window=ConnectionRetryWindow(1.0)
            )
    assert client.request.await_count == 1
    assert log.transport_resends == 0


async def test_a_backoff_cancelled_before_the_next_attempt_counts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [RuntimeError("boom"), "ok"])
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError))
    with RequestFailureLog.capture() as log:
        with pytest.raises(asyncio.CancelledError):
            await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert log.transport_resends == 0


async def test_requests_outside_a_server_request_still_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), _response({RESEND_HEADER: "2"})])
    response = await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert response.status == 200


@pytest.mark.parametrize(
    ("internal", "headers", "expected"),
    [
        (True, {RESEND_HEADER: "4"}, 4),
        (True, {RESEND_HEADER.lower(): "4"}, 4),
        (True, {}, 0),
        (False, {RESEND_HEADER: "4"}, 0),
        *[(True, {RESEND_HEADER: raw}, 0) for raw in ("-1", "+2", " 3", "3.0", "abc", "", "\u0663")],
    ],
    ids=[
        "a Gym reply reports downstream resends",
        "header name is case-insensitive",
        "header absent adds nothing",
        "external reply is never trusted",
        *[f"malformed {raw!r} ignored" for raw in ("-1", "+2", " 3", "3.0", "abc", "", "\u0663")],
    ],
)
async def test_downstream_resends_reach_the_callers_log(
    monkeypatch: pytest.MonkeyPatch, internal: bool, headers: dict[str, str], expected: int
) -> None:
    _install_transport(monkeypatch, [_response(headers)])
    with RequestFailureLog.capture() as log:
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=internal)
    assert log.transport_resends == expected


@pytest.mark.parametrize(
    "raw",
    ["9" * (TransportResendHeader.MAX_DIGITS + 1), "9" * 4301],
    ids=["one digit over the bound", "over int's digit limit"],
)
def test_an_overlong_count_adds_nothing(raw: str) -> None:
    with RequestFailureLog.capture() as log:
        observe_reported_transport_resends(CIMultiDictProxy(CIMultiDict({RESEND_HEADER: raw})))
    assert log.transport_resends == 0


def test_a_count_at_the_bound_is_read() -> None:
    with RequestFailureLog.capture() as log:
        observe_reported_transport_resends({RESEND_HEADER: "9" * TransportResendHeader.MAX_DIGITS})
    assert log.transport_resends == int("9" * TransportResendHeader.MAX_DIGITS)


async def test_a_response_without_headers_is_returned_and_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_transport(monkeypatch, [])
    client.request.side_effect = ["opaque response"]
    with RequestFailureLog.capture() as log:
        assert await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True) == "opaque response"
    assert log.transport_resends == 0


async def test_own_resends_and_downstream_resends_add_up(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, [ServerDisconnectedError(), _response({RESEND_HEADER: "4"})])
    with RequestFailureLog.capture() as log:
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True)
    assert log.transport_resends == 5


async def test_a_reply_whose_body_is_cut_off_still_reports_its_resends(monkeypatch: pytest.MonkeyPatch) -> None:
    cut_off = _response({RESEND_HEADER: "4"})
    cut_off.ok = True
    cut_off.read = AsyncMock(side_effect=RuntimeError("body cut off"))
    complete = _response({RESEND_HEADER: "2"})
    complete.ok = True
    complete.read = AsyncMock()
    _install_transport(monkeypatch, [cut_off, complete])
    with RequestFailureLog.capture() as log:
        await nemo_gym.client.http_client.request(
            "POST", "http://gym/run", _internal=True, _read_body=True, _max_num_tries=3
        )
    # 4 downstream for the cut-off reply, 1 for our resend, 2 downstream for the complete one.
    assert log.transport_resends == 7


async def test_a_bookkeeping_error_never_resends_an_accepted_request(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _install_transport(monkeypatch, [_response({RESEND_HEADER: "1"}), "ok"])
    monkeypatch.setattr(
        nemo_gym.client.http_client, "observe_reported_transport_resends", Mock(side_effect=RuntimeError("handler"))
    )
    with RequestFailureLog.capture():
        # Bounded, so a resend fails this test instead of retrying the internal call forever.
        await nemo_gym.client.http_client.request("POST", "http://gym/run", _internal=True, _max_num_tries=2)
    assert client.request.await_count == 1


async def test_concurrent_requests_count_separately_and_child_tasks_share_their_log() -> None:
    async def sample(resends: int) -> int:
        with RequestFailureLog.capture() as log:
            await asyncio.gather(*(_resend_after_yield() for _ in range(resends)))
        return log.transport_resends

    with RequestFailureLog.capture() as outer:
        assert await asyncio.gather(sample(2), sample(3), sample(0)) == [2, 3, 0]
    assert outer.transport_resends == 0


async def test_an_operation_log_forwards_its_resends_to_the_enclosing_request() -> None:
    with RequestFailureLog.capture() as request_log:
        with RequestFailureLog.operation() as operation_log:
            observe_transport_resend()
            observe_reported_transport_resends({RESEND_HEADER: "2"})
        assert operation_log.transport_resends == 3
    assert request_log.transport_resends == 3
    with RequestFailureLog.operation() as alone:
        observe_transport_resend()
    assert alone.transport_resends == 1


@pytest.mark.parametrize(
    ("query", "expected_status", "expected_count"),
    [
        ({}, 200, []),
        ({"resends": 2}, 200, ["2"]),
        ({"resends": 1, "downstream": "3"}, 200, ["4"]),
        ({"resends": 2, "fail": True}, 500, ["2"]),
        ({"resends": 2, "raise_error": True}, 500, ["2"]),
        ({"app_header": "1", "downstream": "1"}, 200, ["1"]),
        ({"resends": 3, "app_header": "1"}, 200, ["3"]),
        ({"app_header": "5"}, 200, ["5"]),
    ],
    ids=[
        "no resends sends no header",
        "resends in child tasks",
        "own and downstream resends",
        "error response still reports",
        "raised route still reports",
        "a copied downstream count is not added twice",
        "the request's own larger count wins",
        "a nested layer's count is kept",
    ],
)
async def test_every_reply_reports_its_requests_resends(
    query: dict[str, Any], expected_status: int, expected_count: list[str]
) -> None:
    async with _client(_resending_app()) as client:
        response = await client.get("/work", params=query)
    assert response.status_code == expected_status
    assert response.headers.get_list(RESEND_HEADER) == expected_count


async def test_concurrent_requests_report_only_their_own_resends() -> None:
    async with _client(_resending_app()) as client:
        responses = await asyncio.gather(*(client.get("/work", params={"resends": n}) for n in (3, 0, 1)))
    assert [response.headers.get(RESEND_HEADER) for response in responses] == ["3", None, "1"]


async def test_nested_middleware_layers_report_every_resend_once() -> None:
    """Each layer opens its own log; resends in both scopes reach the reply once, and a parent request sees them."""

    async def route(scope, receive, send) -> None:
        for _ in range(3):
            observe_transport_resend()
        await Response()(scope, receive, send)

    def failure_reply(error: BaseException) -> JSONResponse:
        return JSONResponse({}, status_code=500)

    inner = ExceptionHandlingMiddleware(route, failure_reply=failure_reply)

    async def between(scope, receive, send) -> None:
        for _ in range(2):
            observe_transport_resend()
        await inner(scope, receive, send)

    outer = ExceptionHandlingMiddleware(between, failure_reply=failure_reply)
    messages: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request"}

    async def send(message: dict) -> None:
        messages.append(message)

    await outer({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
    assert dict(messages[0]["headers"])[RESEND_HEADER.lower().encode()] == b"5"


async def test_an_operation_log_forwards_its_model_operations_to_the_enclosing_request() -> None:
    from nemo_gym.core.failure_transport import observe_model_operation

    with RequestFailureLog.capture() as request_log:
        with RequestFailureLog.operation():
            observe_model_operation(operation_id="op", outcome="completed", final_attempt_id="a", http_attempts=2)
    assert request_log.model_operations["op"].http_attempts == 2


def _install_transport(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> SimpleNamespace:
    """Replace the shared aiohttp client with one whose attempts follow `outcomes`; "ok" is a plain response."""
    side_effect = [_response({}) if outcome == "ok" else outcome for outcome in outcomes]
    client = SimpleNamespace(request=AsyncMock(side_effect=side_effect))
    monkeypatch.setattr(nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(nemo_gym.client.http_client.asyncio, "sleep", AsyncMock())
    return client


def _response(headers: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(status=200, headers=CIMultiDictProxy(CIMultiDict(headers)))


async def _resend_after_yield() -> None:
    await asyncio.sleep(0)
    observe_transport_resend()


def _resending_app() -> FastAPI:
    app = FastAPI()

    @app.get("/work")
    async def work(
        resends: int = 0,
        downstream: Optional[str] = None,
        fail: bool = False,
        raise_error: bool = False,
        app_header: Optional[str] = None,
    ) -> Response:
        await asyncio.gather(*(_resend_after_yield() for _ in range(resends)))
        if downstream is not None:
            observe_reported_transport_resends({RESEND_HEADER: downstream})
        if raise_error:
            raise RuntimeError("route failed")
        if fail:
            return JSONResponse("failed", status_code=500)
        return Response(headers={RESEND_HEADER: app_header} if app_header is not None else None)

    app.add_middleware(
        ExceptionHandlingMiddleware, failure_reply=lambda error: JSONResponse({"detail": str(error)}, status_code=500)
    )
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://gym")
