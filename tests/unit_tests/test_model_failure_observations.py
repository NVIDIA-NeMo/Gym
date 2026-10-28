# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model transport observations preserve retries, cancellation and readable HTTP responses."""

import asyncio
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import nemo_gym.server_utils as server_utils
from nemo_gym.episode_types import FailureBoundary, FailureEvent, FailureReport
from nemo_gym.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.openai_utils import MODEL_RESPONSE_TIMEOUT_REASON, NeMoGymAsyncOpenAI
from nemo_gym.typed_exceptions import failure_report_for


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError])
async def test_a_recovered_transport_attempt_is_observed_without_an_episode_failure(monkeypatch, error_type):
    success = SimpleNamespace(status=200, headers={})
    error = error_type("sensitive connection text")
    wire = SimpleNamespace(request=AsyncMock(side_effect=[error, success]))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_connection_retries=3)

    with RequestFailureLog.capture() as log:
        assert await model._request(method="POST", url="http://model.test/responses") is success
        report = log.reply_report(None)

    assert wire.request.await_count == 2 and report.failure_id is None
    assert len(report.events) == 1
    event = next(event for event in report.events.values() if event.boundary is FailureBoundary.MODEL)
    assert event.failure_kind == MODEL_REQUEST_ERROR and event.http_status is None
    assert event.cause_id is None
    assert event.foreign_error.exception_type == f"{error_type.__module__}.{error_type.__qualname__}"
    assert event.foreign_error.http_status is None
    assert event.attempt_id.startswith(event.operation_id)
    assert FailureHeader.decode(FailureHeader.encode(report)) == report


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, TimeoutError])
async def test_a_terminal_transport_attempt_is_observed_without_an_extra_retry(monkeypatch, error_type):
    error = error_type("selected transport message")
    error.request_payload = "private request"
    wire = SimpleNamespace(request=AsyncMock(side_effect=error))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_num_tries=1)

    with RequestFailureLog.capture() as log:
        with pytest.raises(error_type) as caught:
            await model._request(method="POST", url="http://model.test/responses")

    assert caught.value is error and wire.request.await_count == 1
    assert len(log.report.events) == 1
    event = next(event for event in log.report.events.values() if event.boundary is FailureBoundary.MODEL)
    assert event.cause_id is None
    assert event.foreign_error.exception_type.endswith(error_type.__qualname__)
    assert "private" not in log.report.model_dump_json()


async def test_model_cancellation_is_neither_retried_nor_observed(monkeypatch):
    wire = SimpleNamespace(request=AsyncMock(side_effect=asyncio.CancelledError()))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test")
    with RequestFailureLog.capture() as log:
        with pytest.raises(asyncio.CancelledError):
            await model._request(method="POST", url="http://model.test/responses")
    assert wire.request.await_count == 1 and not log.report.events


async def test_an_oversized_report_through_the_gym_http_client_keeps_body_status_and_one_call(monkeypatch):
    event = FailureEvent(failure_kind="test:failure", boundary=FailureBoundary.MODEL, failure_reason="界" * 2000)
    report = FailureReport(failure_id=event.event_id, events={event.event_id: event})
    hits = 0

    async def endpoint(request):
        nonlocal hits
        hits += 1
        return web.Response(status=503, text="original reply", headers=FailureHeader.encode(report))

    app = web.Application()
    app.router.add_get("/failure", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
        response = await server_utils.request(method="GET", url=str(server.make_url("/failure")), _max_num_tries=3)
        assert response.status == 503
        assert FailureHeader.decode(response.headers) == FailureReport(omitted=True)
        with pytest.raises(aiohttp.ClientResponseError) as caught:
            await server_utils.raise_for_status(response)
    assert caught.value.status == 503 and caught.value.response_content == b"original reply"
    assert hits == 1


async def test_the_raised_transport_error_names_the_attempt_it_ended_on(monkeypatch):
    recovered, final = aiohttp.ClientOSError("first"), TimeoutError()
    wire = SimpleNamespace(request=AsyncMock(side_effect=[recovered, TimeoutError(), final]))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(server_utils.asyncio, "sleep", AsyncMock())
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_connection_retries=2)

    with RequestFailureLog.capture() as log:
        with pytest.raises(TimeoutError) as caught:
            await model._request(method="POST", url="http://model.test/responses")
        report = log.reply_report(caught.value)

    assert caught.value is final and wire.request.await_count == 3
    first, _, last = sorted(report.events.values(), key=lambda event: event.attempt_id)
    assert report.failure_id == last.event_id == failure_report_for(caught.value).failure_id
    assert last.foreign_error.exception_type == "TimeoutError"
    assert first.operation_id == last.operation_id and first.foreign_error.exception_type.endswith("ClientOSError")


def _never_accepting_listener() -> tuple[socket.socket, list[socket.socket], int]:
    """A listener whose accept queue is full, so a new connection is never established."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(0)
    port = listener.getsockname()[1]
    queued = []
    for _ in range(4):
        pending = socket.socket()
        pending.setblocking(False)
        try:
            pending.connect(("127.0.0.1", port))
        except BlockingIOError:
            pass
        queued.append(pending)
    return listener, queued, port


@pytest.mark.parametrize(
    ("phase", "connect_timeout", "waited_for_response"),
    [
        pytest.param("sent", None, True, id="sent-and-never-answered"),
        pytest.param("sent", 5.0, True, id="sent-and-never-answered-with-a-connect-timeout"),
        pytest.param("never-connected", None, False, id="never-connected"),
        pytest.param("never-connected", 0.2, False, id="connect-timeout"),
        pytest.param("pool", None, False, id="no-free-pooled-connection"),
    ],
)
async def test_only_a_timeout_after_the_request_was_sent_is_a_response_timeout(
    monkeypatch, phase: str, connect_timeout: float | None, waited_for_response: bool
):
    async def never_answer(reader, writer):
        await reader.read()
        writer.close()

    server = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    listener = None
    if phase == "never-connected":
        listener, _queued, port = _never_accepting_listener()
    url = f"http://127.0.0.1:{port}/v1/responses"
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=1))
    held = asyncio.create_task(session.post(url, data=b"x")) if phase == "pool" else None
    await asyncio.sleep(0.1)
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: session)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(
        api_key="test",
        base_url=f"http://127.0.0.1:{port}/v1",
        max_num_tries=1,
        request_timeout_seconds=0.5,
        connect_timeout_seconds=connect_timeout,
    )
    try:
        with RequestFailureLog.capture() as log:
            with pytest.raises(TimeoutError) as caught:
                await model._request(method="POST", url=url, json={"model": "m"})
            report = log.reply_report(caught.value)
    finally:
        if held is not None:
            held.cancel()
        await session.close()
        server.close()
        if listener is not None:
            listener.close()

    [attempt] = report.events.values()
    assert report.failure_id == attempt.event_id
    assert (attempt.failure_reason == MODEL_RESPONSE_TIMEOUT_REASON) is waited_for_response
