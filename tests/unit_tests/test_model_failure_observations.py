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

import nemo_gym.client.http_client as http_client
from nemo_gym.core.failure_kinds import MODEL_REQUEST_ERROR
from nemo_gym.failure_transport import FailureHeader, RequestFailureLog
from nemo_gym.openai_utils import MODEL_RESPONSE_TIMEOUT_REASON, NeMoGymAsyncOpenAI, PermanentEndpointError
from nemo_gym.server_runtime.episode_types import FailureBoundary, FailureEvent, FailureReport
from nemo_gym.typed_exceptions import failure_report_for


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError])
async def test_a_recovered_transport_attempt_is_observed_without_an_episode_failure(monkeypatch, error_type):
    success = SimpleNamespace(status=200, headers={})
    error = error_type("sensitive connection text")
    wire = SimpleNamespace(request=AsyncMock(side_effect=[error, success]))
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(http_client.asyncio, "sleep", AsyncMock())
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
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
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
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
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
        monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
        response = await http_client.request(method="GET", url=str(server.make_url("/failure")), _max_num_tries=3)
        assert response.status == 503
        assert FailureHeader.decode(response.headers) == FailureReport(omitted=True)
        with pytest.raises(aiohttp.ClientResponseError) as caught:
            await http_client.raise_for_status(response)
    assert caught.value.status == 503 and caught.value.response_content == b"original reply"
    assert hits == 1


async def test_the_raised_transport_error_names_the_attempt_it_ended_on(monkeypatch):
    recovered, final = aiohttp.ClientOSError("first"), TimeoutError()
    wire = SimpleNamespace(request=AsyncMock(side_effect=[recovered, TimeoutError(), final]))
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(http_client.asyncio, "sleep", AsyncMock())
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
        pytest.param("uploading", None, False, id="upload-still-in-progress"),
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
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
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

                async def unfinished_body():
                    yield b"started"
                    await asyncio.Event().wait()

                payload = {"data": unfinished_body()} if phase == "uploading" else {"json": {"model": "m"}}
                await model._request(method="POST", url=url, **payload)
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
    assert (attempt.request_upload_complete is True) is waited_for_response


def _reply(status: int, *, headers: dict | None = None, body: bytes = b"") -> SimpleNamespace:
    return SimpleNamespace(
        status=status, headers=headers or {}, content=SimpleNamespace(read=AsyncMock(return_value=body))
    )


@pytest.mark.parametrize("status", [400, 401, 403, 422])
async def test_a_final_non_retryable_status_is_the_calls_observed_attempt(monkeypatch, status):
    reply = _reply(status, body=b'{"error": {"code": "budget_exceeded"}}')
    wire = SimpleNamespace(request=AsyncMock(return_value=reply))
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test")

    with RequestFailureLog.capture() as log:
        returned = await model._request(method="POST", url="http://model.test/responses")

    # Not retried, and handed back for the caller to raise or handle.
    assert returned is reply and wire.request.await_count == 1
    [event] = log.report.events.values()
    assert (event.boundary, event.failure_kind, event.operation, event.http_status) == (
        FailureBoundary.MODEL,
        MODEL_REQUEST_ERROR,
        "model_request",
        status,
    )
    assert event.attempt_id == f"{event.operation_id}-1" and log.report.failure_id is None


async def test_a_failure_a_gym_server_forwarded_is_not_recorded_twice(monkeypatch):
    forwarded = FailureEvent(failure_kind="test:failure", boundary=FailureBoundary.MODEL, failure_reason="upstream")
    report = FailureReport(failure_id=forwarded.event_id, events={forwarded.event_id: forwarded})
    wire = SimpleNamespace(request=AsyncMock(return_value=_reply(422, headers=FailureHeader.encode(report))))
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test")

    with RequestFailureLog.capture() as log:
        await model._request(method="POST", url="http://model.test/responses")

    assert list(log.report.events) == [forwarded.event_id]


async def test_a_successful_call_records_nothing(monkeypatch):
    wire = SimpleNamespace(request=AsyncMock(return_value=_reply(200)))
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test")

    with RequestFailureLog.capture() as log:
        await model._request(method="POST", url="http://model.test/responses")

    assert not log.report.events


async def _scripted_model(monkeypatch, replies: list[tuple[int, bytes, dict]]):
    """A real local model endpoint that answers each POST /v1/responses with the next scripted reply."""
    script = list(replies)

    async def endpoint(request):
        status, body, headers = script.pop(0)
        return web.Response(status=status, body=body, headers=headers, content_type="application/json")

    app = web.Application()
    app.router.add_post("/v1/responses", endpoint)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
    monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
    monkeypatch.setattr(http_client.asyncio, "sleep", AsyncMock())
    model = NeMoGymAsyncOpenAI(api_key="test", base_url=str(server.make_url("/v1")), max_rate_limit_wait_seconds=0)
    return model, script, server, session


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(400, b'{"error": {"message": "bad request"}}', id="400"),
        pytest.param(401, b'{"error": {"message": "denied for this request"}}', id="401-not-permanent"),
        pytest.param(403, b'{"error": {"message": "forbidden"}}', id="403-not-permanent"),
        pytest.param(
            422, b'{"error": {"type": "budget_exceeded", "message": "Budget has been exceeded"}}', id="422-budget-body"
        ),
    ],
)
async def test_the_error_raised_for_a_final_non_retryable_status_names_that_attempt(monkeypatch, status, body):
    model, script, server, session = await _scripted_model(monkeypatch, [(status, body, {})])
    try:
        with RequestFailureLog.capture() as log:
            with pytest.raises(aiohttp.ClientResponseError) as caught:
                await model.create_response(model="m", input="x")
            report = log.reply_report(caught.value)
    finally:
        await session.close()
        await server.close()

    [attempt] = report.events.values()
    assert not script and caught.value.status == status
    assert report.failure_id == attempt.event_id and attempt.http_status == status
    assert attempt.attempt_id == f"{attempt.operation_id}-1" and attempt.cause_id is None


@pytest.mark.parametrize(
    ("status", "body"),
    [
        pytest.param(401, b'{"error": {"code": "invalid_api_key"}}', id="401-permanent-auth"),
        pytest.param(429, b'{"error": {"code": "insufficient_quota"}}', id="429-permanent-quota"),
    ],
)
async def test_a_permanent_trip_names_the_attempt_that_tripped_it_and_later_calls_name_none(monkeypatch, status, body):
    model, script, server, session = await _scripted_model(monkeypatch, [(status, body, {})])
    try:
        with RequestFailureLog.capture() as log:
            with pytest.raises(PermanentEndpointError) as tripped:
                await model.create_response(model="m", input="x")
            report = log.reply_report(tripped.value)
        with RequestFailureLog.capture() as later_log:
            with pytest.raises(PermanentEndpointError) as later:
                await model.create_response(model="m", input="x")
            later_report = later_log.reply_report(later.value)
    finally:
        await session.close()
        await server.close()

    [attempt] = report.events.values()
    assert report.failure_id == attempt.event_id and attempt.http_status == status
    # No request was sent for the later call, so it has no attempt and no cause.
    assert not script and later_report.failure_id is None and not later_report.events


@pytest.mark.parametrize(
    "earlier",
    [
        pytest.param([(429, b'{"error": {"code": "rate_limit_exceeded"}}', {"Retry-After": "0"})], id="recovered-429"),
        pytest.param([], id="handled-400"),
    ],
)
async def test_an_earlier_operations_attempt_never_becomes_a_later_operations_cause(monkeypatch, earlier):
    replies = [*earlier, (200, b"{}", {})] if earlier else [(400, b"{}", {})]
    model, script, server, session = await _scripted_model(monkeypatch, [*replies, (422, b"{}", {})])
    try:
        with RequestFailureLog.capture() as log:
            if earlier:
                assert await model.create_response(model="m", input="x") == {}
            else:
                handled = await model._request(method="POST", url=f"{model.base_url}/responses", json={})
                assert handled.status == 400
            [before] = log.report.events.values()
            assert log.reply_report(None).failure_id is None
            with pytest.raises(aiohttp.ClientResponseError) as caught:
                await model.create_response(model="m", input="x")
            report = log.reply_report(caught.value)
    finally:
        await session.close()
        await server.close()

    current = report.events[report.failure_id]
    assert not script and current.http_status == 422 and current.operation_id != before.operation_id
    assert current.cause_id is None and report.events[before.event_id] == before


async def test_a_forwarded_gym_failure_stays_the_cause_and_no_attempt_is_added(monkeypatch):
    forwarded = FailureEvent(failure_kind="test:failure", boundary=FailureBoundary.MODEL, failure_reason="upstream")
    header = FailureHeader.encode(FailureReport(failure_id=forwarded.event_id, events={forwarded.event_id: forwarded}))
    model, script, server, session = await _scripted_model(monkeypatch, [(422, b"{}", header)])
    try:
        with RequestFailureLog.capture() as log:
            with pytest.raises(aiohttp.ClientResponseError) as caught:
                await model.create_response(model="m", input="x")
            report = log.reply_report(caught.value)
    finally:
        await session.close()
        await server.close()

    assert not script and report.failure_id == forwarded.event_id and list(report.events) == [forwarded.event_id]


async def test_a_recovered_response_timeout_keeps_its_confirmed_upload_observation(monkeypatch):
    calls = 0

    async def endpoint(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.08)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/responses", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
        model = NeMoGymAsyncOpenAI(
            api_key="test", base_url=str(server.make_url("/")), max_connection_retries=2, request_timeout_seconds=0.04
        )
        with RequestFailureLog.capture() as log:
            response = await model._request(
                method="POST", url=str(server.make_url("/responses")), json={"input": "fixture"}
            )
            assert response.status == 200
            await response.read()
            report = log.reply_report(None)
    assert calls == 2 and report.failure_id is None
    [attempt] = report.events.values()
    assert attempt.request_upload_complete is True
    assert attempt.failure_reason == MODEL_RESPONSE_TIMEOUT_REASON
    assert attempt.operation_id and attempt.attempt_id
    assert FailureHeader.decode(FailureHeader.encode(report)) == report


async def test_failed_body_upload_never_reports_a_response_wait_timeout(monkeypatch):
    async def endpoint(request):
        await request.read()
        return web.json_response({})

    async def broken_body():
        yield b"prefix"
        raise OSError("controlled upload failure")

    app = web.Application()
    app.router.add_post("/responses", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
        model = NeMoGymAsyncOpenAI(
            api_key="test", base_url=str(server.make_url("/")), max_num_tries=1, request_timeout_seconds=0.2
        )
        with RequestFailureLog.capture() as log:
            with pytest.raises((aiohttp.ClientError, OSError)) as caught:
                await model._request(method="POST", url=str(server.make_url("/responses")), data=broken_body())
            report = log.reply_report(caught.value)
    [attempt] = report.events.values()
    assert attempt.request_upload_complete is False
    assert attempt.failure_reason != MODEL_RESPONSE_TIMEOUT_REASON


@pytest.mark.parametrize("mode", ["truncated", "timeout", "complete"])
async def test_actual_200_body_failure_keeps_one_model_attempt(monkeypatch, mode):
    hits = 0

    async def endpoint(request):
        nonlocal hits
        hits += 1
        if mode == "complete":
            return web.json_response({"id": "reply", "output": []})
        response = web.StreamResponse(headers={"Content-Type": "application/json", "Content-Length": "1000"})
        await response.prepare(request)
        await response.write(b'{"id":')
        if mode == "truncated":
            await asyncio.sleep(0.01)
            request.transport.close()
        else:
            await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/responses", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
        model = NeMoGymAsyncOpenAI(
            api_key="test",
            base_url=str(server.make_url("")).rstrip("/"),
            request_timeout_seconds=0.05,
            max_num_tries=1,
        )
        with RequestFailureLog.capture() as log:
            if mode == "complete":
                assert (await model.create_response(input=[]))["id"] == "reply"
                assert not log.report.events
                [operation] = log.model_operations.values()
                assert operation.outcome == "completed" and operation.failure_id is None
            else:
                with pytest.raises((aiohttp.ClientPayloadError, TimeoutError)) as caught:
                    await model.create_response(input=[])
                report = log.reply_report(caught.value)
                assert report.failure_id is not None
                event = report.events[report.failure_id]
                assert event.failure_kind == MODEL_REQUEST_ERROR
                assert event.failure_reason != MODEL_RESPONSE_TIMEOUT_REASON
                assert event.http_status == 200
                assert event.attempt_id.startswith(event.operation_id)
                assert len(report.events) == 1
                [operation] = log.model_operations.values()
                assert operation.operation_id == event.operation_id
                assert operation.final_attempt_id == event.attempt_id
                assert operation.outcome == "failed" and operation.failure_id == event.event_id
                assert caught.value.__cause__.event_id == event.event_id
                assert operation.request_elapsed_ms >= (40 if mode == "timeout" else 8)
            assert operation.http_attempts == 1
            assert operation.transport_failures == 0 and operation.retry_wait_s == 0
    assert hits == 1


@pytest.mark.parametrize("forwarded", [False, True])
@pytest.mark.parametrize("mode", ["truncated", "timeout"])
@pytest.mark.parametrize("status", [401, 422, 503])
async def test_a_failed_error_body_read_selects_the_attempt_that_returned_it(monkeypatch, status, mode, forwarded):
    from nemo_gym.failure_transport import FailureHeader
    from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for

    hits = 0
    remote = NgFailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind=MODEL_REQUEST_ERROR,
        failure_reason="remote model server failure",
        operation="model_request",
        operation_id="remote-op",
        attempt_id="remote-op-1",
        http_status=status,
    )

    async def endpoint(request):
        nonlocal hits
        hits += 1
        headers = {"Content-Type": "application/json", "Content-Length": "1000"}
        if forwarded:
            headers.update(FailureHeader.encode(failure_report_for(remote)))
        response = web.StreamResponse(status=status, headers=headers)
        await response.prepare(request)
        await response.write(b'{"error":')
        if mode == "truncated":
            request.transport.close()
        else:
            await asyncio.sleep(0.15)
        return response

    app = web.Application()
    app.router.add_post("/responses", endpoint)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        monkeypatch.setattr(http_client, "get_global_aiohttp_client", lambda: session)
        monkeypatch.setattr(http_client, "is_span_group_enabled", lambda _: False)
        model = NeMoGymAsyncOpenAI(
            api_key="test",
            base_url=str(server.make_url("")).rstrip("/"),
            request_timeout_seconds=0.05,
            max_num_tries=1,
        )
        with RequestFailureLog.capture() as log:
            with pytest.raises((aiohttp.ClientPayloadError, TimeoutError)) as caught:
                await model.create_response(input=[])
            report = log.reply_report(caught.value)
    assert hits == 1, "a failed body read sends no second request"
    assert report.failure_id is not None
    event = report.events[report.failure_id]
    assert event.http_status == status
    assert event.failure_kind == MODEL_REQUEST_ERROR
    assert event.failure_reason != MODEL_RESPONSE_TIMEOUT_REASON
    assert event.operation_id and event.attempt_id
    if forwarded:
        assert report.failure_id == remote.event_id, "the forwarded failure keeps its original ids"
        assert list(report.events) == [remote.event_id], "no local attempt is added beside it"
    else:
        assert len(report.events) == 1
