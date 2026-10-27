# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model transport observations preserve retries, cancellation and readable HTTP responses."""

import asyncio
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
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


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
    assert len(report.events) == 2
    event = next(event for event in report.events.values() if event.boundary is FailureBoundary.MODEL)
    assert event.failure_kind == MODEL_REQUEST_ERROR and event.http_status is None
    cause = report.events[event.cause_id]
    assert cause.exception_type == f"{error_type.__module__}.{error_type.__qualname__}"
    assert cause.boundary is None and cause.failure_reason == ""
    assert "sensitive" not in report.model_dump_json()
    assert event.attempt_id.startswith(event.operation_id)
    assert FailureHeader.decode(FailureHeader.encode(report)) == report


@pytest.mark.parametrize("error_type", [aiohttp.ServerDisconnectedError, aiohttp.ClientOSError, TimeoutError])
async def test_a_terminal_transport_attempt_is_observed_without_an_extra_retry(monkeypatch, error_type):
    error = error_type("private request")
    wire = SimpleNamespace(request=AsyncMock(side_effect=error))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: wire)
    monkeypatch.setattr(server_utils, "is_span_group_enabled", lambda _: False)
    model = NeMoGymAsyncOpenAI(api_key="test", base_url="http://model.test", max_num_tries=1)

    with RequestFailureLog.capture() as log:
        with pytest.raises(error_type) as caught:
            await model._request(method="POST", url="http://model.test/responses")

    assert caught.value is error and wire.request.await_count == 1
    assert len(log.report.events) == 2
    event = next(event for event in log.report.events.values() if event.boundary is FailureBoundary.MODEL)
    assert log.report.events[event.cause_id].failure_reason == ""
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
    event = FailureEvent(boundary=FailureBoundary.MODEL, failure_reason="界" * 2000)
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
