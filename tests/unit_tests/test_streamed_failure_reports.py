# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stream whose headers went out before its backend finished still delivers the request's failure report."""

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from fastapi import Body
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig
from nemo_gym.client.server_client import ServerClient
from nemo_gym.failure_transport import FailureHeader, observe_failure
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.typed_exceptions import NgFailureEvent
from tests.unit_tests.test_chat_completions_streaming import _EchoChatModel
from tests.unit_tests.test_responses_api_model_streaming import _EchoModel


_BACKEND_S = 0.04
_KEEPALIVE_S = 0.005


def _model_event(reason: str, attempt: str) -> NgFailureEvent:
    return NgFailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind="model_request_error",
        failure_reason=reason,
        operation="model_request",
        attempt_id=attempt,
        http_status=503,
    )


RECOVERED = _model_event("recovered 503", "a1")
FINAL = _model_event("final 503", "a2")


class _LateResponses(_EchoModel):
    fail: bool = False

    async def responses(self, body=Body()):
        await asyncio.sleep(_BACKEND_S)
        observe_failure(RECOVERED)
        if self.fail:
            raise FINAL
        return await super().responses(body)


class _LateChat(_EchoChatModel):
    fail: bool = False

    async def chat_completions(self, body=Body()):
        await asyncio.sleep(_BACKEND_S)
        observe_failure(RECOVERED)
        if self.fail:
            raise FINAL
        return await super().chat_completions(body)


def _client(model_cls, *, fail: bool, **config) -> TestClient:
    server = model_cls(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="", **config),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    object.__setattr__(server, "fail", fail)
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    return TestClient(app)


def _payloads(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _assert_report(payload: dict, *, final: bool) -> None:
    report = FailureHeader.decode_stream_event(payload)
    assert report is not None and not report.omitted
    assert RECOVERED.event_id in report.events
    # A recovered attempt is an observation; only the error that ended the request is its failure.
    assert report.failure_id == (FINAL.event_id if final else None)


@pytest.mark.parametrize("fail", [False, True])
def test_a_keepalive_responses_stream_reports_late_failures_on_its_terminal_event(fail: bool) -> None:
    client = _client(_LateResponses, fail=fail, stream_keepalive_s=_KEEPALIVE_S)
    reply = client.post("/v1/responses", json={"stream": True, "input": [{"role": "user", "content": "hi"}]})
    assert reply.status_code == 200
    events = _payloads(reply.text)
    assert events[1]["type"] == "response.in_progress", "keepalives opened the stream before the backend finished"
    # The header went out first, so it cannot hold what the backend observed later.
    assert FailureHeader.decode(reply.headers) is None
    terminal = events[-1]
    assert terminal["type"] == ("response.failed" if fail else "response.completed")
    _assert_report(terminal, final=fail)
    assert all(FailureHeader.STREAM_FIELD not in event for event in events[:-1])


@pytest.mark.parametrize("fail", [False, True])
def test_a_keepalive_chat_stream_reports_late_failures_on_its_last_event(monkeypatch, fail: bool) -> None:
    monkeypatch.setattr("nemo_gym.base_responses_api_model._CHAT_KEEPALIVE_SECONDS", _KEEPALIVE_S)
    client = _client(_LateChat, fail=fail)
    reply = client.post("/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert reply.status_code == 200
    assert reply.text.startswith(": keep-alive\n\n"), "a keepalive went out before the backend finished"
    assert FailureHeader.decode(reply.headers) is None
    payloads = _payloads(reply.text)
    if fail:
        [error] = payloads
        assert "error" in error
        _assert_report(error, final=True)
    else:
        assert reply.text.endswith("data: [DONE]\n\n")
        _assert_report(payloads[-1], final=False)
        assert all(FailureHeader.STREAM_FIELD not in payload for payload in payloads[:-1])
        assert "".join(
            (choice.get("delta") or {}).get("content") or "" for payload in payloads for choice in payload["choices"]
        )


def test_a_stream_report_that_is_invalid_or_too_large_stays_omitted() -> None:
    assert FailureHeader.decode_stream_event({"x": 1}) is None
    assert FailureHeader.decode_stream_event({FailureHeader.STREAM_FIELD: {"events": "bad"}}).omitted
    assert FailureHeader.decode_stream_event({FailureHeader.STREAM_FIELD: {"omitted": True}}).omitted


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_the_codex_relay_observes_the_report_a_keepalive_stream_ends_with(monkeypatch, fail: bool) -> None:
    """Producer bytes from the real model server, relayed by the real Codex proxy into the serving request's log."""
    from aiohttp import ClientSession, web

    from nemo_gym.failure_transport import RequestFailureLog
    from responses_api_agents.codex_agent.transport import rate_limit_proxy

    produced = _client(_LateResponses, fail=fail, stream_keepalive_s=_KEEPALIVE_S).post(
        "/v1/responses", json={"stream": True, "input": [{"role": "user", "content": "hi"}]}
    )
    body = produced.content
    import nemo_gym.client.http_client as http_client

    # The relay's central client must belong to this test's event loop, not the TestClient's.
    monkeypatch.setattr(http_client, "_GLOBAL_AIOHTTP_CLIENT", None)

    async def upstream_handler(request: web.Request) -> web.StreamResponse:
        reply = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await reply.prepare(request)
        for start in range(0, len(body), 37):  # arbitrary chunk boundaries
            await reply.write(body[start : start + 37])
        await reply.write_eof()
        return reply

    app = web.Application()
    app.router.add_post("/v1/responses", upstream_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        with RequestFailureLog.capture() as log:
            async with rate_limit_proxy(f"http://127.0.0.1:{port}/v1", "invocation") as proxy:
                async with ClientSession() as session:
                    async with session.post(f"{proxy}/responses", data=b"{}") as relayed:
                        assert await relayed.read() == body  # the relay changes no byte
    finally:
        await runner.cleanup()
    # Recovered and final model failures are observations of the serving request, never its own failure.
    assert log.report.failure_id is None
    assert RECOVERED.event_id in log.report.events
    assert (FINAL.event_id in log.report.events) is fail


def test_the_reader_keeps_malformed_and_omitted_reports_unaccepted() -> None:
    from nemo_gym.failure_transport import StreamFailureReportReader

    field = FailureHeader.STREAM_FIELD
    for line, omitted in (
        (b'data: {"type": "response.completed", "' + field.encode() + b'": {"omitted": true}}\n', True),
        (b'data: {"type": "response.completed", "' + field.encode() + b'": {"events": 3}}\n', True),
        (b'data: {"' + field.encode() + b'": truncated\n', True),
        (b'data: {"type": "response.completed"}\n', None),
    ):
        reader = StreamFailureReportReader()
        reader.feed(line)
        assert (None if reader.report is None else reader.report.omitted) is omitted


def test_invalid_terminal_report_locations_are_omitted() -> None:
    from nemo_gym.failure_transport import RequestFailureLog, StreamFailureReportReader

    field = FailureHeader.STREAM_FIELD
    assert FailureHeader.decode_stream_event({field: None}).omitted
    reader = StreamFailureReportReader()
    reader.feed(b'data: {"type": "response.in_progress", "' + field.encode() + b'": {"omitted": false}}\n')
    assert reader.report.omitted  # an explicit report at an unsupported location is not silently discarded
    reader.feed(b'data: {"type": "response.output_text.delta", "delta": "' + field.encode() + b'"}\n')
    assert reader.report.omitted  # content cannot clear an earlier omission
    clean = StreamFailureReportReader()
    clean.feed(b'data: {"type": "response.output_text.delta", "delta": "' + field.encode() + b'"}\n')
    assert clean.report is None
    cut = StreamFailureReportReader()
    cut.feed(b'data: {"type": "response.failed", "' + field.encode() + b'": {"events": {')
    with RequestFailureLog.capture() as log:
        cut.observe()  # the stream ended inside the report's line
    assert log.report.omitted
