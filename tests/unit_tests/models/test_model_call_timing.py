# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from typing import Any

import pytest

from nemo_gym.models import model_call_timing
from nemo_gym.models.model_call_timing import ModelCallTimingMiddleware
from nemo_gym.server_runtime.request_handling.disconnect import CLIENT_DISCONNECTED_STATE_KEY
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup


KEEPALIVE = b": keep-alive\n\n"
CREATED = b'event: response.created\ndata: {"type": "response.created"}\n\n'
DELTA = b'event: response.output_text.delta\ndata: {"delta": "hi"}\n\n'


@pytest.fixture
def collected_metrics(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("model-call-timing-test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()

    def collect() -> dict[str, list[Any]]:
        data = reader.get_metrics_data()
        result: dict[str, list[Any]] = {}
        for resource_metric in data.resource_metrics if data is not None else ():
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    result[metric.name] = list(metric.data.data_points)
        return result

    yield collect
    provider.shutdown()


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(model_call_timing.time, "perf_counter", lambda: clock.now)
    return clock


def _app(status: int, chunks: list[bytes], clock: _Clock | None = None):
    """Sends ``chunks``, one second apart on ``clock`` when given."""

    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        for index, chunk in enumerate(chunks):
            if clock is not None:
                clock.now += 1.0
            await send({"type": "http.response.body", "body": chunk, "more_body": index < len(chunks) - 1})

    return app


async def _call(app, path: str = "/v1/responses", method: str = "POST", disconnect: bool = False) -> None:
    async def receive() -> dict[str, Any]:
        if disconnect:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        pass

    await ModelCallTimingMiddleware(app, server_name="policy_model")(
        {
            "type": "http",
            "path": path,
            "method": method,
            "state": {CLIENT_DISCONNECTED_STATE_KEY: asyncio.Event()},
        },
        receive,
        send,
    )


def _only(points: list[Any]) -> tuple[dict[str, Any], float]:
    (point,) = points
    return dict(point.attributes), point.sum


async def test_a_streamed_call_is_timed_to_its_last_byte_and_its_first_content_byte(collected_metrics, clock) -> None:
    await _call(_app(200, [KEEPALIVE, CREATED, DELTA, b"data: [DONE]\n\n"], clock))

    metrics = collected_metrics()
    attributes, first_byte_ms = _only(metrics[gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT])
    assert attributes == {"nemo.gym.model.dialect": "responses", "nemo.gym.server.name": "policy_model"}
    # The keepalive (1 s) and the lifecycle event (2 s) are skipped: the delta arrives at 3 s.
    assert first_byte_ms == pytest.approx(3000)
    attributes, duration_ms = _only(metrics[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "2xx"
    assert duration_ms == pytest.approx(4000)


@pytest.mark.parametrize(
    ("path", "dialect"),
    [
        ("/v1/chat/completions", "chat"),
        ("/v1/messages", "messages"),
        ("/ng-rollout/3-0-a1/v1/responses", "responses"),
        ("/ng-rollout/3-0-a1/training-token-capture/v1/chat/completions", "chat"),
        ("/ng-sampling/null/v1/responses", "responses"),
        ("/ng-rollout/task0-r1-a0/ng-agent-instance/judge/v1/responses", "responses"),
        ("/ng-rollout/3-0-a1/ng-sampling/null/v1/messages", "messages"),
    ],
)
async def test_each_dialect_route_is_timed_under_any_correlation_prefix(collected_metrics, path, dialect) -> None:
    await _call(_app(200, [b'{"id": "x"}']), path=path)

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.dialect"] == dialect


@pytest.mark.parametrize(("status", "outcome"), [(200, "2xx"), (429, "4xx"), (502, "5xx")])
async def test_the_outcome_is_the_status_class(collected_metrics, status, outcome) -> None:
    await _call(_app(status, [b"body"]))

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == outcome


@pytest.mark.parametrize(
    "terminal",
    [
        b'event: response.failed\ndata: {"type": "response.failed"}\n\n',
        b'event: error\ndata: {"error": {"message": "boom"}}\n\n',
    ],
)
async def test_a_stream_that_reports_its_failure_in_band_keeps_its_status_class(collected_metrics, terminal) -> None:
    await _call(_app(200, [CREATED, terminal]))

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "2xx"


@pytest.mark.parametrize(
    "failure", [RuntimeError("boom"), asyncio.CancelledError()], ids=["raised", "server cancelled"]
)
async def test_a_call_that_fails_mid_response_is_an_error(collected_metrics, failure) -> None:
    with pytest.raises(type(failure)):
        await _call(_failing(failure))

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "error"


async def test_a_failure_after_the_client_left_is_a_cancelled_call(collected_metrics) -> None:
    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": DELTA, "more_body": True})
        scope["state"][CLIENT_DISCONNECTED_STATE_KEY].set()
        raise RuntimeError("cleanup after the cancellation failed")

    with pytest.raises(RuntimeError):
        await _call(app)

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "cancelled"


@pytest.mark.parametrize("disconnect", [True, False], ids=["client disconnected", "response abandoned"])
async def test_a_call_that_returns_without_its_last_byte_is_still_timed(collected_metrics, disconnect) -> None:
    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": DELTA, "more_body": True})
        await receive()

    await _call(app, disconnect=disconnect)

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == ("cancelled" if disconnect else "error")


async def test_other_routes_and_methods_are_not_timed(collected_metrics) -> None:
    await _call(_app(200, [b"ok"]), path="/v1/models", method="GET")
    await _call(_app(200, [b"ok"]), path="/v1/embeddings")
    await _call(_app(200, [b"ok"]), path="/foo/v1/responses")
    await _call(_app(200, [b"ok"]), path="/ng-rollout/3-0-a1/v1/models")
    await _call(_app(200, [b"ok"]), path="/ng-sampling/v1/responses")

    metrics = collected_metrics()
    assert gym_metrics.MODEL_CALL_DURATION_INSTRUMENT not in metrics
    assert gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT not in metrics


def _failing(fail: BaseException):
    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise fail

    return app


def _model_server_app(responses, *, stream_keepalive_s: float | None = None):
    """A minimal model server whose ``responses()`` awaits ``responses``, with the request middleware
    ``run_webserver`` adds, in its order."""
    from unittest.mock import MagicMock

    from fastapi import Body

    from nemo_gym.client.server_client import ServerClient
    from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
    from nemo_gym.server_runtime.request_handling.trace_context import install_trace_context_path

    class _Model(SimpleResponsesAPIModel):
        def _mints_response_ids(self) -> bool:
            return True

        async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()):
            return await responses()

        async def chat_completions(self, body: Any = Body()):
            raise NotImplementedError

    server = _Model(
        config=BaseResponsesAPIModelConfig(
            host="0.0.0.0", port=8099, entrypoint="", name="policy_model", stream_keepalive_s=stream_keepalive_s
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    server.setup_request_timing(app)
    server.setup_cancellation_middleware(app)
    install_trace_context_path(app)
    return app


def _response():
    from nemo_gym.core.responses_api_types import NeMoGymResponse

    return NeMoGymResponse(
        id="resp_1",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


async def _serve(
    app, path: str, *, stream: bool = False, disconnect_once: asyncio.Event | None = None
) -> list[dict[str, Any]]:
    """POST a Responses request to ``app`` over raw ASGI and return what reached the client.

    The client leaves once ``disconnect_once`` is set; without it, a streaming client leaves after its first body
    message."""
    sent: list[dict[str, Any]] = []
    requested = False
    leave = disconnect_once or asyncio.Event()
    body = json.dumps({"input": "hi", **({"stream": True} if stream else {})}).encode()

    async def receive() -> dict[str, Any]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": body, "more_body": False}
        await leave.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)
        if stream and disconnect_once is None and message["type"] == "http.response.body":
            leave.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, receive, send)
    return sent


async def test_a_client_that_leaves_before_the_reply_starts_is_a_cancelled_call_without_a_first_byte(
    collected_metrics,
) -> None:
    started = asyncio.Event()

    async def generating() -> None:
        started.set()
        await asyncio.sleep(3600)

    sent = await _serve(_model_server_app(generating), "/v1/responses", disconnect_once=started)

    assert sent == []
    metrics = collected_metrics()
    attributes, _ = _only(metrics[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "cancelled"
    assert gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT not in metrics


async def test_a_client_that_leaves_a_held_open_stream_is_a_cancelled_call_without_a_first_byte(
    collected_metrics,
) -> None:
    async def generating() -> None:
        await asyncio.sleep(3600)

    sent = await _serve(_model_server_app(generating, stream_keepalive_s=0.01), "/v1/responses", stream=True)

    (status,) = [message["status"] for message in sent if message["type"] == "http.response.start"]
    assert status == 200
    assert [message for message in sent if message["type"] == "http.response.body"], "the keepalive opened the stream"
    metrics = collected_metrics()
    attributes, _ = _only(metrics[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "cancelled"
    assert gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT not in metrics


@pytest.mark.parametrize(
    "path",
    [
        "/v1/responses",
        "/ng-sampling/null/v1/responses",
        f"/ng-rollout/r1/ng-agent-instance/judge/ng-sampling/null/ng-trace/00-{'a' * 32}-{'b' * 16}-01/v1/responses",
    ],
)
async def test_a_model_server_times_a_call_under_any_correlation_prefix(collected_metrics, path) -> None:
    reached: list[bool] = []

    async def answered():
        reached.append(True)
        return _response()

    sent = await _serve(_model_server_app(answered), path)

    assert sent[0]["status"] == 200
    assert reached == [True]
    metrics = collected_metrics()
    attributes, _ = _only(metrics[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes == {
        "nemo.gym.model.dialect": "responses",
        "nemo.gym.server.name": "policy_model",
        "nemo.gym.model.call_outcome": "2xx",
    }
    _only(metrics[gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT])


async def test_a_model_server_times_a_failed_call_with_the_status_the_client_gets(collected_metrics) -> None:
    async def failing() -> None:
        raise RuntimeError("upstream broke")

    sent = await _serve(_model_server_app(failing), "/v1/responses")

    assert sent[0]["status"] == 500
    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "5xx"
