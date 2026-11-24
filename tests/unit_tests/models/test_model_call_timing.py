# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any

import pytest

from nemo_gym.models import model_call_timing
from nemo_gym.models.model_call_timing import ModelCallTimingMiddleware
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
        {"type": "http", "path": path, "method": method}, receive, send
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


@pytest.mark.parametrize("failure", [RuntimeError("boom"), asyncio.CancelledError()], ids=["raised", "server cancelled"])
async def test_a_call_that_fails_mid_response_is_an_error(collected_metrics, failure) -> None:
    with pytest.raises(type(failure)):
        await _call(_failing(failure))

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "error"


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


def _model_server_app(route):
    """``route`` on every path, behind the request middleware a model server runs, in the order it runs it."""
    from types import SimpleNamespace

    from fastapi import FastAPI

    from nemo_gym.models.base_responses_api_model import SimpleResponsesAPIModel
    from nemo_gym.server_runtime.base_server import SimpleServer
    from nemo_gym.server_runtime.request_handling.model_sampling import ModelSamplingMiddleware
    from nemo_gym.server_runtime.request_handling.trace_context import install_trace_context_path

    app = FastAPI()
    app.post("/{path:path}")(route)
    app.add_middleware(ModelSamplingMiddleware)
    server = SimpleNamespace(config=SimpleNamespace(name="policy_model"))
    SimpleServer.setup_exception_middleware(server, app)
    SimpleResponsesAPIModel.setup_request_timing(server, app)
    SimpleServer.setup_cancellation_middleware(server, app)
    install_trace_context_path(app)
    return app


async def _serve(app, path: str, *, disconnect_once: asyncio.Event | None = None) -> list[dict[str, Any]]:
    """POST to ``app`` over raw ASGI and return what reached the client; the client leaves once ``disconnect_once``
    is set."""
    sent: list[dict[str, Any]] = []
    requested = False

    async def receive() -> dict[str, Any]:
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await (disconnect_once or asyncio.Event()).wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

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
        "headers": [(b"content-type", b"application/json")],
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


async def test_a_client_that_leaves_mid_stream_is_a_cancelled_call_with_its_first_byte(collected_metrics) -> None:
    from fastapi.responses import StreamingResponse

    first_delta_sent = asyncio.Event()

    async def stream():
        yield DELTA
        first_delta_sent.set()
        await asyncio.sleep(3600)

    async def streaming() -> StreamingResponse:
        return StreamingResponse(stream(), media_type="text/event-stream")

    sent = await _serve(_model_server_app(streaming), "/v1/responses", disconnect_once=first_delta_sent)

    assert [message.get("body") for message in sent if message["type"] == "http.response.body"] == [DELTA]
    metrics = collected_metrics()
    attributes, _ = _only(metrics[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "cancelled"
    _only(metrics[gym_metrics.MODEL_TIME_TO_FIRST_BYTE_INSTRUMENT])


@pytest.mark.parametrize(
    "path",
    [
        "/v1/responses",
        "/ng-sampling/null/v1/responses",
        f"/ng-rollout/r1/ng-agent-instance/judge/ng-sampling/null/ng-trace/00-{'a' * 32}-{'b' * 16}-01/v1/responses",
    ],
)
async def test_a_model_server_times_a_call_under_any_correlation_prefix(collected_metrics, path) -> None:
    async def answered() -> dict[str, str]:
        return {"id": "resp_1"}

    sent = await _serve(_model_server_app(answered), path)

    assert sent[0]["status"] == 200
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
