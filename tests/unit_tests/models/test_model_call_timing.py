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


@pytest.mark.parametrize(
    ("failure", "outcome"), [(RuntimeError("boom"), "error"), (asyncio.CancelledError(), "cancelled")]
)
async def test_a_call_that_fails_mid_response_is_still_timed(collected_metrics, failure, outcome) -> None:
    with pytest.raises(type(failure)):
        await _call(_failing(failure))

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == outcome


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


def test_model_servers_install_the_timing_through_the_request_timing_hook() -> None:
    from types import SimpleNamespace

    from fastapi import FastAPI

    from nemo_gym.models.base_responses_api_model import SimpleResponsesAPIModel

    app = FastAPI()
    SimpleResponsesAPIModel.setup_request_timing(SimpleNamespace(config=SimpleNamespace(name="policy_model")), app)

    (middleware,) = app.user_middleware
    assert middleware.cls is ModelCallTimingMiddleware
    assert middleware.kwargs == {"server_name": "policy_model"}


def test_a_call_the_exception_middleware_answers_is_timed_with_the_status_the_client_gets(collected_metrics) -> None:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/v1/responses")
    async def responses() -> None:
        raise RuntimeError("upstream broke")

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except RuntimeError:
            return JSONResponse("failed", status_code=500)

    # Added after the exception middleware, as the request-timing hook is, so it wraps it.
    app.add_middleware(ModelCallTimingMiddleware, server_name="policy_model")

    assert TestClient(app).post("/v1/responses", json={}).status_code == 500

    attributes, _ = _only(collected_metrics()[gym_metrics.MODEL_CALL_DURATION_INSTRUMENT])
    assert attributes["nemo.gym.model.call_outcome"] == "5xx"
