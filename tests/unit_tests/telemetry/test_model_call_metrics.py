# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ``gym.model_call.*`` instruments the capture middleware records, against a real in-memory reader.

Every test drives the real middleware over a FastAPI app so the metrics are checked where they
are produced: off the event loop, from the same normalised record the capture file is built from.
"""

import asyncio

import pytest
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    CaptureStore,
    ModelCallCaptureConfig,
    _CaptureMiddleware,
    install_model_call_capture,
    read_model_call_records,
)
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry import setup as telemetry_setup
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

SERVER = gym_metrics.MODEL_CALL_SERVER_NAME_ATTRIBUTE
DIALECT = gym_metrics.MODEL_CALL_DIALECT_ATTRIBUTE
OUTCOME = gym_metrics.MODEL_CALL_OUTCOME_ATTRIBUTE
TOKEN_TYPE = gym_metrics.MODEL_CALL_TOKEN_TYPE_ATTRIBUTE
FINISH = gym_metrics.MODEL_CALL_FINISH_REASON_ATTRIBUTE
DURATION = gym_metrics.MODEL_CALL_DURATION_INSTRUMENT
TTFT = gym_metrics.MODEL_CALL_TTFT_INSTRUMENT
TOKENS = gym_metrics.MODEL_CALL_TOKENS_INSTRUMENT
FINISH_TOTAL = gym_metrics.MODEL_CALL_FINISH_INSTRUMENT

CHAT_RESPONSE = {
    "object": "chat.completion",
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "completion_tokens_details": {"reasoning_tokens": 200},
    },
}


def _app_with_capture(tmp_path, handler, *, server_name: str = "policy") -> TestClient:
    app = FastAPI()
    app.post("/v1/chat/completions")(handler)
    app.post("/v1/responses")(handler)
    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path),
        model_server_name=server_name,
    )
    return TestClient(app, raise_server_exceptions=False)


def _by_attributes(points, key):
    return {p.attributes[key]: p for p in points}


def test_successful_chat_call_records_duration_tokens_and_finish(collected_metrics, tmp_path):  # noqa: F811
    async def ok(body: dict = Body()) -> dict:
        return CHAT_RESPONSE

    client = _app_with_capture(tmp_path, ok)
    assert client.post("/ng-rollout/r1/v1/chat/completions", json={"messages": []}).status_code == 200

    collected = collected_metrics()
    (duration,) = collected[DURATION]
    assert duration.attributes == {SERVER: "policy", DIALECT: "chat", OUTCOME: "ok"}
    assert duration.count == 1 and duration.sum >= 0
    assert list(duration.explicit_bounds) == list(gym_metrics.MODEL_CALL_DURATION_BOUNDARIES_MS)

    tokens = _by_attributes(collected[TOKENS], TOKEN_TYPE)
    assert {k: p.sum for k, p in tokens.items()} == {"input": 1200, "output": 300, "reasoning": 200}
    assert all(p.attributes[SERVER] == "policy" and p.attributes[DIALECT] == "chat" for p in tokens.values())
    assert list(tokens["input"].explicit_bounds) == list(gym_metrics.MODEL_CALL_TOKEN_BOUNDARIES)

    (finish,) = collected[FINISH_TOTAL]
    assert finish.attributes == {SERVER: "policy", DIALECT: "chat", FINISH: "stop"}
    assert finish.value == 1

    # A JSON response has no first-chunk latency distinct from its total.
    assert TTFT not in collected


def test_metrics_agree_with_the_capture_file(collected_metrics, tmp_path):  # noqa: F811
    async def ok(body: dict = Body()) -> dict:
        return CHAT_RESPONSE

    client = _app_with_capture(tmp_path, ok)
    client.post("/ng-rollout/r-same/v1/chat/completions", json={"messages": []})

    (record,) = read_model_call_records(CaptureStore(tmp_path), "r-same")
    (duration,) = collected_metrics()[DURATION]
    assert duration.sum == pytest.approx(record.latency_total_ms, abs=0.01)
    tokens = _by_attributes(collected_metrics()[TOKENS], TOKEN_TYPE)
    assert (tokens["input"].sum, tokens["output"].sum) == (record.tokens_in, record.tokens_out)


def test_failed_call_records_outcome_and_no_tokens(collected_metrics, tmp_path):  # noqa: F811
    async def boom(body: dict = Body()) -> JSONResponse:
        return JSONResponse(content={"error": "boom"}, status_code=503)

    client = _app_with_capture(tmp_path, boom, server_name="judge")
    assert client.post("/ng-rollout/r-err/v1/responses", json={"input": "x"}).status_code == 503

    collected = collected_metrics()
    (duration,) = collected[DURATION]
    assert duration.attributes == {SERVER: "judge", DIALECT: "responses", OUTCOME: "upstream_error"}
    assert TOKENS not in collected and FINISH_TOTAL not in collected


def test_raised_call_records_exception_outcome(collected_metrics, tmp_path):  # noqa: F811
    async def raises(body: dict = Body()) -> dict:
        raise RuntimeError("kaboom")

    client = _app_with_capture(tmp_path, raises)
    assert client.post("/ng-rollout/r-raise/v1/responses", json={"input": "x"}).status_code == 500

    (duration,) = collected_metrics()[DURATION]
    assert duration.attributes[OUTCOME] == "exception"


def test_responses_dialect_counts_status_as_finish_reason(collected_metrics, tmp_path):  # noqa: F811
    async def completed(body: dict = Body()) -> dict:
        return {
            "object": "response",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0}},
        }

    client = _app_with_capture(tmp_path, completed)
    client.post("/ng-rollout/r-resp/v1/responses", json={"input": "x"})

    collected = collected_metrics()
    (finish,) = collected[FINISH_TOTAL]
    assert finish.attributes[FINISH] == "completed" and finish.attributes[DIALECT] == "responses"
    tokens = _by_attributes(collected[TOKENS], TOKEN_TYPE)
    assert {k: p.sum for k, p in tokens.items()} == {"input": 10, "output": 5, "reasoning": 0}


def test_streamed_call_records_ttft(collected_metrics, tmp_path):  # noqa: F811
    store = CaptureStore(tmp_path)
    events = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant",'
        b'"model":"claude","usage":{"input_tokens":40,"output_tokens":1}}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":7}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def app(_scope, receive, send):
        await receive()
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        for chunk in events:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b'{"messages":[]}', "more_body": False}

    async def send(_message):
        pass

    asyncio.run(
        _CaptureMiddleware(app, store=store, model_server_name="policy")(
            {"type": "http", "path": "/ng-rollout/r-sse/v1/messages", "raw_path": b"", "headers": []},
            receive,
            send,
        )
    )

    collected = collected_metrics()
    (ttft,) = collected[TTFT]
    assert ttft.attributes == {SERVER: "policy", DIALECT: "messages"}
    assert ttft.count == 1
    assert list(ttft.explicit_bounds) == list(gym_metrics.MODEL_CALL_TTFT_BOUNDARIES_MS)
    (duration,) = collected[DURATION]
    assert duration.attributes[OUTCOME] == "ok"
    (finish,) = collected[FINISH_TOTAL]
    assert finish.attributes[FINISH] == "end_turn"
    tokens = _by_attributes(collected[TOKENS], TOKEN_TYPE)
    assert {k: p.sum for k, p in tokens.items()} == {"input": 40, "output": 7}


def test_without_telemetry_nothing_is_recorded_and_capture_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", None)
    gym_metrics._reset_for_testing()

    async def ok(body: dict = Body()) -> dict:
        return CHAT_RESPONSE

    client = _app_with_capture(tmp_path, ok)
    assert client.post("/ng-rollout/r-off/v1/chat/completions", json={"messages": []}).status_code == 200
    (record,) = read_model_call_records(CaptureStore(tmp_path), "r-off")
    assert record.tokens_in == 1200


def test_metric_failure_never_reaches_the_capture(monkeypatch, tmp_path):
    class _Broken:
        is_exporting = True

        @property
        def meter(self):
            raise RuntimeError("no meter for you")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Broken())
    gym_metrics._reset_for_testing()

    async def ok(body: dict = Body()) -> dict:
        return CHAT_RESPONSE

    client = _app_with_capture(tmp_path, ok)
    assert client.post("/ng-rollout/r-broken/v1/chat/completions", json={"messages": []}).status_code == 200
    (record,) = read_model_call_records(CaptureStore(tmp_path), "r-broken")
    assert record.error_category is None
    assert not CaptureStore(tmp_path).is_incomplete("r-broken")
