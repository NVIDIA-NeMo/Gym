# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib
import json
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi import Body, FastAPI, Response
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    CaptureStore,
    ModelCallCaptureConfig,
    SimpleResponsesAPIModel,
    _orjson_dispatch_response,
    install_model_call_capture,
    make_capture_store,
)
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.observability.model_calls import read_model_call_records
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient, apply_rollout_prefix, rollout_path_prefix


class TestBaseResponsesAPIModel:
    def test_BaseResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")
        BaseResponsesAPIModel(config=config)
        assert "observability_enabled" not in BaseResponsesAPIModelConfig.model_fields
        assert "model_call_capture_dir" not in BaseResponsesAPIModelConfig.model_fields

    def test_SimpleResponsesAPIModel(self) -> None:
        config = BaseResponsesAPIModelConfig(host="", port=0, openai_api_key="123", entrypoint="", name="")

        class TestSimpleResponsesAPIModel(SimpleResponsesAPIModel):
            async def chat_completions(
                self, request: NeMoGymResponseCreateParamsNonStreaming
            ) -> NeMoGymChatCompletion:
                raise NotImplementedError

            async def responses(self, request: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
                raise NotImplementedError

        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {}
        model = TestSimpleResponsesAPIModel(config=config, server_client=server_client)
        model.setup_webserver()


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ModelCallRecord", "nemo_gym.observability.model_calls"),
        ("read_model_call_records", "nemo_gym.observability.model_calls"),
        ("aggregate_model_call_metrics", "nemo_gym.observability.model_calls"),
        ("maybe_rollout_id_from_run_body", "nemo_gym.rollout_correlation"),
    ],
)
def test_model_call_legacy_import_resolves_with_deprecation(name, module):
    legacy = importlib.import_module("nemo_gym.base_responses_api_model")
    with pytest.warns(DeprecationWarning, match=f"{module}.{name}"):
        value = getattr(legacy, name)
    assert value is getattr(importlib.import_module(module), name)


class _DispatchPayload(BaseModel):
    text: str
    token_ids: list[int]


class _AliasedDispatchPayload(BaseModel):
    schema_: dict = Field(alias="schema")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ({"text": "café", "token_ids": [1, 2, 3]}, {"text": "café", "token_ids": [1, 2, 3]}),
        (_DispatchPayload(text="café", token_ids=[1, 2, 3]), {"text": "café", "token_ids": [1, 2, 3]}),
        (_AliasedDispatchPayload(schema={"type": "object"}), {"schema": {"type": "object"}}),
    ],
)
def test_orjson_dispatch_response_serializes_json(content, expected):
    response = _orjson_dispatch_response(content)

    assert type(response) is Response
    assert response.body == orjson.dumps(expected)
    assert response.headers["content-type"] == "application/json"


def test_orjson_dispatch_response_preserves_pydantic_aliases():
    content = NeMoGymResponse(
        id="resp_1",
        created_at=0,
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        text={
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
            }
        },
    )

    body = orjson.loads(_orjson_dispatch_response(content).body)
    wire_format = body["text"]["format"]

    assert "schema" in wire_format
    assert "schema_" not in wire_format


def test_orjson_dispatch_response_preserves_existing_response():
    existing_response = Response(content=b"already encoded", media_type="application/octet-stream", status_code=202)

    assert _orjson_dispatch_response(existing_response) is existing_response


def test_orjson_dispatch_response_preserves_json_schema_alias() -> None:
    """Structured judge replies must retain API field names across the HTTP boundary."""
    schema = {
        "type": "object",
        "properties": {"correct": {"type": "boolean"}},
        "required": ["correct"],
        "additionalProperties": False,
    }
    original = NeMoGymResponse(
        id="structured_judge",
        created_at=0,
        model="fixture",
        object="response",
        status="completed",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        text={"format": {"type": "json_schema", "name": "judge", "schema": schema, "strict": True}},
    )

    payload = orjson.loads(_orjson_dispatch_response(original).body)

    assert payload["text"]["format"]["schema"] == schema
    assert "schema_" not in payload["text"]["format"]
    assert NeMoGymResponse.model_validate(payload) == original


def _capture_config(tmp_path, *, enabled: bool = True) -> ModelCallCaptureConfig:
    return ModelCallCaptureConfig(
        observability_enabled=enabled,
        model_call_capture_dir=tmp_path if enabled else None,
    )


def _install_capture(app, tmp_path, *, model_server_name: str = "srv") -> None:
    install_model_call_capture(
        app,
        _capture_config(tmp_path),
        model_server_name=model_server_name,
        assistant_message_header=b"x-assistant-message-id",
    )


@pytest.mark.parametrize("rollout_id", ["", "a/b", "../a", "a b", "röllout"])
def test_capture_store_rejects_unsafe_rollout_ids(tmp_path, rollout_id):
    store = CaptureStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid rollout id"):
        store.path_for(rollout_id)


def test_capture_store_preserves_valid_rollout_id(tmp_path):
    store = CaptureStore(tmp_path)

    assert store.path_for("task-1_a.2").name == "task-1_a.2.capture.jsonl"


def test_capture_store_orjson_round_trip_preserves_unicode_and_blank_lines(tmp_path):
    store = CaptureStore(tmp_path)
    store.path_for("rollout-1").write_bytes(b"\n")
    exchange = {
        "request": {"text": "Unicode payload: café 東京", "path": tmp_path / "payload"},
        "response": {},
    }

    store.record("rollout-1", exchange)

    assert store.read("rollout-1") == [
        {
            "request": {"text": "Unicode payload: café 東京", "path": str(tmp_path / "payload")},
            "response": {},
        }
    ]
    store.record("rollout-1", {"request": {"text": "second"}, "response": {}})
    assert [record.call_index for record in read_model_call_records(store, "rollout-1")] == [0, 1]


def test_capture_store_raises_on_malformed_nonblank_json(tmp_path):
    store = CaptureStore(tmp_path)
    store.path_for("rollout-1").write_bytes(b'{"request": {}}\n{not-json}\n')

    with pytest.raises(orjson.JSONDecodeError):
        store.read("rollout-1")


@pytest.mark.parametrize(
    "headers,expected",
    [
        ([], None),
        ([(b"X-Session-Id", b"session-1")], "session-1"),
        ([(b"x-session-id", b"session-1"), (b"X-Session-Id", b"session-1")], "session-1"),
        ([(b"x-session-id", b"session-1"), (b"x-session-id", b"session-2")], None),
    ],
)
def test_unique_request_header_requires_one_value(headers, expected):
    from nemo_gym.base_responses_api_model import _unique_request_header

    assert _unique_request_header(headers, b"x-session-id") == expected


@pytest.mark.parametrize(
    "headers,expected",
    [
        ([], None),
        ([(b"X-Assistant-Message-Id", b"assistant-1")], "assistant-1"),
        (
            [
                (b"x-assistant-message-id", b"assistant-1"),
                (b"X-Assistant-Message-Id", b"assistant-1"),
            ],
            "assistant-1",
        ),
        (
            [
                (b"x-assistant-message-id", b"assistant-1"),
                (b"X-Assistant-Message-Id", b"assistant-2"),
            ],
            None,
        ),
        ([(b"x-assistant-message-id", b"")], None),
    ],
)
def test_capture_assistant_message_header_round_trip(tmp_path, headers, expected):
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    store = CaptureStore(tmp_path)
    forwarded_headers = []

    async def app(scope, receive, send):
        forwarded_headers.extend(scope["headers"])
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"output":[]}', "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b'{"input":"hi"}', "more_body": False}

    async def send(_message):
        pass

    request_headers = [(b"x-session-id", b"opencode-session"), *headers]
    asyncio.run(
        _CaptureMiddleware(
            app, store=store, model_server_name="srv", assistant_message_header=b"x-assistant-message-id"
        )(
            {
                "type": "http",
                "path": "/ng-rollout/header-round-trip/v1/responses",
                "raw_path": b"/ng-rollout/header-round-trip/v1/responses",
                "headers": request_headers,
            },
            receive,
            send,
        )
    )

    assert forwarded_headers == request_headers
    [exchange] = store.read("header-round-trip")
    assert exchange.get("client_assistant_message_id") == expected
    if expected is None:
        assert "client_assistant_message_id" not in exchange
    [call] = read_model_call_records(store, "header-round-trip")
    assert call.client_assistant_message_id == expected
    assert call.client_session_id == "opencode-session"
    assert call.response == {"output": []}


def test_capture_is_durable_before_stream_terminal_event_is_sent(tmp_path):
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    store = CaptureStore(tmp_path)
    durable_call_counts = []

    async def app(_scope, receive, send):
        await receive()
        messages = [
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]},
            {"type": "http.response.body", "body": b"event: message_", "more_body": True},
            {
                "type": "http.response.body",
                "body": b'stop\ndata: {"type":"message_stop"}\n\n',
                "more_body": True,
            },
            {"type": "http.response.body", "body": b"", "more_body": False},
        ]
        for message in messages:
            await send(message)

    async def receive():
        return {"type": "http.request", "body": b'{"input":"hi"}', "more_body": False}

    async def send(message):
        if message["type"] == "http.response.body":
            exchanges = store.read("fast-rollout")
            durable_call_counts.append(len(exchanges))
            if exchanges:
                assert exchanges[0]["client_assistant_message_id"] == "assistant-stream"

    asyncio.run(
        _CaptureMiddleware(
            app, store=store, model_server_name="srv", assistant_message_header=b"x-assistant-message-id"
        )(
            {
                "type": "http",
                "path": "/ng-rollout/fast-rollout/v1/messages",
                "raw_path": b"/ng-rollout/fast-rollout/v1/messages",
                "headers": [
                    (b"x-session-id", b"opencode-session"),
                    (b"x-assistant-message-id", b"assistant-stream"),
                ],
            },
            receive,
            send,
        )
    )

    assert durable_call_counts == [0, 1, 1]
    [call] = read_model_call_records(store, "fast-rollout")
    assert call.client_session_id == "opencode-session"
    assert call.client_assistant_message_id == "assistant-stream"


def test_capture_retains_partial_stream_when_downstream_raises(tmp_path):
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    store = CaptureStore(tmp_path)
    partial = b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'

    async def app(_scope, receive, send):
        await receive()
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]}
        )
        await send({"type": "http.response.body", "body": partial, "more_body": True})
        raise RuntimeError("stream failed")

    async def receive():
        return {"type": "http.request", "body": b'{"input":"hi"}', "more_body": False}

    async def send(_message):
        pass

    with pytest.raises(RuntimeError, match="stream failed"):
        asyncio.run(
            _CaptureMiddleware(
                app, store=store, model_server_name="srv", assistant_message_header=b"x-assistant-message-id"
            )(
                {
                    "type": "http",
                    "path": "/ng-rollout/partial/v1/responses",
                    "raw_path": b"/ng-rollout/partial/v1/responses",
                    "headers": [(b"x-assistant-message-id", b"assistant-partial")],
                },
                receive,
                send,
            )
        )

    [exchange] = store.read("partial")
    assert exchange["client_assistant_message_id"] == "assistant-partial"
    [call] = read_model_call_records(store, "partial")
    assert call.client_assistant_message_id == "assistant-partial"
    assert call.status_code == 200
    assert call.error_category == "exception"
    assert call.response_raw == partial.decode()


def test_stream_error_events_are_terminal():
    from nemo_gym.base_responses_api_model import _consume_terminal_sse_event

    for dialect in ("responses", "messages"):
        assert _consume_terminal_sse_event(bytearray(b'event: error\ndata: {"error":"boom"}\n\n'), dialect) == "error"
    assert _consume_terminal_sse_event(bytearray(b"event: response.incomplete\n\n"), "responses") == "incomplete"
    assert _consume_terminal_sse_event(bytearray(b'event:error\ndata:{"error":"boom"}\n\n'), "chat") == "error"
    assert _consume_terminal_sse_event(bytearray(b'data: {"error":{"message":"boom"}}\n\n'), "chat") == "error"
    assert _consume_terminal_sse_event(bytearray(b"data:[DONE]\n\n"), "chat") == "complete"


def test_http_200_stream_error_is_not_recorded_as_success(tmp_path):
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/messages")
    async def _messages() -> StreamingResponse:
        return StreamingResponse(
            iter([b'event: error\ndata: {"type":"error","error":{"message":"boom"}}\n\n']),
            media_type="text/event-stream",
        )

    _install_capture(app, tmp_path)

    response = TestClient(app).post(
        "/ng-rollout/r-error/v1/messages",
        json={"messages": []},
        headers={"X-Assistant-Message-Id": "assistant-stream-error"},
    )

    assert response.status_code == 200
    [exchange] = CaptureStore(tmp_path).read("r-error")
    assert exchange["client_assistant_message_id"] == "assistant-stream-error"
    calls = read_model_call_records(CaptureStore(tmp_path), "r-error")
    assert calls[0].client_assistant_message_id == "assistant-stream-error"
    assert len(calls) == 1 and calls[0].error_category == "upstream_error"


def test_failed_call_is_captured_with_error_category(tmp_path):
    """A non-2xx model call is captured (replacing generic exception catching) with a
    normalized error_category + status_code on the ModelCallRecord."""
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> JSONResponse:
        return JSONResponse(content={"error": "boom"}, status_code=500)

    _install_capture(app, tmp_path)
    client = TestClient(app)

    r = client.post(
        "/ng-rollout/r-err/v1/responses",
        json={"input": "x"},
        headers={"X-Assistant-Message-Id": "assistant-http-error"},
    )
    assert r.status_code == 500  # response unchanged

    [exchange] = CaptureStore(tmp_path).read("r-err")
    assert exchange["client_assistant_message_id"] == "assistant-http-error"
    calls = read_model_call_records(CaptureStore(tmp_path), "r-err")
    assert calls[0].client_assistant_message_id == "assistant-http-error"
    assert len(calls) == 1
    assert calls[0].model_call_id
    assert calls[0].model_ref is not None and calls[0].model_ref.name == "srv"
    assert calls[0].started_at is not None
    assert calls[0].completed_at is not None
    assert calls[0].started_at <= calls[0].completed_at
    assert calls[0].error_category == "upstream_error"
    assert calls[0].status_code == 500


def test_raised_call_is_captured_then_reraised(tmp_path):
    """A model call that raises (not just a non-2xx) is captured with an exception category and the
    error is re-raised (response unchanged for the caller)."""
    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> dict:
        raise RuntimeError("kaboom")

    _install_capture(app, tmp_path)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post(
        "/ng-rollout/r-raise/v1/responses",
        json={"input": "x"},
        headers={"X-Assistant-Message-Id": "assistant-exception"},
    )
    assert r.status_code == 500  # error propagated, response unchanged

    [exchange] = CaptureStore(tmp_path).read("r-raise")
    assert exchange["client_assistant_message_id"] == "assistant-exception"
    calls = read_model_call_records(CaptureStore(tmp_path), "r-raise")
    assert calls[0].client_assistant_message_id == "assistant-exception"
    assert len(calls) == 1
    assert calls[0].model_call_id
    assert calls[0].model_ref is not None and calls[0].model_ref.name == "srv"
    assert calls[0].started_at is not None
    assert calls[0].completed_at is not None
    assert calls[0].started_at <= calls[0].completed_at
    assert calls[0].error_category == "exception" and calls[0].response is None
    assert calls[0].latency_ttft_ms is None  # nothing streamed before the raise


def test_cancelled_call_is_captured_then_reraised(tmp_path):
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    store = CaptureStore(tmp_path)
    seen_paths: list[str] = []

    async def run_cancelled_request() -> None:
        request_read = asyncio.Event()

        async def app(scope, receive, send) -> None:
            del send
            seen_paths.append(scope["path"])
            message = await receive()
            assert message["body"] == b'{"input":"x"}'
            request_read.set()
            await asyncio.sleep(3600)

        async def receive() -> dict:
            return {"type": "http.request", "body": b'{"input":"x"}', "more_body": False}

        async def send(_message: dict) -> None:
            pass

        task = asyncio.create_task(
            _CaptureMiddleware(
                app, store=store, model_server_name="srv", assistant_message_header=b"x-assistant-message-id"
            )(
                {
                    "type": "http",
                    "path": "/ng-rollout/r-cancel/v1/responses",
                    "raw_path": b"/ng-rollout/r-cancel/v1/responses",
                    "headers": [(b"x-assistant-message-id", b"assistant-cancel")],
                },
                receive,
                send,
            )
        )
        await request_read.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_cancelled_request())

    assert seen_paths == ["/v1/responses"]
    [exchange] = store.read("r-cancel")
    assert exchange["request"] == {"input": "x"}
    assert exchange["response"] is None
    assert exchange["status_code"] is None
    assert exchange["error_category"] == "cancelled"
    assert exchange["client_assistant_message_id"] == "assistant-cancel"

    [call] = read_model_call_records(store, "r-cancel")
    assert call.error_category == "cancelled"
    assert call.client_assistant_message_id == "assistant-cancel"
    assert call.response is None


def test_raised_upstream_error_preserves_status_and_body(tmp_path):
    class UpstreamError(RuntimeError):
        response = SimpleNamespace(status_code=403, content=b'{"error":{"message":"denied"}}')

    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> dict:
        raise UpstreamError

    _install_capture(app, tmp_path)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/ng-rollout/r-upstream/v1/responses",
        json={"input": "x"},
    )

    assert response.status_code == 500
    [exchange] = CaptureStore(tmp_path).read("r-upstream")
    assert exchange["status_code"] == 403
    assert exchange["error_category"] == "auth"
    assert json.loads(exchange["response_raw"]) == {"error": {"message": "denied"}}


@pytest.mark.parametrize("request_bytes", [b"{not-json", b"[]"])
def test_invalid_request_body_does_not_drop_capture(tmp_path, request_bytes):
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        return {"output": []}

    _install_capture(app, tmp_path)
    response = TestClient(app).post(
        "/ng-rollout/invalid-request/v1/responses",
        content=request_bytes,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    [exchange] = CaptureStore(tmp_path).read("invalid-request")
    assert exchange["request"] is None
    assert exchange["request_raw"] == request_bytes.decode()
    assert exchange["status_code"] == 422
    assert exchange["error_category"] == "client_error"
    assert exchange["response"] is not None

    [record] = read_model_call_records(CaptureStore(tmp_path), "invalid-request")
    assert record.request is None
    assert record.request_raw == request_bytes.decode()


def test_per_rollout_url_prefix_correlates_and_is_openai_compatible(tmp_path):
    """A caller attributes calls through the model base URL. The prefix is stripped before routing,
    while an ordinary unprefixed request remains unobserved."""
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        return {"output": [], "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}

    _install_capture(app, tmp_path)
    client = TestClient(app)

    # Prefixed base_url: routes to /v1/responses and correlates capture by the path id.
    r = client.post("/ng-rollout/task7-roll2/v1/responses", json={"input": "hi"})
    assert r.status_code == 200 and r.json()["usage"]["total_tokens"] == 4
    r_repeat = client.post("/ng-rollout/task7-roll2/v1/responses", json={"input": "again"})
    assert r_repeat.status_code == 200

    store = CaptureStore(tmp_path)
    exchanges = store.read("task7-roll2")
    assert len(exchanges) == 2
    assert exchanges[0]["model_ref"] == {"type": "responses_api_models", "name": "srv"}
    assert "model_server" not in exchanges[0]
    assert "request_raw" not in exchanges[0]

    calls = read_model_call_records(store, "task7-roll2")
    assert len(calls) == 2 and all(call.tokens_total == 4 for call in calls)
    assert len({call.model_call_id for call in calls}) == 2
    assert calls[0].model_call_id
    assert calls[0].model_ref is not None and calls[0].model_ref.name == "srv"
    assert calls[0].started_at is not None
    assert calls[0].completed_at is not None
    assert calls[0].started_at <= calls[0].completed_at

    # Plain /v1 URL is routed normally but is not captured without an explicit rollout prefix.
    r2 = client.post("/v1/responses", json={"input": "hi"})
    assert r2.status_code == 200
    assert read_model_call_records(CaptureStore(tmp_path), "rollout") == []


def test_per_rollout_prefix_strips_for_non_observed_paths_too(tmp_path):
    """A prefixed but non-observed path (e.g. /v1/models) is still stripped and routed normally,
    and is not captured (composes with arbitrary endpoints, not just the observed dialects)."""
    app = FastAPI()

    @app.get("/v1/models")
    async def _models() -> dict:
        return {"object": "list", "data": []}

    _install_capture(app, tmp_path)
    client = TestClient(app)

    r = client.get("/ng-rollout/abc/v1/models")
    assert r.status_code == 200 and r.json()["object"] == "list"
    assert CaptureStore(tmp_path).read("abc") == []  # non-observed path -> routed, not captured


def test_maybe_rollout_id_from_run_body_reads_canonical_indices():
    """The shared accessor agents use to derive the rollout id from a /run request body."""
    from pydantic import BaseModel, ConfigDict

    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

    mapping = MappingProxyType({TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1})
    assert maybe_rollout_id_from_run_body(mapping) == "3-1"
    assert maybe_rollout_id_from_run_body({TASK_INDEX_KEY_NAME: 3}) is None  # partial -> None
    assert maybe_rollout_id_from_run_body({}) is None
    assert maybe_rollout_id_from_run_body(None) is None

    # The shape agents actually receive: a run-request model with extra="allow".
    class _Body(BaseModel):
        model_config = ConfigDict(extra="allow")

    body = _Body.model_validate({TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 2})
    assert maybe_rollout_id_from_run_body(body) == "5-2"


# --- error classification ---
def test_classify_status_branches():
    from nemo_gym.base_responses_api_model import _classify_status

    assert _classify_status(200) is None
    assert _classify_status(408) == "timeout"
    assert _classify_status(504) == "timeout"
    assert _classify_status(429) == "rate_limit"
    assert _classify_status(401) == "auth"
    assert _classify_status(403) == "auth"
    assert _classify_status(404) == "not_found"
    assert _classify_status(422) == "client_error"
    assert _classify_status(500) == "upstream_error"


def test_classify_exception_branches():
    import asyncio

    from nemo_gym.base_responses_api_model import _classify_exception

    class _ReadTimeout(Exception):
        pass

    assert _classify_exception(asyncio.CancelledError()) == "cancelled"
    assert _classify_exception(asyncio.TimeoutError()) == "timeout"
    assert _classify_exception(_ReadTimeout()) == "timeout"  # name contains "timeout"
    assert _classify_exception(ConnectionError()) == "connection"  # name contains "conn"
    assert _classify_exception(ValueError("x")) == "exception"


def test_exception_http_details_tolerates_lazy_response_failure():
    from nemo_gym.base_responses_api_model import _exception_http_details

    class _Response:
        status_code = 503

        @property
        def content(self):
            raise RuntimeError("body unavailable")

        @property
        def text(self):
            raise RuntimeError("body unavailable")

    error = RuntimeError("upstream")
    error.response = _Response()
    assert _exception_http_details(error) == (503, b"")


def test_context_overflow_error_fields_reads_the_engine_body_and_the_marked_code():
    from nemo_gym.base_responses_api_model import (
        CONTEXT_LENGTH_EXCEEDED_ERROR_CODE,
        PROPAGATED_REFUSAL_CODE_ATTRIBUTE,
        context_overflow_error_fields,
        is_propagated_refusal,
    )

    overflow = (
        "This model's maximum context length is 4096 tokens. However, you requested 256 output tokens and "
        "your prompt contains 4000 input tokens, for a total of 4256 tokens. Please reduce the length of the "
        "input prompt or the number of requested output tokens."
    )

    # vLLM 0.10.1 and later nest the message, param and integer code inside an "error" object: the
    # message and param come back verbatim, and the code defaults to OpenAI's overflow code.
    error = RuntimeError("400, message='Bad Request'")
    error.response_content = json.dumps(
        {"error": {"message": overflow, "type": "BadRequestError", "param": "input_tokens", "code": 400}}
    ).encode()
    assert context_overflow_error_fields(error) == (overflow, CONTEXT_LENGTH_EXCEEDED_ERROR_CODE, "input_tokens")
    assert not is_propagated_refusal(error)

    # The code is whatever the server stored when it marked the error; a refusal of the request's own
    # max_tokens carries none, and the attribute's presence still marks the error.
    setattr(error, PROPAGATED_REFUSAL_CODE_ATTRIBUTE, None)
    assert is_propagated_refusal(error)
    error.response_content = json.dumps(
        {"error": {"message": "max_tokens must be at least 1, got 0.", "param": None, "code": 400}}
    ).encode()
    assert context_overflow_error_fields(error) == ("max_tokens must be at least 1, got 0.", None, None)

    # Older engines put the message at the top level; that shape is still read.
    flat = RuntimeError("400, message='Bad Request'")
    flat.response_content = json.dumps(
        {"object": "error", "message": "maximum context length is 4096 tokens", "code": 400}
    ).encode()
    assert context_overflow_error_fields(flat) == (
        "maximum context length is 4096 tokens",
        CONTEXT_LENGTH_EXCEEDED_ERROR_CODE,
        None,
    )

    # A non-JSON body, or JSON without a message string in either shape, falls back to the raw text
    # so no information is lost; a missing body yields an empty message rather than an exception.
    flat.response_content = b"upstream exploded in a non-JSON way"
    assert context_overflow_error_fields(flat)[0] == "upstream exploded in a non-JSON way"
    flat.response_content = b'{"error": "nested string"}'
    assert context_overflow_error_fields(flat)[0] == '{"error": "nested string"}'
    assert context_overflow_error_fields(RuntimeError("bare")) == ("", CONTEXT_LENGTH_EXCEEDED_ERROR_CODE, None)


# --- capture-store config + init failure ---
def test_model_call_capture_keys_are_reserved_global_config():
    assert {"observability_enabled", "model_call_capture_dir", "token_id_capture"} <= set(
        NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    )


def test_model_call_capture_config_requires_absolute_dir_when_enabled(tmp_path, monkeypatch):
    from nemo_gym.observability.model_calls import model_call_capture_dirs_from_config

    assert make_capture_store(ModelCallCaptureConfig()) is None
    with pytest.raises(ValueError, match="required"):
        ModelCallCaptureConfig(observability_enabled=True)
    with pytest.raises(ValueError, match="absolute"):
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir="relative")

    global_config = OmegaConf.create({"observability_enabled": True, "model_call_capture_dir": str(tmp_path)})
    config = ModelCallCaptureConfig.model_validate(global_config)
    store = make_capture_store(config)
    assert store is not None and store.root == tmp_path
    assert model_call_capture_dirs_from_config(global_config) == [store.root]

    monkeypatch.setenv("NEMO_GYM_MODEL_CALL_CAPTURE_DIR", str(tmp_path))
    assert model_call_capture_dirs_from_config({}) == []
    nested_config = {"policy_model": {"responses_api_models": {"model": {"observability_enabled": True}}}}
    assert model_call_capture_dirs_from_config(nested_config) == []


def test_make_capture_store_init_failure_returns_none(monkeypatch):
    import nemo_gym.base_responses_api_model as obs

    def _boom(_root):
        raise OSError("cannot create")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    config = ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir="/tmp/x")
    assert obs.make_capture_store(config) is None


def test_record_swallows_store_failure_and_marks_capture_incomplete(tmp_path, monkeypatch):
    from nemo_gym.base_responses_api_model import _record

    store = CaptureStore(tmp_path)
    monkeypatch.setattr(store, "record", MagicMock(side_effect=RuntimeError("disk full")))

    # Best-effort: a failing store must not raise out of _record.
    _record(
        store,
        "chat",
        "srv",
        b"{}",
        rollout_id="r",
        model_call_id="call-1",
        started_at=100.0,
        completed_at=100.01,
        response_body={},
        status_code=200,
        error_category=None,
        latency_ms=1.0,
    )
    assert store.is_incomplete("r")


def test_record_falls_back_to_raw_when_request_parser_raises(tmp_path, monkeypatch):
    import nemo_gym.base_responses_api_model as obs

    monkeypatch.setattr(obs.json, "loads", MagicMock(side_effect=RecursionError))
    store = CaptureStore(tmp_path)
    obs._record(
        store,
        "responses",
        "srv",
        b"deeply-nested-request",
        rollout_id="r",
        model_call_id="call-1",
        started_at=100.0,
        completed_at=100.01,
        response_body={},
        status_code=400,
        error_category="client_error",
        latency_ms=1.0,
    )

    [exchange] = store.read("r")
    assert exchange["request"] is None
    assert exchange["request_raw"] == "deeply-nested-request"


def test_capture_records_non_json_response_as_none(tmp_path):
    from fastapi.responses import PlainTextResponse

    app = FastAPI()

    @app.post("/v1/responses")
    async def _r(body: dict = Body()) -> PlainTextResponse:
        return PlainTextResponse("not json")

    _install_capture(app, tmp_path)
    client = TestClient(app)

    r = client.post("/ng-rollout/rnj/v1/responses", json={"input": "x"})
    assert r.status_code == 200 and r.text == "not json"  # response passed through unaltered
    records = CaptureStore(tmp_path).read("rnj")
    assert len(records) == 1 and records[0]["response"] is None  # non-JSON body -> None
    assert records[0]["response_raw"] == "not json"
    # a 2xx whose body we couldn't parse is flagged, not silently counted as a clean success
    assert records[0]["error_category"] == "capture_parse_error"
    [record] = read_model_call_records(CaptureStore(tmp_path), "rnj")
    assert record.response_raw == "not json"


# --- base-agent correlation helpers ---
def test_leading_rollout_prefix_round_trips_with_server_parser():
    from nemo_gym.base_responses_api_model import _ROLLOUT_PATH_RE

    assert apply_rollout_prefix("http://h:1", "r1") == "http://h:1/ng-rollout/r1"
    assert apply_rollout_prefix("http://h:1/", None) == "http://h:1/"
    assert rollout_path_prefix(None) == ""

    client_path = f"{rollout_path_prefix('task-7')}/v1/chat/completions"
    match = _ROLLOUT_PATH_RE.match(client_path)
    assert match is not None
    assert match.group("rollout_id") == "task-7"
    assert match.group("rest") == "/v1/chat/completions"


def test_base_agent_resolve_model_base_url(monkeypatch):
    import nemo_gym.base_responses_api_agent as base_agent

    monkeypatch.setattr(base_agent, "get_first_server_config_dict", lambda _config, _name: {"host": "h", "port": 1})
    agent = SimpleNamespace(
        resolved_model_base_url=None,
        server_client=SimpleNamespace(
            global_config_dict={},
            _build_server_base_url=lambda _config: "http://h:1",
        ),
        _token_id_capture_enabled=lambda: False,
    )

    assert SimpleResponsesAPIAgent.resolve_model_base_url(agent, "model", "rid") == "http://h:1/ng-rollout/rid/v1"
    assert SimpleResponsesAPIAgent.resolve_model_base_url(agent, "model", None) == "http://h:1/v1"
    agent._token_id_capture_enabled = lambda: True
    assert (
        SimpleResponsesAPIAgent.resolve_model_base_url(agent, "model", "rid")
        == "http://h:1/ng-rollout/rid/training-token-capture/v1"
    )


def _make_base_agent(global_config, *, token_id_capture=False):
    from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig

    class _Agent(SimpleResponsesAPIAgent):
        async def responses(self, body=...):
            raise NotImplementedError

        async def run(self, body=...):
            raise NotImplementedError

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = global_config
    config = BaseResponsesAPIAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="agent",
        token_id_capture=token_id_capture,
    )
    return _Agent(config=config, server_client=server_client)


def test_base_agent_url_path_for_run_gates_on_observability_and_indices():
    body = {TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1}

    enabled = _make_base_agent({"observability_enabled": True})
    assert enabled.rollout_id_from_run(body) == "3-1"
    assert enabled.url_path_for_run("/v1/responses", body) == "/ng-rollout/3-1/v1/responses"
    assert enabled.base_url_for_run("http://h:1", body) == "http://h:1/ng-rollout/3-1"
    assert enabled.url_path_for_run("/v1/responses", {}) == "/v1/responses"
    assert enabled.base_url_for_run("http://h:1", {}) == "http://h:1"

    disabled = _make_base_agent({"observability_enabled": False})
    assert disabled.rollout_id_from_run(body) is None
    assert disabled.url_path_for_run("/v1/responses", body) == "/v1/responses"
    assert disabled.base_url_for_run("http://h:1", body) == "http://h:1"

    assert _make_base_agent(MagicMock()).url_path_for_run("/v1/responses", body) == "/v1/responses"


def test_base_agent_propagates_explicit_token_capture_intent():
    body = {TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1}
    global_config = {
        "observability_enabled": True,
        "token_id_capture": {"enabled": True},
    }
    opted_in = _make_base_agent(global_config, token_id_capture=True)
    assert opted_in.url_path_for_run("/v1/responses", body) == "/ng-rollout/3-1/training-token-capture/v1/responses"
    assert opted_in.base_url_for_run("http://h:1", body) == "http://h:1/ng-rollout/3-1/training-token-capture"

    opted_out = _make_base_agent(global_config, token_id_capture=False)
    assert opted_out.url_path_for_run("/v1/responses", body) == "/ng-rollout/3-1/v1/responses"


def test_base_agent_all_agents_overrides_the_agent_opt_in():
    body = {TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1}
    global_config = {
        "token_id_capture": {
            "enabled": True,
            "all_agents": True,
        }
    }
    agent = _make_base_agent(global_config, token_id_capture=False)

    assert agent.url_path_for_run("/v1/responses", body) == ("/ng-rollout/3-1/training-token-capture/v1/responses")


def test_base_agent_url_path_for_request_propagates_inbound_prefix():
    agent = _make_base_agent({})

    prefixed = SimpleNamespace(path_params={"rollout_id": "7-0"})
    assert agent.url_path_for_request("/v1/responses", prefixed) == "/ng-rollout/7-0/v1/responses"
    assert agent.url_path_for_request("/v1/responses", SimpleNamespace(path_params={})) == "/v1/responses"
    assert agent.url_path_for_request("/v1/responses", SimpleNamespace()) == "/v1/responses"
    assert agent.url_path_for_request("/v1/responses", None) == "/v1/responses"

    capture_prefixed = SimpleNamespace(
        path_params={"rollout_id": "7-0"},
        url=SimpleNamespace(path="/ng-rollout/7-0/training-token-capture/v1/responses"),
    )
    assert (
        agent.url_path_for_request("/v1/responses", capture_prefixed)
        == "/ng-rollout/7-0/training-token-capture/v1/responses"
    )


def test_base_agent_registers_prefixed_self_call_route():
    from nemo_gym.server_utils import ROLLOUT_PATH_PREFIX

    routes = {route.path for route in _make_base_agent({}).setup_webserver().routes}
    assert f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses" in routes
    assert f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/training-token-capture/v1/responses" in routes
    assert "/v1/responses" in routes


def _sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/responses", _sse("response.in_progress", {"response": {}})),
        ("/v1/chat/completions", _sse("", {"choices": [{"delta": {}}]})),
        ("/v1/messages", _sse("message_start", {"type": "message_start", "message": {}})),
    ],
    ids=("responses", "chat", "messages"),
)
def test_stream_without_terminal_event_is_truncated(tmp_path, path, payload):
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    async def _stream() -> StreamingResponse:
        return StreamingResponse(iter([payload]), media_type="text/event-stream")

    app.post(path)(_stream)
    _install_capture(app, tmp_path)

    response = TestClient(app).post(f"/ng-rollout/truncated{path}", json={})

    assert response.status_code == 200
    assert response.content == payload
    [exchange] = CaptureStore(tmp_path).read("truncated")
    assert exchange["response"] is not None
    assert exchange["error_category"] == "stream_truncated"


def test_capture_reassembles_streamed_anthropic_sse(tmp_path):
    """Streamed (SSE) /v1/messages calls are forwarded unchanged AND reassembled into the final
    response, so token stats / tool calls / reasoning are captured like a non-streamed call."""
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/messages")
    async def _stream(body: dict = Body()) -> StreamingResponse:
        async def gen():
            yield _sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 5,
                            "cache_creation_input_tokens": 2,
                        },
                        "content": [],
                    },
                },
            )
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "let me think"},
                },
            )
            yield _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            )
            yield _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi there"}},
            )
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"x": 1}'},
                },
            )
            yield _sse(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
            )
            yield _sse("message_stop", {"type": "message_stop"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    _install_capture(app, tmp_path)
    client = TestClient(app)

    r = client.post("/ng-rollout/3-0/v1/messages", json={"stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")  # stream preserved
    assert "event:" in r.text and "data:" in r.text  # SSE content flowed through

    records = CaptureStore(tmp_path).read("3-0")
    assert len(records) == 1 and records[0]["response"] is not None  # reassembled, not dropped
    assert records[0]["response_raw"] == r.text  # lossless SSE evidence is retained alongside the projection

    calls = read_model_call_records(CaptureStore(tmp_path), "3-0")
    assert len(calls) == 1
    call = calls[0]
    assert call.model_call_id
    assert call.response_id == "msg_1"
    assert call.model_ref is not None and call.model_ref.name == "srv"
    assert call.started_at is not None and call.completed_at is not None
    assert call.started_at <= call.completed_at
    assert call.dialect == "messages"
    assert call.tokens_in == 17 and call.tokens_out == 7  # 10 + cache_read 5 + cache_creation 2; output from delta
    assert call.cached_tokens == 5 and call.cache_creation_tokens == 2
    assert call.reasoning_content == "let me think"
    assert call.tool_calls == [{"call_id": "t1", "name": "calc", "arguments": {"x": 1}}]
    assert call.latency_ttft_ms is not None
    assert call.error_category is None
    assert call.response_raw == r.text


def test_reconstruct_chat_sse():
    from nemo_gym.base_responses_api_model import _reconstruct_streamed_response

    chunks = [
        {
            "id": "chatcmpl-1",
            "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
        },
        {"choices": [{"index": 0, "delta": {"content": "lo", "reasoning": "hmm"}}]},  # vLLM `reasoning` alias
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": '{"a":'}}]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
    ]
    raw = (b"".join(_sse("", c) for c in chunks) + b"data: [DONE]\n\n").replace(b"\n", b"\r\n")
    resp = _reconstruct_streamed_response(raw, "chat")
    msg = resp["choices"][0]["message"]
    assert resp["id"] == "chatcmpl-1"
    assert msg["content"] == "Hello" and msg["reasoning_content"] == "hmm"
    assert msg["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a":1}'}
    assert resp["usage"]["total_tokens"] == 8


def test_reconstruct_responses_sse_uses_terminal_envelope():
    from nemo_gym.base_responses_api_model import _reconstruct_streamed_response

    raw = b"".join(
        _sse(c["type"], c)
        for c in [
            {"type": "response.created", "response": {"id": "r", "output": []}},
            {
                "type": "response.completed",
                "response": {
                    "id": "r",
                    "output": [{"type": "message"}],
                    "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                },
            },
        ]
    )
    resp = _reconstruct_streamed_response(raw, "responses")
    assert resp["output"] == [{"type": "message"}] and resp["usage"]["total_tokens"] == 6

    # Fallback: no terminal envelope, but an interim event still carries a response object.
    interim = _sse("response.in_progress", {"type": "response.in_progress", "response": {"id": "r2", "output": []}})
    assert _reconstruct_streamed_response(interim, "responses")["id"] == "r2"


def test_reconstruct_streamed_response_best_effort_none():
    from nemo_gym.base_responses_api_model import _reconstruct_streamed_response

    assert _reconstruct_streamed_response(b"", "chat") is None  # no events
    assert _reconstruct_streamed_response(b"event: ping\ndata: not-json\n\n", "messages") is None  # unparseable
    assert _reconstruct_streamed_response(b"data: 123\n\n", "chat") is None  # non-dict JSON skipped
    # Non-empty events that carry nothing reconstructable -> None for each dialect.
    ping = _sse("ping", {"type": "ping"})
    assert _reconstruct_streamed_response(ping, "messages") is None
    assert _reconstruct_streamed_response(ping, "chat") is None
    assert _reconstruct_streamed_response(ping, "responses") is None


def test_maybe_rollout_id_from_run_body_attempt_suffix():
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

    base = {"_ng_task_index": 3, "_ng_rollout_index": 2}
    assert maybe_rollout_id_from_run_body(base) == "3-2"  # no attempt -> bare key
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 0}) == "3-2"  # first attempt -> bare
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 1}) == "3-2-a1"
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "2"}) == "3-2-a2"  # coerced
    assert maybe_rollout_id_from_run_body({"_ng_rollout_index": 2}) is None  # missing task -> None
    with pytest.raises(ValueError):
        maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "invalid"})


def test_maybe_rollout_id_from_run_body_prefers_an_explicit_id():
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

    base = {"_ng_task_index": 3, "_ng_rollout_index": 2}
    # Restarted index numbering derives the same id twice.
    # An explicit id keeps the dispatches separate.
    assert maybe_rollout_id_from_run_body({**base, "_ng_rollout_id": "s7-3-2"}) == "s7-3-2"
    # A retry of an explicitly keyed rollout still gets a distinct key.
    # Otherwise retry calls would append to the first attempt.
    assert maybe_rollout_id_from_run_body({"_ng_rollout_id": "s7-3-2", "_ng_attempt_index": 1}) == "s7-3-2-a1"
    # The explicit id requires no indices.
    assert maybe_rollout_id_from_run_body({"_ng_rollout_id": "abc"}) == "abc"


@pytest.mark.parametrize("bad", [".hidden", "has/slash", "has space", "", 7, None])
def test_maybe_rollout_id_from_run_body_refuses_an_unusable_explicit_id(bad):
    from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body

    body = {"_ng_task_index": 3, "_ng_rollout_index": 2, "_ng_rollout_id": bad}
    if bad is None:
        # Absent and null both mean "no explicit id", so the derivation still runs.
        assert maybe_rollout_id_from_run_body(body) == "3-2"
        return
    # Reject ids that cannot survive the path round trip.
    # Sanitizing would create a key the caller cannot look up.
    with pytest.raises(ValueError):
        maybe_rollout_id_from_run_body(body)


def test_explicit_rollout_ids_round_trip_through_the_path_prefix():
    from nemo_gym.rollout_correlation import RolloutContextMiddleware, maybe_rollout_id_from_run_body

    # The id becomes a path segment.
    # Middleware must return every accepted id unchanged.
    for candidate in ["s7-3-2", "step7.task3", "a", "A_b-1.2"]:
        rollout_id = maybe_rollout_id_from_run_body({"_ng_rollout_id": candidate})
        match = RolloutContextMiddleware._PREFIX.match(f"/ng-rollout/{rollout_id}/v1/responses")
        assert match is not None and match.group("rollout_id") == candidate
        assert match.group("rest") == "/v1/responses"


def test_rollout_prefix_stripped_when_capture_disabled():
    # The /ng-rollout/<id> prefix must be stripped + routed even when capture is OFF (the default),
    # otherwise a default `gym eval` 404s on every prefixed model call.
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _cc() -> dict:
        return {"ok": True}

    install_model_call_capture(app, ModelCallCaptureConfig())
    client = TestClient(app)
    assert client.post("/v1/chat/completions", json={}).status_code == 200
    assert client.post("/ng-rollout/3-0/v1/chat/completions", json={}).status_code == 200


def test_capture_store_concurrent_append_no_loss(tmp_path):
    import threading

    store = CaptureStore(tmp_path)

    def _write(i: int) -> None:
        store.record("0-0", {"dialect": "chat", "request": {"i": i}, "response": {}})

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = store.read("0-0")
    assert len(rows) == 20  # flock prevents lost or corrupted appends
    assert sorted(r["request"]["i"] for r in rows) == list(range(20))


def test_capture_store_does_not_serialize_independent_rollouts(tmp_path, monkeypatch):
    import os
    import threading

    store = CaptureStore(tmp_path)
    first_fsync = threading.Event()
    release_first = threading.Event()
    original_fsync = os.fsync

    def _fsync(fd):
        if threading.current_thread().name == "first-rollout":
            first_fsync.set()
            assert release_first.wait(timeout=5)
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)
    first = threading.Thread(name="first-rollout", target=store.record, args=("0-0", {"request": {}, "response": {}}))
    second = threading.Thread(target=store.record, args=("1-0", {"request": {}, "response": {}}))
    first.start()
    assert first_fsync.wait(timeout=5)
    second.start()
    second.join(timeout=1)
    try:
        assert not second.is_alive()
        assert store.read("1-0") == [{"request": {}, "response": {}}]
    finally:
        release_first.set()
        first.join(timeout=5)


def test_capture_store_read_waits_for_in_progress_append(tmp_path):
    import fcntl
    import threading

    store = CaptureStore(tmp_path)
    path = store.path_for("0-0")
    writer_ready = threading.Event()
    finish_write = threading.Event()
    reader_done = threading.Event()
    rows = []

    def _write() -> None:
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(b'{"request":')
                handle.flush()
                writer_ready.set()
                assert finish_write.wait(timeout=5)
                handle.write(b'{},"response":{}}\n')
                handle.flush()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _read() -> None:
        rows.extend(store.read("0-0"))
        reader_done.set()

    writer = threading.Thread(target=_write)
    reader = threading.Thread(target=_read)
    writer.start()
    assert writer_ready.wait(timeout=5)
    reader.start()
    try:
        assert not reader_done.wait(timeout=0.1)
    finally:
        finish_write.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert rows == [{"request": {}, "response": {}}]


def _cross_process_writer(root: str, base: int) -> None:
    # Module-level so it is picklable under the "spawn" start method too.
    store = CaptureStore(root)
    for i in range(base, base + 100):
        store.record("0-0", {"dialect": "chat", "request": {"i": i}, "response": {}})


def test_capture_store_cross_process_append_no_loss(tmp_path):
    # The threads-only test above exercises the in-process lock; this exercises fcntl.flock across
    # *processes* -- the num_workers>1 case the in-process lock cannot coordinate.
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_cross_process_writer, args=(str(tmp_path), b * 100)) for b in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    rows = CaptureStore(tmp_path).read("0-0")
    assert len(rows) == 400
    assert sorted(r["request"]["i"] for r in rows) == list(range(400))


def _run_capture_middleware_on(path: str, *, token_store, sent: list | None = None, forwarded: list | None = None):
    """Drive _CaptureMiddleware over one request to ``path`` with a token store."""
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware

    forwarded = forwarded if forwarded is not None else []
    sent = sent if sent is not None else []

    async def app(scope, receive, send):
        forwarded.append(scope["path"])
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    async def receive():
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(
        _CaptureMiddleware(
            app,
            store=None,
            model_server_name="srv",
            token_store=token_store,
            token_capture_enabled=True,
        )({"type": "http", "path": path, "raw_path": path.encode(), "headers": []}, receive, send)
    )
    return forwarded, sent


def test_unobserved_dialect_under_capture_prefix_marks_incomplete(tmp_path):
    # The middleware cannot capture tokens from /v1/completions.
    # Its output can still feed later prompts.
    from nemo_gym.token_id_capture import TokenCaptureStore

    token_store = TokenCaptureStore(tmp_path)
    forwarded, sent = _run_capture_middleware_on(
        "/ng-rollout/hole-0/training-token-capture/v1/completions", token_store=token_store
    )

    assert forwarded == ["/v1/completions"]
    assert sent[0]["status"] == 200
    assert token_store.is_incomplete("hole-0")


def test_unobserved_dialect_marking_failure_still_forwards(tmp_path):
    class _BrokenSink:
        async def mark_incomplete(self, rollout_id, model_call_id=""):
            raise RuntimeError("sink down")

    forwarded, sent = _run_capture_middleware_on(
        "/ng-rollout/hole-1/training-token-capture/v1/completions", token_store=_BrokenSink()
    )

    assert forwarded == ["/v1/completions"]
    assert sent[0]["status"] == 200


def test_observed_dialect_under_capture_prefix_is_not_marked_incomplete(tmp_path):
    from nemo_gym.token_id_capture import TokenCaptureStore

    token_store = TokenCaptureStore(tmp_path)
    forwarded, _sent = _run_capture_middleware_on(
        "/ng-rollout/hole-2/training-token-capture/v1/chat/completions", token_store=token_store
    )

    assert forwarded == ["/v1/chat/completions"]
    assert not token_store.is_incomplete("hole-2")


@pytest.mark.parametrize("header", [b"X-Custom-Reply-Id", None])
def test_capture_uses_supplied_assistant_header_or_none(tmp_path, header):
    app = FastAPI()

    @app.post("/v1/responses")
    async def respond():
        return {"output": []}

    config = ModelCallCaptureConfig(
        observability_enabled=True,
        model_call_capture_dir=tmp_path,
    )
    install_model_call_capture(app, config, model_server_name="policy", assistant_message_header=header)
    with TestClient(app) as client:
        response = client.post(
            "/ng-rollout/configured-header/v1/responses",
            json={"input": "hello"},
            headers={"x-custom-reply-id": "persisted-reply", "x-assistant-message-id": "other-reply"},
        )
        assert response.status_code == 200
        response = client.post(
            "/ng-rollout/no-configured-header/v1/responses",
            json={"input": "hello"},
            headers={"x-assistant-message-id": "other-reply"},
        )
        assert response.status_code == 200
    [captured] = read_model_call_records(CaptureStore(tmp_path), "configured-header")
    assert captured.client_assistant_message_id == ("persisted-reply" if header else None)
    [absent] = read_model_call_records(CaptureStore(tmp_path), "no-configured-header")
    assert absent.client_assistant_message_id is None


def test_capture_store_pending_claims_are_scoped_to_one_rollout(tmp_path):
    store = CaptureStore(tmp_path)

    store.mark_pending("1-0-retry1", "call-1")

    # The dot after the rollout id keeps a prefix from matching a longer id.
    assert store.has_pending("1-0-retry1") is True
    assert store.has_pending("1-0") is False

    store.clear_pending("1-0-retry1", "call-1")
    assert store.has_pending("1-0-retry1") is False


def test_capture_store_pending_token_survives_an_unsafe_model_call_id(tmp_path):
    store = CaptureStore(tmp_path)

    store.mark_pending("1-0", "a/b c.d")

    assert store.has_pending("1-0") is True
    assert store.pending_path_for("1-0", "a/b c.d").name == "1-0.a_b_c_d.capture.pending"


def test_capture_claim_is_visible_while_the_call_is_still_in_flight(tmp_path):
    """The claim must exist for the call's whole lifetime, not just its write.

    A claim taken at write time does not exist yet while the call is in flight, so a reader can
    poll, see nothing outstanding, and snapshot the rollout before the row lands.
    """
    import asyncio

    from nemo_gym.base_responses_api_model import _CaptureMiddleware
    from nemo_gym.observability.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)

    async def run() -> None:
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def app(scope, receive, send) -> None:
            del scope
            await receive()
            in_flight.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"id":"resp-1"}', "more_body": False})

        async def receive() -> dict:
            return {"type": "http.request", "body": b'{"input":"x"}', "more_body": False}

        async def send(_message: dict) -> None:
            pass

        task = asyncio.create_task(
            _CaptureMiddleware(
                app, store=store, model_server_name="srv", assistant_message_header=b"x-assistant-message-id"
            )(
                {
                    "type": "http",
                    "path": "/ng-rollout/r-inflight/v1/responses",
                    "raw_path": b"/ng-rollout/r-inflight/v1/responses",
                    "headers": [(b"x-assistant-message-id", b"assistant-inflight")],
                },
                receive,
                send,
            )
        )
        await in_flight.wait()
        # The row is not on disk yet, so only the claim can tell a reader to wait.
        assert store.read("r-inflight") == []
        assert store.has_pending("r-inflight") is True
        assert await await_model_call_capture_quiescence("r-inflight", [tmp_path], timeout_s=0.05) is False

        release.set()
        await task
        assert await await_model_call_capture_quiescence("r-inflight", [tmp_path], timeout_s=5.0) is True
        assert store.has_pending("r-inflight") is False
        assert len(store.read("r-inflight")) == 1

    asyncio.run(run())
