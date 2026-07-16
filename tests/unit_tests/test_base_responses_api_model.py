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
import json
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import orjson
import pytest
from fastapi import Body, FastAPI
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModel,
    BaseResponsesAPIModelConfig,
    CaptureStore,
    ModelCallCaptureConfig,
    SimpleResponsesAPIModel,
    build_model_call_record,
    install_model_call_capture,
    make_capture_store,
    read_model_call_records,
)
from nemo_gym.global_config import NEMO_GYM_RESERVED_TOP_LEVEL_KEYS, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
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


def test_build_model_call_record_from_exchange():
    exchange = {
        "model_call_id": "call-1",
        "dialect": "responses",
        "model_ref": {"type": "responses_api_models", "name": "srv"},
        "started_at": 100.0,
        "completed_at": 100.02,
        "latency_ms": 18.4,
        "request": {"input": "hi"},
        "response": {
            "model": "m",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "output": [
                {"type": "reasoning", "summary": [{"text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "function_call", "call_id": "c1", "name": "calc", "arguments": '{"x": 1}'},
            ],
        },
    }
    rec = build_model_call_record(exchange, call_index=3)
    assert rec.model_call_id == "call-1"
    assert rec.call_index == 3
    assert rec.model_ref is not None and rec.model_ref.name == "srv"
    assert rec.dialect == "responses"
    assert rec.started_at == 100.0 and rec.completed_at == 100.02
    assert (rec.tokens_in, rec.tokens_out, rec.tokens_total, rec.tokens_reasoning) == (10, 5, 15, 3)
    assert rec.cache_hit is True and rec.cached_tokens == 4
    assert rec.reasoning_content == "thinking..."
    assert rec.tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert rec.latency_total_ms == 18.4
    assert {
        "model_call_id",
        "call_index",
        "model_ref",
        "dialect",
        "status_code",
        "started_at",
        "completed_at",
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "tokens_total",
        "request",
        "response",
        "tool_calls",
        "reasoning_content",
        "cache_hit",
        "cached_tokens",
        "cache_creation_tokens",
        "error_category",
        "latency_total_ms",
        "latency_ttft_ms",
    } <= type(rec).model_json_schema()["properties"].keys()


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
            durable_call_counts.append(len(store.read("fast-rollout")))

    asyncio.run(
        _CaptureMiddleware(app, store=store, model_server_name="srv")(
            {
                "type": "http",
                "path": "/ng-rollout/fast-rollout/v1/messages",
                "raw_path": b"/ng-rollout/fast-rollout/v1/messages",
                "headers": [],
            },
            receive,
            send,
        )
    )

    assert durable_call_counts == [0, 1, 1]


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

    response = TestClient(app).post("/ng-rollout/r-error/v1/messages", json={"messages": []})

    assert response.status_code == 200
    calls = read_model_call_records(CaptureStore(tmp_path), "r-error")
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

    r = client.post("/ng-rollout/r-err/v1/responses", json={"input": "x"})
    assert r.status_code == 500  # response unchanged

    calls = read_model_call_records(CaptureStore(tmp_path), "r-err")
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

    r = client.post("/ng-rollout/r-raise/v1/responses", json={"input": "x"})
    assert r.status_code == 500  # error propagated, response unchanged

    calls = read_model_call_records(CaptureStore(tmp_path), "r-raise")
    assert len(calls) == 1
    assert calls[0].model_call_id
    assert calls[0].model_ref is not None and calls[0].model_ref.name == "srv"
    assert calls[0].started_at is not None
    assert calls[0].completed_at is not None
    assert calls[0].started_at <= calls[0].completed_at
    assert calls[0].error_category == "exception" and calls[0].response is None
    assert calls[0].latency_ttft_ms is None  # nothing streamed before the raise


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
    assert "request_raw" not in record.model_dump()


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

    from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body
    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME

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

    assert _classify_exception(asyncio.TimeoutError()) == "timeout"
    assert _classify_exception(_ReadTimeout()) == "timeout"  # name contains "timeout"
    assert _classify_exception(ConnectionError()) == "connection"  # name contains "conn"
    assert _classify_exception(ValueError("x")) == "exception"


# --- capture-store config + init failure ---
def test_model_call_capture_keys_are_reserved_global_config():
    assert {"observability_enabled", "model_call_capture_dir"} <= set(NEMO_GYM_RESERVED_TOP_LEVEL_KEYS)


def test_model_call_capture_config_requires_absolute_dir_when_enabled(tmp_path, monkeypatch):
    from nemo_gym.base_responses_api_model import model_call_capture_dirs_from_config

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


def test_record_swallows_store_failure():
    from nemo_gym.base_responses_api_model import _record

    class _BadStore:
        def record(self, *args, **kwargs):
            raise RuntimeError("disk full")

    # Best-effort: a failing store must not raise out of _record.
    _record(
        _BadStore(),
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
    # a 2xx whose body we couldn't parse is flagged, not silently counted as a clean success
    assert records[0]["error_category"] == "capture_parse_error"


def test_as_arguments():
    from nemo_gym.base_responses_api_model import _as_arguments

    assert _as_arguments({"x": 1}) == {"x": 1}
    assert _as_arguments('{"y": 2}') == {"y": 2}
    assert _as_arguments("not json") == {"_raw": "not json"}
    assert _as_arguments(123) == {}


def test_cache_signal():
    from nemo_gym.base_responses_api_model import _cache_signal

    assert _cache_signal(None) == (None, None)
    assert _cache_signal({"prompt_tokens_details": {"cached_tokens": 4}}) == (True, 4)
    assert _cache_signal({"input_tokens_details": {"cached_tokens": 0}}) == (False, 0)
    assert _cache_signal({"cache_read_input_tokens": 5}) == (True, 5)  # Anthropic
    assert _cache_signal({"unrelated": 1}) == (None, None)


def test_extract_token_stats_chat_fallback():
    from nemo_gym.base_responses_api_model import extract_token_stats

    assert extract_token_stats(None)["tokens_total"] is None
    stats = extract_token_stats(
        {"prompt_tokens": 5, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}
    )
    assert (stats["tokens_in"], stats["tokens_out"], stats["tokens_total"], stats["tokens_reasoning"]) == (5, 3, 8, 1)


def test_tool_calls_and_reasoning_chat_and_anthropic():
    from nemo_gym.base_responses_api_model import _tool_calls_and_reasoning

    chat = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}],
                    "reasoning_content": "think",
                }
            }
        ]
    }
    assert _tool_calls_and_reasoning(chat) == ([{"call_id": "c1", "name": "f", "arguments": {"a": 1}}], "think")

    anthropic = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "g", "input": {"b": 2}},
            {"type": "thinking", "thinking": "hmm"},
        ]
    }
    assert _tool_calls_and_reasoning(anthropic) == ([{"call_id": "t1", "name": "g", "arguments": {"b": 2}}], "hmm")
    assert _tool_calls_and_reasoning({"unrelated": 1}) == ([], None)


def test_tool_calls_and_reasoning_accepts_reasoning_alias():
    from nemo_gym.base_responses_api_model import _tool_calls_and_reasoning

    # vLLM / newer servers emit `reasoning` (no `reasoning_content`).
    assert _tool_calls_and_reasoning({"choices": [{"message": {"content": "hi", "reasoning": "because"}}]}) == (
        [],
        "because",
    )
    # `reasoning_content` wins when both are present.
    both = {"choices": [{"message": {"reasoning_content": "rc", "reasoning": "r"}}]}
    assert _tool_calls_and_reasoning(both) == ([], "rc")


def test_tool_calls_and_reasoning_skips_non_dict_output_items():
    from nemo_gym.base_responses_api_model import _tool_calls_and_reasoning

    # A Responses `output` list may contain non-dict junk; it must be skipped, not crash.
    resp = {"output": [None, "junk", {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]}
    assert _tool_calls_and_reasoning(resp) == ([{"call_id": "c", "name": "f", "arguments": {}}], None)
    # Same guard on the Anthropic content blocks.
    anth = {"content": [None, "junk", {"type": "tool_use", "id": "t", "name": "g", "input": {"b": 2}}]}
    assert _tool_calls_and_reasoning(anth) == ([{"call_id": "t", "name": "g", "arguments": {"b": 2}}], None)


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
        server_client=SimpleNamespace(
            global_config_dict={},
            _build_server_base_url=lambda _config: "http://h:1",
        )
    )

    assert SimpleResponsesAPIAgent.resolve_model_base_url(agent, "model", "rid") == "http://h:1/ng-rollout/rid/v1"
    assert SimpleResponsesAPIAgent.resolve_model_base_url(agent, "model", None) == "http://h:1/v1"


def _make_base_agent(global_config):
    from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig

    class _Agent(SimpleResponsesAPIAgent):
        async def responses(self, body=...):
            raise NotImplementedError

        async def run(self, body=...):
            raise NotImplementedError

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = global_config
    config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent")
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


def test_base_agent_url_path_for_request_propagates_inbound_prefix():
    agent = _make_base_agent({})

    prefixed = SimpleNamespace(path_params={"rollout_id": "7-0"})
    assert agent.url_path_for_request("/v1/responses", prefixed) == "/ng-rollout/7-0/v1/responses"
    assert agent.url_path_for_request("/v1/responses", SimpleNamespace(path_params={})) == "/v1/responses"
    assert agent.url_path_for_request("/v1/responses", SimpleNamespace()) == "/v1/responses"
    assert agent.url_path_for_request("/v1/responses", None) == "/v1/responses"


def test_base_agent_registers_prefixed_self_call_route():
    from nemo_gym.server_utils import ROLLOUT_PATH_PREFIX

    routes = {route.path for route in _make_base_agent({}).setup_webserver().routes}
    assert f"/{ROLLOUT_PATH_PREFIX}/{{rollout_id}}/v1/responses" in routes
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

    calls = read_model_call_records(CaptureStore(tmp_path), "3-0")
    assert len(calls) == 1
    call = calls[0]
    assert call.model_call_id
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


def test_reconstruct_chat_sse():
    from nemo_gym.base_responses_api_model import _reconstruct_streamed_response

    chunks = [
        {"model": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}]},
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
    from nemo_gym.base_responses_api_model import maybe_rollout_id_from_run_body

    base = {"_ng_task_index": 3, "_ng_rollout_index": 2}
    assert maybe_rollout_id_from_run_body(base) == "3-2"  # no attempt -> bare key
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 0}) == "3-2"  # first attempt -> bare
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": 1}) == "3-2-a1"
    assert maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "2"}) == "3-2-a2"  # coerced
    assert maybe_rollout_id_from_run_body({"_ng_rollout_index": 2}) is None  # missing task -> None
    with pytest.raises(ValueError):
        maybe_rollout_id_from_run_body({**base, "_ng_attempt_index": "invalid"})


def _capture_exchange(dialect, model_server, usage, response):
    return {
        "model_call_id": f"call-{model_server}",
        "dialect": dialect,
        "model_ref": {"type": "responses_api_models", "name": model_server},
        "started_at": 100.0,
        "completed_at": 100.01,
        "latency_ms": 1.0,
        "status_code": 200,
        "error_category": None,
        "request": {"input": "hi"},
        "response": {"model": "m", "usage": usage, **response},
    }


def test_merge_capture_attaches_metrics_without_raw_payloads(tmp_path):
    from nemo_gym.base_responses_api_model import CaptureStore, merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record(
        "0-0",
        _capture_exchange(
            "responses",
            "A",
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]},
        ),
    )

    record = {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "response": {"harness": "A"}}
    merge_model_call_capture_into_record(record, [tmp_path])

    capture = record["ng_model_call_capture"]
    assert set(capture) == {"rollout_id", "metrics", "calls"}
    assert capture["rollout_id"] == "0-0"
    assert capture["metrics"]["num_calls"] == 1
    attached_call = capture["calls"][0]
    assert attached_call["model_call_id"] == "call-A"
    assert attached_call["model_ref"] == {"type": "responses_api_models", "name": "A"}
    assert attached_call["started_at"] == 100.0 and attached_call["completed_at"] == 100.01
    assert attached_call["tokens_in"] == 3
    assert "request" not in attached_call and "response" not in attached_call
    assert record["response"] == {"harness": "A"} and record["reward"] == 1.0


def test_merge_capture_noop_without_capture(tmp_path):
    from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record

    rec = {"_ng_task_index": 9, "_ng_rollout_index": 9, "reward": 1.0}
    merge_model_call_capture_into_record(rec, [tmp_path])  # no capture file for 9-9
    assert "ng_model_call_capture" not in rec
    merge_model_call_capture_into_record(rec, [])  # no dirs
    assert "ng_model_call_capture" not in rec


def test_merge_capture_surfaces_malformed_data_only_when_active(tmp_path):
    from nemo_gym.base_responses_api_model import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.path_for("9-9").write_bytes(b"{not-json}\n")
    record = {"_ng_task_index": 9, "_ng_rollout_index": 9}

    merge_model_call_capture_into_record(record, [])
    with pytest.raises(orjson.JSONDecodeError):
        merge_model_call_capture_into_record(record, [tmp_path])


def test_clear_model_call_captures_for_rollouts_run_scoping(tmp_path, monkeypatch):
    import nemo_gym.base_responses_api_model as obs
    from nemo_gym.base_responses_api_model import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.record("0-0", {"dialect": "chat", "request": {}, "response": {}})
    store.record("1-0", {"dialect": "chat", "request": {}, "response": {}})
    assert store.read("0-0") and store.read("1-0")

    # Clears only the rollout ids about to be (re)run; rows without indices are skipped, others stay.
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"no": "id"}], [tmp_path])
    assert store.read("0-0") == [] and store.read("1-0")
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [])  # no dirs -> no-op
    assert store.read("1-0")

    # A stale-capture cleanup failure must be visible rather than mixing old and new calls.
    def _boom(_directory):
        raise OSError("cannot open")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    with pytest.raises(OSError, match="cannot open"):
        clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [tmp_path])


def test_aggregate_model_call_records_sums_and_counts():
    from nemo_gym.base_responses_api_model import ModelCallRecord, aggregate_model_call_records

    calls = [
        ModelCallRecord(call_index=0, tokens_in=10, tokens_out=5, tokens_total=15, latency_total_ms=2.0),
        ModelCallRecord(call_index=1, tokens_in=20, tokens_out=3, tokens_total=23, latency_total_ms=1.0),
    ]
    agg = aggregate_model_call_records(calls)
    assert (agg["tokens_in"], agg["tokens_out"], agg["tokens_total"]) == (30, 8, 38)
    assert agg["latency_total_ms"] == 3.0 and agg["num_calls"] == 2
    # empty -> all-None totals but a well-formed shape (num_calls 0)
    assert aggregate_model_call_records([]) == {
        "tokens_in": None,
        "tokens_out": None,
        "tokens_reasoning": None,
        "tokens_total": None,
        "latency_total_ms": None,
        "num_calls": 0,
    }


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


def test_extract_token_stats_anthropic_cache_fold():
    from nemo_gym.base_responses_api_model import extract_token_stats

    # Anthropic native: input_tokens is the uncached remainder; cache-read + cache-creation fold in.
    stats = extract_token_stats(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 30,
        }
    )
    assert stats["tokens_in"] == 330  # 100 + 200 + 30
    assert stats["tokens_out"] == 50
    assert stats["tokens_total"] == 380  # 330 + 50 (Anthropic sends no total_tokens)
    assert stats["cache_creation_tokens"] == 30


def test_extract_token_stats_anthropic_fully_cached_zero_base():
    from nemo_gym.base_responses_api_model import extract_token_stats

    # A fully-cached Anthropic response omits input_tokens; a 0 base preserves the folded prompt
    # size instead of leaving tokens_in null.
    stats = extract_token_stats(
        {
            "output_tokens": 12,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        }
    )
    assert stats["tokens_in"] == 500  # 0 base + cache_read 500 + cache_creation 0
    assert stats["tokens_out"] == 12
    assert stats["cache_creation_tokens"] == 0


def test_extract_token_stats_openai_cached_not_double_counted():
    from nemo_gym.base_responses_api_model import extract_token_stats

    # OpenAI: input_tokens already includes cached tokens (cached_tokens is a subset) -> no fold.
    stats = extract_token_stats(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
    )
    assert stats["tokens_in"] == 10
    assert stats["tokens_total"] == 15
    assert stats["cache_creation_tokens"] is None


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
    assert len(rows) == 20  # flock + in-process lock: no lost or corrupted appends
    assert sorted(r["request"]["i"] for r in rows) == list(range(20))


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
