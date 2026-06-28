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
from types import SimpleNamespace

from fastapi import Body, FastAPI
from fastapi.testclient import TestClient

from nemo_gym.observability import install_trajectory_capture, make_capture_store
from nemo_gym.trajectory_capture import (
    CaptureStore,
    StepRecord,
    aggregate_rollout_metrics,
    assemble_rollout,
    assemble_step_records,
    build_step_record,
)


def test_make_capture_store_disabled_returns_none():
    assert make_capture_store(SimpleNamespace(observability_enabled=False)) is None


# --- StepRecord contract ---
def test_build_step_record_from_exchange():
    exchange = {
        "dialect": "responses",
        "model_server": "srv",
        "trial_index": 2,
        "turn_index": 1,
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
    rec = build_step_record(exchange, step_index=3, run_id="run-1")
    assert (rec.step_index, rec.trial_index, rec.turn_index) == (3, 2, 1)
    assert rec.run_id == "run-1" and rec.model_server == "srv" and rec.dialect == "responses"
    assert (rec.tokens_in, rec.tokens_out, rec.tokens_total, rec.tokens_reasoning) == (10, 5, 15, 3)
    assert rec.cache_hit is True and rec.cached_tokens == 4
    assert rec.reasoning_content == "thinking..."
    assert rec.tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert rec.latency_total_ms == 18.4


def test_step_record_json_schema_has_contract_fields():
    props = StepRecord.model_json_schema()["properties"]
    for field in (
        "step_index",
        "trial_index",
        "turn_index",
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "tokens_total",
        "request",
        "response",
        "tool_calls",
        "reasoning_content",
        "cache_hit",
        "error_category",
        "latency_total_ms",
        "latency_ttft_ms",
    ):
        assert field in props


# --- full per-rollout capture + assembly (trajectory + StepRecords) ---
def test_capture_assembles_trajectory_and_step_records(tmp_path):
    """Capture two model calls and assemble both the eval-only trajectory (no token-ids) and the
    typed StepRecords; also exercises request-body replay."""
    turns = [
        {
            "model": "m",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 7,
                "total_tokens": 19,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "let me check"}],
                    "prompt_token_ids": [9],
                    "generation_token_ids": [1, 2, 3],
                    "generation_log_probs": [-0.1, -0.2, -0.3],
                },
                {"type": "function_call", "name": "calc", "call_id": "c1", "arguments": '{"x": 1}'},
            ],
        },
        {
            "model": "m",
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer is 42"}],
                    "prompt_token_ids": [9, 1, 2, 3],
                    "generation_token_ids": [4, 5],
                    "generation_log_probs": [-0.1, -0.2],
                }
            ],
        },
    ]
    seen_requests: list[dict] = []

    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        seen_requests.append(body)
        return turns[len(seen_requests) - 1]

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)
    base_headers = {"x-nemo-gym-rollout-id": "rollout-x", "x-nemo-gym-trial-index": "2"}

    r1 = client.post(
        "/v1/responses", json={"input": "solve it"}, headers={**base_headers, "x-nemo-gym-turn-index": "0"}
    )
    r2 = client.post(
        "/v1/responses",
        json={"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]},
        headers={**base_headers, "x-nemo-gym-turn-index": "1"},
    )

    assert r1.status_code == 200 and r2.status_code == 200
    assert seen_requests[0] == {"input": "solve it"}  # request-body replay worked

    # Eval-only NeMoGym trajectory (ordered content view; token-ids deliberately not surfaced —
    # on-policy RL trajectory assembly is the RL side's, WIP).
    items = assemble_rollout(CaptureStore(tmp_path), "rollout-x")
    assert [type(i).__name__ for i in items] == [
        "NeMoGymResponseOutputMessage",
        "NeMoGymResponseFunctionToolCall",
        "NeMoGymFunctionCallOutput",
        "NeMoGymResponseOutputMessage",
    ]
    assert not hasattr(items[0], "generation_token_ids")  # eval view carries no token-ids

    # Typed StepRecords.
    steps = assemble_step_records(CaptureStore(tmp_path), "rollout-x", run_id="run-1")
    assert [s.step_index for s in steps] == [0, 1]
    assert [s.turn_index for s in steps] == [0, 1]
    assert all(s.trial_index == 2 and s.run_id == "run-1" and s.model_server == "srv" for s in steps)
    assert (steps[0].tokens_in, steps[0].tokens_out, steps[0].tokens_total) == (12, 7, 19)
    assert steps[0].cache_hit is True and steps[0].cached_tokens == 4
    assert steps[0].tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert steps[0].latency_total_ms is not None
    assert steps[1].tokens_in == 20

    # Per-rollout aggregates for the rollout record.
    agg = aggregate_rollout_metrics(CaptureStore(tmp_path), "rollout-x")
    assert agg["tokens_in"] == 32 and agg["tokens_out"] == 12
    assert agg["num_turns"] == 2 and agg["num_steps"] == 2


def test_failed_call_is_captured_with_error_category(tmp_path):
    """A non-2xx model call is captured (replacing generic exception catching) with a
    normalized error_category + status_code on the StepRecord."""
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> JSONResponse:
        return JSONResponse(content={"error": "boom"}, status_code=500)

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "r-err"})
    assert r.status_code == 500  # response unchanged

    steps = assemble_step_records(CaptureStore(tmp_path), "r-err")
    assert len(steps) == 1
    assert steps[0].error_category == "upstream_error"
    assert steps[0].status_code == 500


def test_raised_call_is_captured_then_reraised(tmp_path):
    """A model call that raises (not just a non-2xx) is captured with an exception category and the
    error is re-raised (response unchanged for the caller)."""
    app = FastAPI()

    @app.post("/v1/responses")
    async def _boom(body: dict = Body()) -> dict:
        raise RuntimeError("kaboom")

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app, raise_server_exceptions=False)

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "r-raise"})
    assert r.status_code == 500  # error propagated, response unchanged

    steps = assemble_step_records(CaptureStore(tmp_path), "r-raise")
    assert len(steps) == 1
    assert steps[0].error_category == "exception" and steps[0].response is None
    assert steps[0].latency_ttft_ms is None  # nothing streamed before the raise


def test_per_rollout_url_prefix_correlates_and_is_openai_compatible(tmp_path):
    """A caller can attribute calls by a base_url path prefix (no header). The /v1/... route is
    reached unchanged (prefix stripped), an explicit header still wins, and plain URLs are
    unaffected."""
    app = FastAPI()

    @app.post("/v1/responses")
    async def _responses(body: dict = Body()) -> dict:
        return {"output": [], "usage": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    # Prefixed base_url: routes to /v1/responses AND correlates capture by the path id (no header).
    r = client.post("/ng-rollout/task7-roll2/v1/responses", json={"input": "hi"})
    assert r.status_code == 200 and r.json()["usage"]["total_tokens"] == 4
    steps = assemble_step_records(CaptureStore(tmp_path), "task7-roll2")
    assert len(steps) == 1 and steps[0].tokens_total == 4

    # Explicit header wins when both are present.
    r2 = client.post(
        "/ng-rollout/path-id/v1/responses", json={"input": "hi"}, headers={"x-nemo-gym-rollout-id": "hdr-id"}
    )
    assert r2.status_code == 200
    assert len(assemble_step_records(CaptureStore(tmp_path), "hdr-id")) == 1
    assert assemble_step_records(CaptureStore(tmp_path), "path-id") == []

    # Plain /v1 URL is unaffected (back-compat: capture falls back to the default key).
    r3 = client.post("/v1/responses", json={"input": "hi"})
    assert r3.status_code == 200
    assert len(assemble_step_records(CaptureStore(tmp_path), "rollout")) == 1


def test_per_rollout_prefix_strips_for_non_observed_paths_too(tmp_path):
    """A prefixed but non-observed path (e.g. /v1/models) is still stripped and routed normally,
    and is not captured (composes with arbitrary endpoints, not just the observed dialects)."""
    app = FastAPI()

    @app.get("/v1/models")
    async def _models() -> dict:
        return {"object": "list", "data": []}

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    r = client.get("/ng-rollout/abc/v1/models")
    assert r.status_code == 200 and r.json()["object"] == "list"
    assert CaptureStore(tmp_path).read("abc") == []  # non-observed path -> routed, not captured


def test_apply_rollout_prefix_is_uniform_and_round_trips_with_server_parser():
    """The shared agent-side builder works for every base_url shape agents use, and round-trips
    with the model server's parser (producer <-> consumer agreement)."""
    from nemo_gym.observability import _ROLLOUT_PATH_RE
    from nemo_gym.server_utils import apply_rollout_prefix

    # Forgiving + uniform across the base_url shapes agents build today.
    assert apply_rollout_prefix("http://h:1/v1", "r1") == "http://h:1/ng-rollout/r1/v1"
    assert apply_rollout_prefix("http://h:1", "r1") == "http://h:1/ng-rollout/r1"
    assert apply_rollout_prefix("http://h:1/v1", None) == "http://h:1/v1"  # no-op
    assert apply_rollout_prefix("http://h:1/v1", "") == "http://h:1/v1"  # no-op

    # Producer (agent) and consumer (server) agree: a prefixed call round-trips to the id + /v1 path.
    client_path = apply_rollout_prefix("/v1", "task-7") + "/chat/completions"
    match = _ROLLOUT_PATH_RE.match(client_path)
    assert match and match.group("rollout_id") == "task-7" and match.group("rest") == "/v1/chat/completions"


def test_rollout_id_from_run_body_reads_canonical_indices():
    """The shared accessor agents use to derive the rollout id from a /run request body."""
    from pydantic import BaseModel, ConfigDict

    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
    from nemo_gym.server_utils import rollout_id_from_run_body

    assert rollout_id_from_run_body({TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 1}) == "3-1"
    assert rollout_id_from_run_body({TASK_INDEX_KEY_NAME: 3}) is None  # partial -> None
    assert rollout_id_from_run_body({}) is None
    assert rollout_id_from_run_body(None) is None

    # The shape agents actually receive: a run-request model with extra="allow".
    class _Body(BaseModel):
        model_config = ConfigDict(extra="allow")

    body = _Body.model_validate({TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 2})
    assert rollout_id_from_run_body(body) == "5-2"


# --- error classification + header parsing ---
def test_classify_status_branches():
    from nemo_gym.observability import _classify_status

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

    from nemo_gym.observability import _classify_exception

    class _ReadTimeout(Exception):
        pass

    assert _classify_exception(asyncio.TimeoutError()) == "timeout"
    assert _classify_exception(_ReadTimeout()) == "timeout"  # name contains "timeout"
    assert _classify_exception(ConnectionError()) == "connection"  # name contains "conn"
    assert _classify_exception(ValueError("x")) == "exception"


def test_scope_header_parsing():
    from nemo_gym.observability import _scope_header, _scope_header_int

    scope = {"headers": [(b"good", b"3"), (b"bad", b"nope")]}
    assert _scope_header(scope, "good") == "3"
    assert _scope_header(scope, "missing") is None
    assert _scope_header_int(scope, "good") == 3
    assert _scope_header_int(scope, "bad") is None  # not an int
    assert _scope_header_int(scope, "missing") is None  # absent


# --- capture-store dir resolution + init failure ---
def test_default_capture_dir(monkeypatch):
    from nemo_gym.observability import _default_capture_dir

    monkeypatch.setenv("NEMO_GYM_TRAJECTORY_DIR", "/tmp/custom_traj")
    assert _default_capture_dir("srv") == "/tmp/custom_traj"
    monkeypatch.delenv("NEMO_GYM_TRAJECTORY_DIR", raising=False)
    assert _default_capture_dir("srv").endswith("nemo_gym_trajectories/srv")


def test_make_capture_store_init_failure_returns_none(monkeypatch):
    import nemo_gym.observability as obs

    def _boom(_root):
        raise OSError("cannot create")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir="/tmp/x", name="srv")
    assert obs.make_capture_store(config) is None


def test_record_swallows_store_failure():
    from nemo_gym.observability import _record

    class _BadStore:
        def record(self, *args, **kwargs):
            raise RuntimeError("disk full")

    # Best-effort: a failing store must not raise out of _record.
    _record(
        _BadStore(),
        SimpleNamespace(headers={}),
        "chat",
        SimpleNamespace(name="srv"),
        b"{}",
        rollout_id="r",
        response_body={},
        status_code=200,
        error_category=None,
        latency_ms=1.0,
    )


def test_capture_store_read_skips_blank_and_bad_lines(tmp_path):
    store = CaptureStore(tmp_path)
    store.path_for("r").write_text('{"a": 1}\n\nnot json\n{"b": 2}\n')
    assert store.read("r") == [{"a": 1}, {"b": 2}]


def test_capture_records_non_json_response_as_none(tmp_path):
    from fastapi.responses import PlainTextResponse

    app = FastAPI()

    @app.post("/v1/responses")
    async def _r(body: dict = Body()) -> PlainTextResponse:
        return PlainTextResponse("not json")

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "rnj"})
    assert r.status_code == 200 and r.text == "not json"  # response passed through unaltered
    records = CaptureStore(tmp_path).read("rnj")
    assert len(records) == 1 and records[0]["response"] is None  # non-JSON body -> None


# --- trajectory_capture helpers ---
def test_content_text():
    from nemo_gym.trajectory_capture import _content_text

    assert _content_text("hi") == "hi"
    assert _content_text([{"text": "a"}, {"text": "b"}]) == "ab"
    assert _content_text(None) == ""
    assert _content_text(123) == "123"


def test_as_arguments():
    from nemo_gym.trajectory_capture import _as_arguments

    assert _as_arguments({"x": 1}) == {"x": 1}
    assert _as_arguments('{"y": 2}') == {"y": 2}
    assert _as_arguments("not json") == {"_raw": "not json"}
    assert _as_arguments(123) == {}


def test_cache_signal():
    from nemo_gym.trajectory_capture import _cache_signal

    assert _cache_signal(None) == (None, None)
    assert _cache_signal({"prompt_tokens_details": {"cached_tokens": 4}}) == (True, 4)
    assert _cache_signal({"input_tokens_details": {"cached_tokens": 0}}) == (False, 0)
    assert _cache_signal({"cache_read_input_tokens": 5}) == (True, 5)  # Anthropic
    assert _cache_signal({"unrelated": 1}) == (None, None)


def test_extract_token_stats_chat_fallback():
    from nemo_gym.trajectory_capture import extract_token_stats

    assert extract_token_stats(None)["tokens_total"] is None
    stats = extract_token_stats(
        {"prompt_tokens": 5, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}
    )
    assert (stats["tokens_in"], stats["tokens_out"], stats["tokens_total"], stats["tokens_reasoning"]) == (5, 3, 8, 1)


def test_tool_calls_and_reasoning_chat_and_anthropic():
    from nemo_gym.trajectory_capture import _tool_calls_and_reasoning

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
    from nemo_gym.trajectory_capture import _tool_calls_and_reasoning

    # vLLM / newer servers emit `reasoning` (no `reasoning_content`).
    assert _tool_calls_and_reasoning({"choices": [{"message": {"content": "hi", "reasoning": "because"}}]}) == (
        [],
        "because",
    )
    # `reasoning_content` wins when both are present.
    both = {"choices": [{"message": {"reasoning_content": "rc", "reasoning": "r"}}]}
    assert _tool_calls_and_reasoning(both) == ([], "rc")


def test_tool_calls_and_reasoning_skips_non_dict_output_items():
    from nemo_gym.trajectory_capture import _tool_calls_and_reasoning

    # A Responses `output` list may contain non-dict junk; it must be skipped, not crash.
    resp = {"output": [None, "junk", {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]}
    assert _tool_calls_and_reasoning(resp) == ([{"call_id": "c", "name": "f", "arguments": {}}], None)


def test_assemble_chat_wire_trajectory(tmp_path):
    """The chat-wire assembler: tool results arrive in the next request as role:tool messages."""
    store = CaptureStore(tmp_path)
    store.record(
        "rc",
        {
            "dialect": "chat",
            "request": {"messages": [{"role": "user", "content": "hi"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "content": "let me check",
                            "tool_calls": [{"id": "c1", "function": {"name": "calc", "arguments": "{}"}}],
                        }
                    }
                ]
            },
        },
    )
    store.record(
        "rc",
        {
            "dialect": "chat",
            "request": {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "c1", "content": "42"},
                ]
            },
            "response": {"choices": [{"message": {"content": "answer is 42"}}]},
        },
    )

    items = assemble_rollout(store, "rc")
    assert [type(i).__name__ for i in items] == [
        "NeMoGymResponseOutputMessage",  # turn 0 assistant
        "NeMoGymResponseFunctionToolCall",  # turn 0 tool call
        "NeMoGymFunctionCallOutput",  # turn 1 tool result (from the role:tool request message)
        "NeMoGymResponseOutputMessage",  # turn 1 assistant
    ]


# --- base-agent correlation helpers ---
def test_base_agent_resolve_model_base_url_and_call_kwargs(monkeypatch):
    import nemo_gym.base_responses_api_agent as ba
    from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

    monkeypatch.setattr(ba, "get_first_server_config_dict", lambda _gc, _name: {"host": "h", "port": 1})
    fake_self = SimpleNamespace(
        server_client=SimpleNamespace(global_config_dict={}, _build_server_base_url=lambda _cfg: "http://h:1"),
        rollout_id_from_run=lambda body: ba.rollout_id_from_run_body(body),
    )

    with_id = SimpleResponsesAPIAgent.resolve_model_base_url(
        fake_self, "m", SimpleNamespace(headers={"x-nemo-gym-rollout-id": "rid"})
    )
    assert with_id == "http://h:1/ng-rollout/rid/v1"
    assert SimpleResponsesAPIAgent.resolve_model_base_url(fake_self, "m", None) == "http://h:1/v1"

    from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME

    kwargs = SimpleResponsesAPIAgent.rollout_call_kwargs(
        fake_self, {TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 2}
    )
    assert kwargs == {"headers": {"x-nemo-gym-rollout-id": "5-2"}}
    assert SimpleResponsesAPIAgent.rollout_call_kwargs(fake_self, {}) == {}


def _sse(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


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

    config = SimpleNamespace(observability_enabled=True, trajectory_capture_dir=str(tmp_path), name="srv")
    install_trajectory_capture(app, config)
    client = TestClient(app)

    r = client.post("/ng-rollout/3-0/v1/messages", json={"stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")  # stream preserved
    assert "event:" in r.text and "data:" in r.text  # SSE content flowed through

    records = CaptureStore(tmp_path).read("3-0")
    assert len(records) == 1 and records[0]["response"] is not None  # reassembled, not dropped

    steps = assemble_step_records(CaptureStore(tmp_path), "3-0")
    assert len(steps) == 1
    step = steps[0]
    assert step.dialect == "messages"
    assert step.tokens_in == 17 and step.tokens_out == 7  # 10 + cache_read 5 + cache_creation 2; output from delta
    assert step.cached_tokens == 5 and step.cache_creation_tokens == 2
    assert step.reasoning_content == "let me think"
    assert step.tool_calls == [{"call_id": "t1", "name": "calc", "arguments": {"x": 1}}]
    assert step.latency_ttft_ms is not None


def test_reconstruct_chat_sse():
    from nemo_gym.observability import _reconstruct_streamed_response

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
    raw = b"".join(_sse("", c) for c in chunks) + b"data: [DONE]\n\n"
    resp = _reconstruct_streamed_response(raw, "chat")
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hello" and msg["reasoning_content"] == "hmm"
    assert msg["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"a":1}'}
    assert resp["usage"]["total_tokens"] == 8


def test_reconstruct_responses_sse_uses_terminal_envelope():
    from nemo_gym.observability import _reconstruct_streamed_response

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
    from nemo_gym.observability import _reconstruct_streamed_response

    assert _reconstruct_streamed_response(b"", "chat") is None  # no events
    assert _reconstruct_streamed_response(b"event: ping\ndata: not-json\n\n", "messages") is None  # unparseable
    assert _reconstruct_streamed_response(b"data: 123\n\n", "chat") is None  # non-dict JSON skipped
    # Non-empty events that carry nothing reconstructable -> None for each dialect.
    ping = _sse("ping", {"type": "ping"})
    assert _reconstruct_streamed_response(ping, "messages") is None
    assert _reconstruct_streamed_response(ping, "chat") is None
    assert _reconstruct_streamed_response(ping, "responses") is None


def test_rollout_id_from_run_body_attempt_suffix():
    from nemo_gym.server_utils import rollout_id_from_run_body

    base = {"_ng_task_index": 3, "_ng_rollout_index": 2}
    assert rollout_id_from_run_body(base) == "3-2"  # no attempt -> bare key
    assert rollout_id_from_run_body({**base, "_ng_attempt_index": 0}) == "3-2"  # attempt 0 -> bare (back-compat)
    assert rollout_id_from_run_body({**base, "_ng_attempt_index": 1}) == "3-2-a1"
    assert rollout_id_from_run_body({**base, "_ng_attempt_index": "2"}) == "3-2-a2"  # coerced
    assert rollout_id_from_run_body({"_ng_rollout_index": 2}) is None  # missing task -> None


def test_capture_dirs_from_config_resolves_env_and_config(tmp_path):
    from nemo_gym.observability import capture_dirs_from_config

    shared = tmp_path / "shared"
    shared.mkdir()
    server_dir = tmp_path / "srv"
    server_dir.mkdir()
    config = {
        "policy_model": {
            "responses_api_models": {
                "openai_model": {
                    "observability_enabled": True,
                    "trajectory_capture_dir": str(server_dir),
                    "name": "srv",
                }
            }
        }
    }
    dirs = capture_dirs_from_config(config, env={"NEMO_GYM_TRAJECTORY_DIR": str(shared)})
    assert shared in dirs and server_dir in dirs
    # capture-off servers contribute nothing; non-existent dirs are dropped
    off = {"a": {"b": {"observability_enabled": False, "trajectory_capture_dir": str(tmp_path / "nope")}}}
    assert capture_dirs_from_config(off, env={}) == []


def _capture_exchange(dialect, model_server, usage, response):
    return {
        "dialect": dialect,
        "model_server": model_server,
        "trial_index": 0,
        "turn_index": 0,
        "latency_ms": 1.0,
        "status_code": 200,
        "error_category": None,
        "request": {"input": "hi"},
        "response": {"model": "m", "usage": usage, **response},
    }


def test_merge_capture_uniform_shape_across_agents(tmp_path):
    from nemo_gym.observability import merge_capture_into_record
    from nemo_gym.trajectory_capture import CaptureStore

    store = CaptureStore(tmp_path)
    # Two different harnesses/dialects -> the attached trajectory must have the identical shape.
    store.record(
        "0-0",
        _capture_exchange(
            "responses",
            "A",
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]},
        ),
    )
    store.record(
        "1-0",
        _capture_exchange(
            "messages",
            "B",
            {"input_tokens": 4, "output_tokens": 1},
            {"content": [{"type": "text", "text": "ok"}]},
        ),
    )

    rec_a = {"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0, "response": {"harness": "A"}}
    rec_b = {"_ng_task_index": 1, "_ng_rollout_index": 0, "reward": 0.0, "response": {"harness": "B"}}
    merge_capture_into_record(rec_a, [tmp_path])
    merge_capture_into_record(rec_b, [tmp_path])

    cap_a, cap_b = rec_a["ng_trajectory_capture"], rec_b["ng_trajectory_capture"]
    assert set(cap_a) == set(cap_b) == {"rollout_id", "metrics", "steps"}  # identical top-level
    assert set(cap_a["metrics"]) == set(cap_b["metrics"])  # identical metrics shape
    assert set(cap_a["steps"][0]) == set(cap_b["steps"][0])  # identical per-step shape
    assert "request" not in cap_a["steps"][0] and "response" not in cap_a["steps"][0]  # payloads stay in the store
    # harness output + reward are untouched; capture carries the wire-level token stats
    assert rec_a["response"] == {"harness": "A"} and rec_a["reward"] == 1.0
    assert cap_a["rollout_id"] == "0-0" and cap_a["steps"][0]["tokens_in"] == 3


def test_merge_capture_noop_without_capture(tmp_path):
    from nemo_gym.observability import merge_capture_into_record

    rec = {"_ng_task_index": 9, "_ng_rollout_index": 9, "reward": 1.0}
    merge_capture_into_record(rec, [tmp_path])  # no capture file for 9-9
    assert "ng_trajectory_capture" not in rec
    merge_capture_into_record(rec, [])  # no dirs
    assert "ng_trajectory_capture" not in rec


def test_aggregate_step_records_sums_and_counts():
    from nemo_gym.trajectory_capture import StepRecord, aggregate_step_records

    steps = [
        StepRecord(step_index=0, turn_index=0, tokens_in=10, tokens_out=5, tokens_total=15, latency_total_ms=2.0),
        StepRecord(step_index=1, turn_index=1, tokens_in=20, tokens_out=3, tokens_total=23, latency_total_ms=1.0),
    ]
    agg = aggregate_step_records(steps)
    assert (agg["tokens_in"], agg["tokens_out"], agg["tokens_total"]) == (30, 8, 38)
    assert agg["latency_total_ms"] == 3.0 and agg["num_turns"] == 2 and agg["num_steps"] == 2
    # empty -> all-None totals but a well-formed shape (num_steps 0)
    assert aggregate_step_records([]) == {
        "tokens_in": None,
        "tokens_out": None,
        "tokens_reasoning": None,
        "tokens_total": None,
        "latency_total_ms": None,
        "num_turns": None,
        "num_steps": 0,
    }


def test_aggregate_num_turns_is_none_without_turn_markers():
    from nemo_gym.trajectory_capture import StepRecord, aggregate_step_records

    # No caller set turn_index -> num_turns is None (a turn != a model step), num_steps stays exact.
    steps = [StepRecord(step_index=0, tokens_in=1), StepRecord(step_index=1, tokens_in=2)]
    agg = aggregate_step_records(steps)
    assert agg["num_steps"] == 2 and agg["num_turns"] is None


# --- P0 review regressions: prefix routing + capture-dir resolution ---
def test_rollout_prefix_stripped_when_capture_disabled():
    # The /ng-rollout/<id> prefix must be stripped + routed even when capture is OFF (the default),
    # otherwise a default `gym eval` 404s on every prefixed model call.
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _cc() -> dict:
        return {"ok": True}

    install_trajectory_capture(app, SimpleNamespace(observability_enabled=False))
    client = TestClient(app)
    assert client.post("/v1/chat/completions", json={}).status_code == 200
    assert client.post("/ng-rollout/3-0/v1/chat/completions", json={}).status_code == 200


def test_capture_dirs_resolves_default_by_config_key(tmp_path, monkeypatch):
    # Bare opt-in (no trajectory_capture_dir / NEMO_GYM_TRAJECTORY_DIR): the consumer must resolve the
    # same default dir the producer used, which is keyed off the asset config key (== config.name),
    # not the "model_server" fallback.
    import nemo_gym.observability as obs

    monkeypatch.delenv("NEMO_GYM_TRAJECTORY_DIR", raising=False)
    monkeypatch.setattr(obs.tempfile, "gettempdir", lambda: str(tmp_path))
    producer_dir = tmp_path / "nemo_gym_trajectories" / "openai_model"
    producer_dir.mkdir(parents=True)
    # Inner server node has no "name"; instance key (policy_model) differs from the asset key.
    gc = {"policy_model": {"responses_api_models": {"openai_model": {"observability_enabled": True}}}}
    dirs = obs.capture_dirs_from_config(gc, env={})
    assert producer_dir in dirs
    assert (tmp_path / "nemo_gym_trajectories" / "model_server") not in dirs


def test_capture_dirs_warns_once_when_unresolved(tmp_path, monkeypatch, caplog):
    import logging

    import nemo_gym.observability as obs

    monkeypatch.delenv("NEMO_GYM_TRAJECTORY_DIR", raising=False)
    monkeypatch.setattr(obs.tempfile, "gettempdir", lambda: str(tmp_path))  # nothing created -> unresolved
    obs._capture_dirs_warned = False
    gc = {"policy_model": {"responses_api_models": {"openai_model": {"observability_enabled": True}}}}
    try:
        with caplog.at_level(logging.WARNING, logger="nemo_gym.observability"):
            assert obs.capture_dirs_from_config(gc, env={}) == []
            assert obs.capture_dirs_from_config(gc, env={}) == []  # second call: no new warning
        warns = [r for r in caplog.records if r.levelno == logging.WARNING and "capture directory" in r.message]
        assert len(warns) == 1
    finally:
        obs._capture_dirs_warned = False


# --- P1 review regressions: Anthropic cache tokens + concurrent-append integrity ---
def test_extract_token_stats_anthropic_cache_fold():
    from nemo_gym.trajectory_capture import extract_token_stats

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


def test_extract_token_stats_openai_cached_not_double_counted():
    from nemo_gym.trajectory_capture import extract_token_stats

    # OpenAI: input_tokens already includes cached tokens (cached_tokens is a subset) -> no fold.
    stats = extract_token_stats(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
    )
    assert stats["tokens_in"] == 10
    assert stats["tokens_total"] == 15
    assert stats["cache_creation_tokens"] is None


def test_build_step_record_surfaces_anthropic_cache_creation():
    rec = build_step_record(
        {
            "dialect": "messages",
            "response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 7,
                    "cache_creation_input_tokens": 3,
                }
            },
        },
        step_index=0,
    )
    assert rec.tokens_in == 20  # 10 + 7 + 3
    assert rec.cached_tokens == 7  # cache-read
    assert rec.cache_creation_tokens == 3


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
