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
"""End-to-end model-call capture checks through a real Gym model server.

Drives the actual SimpleModelServer.setup_webserver() install path (only the upstream OpenAI
client mocked), so the captured response is a real NeMoGymResponse that went through
model validation and serialization.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.model_call_capture import (
    CaptureStore,
    aggregate_model_call_metrics,
    read_model_call_records,
)
from nemo_gym.observability import merge_model_call_capture_into_record
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import (
    NeMoGymAsyncOpenAI,
    SimpleModelServer,
    SimpleModelServerConfig,
)


def _server(tmp_path, *, enabled: bool = True) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="dummy_model",
        entrypoint="",
        name="srv-e2e",
        observability_enabled=enabled,
        model_call_capture_dir=str(tmp_path),
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text, *, tool=None, reasoning=None, cached=0, reasoning_tokens=0, in_tok=12, out_tok=7) -> dict:
    """A valid OpenAI Responses payload (the shape the upstream model returns)."""
    output = []
    if reasoning is not None:
        output.append({"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": reasoning}]})
    output.append(
        {
            "type": "message",
            "id": "m",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )
    if tool is not None:
        output.append(
            {
                "type": "function_call",
                "id": "fc",
                "call_id": tool["call_id"],
                "name": tool["name"],
                "arguments": tool["arguments"],
                "status": "completed",
            }
        )
    return {
        "id": "resp_x",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def test_e2e_model_call_capture_through_real_model_server(tmp_path):
    server = _server(tmp_path)
    app = server.setup_webserver()  # real install path -> install_model_call_capture(app, config)
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(
        side_effect=[
            _response(
                "let me check",
                reasoning="weigh the options",
                tool={"call_id": "c1", "name": "calc", "arguments": '{"x": 1}'},
                cached=4,
                reasoning_tokens=2,
                in_tok=12,
                out_tok=7,
            ),
            _response("the answer is 42", in_tok=20, out_tok=5),
        ]
    )

    r1 = client.post("/ng-rollout/0-0/v1/responses", json={"input": "solve"})
    r2 = client.post(
        "/ng-rollout/0-0/v1/responses",
        json={"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]},
    )
    assert r1.status_code == 200 and r2.status_code == 200

    store = CaptureStore(tmp_path)
    calls = read_model_call_records(store, "0-0")

    # indices + attribution survive the round trip
    assert [call.call_index for call in calls] == [0, 1]
    assert all(call.model_server == "srv-e2e" and call.dialect == "responses" for call in calls)

    # Token statistics survive NeMoGymResponse validation and serialization.
    assert (calls[0].tokens_in, calls[0].tokens_out, calls[0].tokens_total) == (12, 7, 19)
    assert calls[0].tokens_reasoning == 2
    assert calls[0].cache_hit is True and calls[0].cached_tokens == 4
    assert calls[0].reasoning_content == "weigh the options"
    assert calls[0].tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert calls[0].latency_total_ms is not None
    assert calls[1].tokens_in == 20
    assert calls[1].request == {"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]}

    # per-rollout aggregates
    agg = aggregate_model_call_metrics(store, "0-0")
    assert agg["tokens_in"] == 32 and agg["tokens_out"] == 12
    assert agg["num_calls"] == 2

    rollout = {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {"preserved": True}, "reward": 1.0}
    merge_model_call_capture_into_record(rollout, [tmp_path])
    assert rollout["response"] == {"preserved": True} and rollout["reward"] == 1.0
    assert rollout["ng_model_call_capture"]["metrics"]["num_calls"] == 2
    assert [call["call_index"] for call in rollout["ng_model_call_capture"]["calls"]] == [0, 1]
    assert "request" not in rollout["ng_model_call_capture"]["calls"][0]


def test_e2e_error_category_through_real_model_server(tmp_path):
    server = _server(tmp_path)
    app = server.setup_webserver()
    client = TestClient(app, raise_server_exceptions=False)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(side_effect=asyncio.TimeoutError())

    r = client.post("/ng-rollout/r-timeout/v1/responses", json={"input": "x"})
    assert r.status_code == 500  # response unchanged for the caller

    calls = read_model_call_records(CaptureStore(tmp_path), "r-timeout")
    assert len(calls) == 1 and calls[0].error_category == "timeout"


def test_e2e_off_by_default_writes_nothing(tmp_path):
    server = _server(tmp_path, enabled=False)
    app = server.setup_webserver()
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(return_value=_response("hi"))

    r = client.post("/ng-rollout/r-off/v1/responses", json={"input": "x"})
    assert r.status_code == 200
    assert CaptureStore(tmp_path).read("r-off") == []  # disabled => no capture


def test_e2e_per_rollout_url_prefix(tmp_path):
    """The per-rollout URL prefix correlates capture through the real server install path; the
    standard /v1/... endpoint is reached unchanged (prefix stripped before routing)."""
    server = _server(tmp_path)
    app = server.setup_webserver()
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "hi", "role": "assistant"}}],
            "created": 1753983922,
            "model": "dummy_model",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    )

    r = client.post(
        "/ng-rollout/task9-roll3/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200

    calls = read_model_call_records(CaptureStore(tmp_path), "task9-roll3")
    assert len(calls) == 1
    assert calls[0].dialect == "chat" and calls[0].tokens_total == 7


def test_e2e_streaming_messages_is_captured_and_correlated(tmp_path):
    """Claude Code always streams /v1/messages. Through the real server install path the SSE is
    forwarded intact (status 200, text/event-stream) and the call is captured + correlated by the
    /ng-rollout/<id> URL prefix. The streamed events are also reassembled, so token stats survive
    on the streamed path too."""
    server = _server(tmp_path)
    app = server.setup_webserver()
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(return_value=_response("hi from claude"))

    r = client.post(
        "/ng-rollout/task5-roll2/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")  # stream preserved
    assert "event:" in r.text and "data:" in r.text  # SSE content flowed through

    records = CaptureStore(tmp_path).read("task5-roll2")
    assert len(records) == 1
    assert records[0]["dialect"] == "messages" and records[0]["status_code"] == 200
    assert records[0]["request"]["stream"] is True  # request captured
    assert records[0]["response"] is not None  # streamed SSE reassembled into the final response

    # the streamed call surfaces a full call record with token stats reassembled from the SSE
    calls = read_model_call_records(CaptureStore(tmp_path), "task5-roll2")
    assert len(calls) == 1 and calls[0].dialect == "messages"
    assert calls[0].tokens_in == 12 and calls[0].tokens_out == 7  # _response defaults, recovered via SSE
    assert calls[0].latency_ttft_ms is not None
