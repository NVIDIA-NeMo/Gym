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
"""Tests for Anthropic<->OpenAI translation (Claude Code against any backend)."""

import asyncio
import json

from nemo_gym.adapters.capture_store import CaptureStore, assemble_trajectory
from nemo_gym.adapters.interceptors.translate_anthropic import (
    anthropic_to_openai_request,
    openai_to_anthropic_error,
    openai_to_anthropic_response,
)
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


def test_registered():
    assert "translate_anthropic" in InterceptorRegistry.available()


def test_request_system_tools_and_forces_non_stream():
    body = {
        "model": "claude-x",
        "max_tokens": 1024,
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
        "stream": True,
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "be terse"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}
    assert out["tools"][0]["type"] == "function"
    assert out["tools"][0]["function"]["name"] == "bash"
    assert out["tool_choice"] == "auto"
    assert out["max_tokens"] == 1024
    assert out["stream"] is False  # proxy buffers => forced non-streaming upstream


def test_request_tool_use_and_tool_result():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me"},
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.py"}]},
        ],
    }
    out = anthropic_to_openai_request(body)
    assistant = out["messages"][0]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "t1"
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"cmd": "ls"}
    assert out["messages"][1] == {"role": "tool", "tool_call_id": "t1", "content": "file.py"}


def test_response_text_and_tool_calls_to_anthropic():
    oai = {
        "id": "c1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [{"id": "t2", "function": {"name": "edit", "arguments": "{\"path\": \"a\"}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    ant = openai_to_anthropic_response(oai, "claude-x")
    assert ant["type"] == "message" and ant["role"] == "assistant"
    assert [b["type"] for b in ant["content"]] == ["text", "tool_use"]
    assert ant["content"][1]["name"] == "edit"
    assert ant["content"][1]["input"] == {"path": "a"}
    assert ant["stop_reason"] == "tool_use"
    assert ant["usage"] == {"input_tokens": 10, "output_tokens": 3}


def _fake_upstream(captured, response_body):
    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        captured["path"] = req.path
        captured["body"] = dict(req.body)
        return AdapterResponse(status_code=200, headers={}, body=response_body, latency_ms=1.0, ctx=req.ctx)

    return _upstream


def test_interceptor_roundtrip_messages_to_chat():
    interceptor = InterceptorRegistry.create("translate_anthropic", {})
    pipeline = AdapterPipeline([interceptor])
    captured: dict = {}
    response_body = {
        "id": "c",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    req = AdapterRequest(
        method="POST",
        path="/v1/messages",
        headers={},
        body={"model": "claude-x", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 50},
        ctx=InterceptorContext(),
    )
    resp = asyncio.run(pipeline.process(req, upstream_call=_fake_upstream(captured, response_body)))

    # upstream saw an OpenAI chat-completions request at the rewritten path
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["messages"][-1] == {"role": "user", "content": "hello"}
    # the agent receives an Anthropic-shaped message
    assert resp.body["type"] == "message"
    assert resp.body["content"][0]["text"] == "hi"
    assert resp.body["stop_reason"] == "end_turn"


def test_translate_then_capture_records_openai_with_token_ids(tmp_path):
    translate = InterceptorRegistry.create("translate_anthropic", {})
    capture = InterceptorRegistry.create("capture", {"store_dir": str(tmp_path), "session_id": "s1"})
    pipeline = AdapterPipeline([translate, capture])
    response_body = {
        "id": "c",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok", "generation_token_ids": [1, 2]},
                "prompt_token_ids": [3],
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    req = AdapterRequest(
        method="POST",
        path="/v1/messages",
        headers={},
        body={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )
    resp = asyncio.run(pipeline.process(req, upstream_call=_fake_upstream({}, response_body)))

    # agent gets Anthropic back...
    assert resp.body["type"] == "message"
    # ...but the captured exchange is OpenAI-shaped, so token-ids/trajectory stay uniform
    exchanges = CaptureStore(tmp_path).read("s1")
    assert len(exchanges) == 1
    assert exchanges[0]["response"]["choices"][0]["message"]["generation_token_ids"] == [1, 2]
    trajectory = assemble_trajectory(exchanges)
    assert trajectory and trajectory[0].generation_token_ids == [1, 2]


def test_openai_error_body_to_anthropic_envelope():
    out = openai_to_anthropic_error({"error": {"message": "bad key", "type": "invalid_api_key"}}, 401)
    assert out == {"type": "error", "error": {"type": "invalid_api_key", "message": "bad key"}}
    # no explicit type -> mapped from the HTTP status
    mapped = openai_to_anthropic_error({"error": {"message": "slow down"}}, 429)
    assert mapped["error"]["type"] == "rate_limit_error"


def test_interceptor_reshapes_upstream_error_to_anthropic():
    interceptor = InterceptorRegistry.create("translate_anthropic", {})
    pipeline = AdapterPipeline([interceptor])

    async def _err_upstream(req: AdapterRequest) -> AdapterResponse:
        # upstream returns an OpenAI-shaped error (no "choices")
        return AdapterResponse(
            status_code=429, headers={}, body={"error": {"message": "rate limited"}}, latency_ms=1.0, ctx=req.ctx
        )

    req = AdapterRequest(
        method="POST",
        path="/v1/messages",
        headers={},
        body={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )
    resp = asyncio.run(pipeline.process(req, upstream_call=_err_upstream))
    # the CLI gets a typed Anthropic error, not an opaque OpenAI body
    assert resp.body == {"type": "error", "error": {"type": "rate_limit_error", "message": "rate limited"}}
