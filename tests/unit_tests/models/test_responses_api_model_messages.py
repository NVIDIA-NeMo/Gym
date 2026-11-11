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
"""Tests for the default ``/v1/messages`` route on ``SimpleResponsesAPIModel``.

Every Gym model server inherits an Anthropic Messages endpoint that maps Messages <-> Responses
around the server's own ``responses()``. Minimal servers cover both ``responses()`` signatures;
provider-cause regressions use the real VLLM adapter with only its upstream boundary mocked.
"""

import json
from pathlib import Path
from time import time
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.client.openai_client import NeMoGymAsyncOpenAI
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.model_call_capture import (
    CaptureStore,
    ModelCallCaptureConfig,
    _reconstruct_anthropic_sse,
    install_model_call_capture,
)
from nemo_gym.rollouts.model_calls import read_model_call_records
from nemo_gym.rollouts.rollout_health import run_health_checks
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.trajectory.model_call_record import ModelCallRecord
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


def _build_response(text: str, model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _BodyOnlyModel(SimpleResponsesAPIModel):
    """A server whose responses() takes only `body` (like openai_model)."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "last_params", body)
        return _build_response("hi from body-only")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


class _RequestAwareModel(SimpleResponsesAPIModel):
    """A server whose responses() also takes `request` (like vllm_model / azure)."""

    config: BaseResponsesAPIModelConfig
    saw_request: bool = False
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return _build_response("hi from request-aware")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


def _config() -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="")


def _client(model_cls) -> TestClient:
    server = model_cls(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    return TestClient(server.setup_webserver()), server


class TestDefaultMessagesRoute:
    def test_messages_route_registered_alongside_openai_routes(self) -> None:
        server = _BodyOnlyModel(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        paths = {route.path for route in server.setup_webserver().routes}
        assert {"/v1/messages", "/v1/responses", "/v1/chat/completions"} <= paths

    def test_body_only_responses_signature(self) -> None:
        client, server = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "hi from body-only"}]
        assert data["model"] == "claude-x"  # request model echoed back
        # the inbound Anthropic request was translated to Responses params before delegating
        assert server.last_params.input[0].content == "hello"
        assert server.last_params.max_output_tokens == 32

    def test_request_aware_responses_signature(self) -> None:
        client, server = _client(_RequestAwareModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == [{"type": "text", "text": "hi from request-aware"}]
        assert server.saw_request is True  # request was forwarded to responses()

    def test_streaming_returns_anthropic_sse(self) -> None:
        client, _ = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-x",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body


def _capture_vllm_messages(tmp_path: Path, *, stream: bool, completions: bool, finish: str, stop: str | int | None):
    server = VLLMModel(
        config=VLLMModelConfig(
            host="0.0.0.0",
            port=8099,
            entrypoint="",
            name="srv",
            base_url="http://127.0.0.1:8001/v1",
            api_key="synthetic",
            model="synthetic-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
            use_completions_api=completions,
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    choice = {"index": 0, "finish_reason": finish}
    choice.update(
        {"text": "Retained answer."}
        if completions
        else {"message": {"role": "assistant", "content": "Retained answer."}}
    )
    if stop is not None:
        choice["stop_reason"] = stop
    provider = MagicMock(spec=NeMoGymAsyncOpenAI, base_url="http://127.0.0.1:8001/v1")
    generation = AsyncMock(
        return_value={
            "id": "provider-response",
            "created": 1,
            "object": "text_completion" if completions else "chat.completion",
            "model": "synthetic-model",
            "choices": [choice],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )
    if completions:
        provider.create_completion = generation
    else:
        provider.create_chat_completion = generation
    server._clients = [provider]
    app = server.setup_webserver()
    capture_dir = tmp_path / "capture"
    install_model_call_capture(
        app,
        ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=capture_dir),
        model_server_name="srv",
        assistant_message_header=b"x-assistant-message-id",
    )
    with TestClient(app) as client:
        received = client.post(
            "/ng-rollout/messages-native/v1/messages",
            json={
                "model": "caller-model",
                "max_tokens": 32,
                "stream": stream,
                "messages": [{"role": "user", "content": "Question"}],
            },
        )
    assert received.status_code == 200, received.text
    generation.assert_awaited_once()
    forwarded = generation.await_args.kwargs
    assert ("prompt" in forwarded) is completions
    assert ("messages" in forwarded) is not completions
    store = CaptureStore(capture_dir)
    [exchange] = store.read("messages-native")
    [call] = read_model_call_records(store, "messages-native")
    return received, exchange, call


def _health_for_messages_call(tmp_path: Path, call: ModelCallRecord, *, bound: bool = True):
    indices = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result = {
        "response": {
            "output": [{"type": "message", "role": "assistant", "content": "Retained answer."}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        "ng_trajectory": {
            "task_id": "0",
            "rollout_id": "0-0",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "0",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "answer": "Retained answer.",
                    "step_count": 1,
                    "model_calls": [{"model_call_id": call.model_call_id if bound else "not-captured"}],
                }
            ],
            "model_calls": [],
        },
        "ng_model_call_capture": {"calls": [call.model_dump(mode="json")]},
    }
    result["ng_trajectory"] = _build_trajectory_record(indices, result).model_dump(mode="json")
    saved = tmp_path / "rollouts.jsonl"
    saved.write_text(json.dumps({**indices, **result}) + "\n")
    before = saved.read_bytes()
    [digest] = run_health_checks(saved, output_dir=tmp_path / "health", workers=1).rollouts
    assert saved.read_bytes() == before
    return digest


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("completions", [False, True], ids=["chat-backend", "completions-backend"])
@pytest.mark.parametrize(
    "finish,stop,runaway",
    [
        ("repetition", "repetition_detected", True),
        ("repetition", None, True),
        ("stop", "repetition_detected", True),
        ("stop", 0, False),
        ("stop", 11, False),
        ("stop", None, False),
    ],
)
def test_messages_preserve_native_provider_causes_and_saved_response_population(
    tmp_path: Path, stream: bool, completions: bool, finish: str, stop: str | int | None, runaway: bool
) -> None:
    received, exchange, call = _capture_vllm_messages(
        tmp_path, stream=stream, completions=completions, finish=finish, stop=stop
    )
    if stream:
        events = [json.loads(line[6:]) for line in received.text.splitlines() if line.startswith("data: ")]
        served = next(event["message"] for event in events if event["type"] == "message_start")
        assert events[-1]["type"] == "message_stop"
        assert [event["delta"]["text"] for event in events if event["type"] == "content_block_delta"] == [
            "Retained answer."
        ]
    else:
        served = received.json()
        assert served["content"] == [{"type": "text", "text": "Retained answer."}]
    assert served["stop_reason"] == "end_turn"
    assert served["usage"] == {"input_tokens": 3, "output_tokens": 2}
    native_finish = "repetition" if finish == "repetition" else None
    for key, value in (("native_stop_reason", stop), ("native_finish_reason", native_finish)):
        assert served.get(key) == value
        assert type(served.get(key)) is type(value)
        assert (key in served) is (value is not None)
        assert exchange["response"].get(key) == value
        assert type(exchange["response"].get(key)) is type(value)
        assert (key in exchange["response"]) is (value is not None)
    if stream:
        assert exchange["response_raw"] == received.text
    else:
        assert exchange["response"] == received.json()
    assert exchange["response"]["content"] == [{"type": "text", "text": "Retained answer."}]
    assert exchange["response"]["stop_reason"] == "end_turn"
    assert exchange["response"]["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert call.dialect == "messages" and call.finish_reason == "end_turn"
    assert (call.tokens_in, call.tokens_out, call.tokens_total) == (3, 2, 5)
    assert call.tokens_reasoning is None
    digest = _health_for_messages_call(tmp_path, call)
    assert digest.responses == {"selected": [call.model_call_id], "faulted": [call.model_call_id] if runaway else []}
    assert ("model_call_runaway_generation" in {finding.check for finding in digest.findings}) is runaway
    assert not digest.unobserved


def test_messages_repetition_capture_does_not_invent_a_bound_response(tmp_path: Path) -> None:
    _, _, call = _capture_vllm_messages(
        tmp_path, stream=True, completions=False, finish="repetition", stop="repetition_detected"
    )
    digest = _health_for_messages_call(tmp_path, call, bound=False)
    assert digest.responses is None
    assert "model_call_runaway_generation" in digest.unobserved


def test_anthropic_sse_reconstruction_preserves_zero_native_causes() -> None:
    response = _reconstruct_anthropic_sse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "model": "model",
                    "stop_reason": "end_turn",
                    "native_stop_reason": 0,
                    "native_finish_reason": 0,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "answer"}},
            {"type": "message_stop"},
        ]
    )
    assert response["stop_reason"] == "end_turn"
    assert response["content"] == [{"type": "text", "text": "answer"}]
    for key in ("native_stop_reason", "native_finish_reason"):
        assert response[key] == 0
        assert type(response[key]) is int
