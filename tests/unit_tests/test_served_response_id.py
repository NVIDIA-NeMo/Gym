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
"""Test the served envelope id recorded on every captured call.

Each dialect route serves a payload with one top-level id.
Capture observes that id; it never mints one.
The Anthropic mapping reuses the inner Responses id on its outer envelope,
so the id a Messages-dialect client keeps is the id capture recorded.
"""

from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import TokenCaptureStore, TokenEntry


PTOKS = [1, 2, 3]
GTOKS = [4, 5]
LPS = [-0.1, -0.2]


def _training_response(text: str = "hi") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model="m",
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
                "prompt_token_ids": PTOKS,
                "generation_token_ids": GTOKS,
                "generation_log_probs": LPS,
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _CapturingModel(SimpleResponsesAPIModel):
    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        return _training_response()

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        return NeMoGymChatCompletion.model_validate(
            {
                "id": f"chatcmpl_{uuid4().hex}",
                "created": int(time()),
                "model": "m",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "hi",
                            "prompt_token_ids": PTOKS,
                            "generation_token_ids": GTOKS,
                            "generation_log_probs": LPS,
                        },
                    }
                ],
            }
        )


def _client(tmp_path) -> TestClient:
    server = _CapturingModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="srv"),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"token_id_capture": {"enabled": True, "dir": str(tmp_path)}},
        ),
    )
    return TestClient(server.setup_webserver())


def test_responses_route_records_the_served_envelope_id(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/ng-rollout/rid-a/training-token-capture/v1/responses", json={"input": "hi"})
    assert resp.status_code == 200
    [entry] = TokenCaptureStore(tmp_path).read_entries("rid-a")
    assert entry.response_id == resp.json()["id"]
    assert entry.response_id and entry.response_id.startswith("resp_")


def test_chat_route_records_the_served_envelope_id(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/ng-rollout/rid-b/training-token-capture/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    [entry] = TokenCaptureStore(tmp_path).read_entries("rid-b")
    assert entry.response_id == resp.json()["id"]
    assert entry.response_id and entry.response_id.startswith("chatcmpl_")


def test_messages_route_serves_the_recorded_id_on_its_envelope(tmp_path):
    # The Anthropic mapping reuses the inner Responses id on its outer envelope,
    # so the id a Messages-dialect client keeps is the id capture recorded.
    client = _client(tmp_path)
    resp = client.post(
        "/ng-rollout/rid-c/training-token-capture/v1/messages",
        json={"model": "claude-x", "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    [entry] = TokenCaptureStore(tmp_path).read_entries("rid-c")
    assert entry.response_id == resp.json()["id"]


def test_anthropic_converter_reuses_the_responses_envelope_id():
    response = _training_response()
    message = AnthropicConverter().responses_to_anthropic_response(response, model="m")
    message_id = message["id"] if isinstance(message, dict) else message.id
    assert message_id == response.id


def test_record_without_response_id_uses_none():
    entry = TokenEntry.model_validate(
        {
            "schema_version": 1,
            "rollout_id": "r",
            "model_call_id": "c",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
        }
    )
    assert entry.response_id is None


def test_newer_record_is_rejected():
    with pytest.raises(Exception):
        TokenEntry.model_validate(
            {
                "schema_version": 99,
                "rollout_id": "r",
                "model_call_id": "c",
                "prompt_token_ids": [1],
                "generation_token_ids": [2],
                "generation_log_probs": [-0.1],
            }
        )
