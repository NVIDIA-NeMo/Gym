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
"""``priority_by_turn``: every engine call carries vLLM's ``priority`` counted from its assistant turns."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import mark, raises

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig, turn_priority


_CHAT_COMPLETION = {
    "id": "chtcmpl",
    "object": "chat.completion",
    "created": 1691418000,
    "model": "dummy_model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "response", "tool_calls": []},
        }
    ],
}

_COMPLETION = {
    "id": "cmpl",
    "object": "text_completion",
    "created": 1691418000,
    "model": "dummy_model",
    "choices": [{"index": 0, "finish_reason": "stop", "text": "response", "logprobs": None}],
}


def _config(**overrides: Any) -> VLLMModelConfig:
    return VLLMModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url="http://api.openai.com/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        entrypoint="",
        name="",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        **overrides,
    )


def _server(**overrides: Any) -> tuple[TestClient, dict[str, Any]]:
    """A served VLLMModel whose engine client records the body of every call it is sent."""
    server = VLLMModel(config=_config(**overrides), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    sent: dict[str, Any] = {}

    async def create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        sent.update(kwargs)
        return json.loads(json.dumps(_CHAT_COMPLETION))

    async def create_completion(**kwargs: Any) -> dict[str, Any]:
        sent.update(kwargs)
        return json.loads(json.dumps(_COMPLETION))

    client = MagicMock(spec=NeMoGymAsyncOpenAI)
    client.create_chat_completion = AsyncMock(side_effect=create_chat_completion)
    client.create_completion = AsyncMock(side_effect=create_completion)
    server._clients = [client]
    return TestClient(server.setup_webserver()), sent


_FIRST_TURN = [{"role": "user", "content": "hi"}]
_THIRD_TURN = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "again"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "found"},
]


def test_turn_priority_counts_assistant_messages() -> None:
    assert turn_priority([]) == 0
    assert turn_priority(_FIRST_TURN) == 0
    assert turn_priority(_THIRD_TURN) == -2


@mark.parametrize("messages, priority", [(_FIRST_TURN, 0), (_THIRD_TURN, -2)])
def test_chat_completions_send_the_turn_priority(messages: list[dict[str, Any]], priority: int) -> None:
    client, sent = _server(priority_by_turn=True)

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": messages})

    assert response.status_code == 200, response.text
    assert sent["priority"] == priority


def test_no_priority_is_sent_by_default() -> None:
    client, sent = _server()

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": _THIRD_TURN})

    assert response.status_code == 200, response.text
    assert "priority" not in sent


def test_responses_send_the_priority_of_the_converted_messages() -> None:
    client, sent = _server(priority_by_turn=True)
    body = {
        "input": [
            {"type": "message", "role": "user", "content": "look both up"},
            {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "call_1"},
            {"type": "function_call_output", "call_id": "call_1", "output": "first"},
            {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "call_2"},
            {"type": "function_call_output", "call_id": "call_2", "output": "second"},
        ],
    }

    response = client.post("/v1/responses", json=body)

    assert response.status_code == 200, response.text
    assert [message["role"] for message in sent["messages"]].count("assistant") == 2
    assert sent["priority"] == -2


def test_a_configured_extra_body_priority_is_kept() -> None:
    client, sent = _server(priority_by_turn=True, extra_body={"priority": 7})

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": _THIRD_TURN})

    assert response.status_code == 200, response.text
    assert sent["priority"] == 7


def test_a_request_extra_body_priority_is_kept() -> None:
    client, sent = _server(priority_by_turn=True)
    body = {
        "input": [{"type": "message", "role": "user", "content": "hi"}],
        "metadata": {"extra_body": json.dumps({"priority": 3})},
    }

    response = client.post("/v1/responses", json=body)

    assert response.status_code == 200, response.text
    assert sent["priority"] == 3


def test_the_completions_api_sends_the_turn_priority() -> None:
    client, sent = _server(priority_by_turn=True, use_completions_api=True)

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": _FIRST_TURN})

    assert response.status_code == 200, response.text
    assert sent["prompt"] == "hi"
    assert sent["priority"] == 0


@mark.parametrize("extra_body, priority", [(None, -2), ({"priority": 5}, 5)])
def test_the_completions_body_counts_the_rendered_history(extra_body: dict[str, Any] | None, priority: int) -> None:
    server = VLLMModel(
        config=_config(priority_by_turn=True, use_completions_api=True, extra_body=extra_body),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )

    body = server._build_completion_body_from_chat_body({"messages": _THIRD_TURN}, "<rendered>")

    assert body["prompt"] == "<rendered>"
    assert body["priority"] == priority


def test_priority_by_turn_is_refused_on_the_native_responses_route() -> None:
    with raises(ValueError, match="priority_by_turn requires is_responses_native=false"):
        _config(priority_by_turn=True, is_responses_native=True)


def _dynamo_extras(sent: dict[str, Any]) -> dict[str, Any]:
    (annotation,) = sent["nvext"]["annotations"]
    assert annotation.startswith("nemo_rl:gym:")
    return json.loads(annotation.removeprefix("nemo_rl:gym:"))


@mark.parametrize(
    "messages, extra_body, priority",
    [(_FIRST_TURN, None, 0), (_THIRD_TURN, None, -2), (_THIRD_TURN, {"priority": 7}, 7)],
)
def test_a_dynamo_frontend_call_carries_the_priority_in_its_annotation(
    messages: list[dict[str, Any]], extra_body: dict[str, Any] | None, priority: int
) -> None:
    # The Dynamo frontend refuses a top-level priority; NeMo RL's chat engine reads it from the annotation.
    # A first call's 0 must move too, or the frontend refuses every first call.
    client, sent = _server(priority_by_turn=True, dynamo_frontend=True, extra_body=extra_body)

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": messages})

    assert response.status_code == 200, response.text
    assert "priority" not in sent
    assert _dynamo_extras(sent) == {"priority": priority}


def test_a_dynamo_frontend_call_without_a_priority_annotates_none() -> None:
    client, sent = _server(dynamo_frontend=True)

    response = client.post("/v1/chat/completions", json={"model": "dummy_model", "messages": _THIRD_TURN})

    assert response.status_code == 200, response.text
    assert _dynamo_extras(sent) == {}
