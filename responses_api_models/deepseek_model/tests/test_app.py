# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from responses_api_models.deepseek_model.app import DeepSeekModel
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def make_server(model_class: type[SimpleModelServer] = DeepSeekModel) -> SimpleModelServer:
    return model_class(
        config=SimpleModelServerConfig(
            host="127.0.0.1",
            port=8081,
            name="policy_model",
            entrypoint="app.py",
            openai_base_url="https://example.invalid/v1",
            openai_api_key="test-only",
            openai_model="deepseek-v4-flash",
        ),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking_type", ["enabled", "disabled"])
def test_deepseek_fields_and_reasoning_tool_history_reach_backend(stream: bool, thinking_type: str) -> None:
    server = make_server()
    server._client = MagicMock(
        create_chat_completion=AsyncMock(
            return_value={
                "id": "chat_test",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "4", "reasoning_content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            }
        )
    )
    messages = [
        {"role": "user", "content": "Compute 2 + 2."},
        {
            "role": "assistant",
            "reasoning_content": "Use a tool.",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"echo 4"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4"},
    ]
    body = {
        "model": "requested",
        "messages": messages,
        "reasoning_effort": "max",
        "thinking": {"type": thinking_type},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    sent = server._client.create_chat_completion.call_args.kwargs
    assert sent["messages"] == messages
    assert sent["reasoning_effort"] == "max"
    assert sent["thinking"] == {"type": thinking_type}
    assert sent["model"] == "deepseek-v4-flash"
    assert sent.get("stream", False) is False
    assert "stream_options" not in sent
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert any(c["choices"] and c["choices"][0]["delta"].get("reasoning_content") == "Done." for c in chunks)
        assert any(c.get("usage", {}).get("total_tokens") == 23 for c in chunks if c.get("usage"))
        assert response.text.endswith("data: [DONE]\n\n")
    else:
        assert response.json()["choices"][0]["message"]["reasoning_content"] == "Done."


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields", [{"reasoning_effort": "invalid"}, {"thinking": {"type": "invalid"}}])
def test_deepseek_extensions_are_validated(stream: bool, fields: dict) -> None:
    server = make_server()
    server._client = MagicMock(create_chat_completion=AsyncMock())
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": stream,
                **fields,
            },
        )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", next(iter(fields))]
    server._client.create_chat_completion.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
def test_openai_model_still_rejects_deepseek_max_effort(stream: bool) -> None:
    server = make_server(SimpleModelServer)
    server._client = MagicMock(create_chat_completion=AsyncMock())
    with TestClient(server.setup_webserver()) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "max",
                "stream": stream,
            },
        )
    assert response.status_code == 422
    server._client.create_chat_completion.assert_not_called()
