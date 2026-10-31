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
"""Normalize the demonstrated provider content shape at the response boundary."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import (
    SimpleModelServer,
    SimpleModelServerConfig,
    _normalize_reasoning_content,
)


def _response(content: object) -> dict:
    return {
        "id": "resp_synthetic",
        "created_at": 0,
        "model": "synthetic-model",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "reasoning": {"effort": "high"},
        "output": [
            {
                "type": "reasoning",
                "id": "rs_synthetic",
                "summary": [],
                "content": content,
                "encrypted_content": "opaque-λ\nsynthetic-ciphertext",
            },
            {
                "type": "function_call",
                "id": "fc_synthetic",
                "call_id": "call_synthetic",
                "name": "shell",
                "arguments": "{}",
            },
            {
                "type": "message",
                "id": "msg_synthetic",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Answer", "annotations": []}],
            },
        ],
        "usage": {
            "input_tokens": 4,
            "output_tokens": 6,
            "total_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }


def _server(response: dict) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        entrypoint="",
        name="synthetic",
        openai_base_url="https://example.invalid/v1",
        openai_api_key="synthetic",  # pragma: allowlist secret
        openai_model="synthetic-model",
    )
    server = SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    server._client = MagicMock(create_response=AsyncMock(return_value=response))
    return server


@pytest.mark.parametrize("text", ["Thoughts.\nMore thoughts.", ""])
@pytest.mark.parametrize("extra", [{}, {"annotations": []}])
async def test_provider_response_is_corrected_before_strict_validation(text: str, extra: dict) -> None:
    response = _response([{"type": "output_text", "text": text, **extra}])
    untouched = deepcopy(response)
    with pytest.raises(ValidationError):
        NeMoGymResponse.model_validate(response)
    server = _server(response)
    result = await server.responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))
    expected = deepcopy(response)
    expected["output"][0]["content"][0]["type"] = "reasoning_text"
    assert result.model_dump() == NeMoGymResponse.model_validate(expected).model_dump()
    assert response == untouched
    server._client.create_response.assert_awaited_once_with(input="Synthetic request", model="synthetic-model")


async def test_continuation_request_is_forwarded_without_normalization() -> None:
    response = _response([{"type": "output_text", "text": "Provider reasoning"}])
    request_item = {
        "type": "reasoning",
        "id": "rs_history",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": "Earlier reasoning"}],
        "encrypted_content": "opaque-history",
    }
    body = NeMoGymResponseCreateParamsNonStreaming(input=[request_item])
    server = _server(response)
    await server.responses(body)
    server._client.create_response.assert_awaited_once_with(input=[request_item], model="synthetic-model")
    assert body.model_dump(exclude_unset=True)["input"] == [request_item]


def test_copy_on_write_preserves_opaque_fields_and_unchanged_objects() -> None:
    response = _response(
        [
            {"type": "output_text", "text": "First", "annotations": []},
            {"type": "reasoning_text", "text": "Already valid"},
            {"type": "output_text", "text": "Last"},
        ]
    )
    response["provider_extension"] = {"opaque": [1, 2]}
    original = deepcopy(response)
    expected = deepcopy(response)
    for index in (0, 2):
        expected["output"][0]["content"][index]["type"] = "reasoning_text"
    result = _normalize_reasoning_content(response)
    assert result == expected
    assert response == original
    assert result is not response
    assert result["output"] is not response["output"]
    assert result["output"][0] is not response["output"][0]
    assert result["output"][0]["content"] is not response["output"][0]["content"]
    assert result["output"][0]["content"][0] is not response["output"][0]["content"][0]
    assert result["output"][0]["content"][0]["annotations"] is response["output"][0]["content"][0]["annotations"]
    assert result["output"][0]["content"][1] is response["output"][0]["content"][1]
    assert result["output"][1] is response["output"][1]
    assert result["output"][2] is response["output"][2]
    assert result["usage"] is response["usage"]
    assert result["provider_extension"] is response["provider_extension"]


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "output_text", "text": "Unknown field", "extra": True}],
        [{"type": "output_text", "text": "Annotation", "annotations": [{"type": "unknown"}]}],
        [{"type": "output_text", "text": "Null annotations", "annotations": None}],
        [{"type": "output_text", "text": "Wrong annotations", "annotations": {}}],
        [{"type": "output_text", "text": 42}],
        [{"type": "output_text", "text": None}],
        [{"type": "output_text"}],
        [{"type": "unknown", "text": "Unknown type"}],
        ["Not a content object"],
        {"type": "output_text", "text": "Not a list"},
        ({"type": "output_text", "text": "Not a list"},),
    ],
)
async def test_unsupported_content_is_left_for_native_validation(content: object) -> None:
    response = _response(content)
    original = deepcopy(response)
    assert _normalize_reasoning_content(response) is response
    with pytest.raises(ValidationError):
        await _server(response).responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))
    assert response == original


@pytest.mark.parametrize("summary", [None, "missing"])
async def test_content_correction_keeps_summary_validation(summary: object) -> None:
    response = _response([{"type": "output_text", "text": "Reasoning"}])
    if summary == "missing":
        del response["output"][0]["summary"]
    else:
        response["output"][0]["summary"] = summary
    with pytest.raises(ValidationError):
        await _server(response).responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))


@pytest.mark.parametrize("content", [None, [], [{"type": "reasoning_text", "text": "Valid"}]])
async def test_valid_reasoning_messages_and_tools_are_unchanged(content: object) -> None:
    response = _response(content)
    assert _normalize_reasoning_content(response) is response
    result = await _server(response).responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))
    assert result.model_dump() == NeMoGymResponse.model_validate(response).model_dump()


@pytest.mark.parametrize("response", [None, [], {}, {"output": None}, {"output": {}}, {"output": ["unknown", None]}])
def test_other_response_shapes_are_unchanged(response: object) -> None:
    assert _normalize_reasoning_content(response) is response


async def test_multiple_reasoning_items_keep_order_and_malformed_neighbors_fail() -> None:
    response = _response([{"type": "output_text", "text": "Before tool"}])
    second = deepcopy(response["output"][0])
    second["id"] = "rs_second"
    second["content"] = [{"type": "output_text", "text": "After tool"}]
    response["output"].insert(2, second)
    result = await _server(response).responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))
    assert [item.type for item in result.output] == ["reasoning", "function_call", "reasoning", "message"]
    assert [result.output[index].content[0].text for index in (0, 2)] == ["Before tool", "After tool"]
    response["output"][2]["content"].append({"type": "output_text", "text": "Unsupported", "extra": True})
    original = deepcopy(response)
    with pytest.raises(ValidationError):
        await _server(response).responses(NeMoGymResponseCreateParamsNonStreaming(input="Synthetic request"))
    assert response == original
