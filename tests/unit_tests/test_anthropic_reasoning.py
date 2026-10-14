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
"""Opaque Responses ciphertext survives an Anthropic client and streaming replay."""

import json

import pytest

from nemo_gym.anthropic_converter import AnthropicConverter
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


PREFIX = "litellm_encrypted_reasoning:"
CIPHERTEXT = '[{"type":"thinking","thinking":"Synthetic reasoning","signature":"synthetic-signature"}]'


def response(encrypted, summaries):
    return NeMoGymResponse.model_validate(
        {
            "id": "resp_synthetic",
            "created_at": 0,
            "model": "synthetic",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {"type": "reasoning", "id": "rs_synthetic", "encrypted_content": encrypted, "summary": summaries},
                {
                    "type": "function_call",
                    "id": "fc_synthetic",
                    "call_id": "call_synthetic",
                    "name": "shell",
                    "arguments": "{}",
                },
            ],
        }
    )


@pytest.mark.parametrize(
    "encrypted", [CIPHERTEXT, "arbitrary-opaque-content", PREFIX + "nested", "[not-json", "", "opaque-λ\nbytes"]
)
@pytest.mark.parametrize(
    "summaries",
    [
        [],
        [{"type": "summary_text", "text": "Synthetic summary."}],
        [{"type": "summary_text", "text": "First."}, {"type": "summary_text", "text": "Second."}],
    ],
)
def test_opaque_carrier_round_trips_without_duplication_or_reordering(encrypted, summaries):
    converter = AnthropicConverter()
    message = converter.responses_to_anthropic_response(response(encrypted, summaries), model="synthetic")
    blocks = message["content"][:-1]
    assert message["content"][-1]["type"] == "tool_use"
    if summaries:
        assert [block["thinking"] for block in blocks] == [item["text"] for item in summaries]
        assert blocks[0]["signature"] == PREFIX + encrypted
        assert all(block["signature"] == "" for block in blocks[1:])
    else:
        assert blocks == [{"type": "redacted_thinking", "data": PREFIX + encrypted}]
    replay = converter.anthropic_request_to_responses(
        {"max_tokens": 10, "messages": [{"role": "assistant", "content": message["content"]}]}
    )
    assert all(item.type == "reasoning" for item in replay.input[:-1])
    assert all(
        item["type"] == "reasoning" for item in replay.model_dump(mode="json", exclude_unset=True)["input"][:-1]
    )
    assert replay.input[0].encrypted_content == encrypted
    assert all(item.encrypted_content in (None, "") for item in replay.input[1:-1])
    assert replay.input[-1].call_id == "call_synthetic"
    assert [summary.text for item in replay.input[:-1] for summary in item.summary] == [
        item["text"] for item in summaries
    ]
    events = [json.loads(event.split("data: ", 1)[1]) for event in converter.anthropic_response_to_sse(message)]
    for index, block in enumerate(blocks):
        deltas = [
            event["delta"] for event in events if event["type"] == "content_block_delta" and event["index"] == index
        ]
        if block["type"] == "thinking":
            assert deltas == [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            starts = [event["content_block"] for event in events if event["type"] == "content_block_start"]
            assert starts[index] == block
            assert deltas == []


def test_native_provider_keeps_unmarked_signature():
    converter = AnthropicConverter()
    request = converter.anthropic_request_to_responses(
        {
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Synthetic reasoning", "signature": "native-signature"}
                    ],
                }
            ],
        }
    )
    assert request.input[0].encrypted_content == "native-signature"
    provider = converter.responses_to_anthropic(
        request, model="synthetic", max_tokens=10, thinking=None, thinking_budget_tokens=None, extra_body={}
    )
    assert provider["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "Synthetic reasoning", "signature": "native-signature"}
    ]


def test_native_provider_rejects_encrypted_only_without_payload():
    converter = AnthropicConverter()
    request = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": [{"type": "reasoning", "id": "rs_synthetic", "encrypted_content": "opaque-secret"}]}
    )
    with pytest.raises(ValueError, match="^Encrypted-only reasoning cannot be sent to a native Anthropic provider$"):
        converter.responses_to_anthropic(
            request, model="synthetic", max_tokens=10, thinking=None, thinking_budget_tokens=None, extra_body={}
        )
