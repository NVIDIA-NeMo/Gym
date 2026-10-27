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
from nemo_gym.openai_utils import NeMoGymResponse


PREFIX = "nemo_gym_responses_reasoning:"
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
    "encrypted", [CIPHERTEXT, "arbitrary-opaque-content", PREFIX + "nested", "[not-json", "opaque-λ\nbytes"]
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
    joined_summary = "\n\n".join(item["text"] for item in summaries)
    if summaries:
        assert blocks == [{"type": "thinking", "thinking": joined_summary, "signature": PREFIX + encrypted}]
    else:
        assert blocks == [{"type": "redacted_thinking", "data": PREFIX + encrypted}]
    replay = converter.anthropic_request_to_responses(
        {"max_tokens": 10, "messages": [{"role": "assistant", "content": message["content"]}]}
    )
    forwarded = replay.model_dump(mode="json", exclude_unset=True)["input"]
    assert [item["type"] for item in forwarded] == ["reasoning", "function_call"]
    assert forwarded[0]["summary"] == ([{"type": "summary_text", "text": joined_summary}] if summaries else [])
    assert forwarded[0]["encrypted_content"] == encrypted
    assert forwarded[1]["call_id"] == "call_synthetic"
    events = [json.loads(event.split("data: ", 1)[1]) for event in converter.anthropic_response_to_sse(message)]
    deltas = [event["delta"] for event in events if event["type"] == "content_block_delta" and event["index"] == 0]
    if summaries:
        assert deltas == [
            {"type": "thinking_delta", "thinking": joined_summary},
            {"type": "signature_delta", "signature": PREFIX + encrypted},
        ]
    else:
        assert events[1]["content_block"] == blocks[0]
        assert deltas == []


@pytest.mark.parametrize("encrypted", [None, ""])
@pytest.mark.parametrize("summaries", [[], [{"type": "summary_text", "text": "Readable."}]])
def test_absent_ciphertext_emits_no_carrier(encrypted, summaries):
    converter = AnthropicConverter()
    message = converter.responses_to_anthropic_response(response(encrypted, summaries), model="synthetic")
    expected = [{"type": "thinking", "thinking": "Readable.", "signature": ""}] if summaries else []
    assert message["content"][:-1] == expected
    replay = converter.anthropic_request_to_responses(
        {"max_tokens": 10, "messages": [{"role": "assistant", "content": message["content"]}]}
    )
    forwarded = replay.model_dump(mode="json", exclude_unset=True)["input"]
    assert len(forwarded) == (2 if summaries else 1)
    for item in forwarded[:-1]:
        assert item["summary"] == summaries
        assert item["encrypted_content"] is None


@pytest.mark.parametrize(
    "signature", [None, "", "native-signature", "litellm_encrypted_reasoning:opaque", PREFIX, 123]
)
def test_only_marked_nonempty_signatures_become_ciphertext(signature):
    request = AnthropicConverter().anthropic_request_to_responses(
        {
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Readable.", "signature": signature}],
                }
            ],
        }
    )
    item = request.model_dump(mode="json", exclude_unset=True)["input"][0]
    assert item["summary"] == [{"type": "summary_text", "text": "Readable."}]
    assert item["encrypted_content"] is None


@pytest.mark.parametrize(
    "block",
    [
        {"type": "redacted_thinking"},
        *({"type": "redacted_thinking", "data": value} for value in [None, "", "native-data", PREFIX, 123]),
    ],
)
def test_unsupported_redacted_history_is_dropped(block):
    request = AnthropicConverter().anthropic_request_to_responses(
        {
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Before"}, block, {"type": "text", "text": "After"}],
                }
            ],
        }
    )
    assert request.model_dump(mode="json", exclude_unset=True)["input"] == [
        {
            "role": "assistant",
            "type": "message",
            "content": [{"type": "input_text", "text": "Before"}, {"type": "input_text", "text": "After"}],
        }
    ]
