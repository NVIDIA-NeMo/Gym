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
import pytest
from pydantic import ValidationError

from nemo_gym.anthropic_utils import (
    NeMoGymAnthropicMessage,
    NeMoGymAnthropicMessageCreateParamsNonStreaming,
    NeMoGymAnthropicMessageForTraining,
)


class TestNeMoGymAnthropicMessageCreateParamsNonStreaming:
    def test_minimal_request(self) -> None:
        params = NeMoGymAnthropicMessageCreateParamsNonStreaming(
            max_tokens=16, messages=[{"role": "user", "content": "hi"}]
        )
        # model is optional so the model server can fill it from its own config.
        assert params.model is None
        assert params.tools == []

    def test_tool_calling_history(self) -> None:
        """A text + tool_use + tool_result conversation round-trips through the strict schema."""
        params = NeMoGymAnthropicMessageCreateParamsNonStreaming(
            max_tokens=1024,
            system="You are helpful.",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "Paris"}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}]},
            ],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            thinking={"type": "adaptive"},
            tool_choice={"type": "auto"},
        )
        assert len(params.messages) == 3
        assert len(params.tools) == 1

    def test_unknown_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymAnthropicMessageCreateParamsNonStreaming(
                max_tokens=16, messages=[{"role": "user", "content": "hi"}], not_a_real_field=1
            )

    def test_max_tokens_required(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymAnthropicMessageCreateParamsNonStreaming(messages=[{"role": "user", "content": "hi"}])

    def test_training_input_message(self) -> None:
        """Assistant turns in history may carry token IDs + log probs for training."""
        params = NeMoGymAnthropicMessageCreateParamsNonStreaming(
            max_tokens=16,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hi"}],
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                }
            ],
        )
        assert params.messages[0]["generation_token_ids"] == [3]


class TestNeMoGymAnthropicMessage:
    _RESPONSE = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "content": [{"type": "text", "text": "Sunny, 72F."}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    def test_response_round_trip(self) -> None:
        resp = NeMoGymAnthropicMessage.model_validate(self._RESPONSE)
        assert resp.content[0].type == "text"
        assert resp.content[0].text == "Sunny, 72F."
        assert resp.usage.output_tokens == 5

    def test_response_training_variant(self) -> None:
        resp = NeMoGymAnthropicMessageForTraining.model_validate(
            self._RESPONSE | {"prompt_token_ids": [1], "generation_token_ids": [2], "generation_log_probs": [-0.2]}
        )
        assert resp.generation_token_ids == [2]
