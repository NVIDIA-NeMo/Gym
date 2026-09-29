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
"""Tests for the Anthropic Messages API boundary types in nemo_gym/anthropic_utils.py."""

import pytest
from pydantic import ValidationError

from nemo_gym.anthropic_utils import (
    NeMoGymAnthropicMessage,
    NeMoGymAnthropicMessageCreateParamsNonStreaming,
)


class TestNeMoGymAnthropicMessageCreateParamsNonStreaming:
    def test_validates_request_and_eagerly_materializes_iterables(self) -> None:
        req = NeMoGymAnthropicMessageCreateParamsNonStreaming.model_validate(
            {
                "max_tokens": 1024,
                "model": "claude-opus-4-8",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "be terse",
                "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
                "stop_sequences": ["STOP"],
                "temperature": 0.2,
            }
        )
        # Iterable fields are real lists, not single-use lazy ValidatorIterators.
        assert isinstance(req.messages, list)
        assert isinstance(req.tools, list)
        assert len(req.messages) == 1

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymAnthropicMessageCreateParamsNonStreaming.model_validate(
                {"max_tokens": 1, "model": "claude-opus-4-8", "messages": [], "bogus": 1}
            )

    def test_requires_max_tokens(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymAnthropicMessageCreateParamsNonStreaming.model_validate(
                {"model": "claude-opus-4-8", "messages": []}
            )


class TestNeMoGymAnthropicMessage:
    def test_validates_response_and_subclasses_sdk_message(self) -> None:
        from anthropic.types import Message

        msg = NeMoGymAnthropicMessage.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        )
        assert isinstance(msg, Message)
        assert isinstance(msg.content, list)
        assert msg.content[0].text == "hello"
