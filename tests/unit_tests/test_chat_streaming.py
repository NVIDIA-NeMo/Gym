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

from nemo_gym.chat_streaming import drop_prompt_cache_hints
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming


def test_drop_prompt_cache_hints_only_touches_content_parts_and_tool_specs() -> None:
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": "plain string content"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"properties": {"cache_control": {"type": "string"}}}},
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }

    cleaned = drop_prompt_cache_hints(body)

    assert "cache_control" not in cleaned["messages"][0]["content"][0]
    assert cleaned["messages"][1] == body["messages"][1]
    assert "cache_control" not in cleaned["tools"][0]
    assert cleaned["tools"][0]["function"]["parameters"]["properties"] == {"cache_control": {"type": "string"}}
    assert body["tools"][0]["cache_control"] == {"type": "ephemeral"}
    NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(cleaned)
