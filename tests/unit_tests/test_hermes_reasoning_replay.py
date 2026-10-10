# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming


@pytest.mark.parametrize("reasoning", [None, "I should inspect the repository first."])
def test_hermes_assistant_history_preserves_reasoning_content(reasoning):
    payload = {
        "model": "policy",
        "messages": [
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": "inspect",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"pwd"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "inspect", "content": "/app"},
        ],
    }
    params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(payload)
    wire = params.model_dump(mode="json")
    assert wire["messages"][1]["reasoning_content"] == reasoning
    assert wire["messages"][2]["tool_call_id"] == "inspect"
