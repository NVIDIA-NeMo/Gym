# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.models.token_capture.harness_notes import OPENCODE_MAX_STEPS_PROMPT, trailing_harness_note


NOTE = OPENCODE_MAX_STEPS_PROMPT
TOOL_CALL = {"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}


@pytest.mark.parametrize(
    "last",
    [
        {"role": "assistant", "content": NOTE},
        {"role": "assistant", "content": [{"type": "text", "text": NOTE}]},
        {"role": "assistant", "content": NOTE, "tool_calls": [], "reasoning_content": None},
    ],
)
def test_the_exact_note_as_the_final_assistant_message_is_a_note(last):
    assert trailing_harness_note([{"role": "user", "content": "task"}, last]) == NOTE


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": NOTE}],
        [{"role": "assistant", "content": NOTE}, {"role": "tool", "tool_call_id": "call-1", "content": "x"}],
        [{"role": "assistant", "content": NOTE + "\n"}],
        [{"role": "assistant", "content": NOTE, "tool_calls": [TOOL_CALL]}],
        [{"role": "assistant", "content": NOTE, "reasoning_content": "thinking"}],
        [{"role": "assistant", "content": [{"type": "text", "text": NOTE}, {"type": "image_url", "image_url": {}}]}],
        [],
    ],
)
def test_anything_else_is_not_a_note(messages):
    assert trailing_harness_note(messages) is None
