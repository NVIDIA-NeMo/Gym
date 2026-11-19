# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

import pytest

from nemo_gym.models.token_capture.harness_notes import HARNESS_NOTES, trailing_harness_note
from nemo_gym.models.token_capture.harness_notes.opencode import OPENCODE_MAX_STEPS_PROMPT


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


@pytest.mark.parametrize("note", HARNESS_NOTES, ids=lambda note: f"{note.harness}-{note.name}")
def test_every_registered_note_is_recognized(note):
    assert trailing_harness_note(
        [{"role": "user", "content": "task"}, {"role": "assistant", "content": note.text}]
    ) == (note.text)


def test_registered_notes_are_distinct_and_name_the_releases_checked():
    assert len({(note.harness, note.name) for note in HARNESS_NOTES}) == len(HARNESS_NOTES)
    assert len({note.text for note in HARNESS_NOTES}) == len(HARNESS_NOTES)
    for note in HARNESS_NOTES:
        assert note.text.strip()
        assert note.checked_versions
        assert all(re.fullmatch(r"\d+\.\d+\.\d+", version) for version in note.checked_versions)
