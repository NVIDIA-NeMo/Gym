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
"""Test the dialect-blind conversation fingerprints.

The same content served in Chat, Responses, or Anthropic shape must hash
identically: terminal attribution compares a harness-dialect transcript
against capture's normalized view, and the comparison is only sound if the
fingerprint erases the dialect.
"""

import hashlib
from types import SimpleNamespace

import pytest

from nemo_gym.models.token_capture import fingerprint
from nemo_gym.models.token_capture.fingerprint import (
    assistant_fingerprint,
    conversation_digest,
    tool_call_turn_fingerprint,
)


# An agent turn with a tool call and a tool result, each field longer than 2 KiB.
LONG_HISTORY = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Fix the failing test."},
    {
        "role": "assistant",
        "content": "x" * 3000,
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"cmd": "pytest"}'}}],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "y" * 5000},
    {"role": "assistant", "content": "Done."},
]


def test_the_three_dialect_shapes_hash_identically():
    chat = {"role": "assistant", "content": "hi"}
    responses = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    }
    anthropic = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    fingerprints = {assistant_fingerprint([shape]) for shape in (chat, responses, anthropic)}
    assert len(fingerprints) == 1 and fingerprints != {""}


def test_tool_calls_normalize_across_dialects():
    responses_call = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 2, "b": 1}'}
    chat_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"b":1,"a":2}'}}],
    }
    anthropic_call = {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "c1", "name": "f", "input": {"b": 1, "a": 2}}],
    }
    fingerprints = {assistant_fingerprint([shape]) for shape in (responses_call, chat_call, anthropic_call)}
    assert len(fingerprints) == 1 and fingerprints != {""}


def test_argument_reserialization_is_tolerated_but_argument_changes_are_not():
    base = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": 1, "b": 2}'}
    reserialized = dict(base, arguments='{"b":2,"a":1}')
    changed = dict(base, arguments='{"a": 1, "b": 3}')
    assert assistant_fingerprint([base]) == assistant_fingerprint([reserialized])
    assert assistant_fingerprint([base]) != assistant_fingerprint([changed])


def test_arguments_with_an_integer_wider_than_64_bits_still_fingerprint():
    base = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": '{"a": %d, "b": 2}' % 2**70}
    reserialized = dict(base, arguments='{"b":2,"a":%d}' % 2**70)
    changed = dict(base, arguments='{"a": %d, "b": 2}' % (2**70 + 1))
    assert assistant_fingerprint([base]) not in ("", None)
    assert assistant_fingerprint([base]) == assistant_fingerprint([reserialized])
    assert assistant_fingerprint([base]) != assistant_fingerprint([changed])


def test_reasoning_items_are_excluded():
    # A harness need not echo standalone reasoning items; including them would
    # make the fingerprint depend on dialect and echo behavior.
    message = {"role": "assistant", "content": "answer"}
    reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]}
    assert assistant_fingerprint([reasoning, message]) == assistant_fingerprint([message])


def test_no_model_authored_turn_yields_the_empty_fingerprint():
    assert assistant_fingerprint([{"role": "user", "content": "question"}]) == ""
    assert assistant_fingerprint([]) == ""


def test_conversation_digest_covers_tool_results_the_fingerprint_ignores():
    # A summarized or truncated tool result changes the request context, so the
    # digest must see it — while the assistant fingerprint must not.
    turns_a = [
        {"role": "assistant", "content": "calling"},
        {"type": "function_call_output", "call_id": "c1", "output": "result A"},
    ]
    turns_b = [
        {"role": "assistant", "content": "calling"},
        {"type": "function_call_output", "call_id": "c1", "output": "result B"},
    ]
    assert conversation_digest(turns_a) != conversation_digest(turns_b)
    assert assistant_fingerprint(turns_a) == assistant_fingerprint(turns_b)


def test_a_non_object_item_raises_for_the_caller_to_handle():
    with pytest.raises(ValueError):
        assistant_fingerprint([42])
    with pytest.raises(ValueError):
        conversation_digest(["not a dict"])


def test_conversation_digest_ignores_instruction_content_but_not_its_position():
    """OpenCode regenerates its system prompt per request with today's date.

    A rollout crossing a UTC day boundary changes only that text; the parent
    link must survive it. Conversation turns are still verified.
    """
    day1 = [
        {"role": "system", "content": "You are opencode. Today's date: Wed Sep 23 2026"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "looking"},
    ]
    day2 = [dict(day1[0], content="You are opencode. Today's date: Thu Sep 24 2026"), *day1[1:]]
    assert conversation_digest(day1) == conversation_digest(day2)
    developer = [{"role": "developer", "content": "v1"}, *day1[1:]]
    assert conversation_digest(developer) == conversation_digest([{"role": "developer", "content": "v2"}, *day1[1:]])
    # The instruction item still counts by role and position.
    assert conversation_digest(day1) != conversation_digest(day1[1:])
    assert conversation_digest(day1) != conversation_digest(developer)
    # Conversation turns are still covered.
    assert conversation_digest(day1) != conversation_digest([day1[0], dict(day1[1], content="fix the test"), day1[2]])
    assert conversation_digest(day1) != conversation_digest([*day1[:2], dict(day1[2], content="done")])
    # The assistant fingerprint never looked at instruction items.
    assert assistant_fingerprint(day1) == assistant_fingerprint(day2)


def test_digests_of_a_long_history_keep_their_values():
    """Lineage stores keep these digests across runs, so a change to how they are computed must not change them.

    The values were computed by hashing each field with its own sha256 update.
    """
    # SHA-256 digests, not credentials.
    fingerprint = "706557779489c60e4b53bfb6068591c2268754103d303174f077870017333bbf"  # pragma: allowlist secret
    context = "aeb8523a8bbbfa0d8ed08ca75347df903f27675eb573afdf544c461a21b602af"  # pragma: allowlist secret
    turn = "bb1a94ddb070f4d11db9272a6032a36dbcb5d28d7b285a9e4d1c7f0376204608"  # pragma: allowlist secret
    assert assistant_fingerprint(LONG_HISTORY) == fingerprint
    assert conversation_digest(LONG_HISTORY) == context
    assert tool_call_turn_fingerprint(LONG_HISTORY[2:3]) == turn


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param(lambda: assistant_fingerprint(LONG_HISTORY), id="assistant_fingerprint"),
        pytest.param(lambda: conversation_digest(LONG_HISTORY), id="conversation_digest"),
        pytest.param(lambda: tool_call_turn_fingerprint(LONG_HISTORY[2:3]), id="tool_call_turn_fingerprint"),
    ],
)
def test_a_long_history_is_hashed_in_one_update(monkeypatch, digest):
    """On Python 3.13 and 3.14, a hash object releases the GIL on every update after one of 2 KiB or more.

    One update per digest keeps a long history from giving up the GIL three times per field.
    """
    updates = []

    class CountingSha256:
        def __init__(self, data=b""):
            self._hasher = hashlib.sha256(data)

        def update(self, data):
            updates.append(len(data))
            self._hasher.update(data)

        def hexdigest(self):
            return self._hasher.hexdigest()

    monkeypatch.setattr(fingerprint, "hashlib", SimpleNamespace(sha256=CountingSha256))
    digest()
    assert len(updates) == 1
