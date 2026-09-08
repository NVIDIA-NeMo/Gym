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

import pytest

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest


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
