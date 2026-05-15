# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``stirrup_utils.convert_stirrup_history_to_output_items``.

Focuses on the rollout-output coverage of tool results: ``ToolMessage``
and ``NeMoUserMessage`` (the ``UserMessage`` subclass produced when
``tool_response_as_user=True``) must both materialise as
``function_call_output`` items, with ``call_id`` paired to the
upstream ``function_call``.
"""

from __future__ import annotations

from stirrup.core.models import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from responses_api_agents.stirrup_agent.nemo_agent import NeMoUserMessage
from responses_api_agents.stirrup_agent.stirrup_utils import (
    convert_stirrup_history_to_output_items,
)


def _types(items):
    return [it.type for it in items]


def _call_ids(items, item_type):
    return [it.call_id for it in items if it.type == item_type]


def test_tool_message_emits_function_call_output_paired_with_call() -> None:
    history = [
        [
            SystemMessage(content="sys"),
            UserMessage(content="user prompt"),
        ],
        [
            AssistantMessage(
                content="thinking",
                tool_calls=[ToolCall(name="code_exec", arguments='{"code": "1+1"}', tool_call_id="abc123")],
            ),
            ToolMessage(content="2", tool_call_id="abc123", name="code_exec", success=True),
        ],
    ]
    input_items, output_items = convert_stirrup_history_to_output_items(history)

    assert _types(input_items) == ["message", "message"]
    assert _types(output_items) == ["message", "function_call", "function_call_output"]
    assert _call_ids(output_items, "function_call") == ["abc123"]
    assert _call_ids(output_items, "function_call_output") == ["abc123"]


def test_nemo_user_message_with_tool_call_id_emits_function_call_output() -> None:
    """``NeMoUserMessage`` is a ``UserMessage`` subclass; without the
    tool_call_id-based dispatch its tool output would either be dropped or
    misclassified as a regular user message."""
    history = [
        [
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(name="code_exec", arguments="{}", tool_call_id="tc-1")],
            ),
            NeMoUserMessage(content="stdout: ok", tool_call_id="tc-1", name="code_exec", success=True),
        ],
    ]
    _, output_items = convert_stirrup_history_to_output_items(history)

    assert _types(output_items) == ["function_call", "function_call_output"]
    assert _call_ids(output_items, "function_call") == ["tc-1"]
    assert _call_ids(output_items, "function_call_output") == ["tc-1"]
    fco = output_items[1]
    assert fco.output == "stdout: ok"


def test_plain_user_message_routes_to_input_items() -> None:
    """A regular ``UserMessage`` (no ``tool_call_id``) must land in input_items,
    not be misclassified as a tool result."""
    history = [[UserMessage(content="hello")]]
    input_items, output_items = convert_stirrup_history_to_output_items(history)

    assert len(input_items) == 1
    assert input_items[0].role == "user"
    assert input_items[0].content == "hello"
    assert output_items == []


def test_assistant_tool_call_uses_stirrup_tool_call_id() -> None:
    """``ToolCall.tool_call_id`` is the canonical id field; verify that the
    function_call output uses it (rather than falling back to a random uuid)
    so it pairs with the corresponding function_call_output."""
    history = [
        [
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(name="run", arguments="{}", tool_call_id="real-id-7")],
            ),
        ],
    ]
    _, output_items = convert_stirrup_history_to_output_items(history)
    assert _call_ids(output_items, "function_call") == ["real-id-7"]
