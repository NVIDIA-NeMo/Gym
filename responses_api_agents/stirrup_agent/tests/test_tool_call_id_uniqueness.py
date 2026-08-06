# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call_id must identify one call, or downstream pairing attributes the wrong output.

Some serving layers mint ``tool_call_id`` per turn ("code_exec:0"), so a long trajectory
repeats the same id dozens of times. Real runs carry 268-300 duplicate ids each. Replay,
training and analysis code that pairs ``function_call`` to ``function_call_output`` by id
then silently matches a call to some other turn's output.
"""

from stirrup.core.models import AssistantMessage, ToolCall, ToolMessage

from responses_api_agents.stirrup_agent.stirrup_utils import convert_stirrup_history_to_output_items


def _turn(call_id: str, args: str, output: str) -> list:
    """One assistant tool call plus the tool result that answers it."""
    return [
        AssistantMessage(content="", tool_calls=[ToolCall(tool_call_id=call_id, name="code_exec", arguments=args)]),
        ToolMessage(content=output, name="code_exec", tool_call_id=call_id),
    ]


def _calls_and_outputs(items):
    calls = [i for i in items if getattr(i, "type", None) == "function_call"]
    outputs = [i for i in items if getattr(i, "type", None) == "function_call_output"]
    return calls, outputs


def test_repeated_ids_across_turns_become_unique():
    history = [_turn("code_exec:0", f"arg{i}", f"out{i}") for i in range(4)]

    _, items = convert_stirrup_history_to_output_items(history)
    calls, _ = _calls_and_outputs(items)

    ids = [c.call_id for c in calls]
    assert len(set(ids)) == len(ids) == 4, f"call ids are not unique: {ids}"


def test_each_call_still_pairs_with_its_own_output():
    """Uniqueness is worthless if it breaks the pairing it exists to protect."""
    history = [_turn("code_exec:0", f"arg{i}", f"out{i}") for i in range(4)]

    _, items = convert_stirrup_history_to_output_items(history)
    calls, outputs = _calls_and_outputs(items)

    by_id = {o.call_id: o.output for o in outputs}
    for i, call in enumerate(calls):
        assert call.arguments == f"arg{i}"
        assert by_id[call.call_id] == f"out{i}", (
            f"call with arguments {call.arguments!r} resolved to output {by_id[call.call_id]!r}"
        )


def test_distinct_ids_are_left_alone():
    history = [_turn(f"code_exec:{i}", f"arg{i}", f"out{i}") for i in range(3)]

    _, items = convert_stirrup_history_to_output_items(history)
    calls, outputs = _calls_and_outputs(items)

    assert [c.call_id for c in calls] == ["code_exec:0", "code_exec:1", "code_exec:2"]
    assert [o.call_id for o in outputs] == ["code_exec:0", "code_exec:1", "code_exec:2"]


def test_interleaved_tools_pair_independently():
    """Two tools each reusing their own id must not cross-contaminate."""
    history = [
        _turn("code_exec:0", "code-a", "code-out-a"),
        _turn("web_search:0", "search-a", "search-out-a"),
        _turn("code_exec:0", "code-b", "code-out-b"),
        _turn("web_search:0", "search-b", "search-out-b"),
    ]

    _, items = convert_stirrup_history_to_output_items(history)
    calls, outputs = _calls_and_outputs(items)

    by_id = {o.call_id: o.output for o in outputs}
    resolved = {c.arguments: by_id[c.call_id] for c in calls}
    assert resolved == {
        "code-a": "code-out-a",
        "code-b": "code-out-b",
        "search-a": "search-out-a",
        "search-b": "search-out-b",
    }
