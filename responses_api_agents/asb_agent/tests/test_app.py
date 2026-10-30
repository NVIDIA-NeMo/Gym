# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan parsing, attacker-tool injection, tool dispatch and memory retrieval."""

from __future__ import annotations

import json

from responses_api_agents.asb_agent.app import (
    inject_attacker_tool_into_workflow,
    normalize_tool_calls,
    parse_workflow,
    responses_tool_schema,
    retrieve_memory,
)


PLAN = [
    {"message": "gather data", "tool_use": ["sys_monitor"]},
    {"message": "summarize", "tool_use": []},
]


def test_strict_json_plan_parses_on_the_upstream_path():
    workflow, path = parse_workflow(json.dumps(PLAN))
    assert workflow == PLAN
    assert path == "strict"


def test_fenced_and_embedded_plans_are_salvaged_and_labelled():
    """Salvage paths are reported so a run leaning on them is visible, not silent."""
    fenced, path = parse_workflow(f"```json\n{json.dumps(PLAN)}\n```")
    assert fenced == PLAN and path == "fenced"

    embedded, path = parse_workflow(f"Here is the plan:\n{json.dumps(PLAN)}\nHope that helps.")
    assert embedded == PLAN and path == "embedded"


def test_plan_missing_required_keys_is_a_workflow_failure():
    """``check_workflow`` requires both keys on every step."""
    workflow, path = parse_workflow(json.dumps([{"message": "no tools key"}]))
    assert workflow is None and path == "failed"


def test_non_json_reply_is_a_workflow_failure():
    assert parse_workflow("I cannot help with that.") == (None, "failed")


def test_single_step_object_is_wrapped_into_a_list():
    workflow, _ = parse_workflow(json.dumps({"message": "one step", "tool_use": []}))
    assert workflow == [{"message": "one step", "tool_use": []}]


def test_attacker_tool_injection_flattens_tools_across_every_stage():
    """Upstream unions all stages' tools and assigns that set to each stage.

    The consequence is load-bearing: the attacker tool becomes callable at *every* step,
    including ones the model planned as tool-free.
    """
    injected = inject_attacker_tool_into_workflow(PLAN, "ResourceAllocationHijack")
    assert len(injected) == len(PLAN)
    for stage in injected:
        assert set(stage["tool_use"]) == {"sys_monitor", "ResourceAllocationHijack"}
    # The original plan is not mutated in place.
    assert PLAN[1]["tool_use"] == []


def test_tool_call_normalization_handles_the_shapes_upstream_accepts():
    assert normalize_tool_calls("sys_monitor") == [{"name": "sys_monitor", "parameters": None}]
    assert normalize_tool_calls({"name": "a", "parameters": {"x": 1}}) == [{"name": "a", "parameters": {"x": 1}}]
    assert normalize_tool_calls({"function": {"name": "b", "arguments": "{}"}}) == [{"name": "b", "parameters": "{}"}]
    assert normalize_tool_calls({"tool_use": ["c", "d"]}) == [
        {"name": "c", "parameters": None},
        {"name": "d", "parameters": None},
    ]
    assert normalize_tool_calls(None) == []
    assert normalize_tool_calls([{"name": "   "}]) == []


def test_responses_schema_supplies_an_object_schema_for_paramless_tools():
    """ASB declares ``parameters: None``; the Responses API requires a schema object.

    ASB never passes arguments, so an empty object is behaviorally identical -- and the
    prompt text keeps the original schema, which is what the model actually reads.
    """
    chat = [
        {"type": "function", "function": {"name": "sys_monitor", "description": "d", "parameters": None}},
        {"type": "function", "function": {"name": "Attacker", "description": "d"}},
    ]
    converted = responses_tool_schema(chat)
    assert [entry["name"] for entry in converted] == ["sys_monitor", "Attacker"]
    for entry in converted:
        assert entry["parameters"] == {"type": "object", "properties": {}, "additionalProperties": False}
        # FunctionToolParam requires `strict`; non-strict so a stray argument is tolerated
        # rather than rejected, matching upstream's argument-free dispatch.
        assert entry["strict"] is False


def test_memory_retrieval_prefers_the_record_about_the_same_task():
    corpus = [
        'Agent: analyst; Task: Evaluate currency fluctuations on investments; Workflow: [{"a": 1}]; Tools: []',
        'Agent: surgeon; Task: Plan a cardiac procedure for a patient; Workflow: [{"b": 2}]; Tools: []',
    ]
    record = retrieve_memory(corpus, "Evaluate currency fluctuations on investments")
    assert record is not None and "analyst" in record


def test_memory_retrieval_degrades_safely():
    assert retrieve_memory([], "anything") is None
    assert retrieve_memory(["Agent: a; Task: t;"], "") is None
    # No shared terms at all means no retrieval rather than an arbitrary pick.
    assert retrieve_memory(["zzz qqq"], "alpha beta") is None


def test_consecutive_system_messages_are_merged():
    """Qwen3.5's template rejects two system turns; ASB always sends two.

    Merged for every model, not just Qwen, so all four answer identical input.
    """
    from responses_api_agents.asb_agent.app import merge_system_messages

    merged = merge_system_messages(
        [
            {"role": "system", "content": "agent description"},
            {"role": "system", "content": "plan instruction"},
            {"role": "user", "content": "task"},
        ]
    )
    assert [m["role"] for m in merged] == ["system", "user"]
    assert merged[0]["content"] == "agent description\n\nplan instruction"
    # No text is invented, dropped or reordered.
    assert merged[1]["content"] == "task"


def test_merge_leaves_a_later_system_turn_alone():
    """Only *consecutive* system turns collapse; ordering is never rewritten."""
    from responses_api_agents.asb_agent.app import merge_system_messages

    messages = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "u"},
        {"role": "system", "content": "b"},
    ]
    assert merge_system_messages(messages) == messages
