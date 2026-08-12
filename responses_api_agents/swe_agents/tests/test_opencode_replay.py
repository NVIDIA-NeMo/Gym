# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Optional

from responses_api_agents.swe_agents.opencode_replay import (
    build_replay_prefix_row,
    build_replay_subagent_manifest,
    completed_tool_turn_cut_indices,
    extract_responses_task_records,
    extract_task_spawn_records,
    merge_replay_subagent_trajectories,
    parse_replay_subagent_payload,
    truncate_replay_subagent_payload,
)


def _task_call(call_id: str, prompt: str, subagent_type: str = "general") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "task",
            "arguments": json.dumps(
                {
                    "description": prompt,
                    "prompt": prompt,
                    "subagent_type": subagent_type,
                }
            ),
        },
    }


def _task_result(call_id: str, session_id: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": f"task_id: {session_id} (for resuming)\n\n<task_result>done</task_result>",
    }


def _responses_task_call(call_id: str, prompt: str, task_id: Optional[str] = None) -> dict:
    arguments = {"description": prompt, "prompt": prompt, "subagent_type": "general"}
    if task_id is not None:
        arguments["task_id"] = task_id
    return {
        "type": "function_call",
        "name": "task",
        "call_id": call_id,
        "arguments": json.dumps(arguments),
    }


def _responses_task_result(call_id: str, session_id: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": f"task_id: {session_id} (for resuming)\n\n<task_result>done</task_result>",
    }


def test_extract_task_spawns_uses_parent_message_and_tool_order() -> None:
    messages = [
        {"role": "user", "content": "root"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_task_call("call_a", "A"), _task_call("call_b", "B", "explore")],
        },
        # Tool results can finish in the opposite order.
        _task_result("call_b", "session_b"),
        _task_result("call_a", "session_a"),
    ]

    spawns = extract_task_spawn_records(messages)
    assert [(spawn["spawn_call_id"], spawn["spawn_index"]) for spawn in spawns] == [
        ("call_a", 0),
        ("call_b", 1),
    ]
    assert [spawn["child_session_id"] for spawn in spawns] == ["session_a", "session_b"]


def test_responses_task_records_and_cut_preserve_parallel_call_order() -> None:
    items = [
        {"type": "reasoning", "summary": []},
        _responses_task_call("call_a", "A"),
        _responses_task_call("call_b", "B"),
        # Results deliberately finish in the opposite order.
        _responses_task_result("call_b", "session_b"),
        _responses_task_result("call_a", "session_a"),
        {"type": "function_call", "name": "read", "call_id": "call_read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_read", "output": "content"},
    ]

    records = extract_responses_task_records(items)
    assert [(record["spawn_call_id"], record["child_session_id"]) for record in records] == [
        ("call_a", "session_a"),
        ("call_b", "session_b"),
    ]
    assert completed_tool_turn_cut_indices(items, require_task=True) == [4]
    assert completed_tool_turn_cut_indices(items) == [4, 6]


def test_truncate_payload_keeps_only_completed_child_invocations_and_nested_branches() -> None:
    child_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first result"},
        {"role": "user", "content": "resume"},
        {"role": "assistant", "content": None, "tool_calls": [_task_call("call_nested", "nested")]},
        _task_result("call_nested", "session_nested"),
        {"role": "assistant", "content": "resume result"},
    ]
    payload = {
        "root_session_id": "session_root",
        "sessions": [
            {
                "session_id": "session_nested",
                "parent_session_id": "session_child",
                "messages": [{"role": "user", "content": "nested"}, {"role": "assistant", "content": "done"}],
            },
            {
                "session_id": "session_child",
                "parent_session_id": "session_root",
                "messages": child_messages,
            },
        ],
    }
    first_call = extract_responses_task_records(
        [_responses_task_call("call_child", "first"), _responses_task_result("call_child", "session_child")]
    )
    first_prefix = truncate_replay_subagent_payload(first_call, payload)
    assert first_prefix is not None
    assert [session["session_id"] for session in first_prefix["sessions"]] == ["session_child"]
    assert first_prefix["sessions"][0]["messages"] == child_messages[:3]

    resumed_calls = extract_responses_task_records(
        [
            _responses_task_call("call_child", "first"),
            _responses_task_result("call_child", "session_child"),
            _responses_task_call("call_resume", "resume", task_id="session_child"),
            _responses_task_result("call_resume", "session_child"),
        ]
    )
    resumed_prefix = truncate_replay_subagent_payload(resumed_calls, payload)
    assert resumed_prefix is not None
    assert [session["session_id"] for session in resumed_prefix["sessions"]] == [
        "session_child",
        "session_nested",
    ]
    assert resumed_prefix["sessions"][0]["messages"] == child_messages


def test_build_prefix_row_moves_legacy_subagents_into_request_metadata() -> None:
    row = {
        "responses_create_params": {
            "input": [{"role": "user", "content": "root"}],
            "metadata": {"instance_id": "example"},
        },
        "response": {
            "output": [
                _responses_task_call("call_a", "A"),
                _responses_task_call("call_b", "B"),
                _responses_task_result("call_b", "session_b"),
                _responses_task_result("call_a", "session_a"),
                {"type": "message", "role": "assistant", "content": "later"},
            ]
        },
        # Metadata order intentionally differs from parent task-call order.
        "subagent_trajectories": [
            {
                "session_id": "session_b",
                "parent_session_id": "session_root",
                "messages": [{"role": "user", "content": "B"}, {"role": "assistant", "content": "b"}],
            },
            {
                "session_id": "session_a",
                "parent_session_id": "session_root",
                "messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "a"}],
            },
        ],
        "reward": 1.0,
    }

    prefix = build_replay_prefix_row(row, source_line=96)
    assert set(prefix) == {"responses_create_params", "replay_provenance"}
    assert prefix["replay_provenance"] == {
        "cut_output_index": 3,
        "source_line": 96,
        "strategy": "first-task-batch",
    }
    assert len(prefix["responses_create_params"]["input"]) == 5
    payload = json.loads(prefix["responses_create_params"]["metadata"]["subagent_trajectories"])
    assert [session["session_id"] for session in payload["sessions"]] == ["session_a", "session_b"]


def test_build_manifest_links_parallel_children_by_task_call_not_metadata_order() -> None:
    main_messages = [
        {"role": "user", "content": "root"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_task_call("call_a", "A"), _task_call("call_b", "B", "explore")],
        },
        _task_result("call_a", "session_a"),
        _task_result("call_b", "session_b"),
    ]
    payload = {
        # Deliberately reverse the metadata array relative to task-call order.
        "sessions": [
            {
                "session_id": "session_b",
                "parent_session_id": "session_root",
                "messages": [{"role": "user", "content": "B"}, {"role": "assistant", "content": "b"}],
            },
            {
                "session_id": "session_a",
                "parent_session_id": "session_root",
                "messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "a"}],
            },
        ]
    }

    manifest = build_replay_subagent_manifest(main_messages, payload)
    assert manifest is not None
    assert manifest["root_session_id"] == "session_root"
    assert [session["session_id"] for session in manifest["sessions"]] == ["session_a", "session_b"]
    by_id = {session["session_id"]: session for session in manifest["sessions"]}
    assert (by_id["session_a"]["spawn_call_id"], by_id["session_a"]["spawn_index"]) == ("call_a", 0)
    assert (
        by_id["session_b"]["spawn_call_id"],
        by_id["session_b"]["spawn_index"],
        by_id["session_b"]["subagent_type"],
    ) == ("call_b", 1, "explore")


def test_build_manifest_links_nested_child_in_its_parent_messages() -> None:
    main_messages = [
        {"role": "user", "content": "root"},
        {"role": "assistant", "content": None, "tool_calls": [_task_call("call_child", "child")]},
        _task_result("call_child", "session_child"),
    ]
    child_messages = [
        {"role": "user", "content": "child"},
        {"role": "assistant", "content": None, "tool_calls": [_task_call("call_nested", "nested", "explore")]},
        _task_result("call_nested", "session_nested"),
    ]
    manifest = build_replay_subagent_manifest(
        main_messages,
        {
            "sessions": [
                {
                    "session_id": "session_nested",
                    "parent_session_id": "session_child",
                    "messages": [{"role": "user", "content": "nested"}],
                },
                {
                    "session_id": "session_child",
                    "parent_session_id": "session_root",
                    "messages": child_messages,
                },
            ]
        },
    )

    assert manifest is not None
    assert [session["session_id"] for session in manifest["sessions"]] == ["session_child", "session_nested"]
    by_id = {session["session_id"]: session for session in manifest["sessions"]}
    assert by_id["session_child"]["spawn_call_id"] == "call_child"
    assert by_id["session_nested"]["spawn_call_id"] == "call_nested"
    assert by_id["session_nested"]["parent_session_id"] == "session_child"


def test_build_manifest_falls_back_to_unique_prompt_when_task_result_is_missing() -> None:
    main_messages = [
        {"role": "user", "content": "root"},
        {"role": "assistant", "content": None, "tool_calls": [_task_call("call_interrupted", "child")]},
    ]
    manifest = build_replay_subagent_manifest(
        main_messages,
        {
            "sessions": [
                {
                    "session_id": "session_child",
                    "parent_session_id": "session_root",
                    "messages": [{"role": "user", "content": "child"}],
                }
            ]
        },
    )
    assert manifest is not None
    assert manifest["sessions"][0]["spawn_call_id"] == "call_interrupted"


def test_parse_payload_accepts_legacy_json_list() -> None:
    sessions = [{"session_id": "child"}]
    assert parse_replay_subagent_payload({"subagent_trajectories": json.dumps(sessions)}) == {
        "version": 1,
        "sessions": sessions,
    }


def test_merge_carries_original_prefix_and_appends_only_live_continuation() -> None:
    original = {
        "version": 1,
        "root_session_id": "recorded_root",
        "sessions": [
            {
                "session_id": "recorded_child",
                "parent_session_id": "recorded_root",
                "spawn_call_id": "call_child",
                "spawn_index": 0,
                "messages": [
                    {"role": "user", "content": "child"},
                    {"role": "assistant", "content": "recorded"},
                ],
            }
        ],
    }
    captured = [
        {
            "session_id": "live_child",
            "parent_session_id": "live_root",
            "recorded_session_id": "recorded_child",
            "replay_prefix_message_count": 2,
            "messages": [
                {"role": "user", "content": "child"},
                {"role": "assistant", "content": "recorded"},
                {"role": "assistant", "content": "live continuation"},
            ],
            "tools": [{"name": "read"}],
        }
    ]

    merged = merge_replay_subagent_trajectories(original, captured)
    assert len(merged) == 1
    assert merged[0]["session_id"] == "recorded_child"
    assert merged[0]["live_session_id"] == "live_child"
    assert [message["content"] for message in merged[0]["messages"]] == [
        "child",
        "recorded",
        "live continuation",
    ]


def test_merge_reparents_a_new_live_child_to_the_stable_recorded_root() -> None:
    manifest = {
        "version": 1,
        "root_session_id": "recorded_root",
        "sessions": [],
    }
    captured = [
        {
            "session_id": "new_live_child",
            "parent_session_id": "live_root",
            "recorded_parent_session_id": "recorded_root",
            "spawn_call_id": "new_call",
            "spawn_index": 2,
            "messages": [{"role": "user", "content": "new work"}],
        }
    ]

    merged = merge_replay_subagent_trajectories(manifest, captured)
    assert merged[0]["parent_session_id"] == "recorded_root"
