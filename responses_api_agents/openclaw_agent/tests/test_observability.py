# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymFunctionCallOutput, NeMoGymResponseFunctionToolCall
from nemo_gym.rollout_observability import (
    AgentInvocation,
    ContextCompactionObservation,
    ToolCallObservation,
)
from responses_api_agents.openclaw_agent.observability import (
    build_openclaw_observation_tree,
    build_openclaw_observations,
    discover_openclaw_session_tree,
)


def _records(bundle, record_type):
    return [record for record in bundle.records if isinstance(record, record_type)]


def _session(path: Path, session_id: str, text: str = "done") -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": session_id}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    }
                ),
            ]
        )
    )


def test_discovers_cross_agent_session_tree(tmp_path: Path) -> None:
    main = tmp_path / "main" / "sessions"
    worker = tmp_path / "worker" / "sessions"
    main.mkdir(parents=True)
    worker.mkdir(parents=True)
    _session(main / "root.jsonl", "root")
    _session(main / "child.jsonl", "child")
    _session(worker / "grandchild.jsonl", "grandchild")
    (main / "sessions.json").write_text(
        json.dumps(
            {
                "agent:main:main": {"sessionId": "root"},
                "agent:main:subagent:child": {"sessionId": "child", "spawnedBy": "agent:main:main"},
            }
        )
    )
    (worker / "sessions.json").write_text(
        json.dumps(
            {
                "agent:worker:subagent:grandchild": {
                    "sessionId": "grandchild",
                    "parentSessionKey": "agent:main:subagent:child",
                }
            }
        )
    )

    sessions, gaps = discover_openclaw_session_tree(tmp_path, "root")

    assert [(item[0], item[1]) for item in sessions] == [
        ("agent:main:main", None),
        ("agent:main:subagent:child", "agent:main:main"),
        ("agent:worker:subagent:grandchild", "agent:main:subagent:child"),
    ]
    assert gaps == []

    bundle = build_openclaw_observation_tree(
        (
            (invocation_id, parent_id, [NeMoGymEasyInputMessage(role="user", content=invocation_id)], events)
            for invocation_id, parent_id, events in sessions
        )
    )
    assert [item.parent_invocation_id for item in _records(bundle, AgentInvocation)] == [
        None,
        "agent:main:main",
        "agent:main:subagent:child",
    ]
    assert "subagent_hierarchy_unavailable" not in {gap.code for gap in bundle.gaps}


def test_missing_child_transcript_is_explicit(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "main" / "sessions"
    sessions_dir.mkdir(parents=True)
    _session(sessions_dir / "root.jsonl", "root")
    (sessions_dir / "sessions.json").write_text(
        json.dumps(
            {
                "root-key": {"sessionId": "root"},
                "child-key": {"sessionId": "deleted", "spawnedBy": "root-key"},
            }
        )
    )

    sessions, gaps = discover_openclaw_session_tree(tmp_path, "root")

    assert [item[0] for item in sessions] == ["root-key", "child-key"]
    assert {(gap.code, gap.invocation_id) for gap in gaps} >= {
        ("agent_transcript_unavailable", "child-key"),
        ("subagent_hierarchy_incomplete", None),
    }


def test_three_duplicate_session_keys_are_reported_without_crashing(tmp_path: Path) -> None:
    for agent in ("one", "two", "three"):
        sessions_dir = tmp_path / agent / "sessions"
        sessions_dir.mkdir(parents=True)
        (sessions_dir / "sessions.json").write_text(json.dumps({"duplicate": {"sessionId": agent}}))

    sessions, gaps = discover_openclaw_session_tree(tmp_path, "one")

    assert sessions == []
    assert "agent_session_identity_ambiguous" in {gap.code for gap in gaps}


def test_session_file_cannot_escape_the_session_archive(tmp_path: Path) -> None:
    agents_root = tmp_path / "agents"
    sessions_dir = agents_root / "main" / "sessions"
    sessions_dir.mkdir(parents=True)
    outside = tmp_path / "outside-session.jsonl"
    _session(outside, "root", "secret")
    (sessions_dir / outside.name).symlink_to(outside)
    (sessions_dir / "sessions.json").write_text(
        json.dumps({"root-key": {"sessionId": "root", "sessionFile": str(outside)}})
    )

    sessions, gaps = discover_openclaw_session_tree(agents_root, "root")

    assert sessions == [("root-key", None, [])]
    assert "agent_transcript_unavailable" in {gap.code for gap in gaps}


def test_builds_root_observation_and_pairs_tools() -> None:
    model_ref = ModelServerRef(type="responses_api_models", name="policy")
    bundle = build_openclaw_observations(
        "session-1",
        [
            NeMoGymResponseFunctionToolCall(
                arguments='{"query":"AAPL"}',
                call_id="call-1",
                name="web_search",
                id="call-1",
                status="completed",
            ),
            NeMoGymFunctionCallOutput(call_id="call-1", output="result", status="completed"),
        ],
        [
            {
                "type": "message",
                "message": {"role": "assistant", "responseId": "response-1"},
            }
        ],
        transcript_available=True,
        model_ref=model_ref,
    )

    [invocation] = _records(bundle, AgentInvocation)
    assert invocation.invocation_id == "session-1"
    assert invocation.conversation
    assert invocation.model_calls[0].response_id == "response-1"
    assert invocation.model_calls[0].model_ref == model_ref
    assert _records(bundle, ToolCallObservation) == []
    assert {gap.code for gap in bundle.gaps} == {
        "subagent_hierarchy_unavailable",
        "tool_timing_unavailable",
    }


def test_marks_missing_transcript_and_incomplete_tool() -> None:
    bundle = build_openclaw_observations(
        "fallback",
        [
            NeMoGymResponseFunctionToolCall(
                arguments="{}",
                call_id="call-1",
                name="tool",
                id="call-1",
                status="completed",
            )
        ],
        [],
        transcript_available=False,
    )

    assert _records(bundle, ToolCallObservation) == []
    assert "agent_transcript_unavailable" in {gap.code for gap in bundle.gaps}


def test_duplicate_tool_ids_are_ambiguous_and_do_not_imply_success() -> None:
    bundle = build_openclaw_observations(
        "session-1",
        [
            NeMoGymResponseFunctionToolCall(arguments="{}", call_id="duplicate", name="tool"),
            NeMoGymResponseFunctionToolCall(arguments="{}", call_id="duplicate", name="tool"),
            NeMoGymFunctionCallOutput(call_id="duplicate", output="result", status="completed"),
        ],
        [],
        transcript_available=True,
    )

    assert _records(bundle, ToolCallObservation) == []
    assert "tool_call_identity_ambiguous" in {gap.code for gap in bundle.gaps}


def test_model_visible_result_does_not_imply_observed_execution() -> None:
    bundle = build_openclaw_observations(
        "run-1",
        [
            NeMoGymResponseFunctionToolCall(arguments="{}", call_id="call-1", name="tool"),
            NeMoGymFunctionCallOutput(call_id="call-1", output="partial", status="incomplete"),
        ],
        [],
        transcript_available=True,
        source="pinchbench",
    )

    assert bundle.source == "pinchbench"
    assert _records(bundle, ToolCallObservation) == []
    assert "tool_timing_unavailable" in {gap.code for gap in bundle.gaps}


def test_extracts_parallel_tool_intervals_and_compaction() -> None:
    conversation = [
        NeMoGymResponseFunctionToolCall(arguments="{}", call_id="call-1", name="tool"),
        NeMoGymFunctionCallOutput(call_id="call-1", output="result", status="completed"),
        NeMoGymResponseFunctionToolCall(arguments="{}", call_id="call-2", name="tool"),
        NeMoGymFunctionCallOutput(call_id="call-2", output="error", status="completed"),
    ]
    events = [
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "timestamp": 1_750_000_002_000,
                "details": {"durationMs": 1000, "status": "completed"},
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-2",
                "timestamp": 1_750_000_002_500,
                "details": {"durationMs": 1000, "status": "failed"},
                "isError": True,
            },
        },
        {
            "type": "compaction",
            "timestamp": "2025-06-15T15:06:43Z",
            "summary": "condensed context",
            "firstKeptEntryId": "entry-7",
            "tokensBefore": 120_000,
        },
    ]

    bundle = build_openclaw_observations("session-1", conversation, events, transcript_available=True)

    first, second = _records(bundle, ToolCallObservation)
    assert first.started_at < second.started_at < first.completed_at < second.completed_at
    assert first.duration_ms == second.duration_ms == 1000
    assert first.timing_source == second.timing_source == "artifact"
    assert first.status == "completed"
    assert second.status == "failed"
    [compaction] = _records(bundle, ContextCompactionObservation)
    assert compaction.summary == "condensed context"
    assert compaction.first_kept_item_id == "entry-7"
    assert compaction.tokens_before == 120_000
    assert {"compaction_tokens_after_unavailable", "compaction_model_call_boundary_unavailable"} <= {
        gap.code for gap in bundle.gaps
    }


def test_reports_intra_session_branching() -> None:
    events = [
        {"type": "message", "id": "first", "parentId": "root", "message": {"role": "assistant", "content": []}},
        {"type": "message", "id": "second", "parentId": "root", "message": {"role": "assistant", "content": []}},
    ]

    bundle = build_openclaw_observations("session-1", [], events, transcript_available=True)

    assert "agent_conversation_branching_unavailable" in {gap.code for gap in bundle.gaps}
