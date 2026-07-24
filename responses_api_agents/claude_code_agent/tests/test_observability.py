# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import TypeVar

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ToolCallObservation,
)
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations


MODEL_REF = ModelServerRef(type="responses_api_models", name="policy")
T = TypeVar("T")


def _records(bundle: AgentObservationBundle, record_type: type[T]) -> list[T]:
    return [record for record in bundle.records if isinstance(record, record_type)]


def _write(path: Path, *events: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(event if isinstance(event, str) else json.dumps(event) for event in events))


def _event(
    session: str,
    role: str,
    timestamp: str,
    content: str | list[dict],
    *,
    agent: str | None = None,
    message_id: str | None = None,
    message_extra: dict | None = None,
    **extra: object,
) -> dict:
    message = {"role": role, "content": content, **(message_extra or {})}
    if message_id:
        message["id"] = message_id
    event = {
        "type": role,
        "sessionId": session,
        "timestamp": timestamp,
        "message": message,
        **extra,
    }
    if agent:
        event["agentId"] = agent
    return event


def _assistant(session: str, timestamp: str, message_id: str, *content: dict, agent: str | None = None) -> dict:
    return _event(
        session,
        "assistant",
        timestamp,
        list(content),
        agent=agent,
        message_id=message_id,
        uuid=f"{message_id}-event",
    )


def _tool_result(
    session: str,
    timestamp: str,
    tool_call_id: str,
    *,
    agent: str | None = None,
    child_id: str | None = None,
    status: str = "completed",
    is_error: bool = False,
) -> dict:
    content = [{"type": "tool_result", "tool_use_id": tool_call_id, "content": "result", "is_error": is_error}]
    event = _event(session, "user", timestamp, content, agent=agent, uuid=f"{tool_call_id}-result")
    if child_id is not None:
        event["toolUseResult"] = {"agentId": child_id, "status": status}
    return event


def test_extracts_nested_tree_model_refs_and_parallel_tool_timing(tmp_path: Path) -> None:
    session = "session-root"
    child = "agent-child"
    grandchild = "agent-grandchild"
    root = tmp_path / "projects" / "work" / f"{session}.jsonl"
    subagents = root.parent / session / "subagents"

    _write(
        root,
        _event(session, "user", "2026-07-22T10:00:00Z", "solve", uuid="root-user"),
        _assistant(
            session,
            "2026-07-22T10:00:01Z",
            "msg-root",
            {"type": "thinking", "thinking": "plan", "signature": "sig"},
        ),
        _assistant(
            session,
            "2026-07-22T10:00:02Z",
            "msg-root",
            {"type": "tool_use", "id": "tool-fast", "name": "Read", "input": {"path": "a"}},
            {"type": "tool_use", "id": "tool-child", "name": "Agent", "input": {"prompt": "delegate"}},
        ),
        _tool_result(session, "2026-07-22T10:00:03Z", "tool-fast"),
        _tool_result(
            session,
            "2026-07-22T10:00:05Z",
            "tool-child",
            child_id=child,
        ),
    )
    _write(
        subagents / f"{child}.jsonl",
        _event(session, "user", "2026-07-22T10:00:02.100Z", "child task", agent=child, uuid="child-user"),
        _assistant(
            session,
            "2026-07-22T10:00:03Z",
            "msg-child",
            {"type": "tool_use", "id": "tool-grandchild", "name": "Agent", "input": {}},
            agent=child,
        ),
        _tool_result(
            session,
            "2026-07-22T10:00:04Z",
            "tool-grandchild",
            agent=child,
            child_id=grandchild,
        ),
    )
    _write(
        subagents / f"{grandchild}.jsonl",
        _assistant(
            session,
            "2026-07-22T10:00:03.100Z",
            "msg-grandchild",
            {"type": "text", "text": "done"},
            agent=grandchild,
        ),
    )

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    invocations = _records(bundle, AgentInvocation)
    assert [invocation.invocation_id for invocation in invocations] == [session, child, grandchild]
    root_invocation, child_invocation, grandchild_invocation = invocations
    assert child_invocation.parent_invocation_id == session
    assert child_invocation.spawned_by_tool_call_id == "tool-child"
    assert grandchild_invocation.parent_invocation_id == child
    assert grandchild_invocation.spawned_by_tool_call_id == "tool-grandchild"
    assert [reference.response_id for reference in root_invocation.model_calls] == ["msg-root"]
    assert [reference.response_id for reference in child_invocation.model_calls] == ["msg-child"]
    assert [reference.response_id for reference in grandchild_invocation.model_calls] == ["msg-grandchild"]
    assert all(reference.model_ref == MODEL_REF for invocation in invocations for reference in invocation.model_calls)
    assert [item.type for item in root_invocation.conversation] == [
        "message",
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]

    timings = {tool.tool_call_id: tool for tool in _records(bundle, ToolCallObservation)}
    assert timings["tool-fast"].duration_ms == pytest.approx(1000)
    assert timings["tool-child"].duration_ms == pytest.approx(3000)
    assert timings["tool-grandchild"].duration_ms == pytest.approx(1000)
    assert all(tool.timing_source == "artifact" for tool in timings.values())
    assert [(gap.code, gap.invocation_id) for gap in bundle.gaps] == [("invocation_outcome_unavailable", session)]


def test_extracts_explicit_compaction_markers(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _assistant("root", "2026-07-22T09:59:59Z", "msg-before", {"type": "text", "text": "before"}),
        _event(
            "root",
            "user",
            "2026-07-22T10:00:00Z",
            "summary",
            message_extra={
                "isCompactSummary": True,
                "compactMetadata": {"tokensBefore": 1000, "tokensAfter": 200, "trigger": "auto"},
            },
        ),
        _event(
            "root",
            "system",
            "2026-07-22T10:00:01Z",
            "",
            subtype="compact_boundary",
            compact_metadata={"preTokens": 900, "postTokens": 180},
        ),
        _assistant("root", "2026-07-22T10:00:02Z", "msg-after", {"type": "text", "text": "after"}),
    )

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    [compaction] = _records(bundle, ContextCompactionObservation)
    assert compaction.trigger == "auto"
    assert compaction.tokens_before == 1000
    assert compaction.tokens_after == 200
    assert compaction.summary == "summary"
    assert compaction.outcome == "completed"
    assert compaction.before_model_call.response_id == "msg-before"
    assert compaction.after_model_call.response_id == "msg-after"


def test_malformed_and_incomplete_artifacts_produce_sanitized_gaps(tmp_path: Path) -> None:
    sentinel = "redacted-payload-line"
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        f'{{"private":"{sentinel}"',
        _assistant(
            "root",
            "bad-timestamp",
            "msg-root",
            {"type": "tool_use", "id": "pending", "name": "Bash", "input": {}},
        ),
        _tool_result("root", "2026-07-22T10:00:03Z", "orphan"),
    )
    _write(
        tmp_path / "projects" / "work" / "subagents" / "agent-orphan.jsonl",
        _assistant(
            "root",
            "2026-07-22T10:00:01Z",
            "msg-orphan",
            {"type": "text", "text": "answer"},
            agent="agent-orphan",
        ),
    )

    bundle = extract_claude_code_observations(tmp_path)
    codes = {gap.code for gap in bundle.gaps}

    assert {
        "malformed_transcript_line",
        "subagent_parent_unavailable",
        "tool_result_missing",
        "tool_start_timestamp_missing",
        "tool_start_missing",
    } <= codes
    assert all(not invocation.model_calls for invocation in _records(bundle, AgentInvocation))
    assert sentinel not in bundle.model_dump_json()


def test_ignores_non_transcript_jsonl_and_reports_no_usable_transcript(tmp_path: Path) -> None:
    _write(
        tmp_path / "skills" / "fixture.jsonl",
        _assistant("unrelated", "2026-07-22T10:00:00Z", "msg-unrelated", {"type": "text", "text": "x"}),
    )
    (tmp_path / "projects").mkdir()

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    assert _records(bundle, AgentInvocation) == []
    assert "agent_transcript_unavailable" in {gap.code for gap in bundle.gaps}


def test_reports_missing_response_id_and_unsupported_content_blocks(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _event(
            "root",
            "assistant",
            "2026-07-22T10:00:00Z",
            [{"type": "image", "source": "omitted"}],
            uuid="assistant-event",
        ),
        _event(
            "root",
            "user",
            "2026-07-22T10:00:01Z",
            [{"type": "image", "source": "omitted"}],
            uuid="user-event",
        ),
    )

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    codes = {gap.code for gap in bundle.gaps}

    assert "model_response_id_missing" in codes
    assert "unsupported_assistant_content_block" in codes
    assert "unsupported_user_content_block" in codes
    assert _records(bundle, AgentInvocation)[0].model_calls == []
