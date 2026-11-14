# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymFunctionCallOutput, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import merge_model_call_capture_into_record
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRecord,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)
from responses_api_agents.claude_code_agent.app import ClaudeCodeAgent, ClaudeCodeAgentConfig
from responses_api_agents.claude_code_agent.observability import (
    associate_claude_code_compaction_calls,
    extract_claude_code_observations,
)
from responses_api_agents.claude_code_agent.sandbox import ClaudeCodeArtifacts, ClaudeCodeSandboxSession


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
            status="timeout",
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

    bundle = extract_claude_code_observations(
        tmp_path,
        model_ref=MODEL_REF,
        root_status="completed",
        root_duration_ms=5000,
        agent_instance="claude-code-agent:a",
        clock_domain="claude-code-wall:a",
    )

    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}
    assert set(invocations) == {session, child, grandchild}
    assert {invocation.agent_instance for invocation in invocations.values()} == {"claude-code-agent:a"}
    root_invocation = invocations[session]
    child_invocation = invocations[child]
    grandchild_invocation = invocations[grandchild]
    assert root_invocation.status == "completed"
    assert root_invocation.duration_ms == 5000
    assert child_invocation.parent_invocation_id == session
    assert child_invocation.spawned_by_tool_call_id == "tool-child"
    assert grandchild_invocation.parent_invocation_id == child
    assert grandchild_invocation.spawned_by_tool_call_id == "tool-grandchild"
    assert grandchild_invocation.status == "incomplete"
    assert [reference.response_id for reference in root_invocation.model_calls] == ["msg-root"]
    assert [reference.response_id for reference in child_invocation.model_calls] == ["msg-child"]
    assert [reference.response_id for reference in grandchild_invocation.model_calls] == ["msg-grandchild"]
    assert all(
        reference.model_ref == MODEL_REF for invocation in invocations.values() for reference in invocation.model_calls
    )
    assert [item.type for item in root_invocation.conversation] == [
        "message",
        "reasoning",
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert all(item.id is None for item in root_invocation.conversation if isinstance(item, NeMoGymFunctionCallOutput))
    [grandchild_result] = [
        item for item in child_invocation.conversation if isinstance(item, NeMoGymFunctionCallOutput)
    ]
    assert grandchild_result.status == "incomplete"

    timings = {tool.tool_call_id: tool for tool in _records(bundle, ToolCallObservation)}
    assert timings["tool-fast"].timings[0].duration_ms == pytest.approx(1000)
    assert timings["tool-child"].timings[0].duration_ms == pytest.approx(3000)
    assert timings["tool-grandchild"].timings[0].duration_ms == pytest.approx(1000)
    assert timings["tool-grandchild"].status == "timeout"
    legacy = bundle.model_copy(update={"schema_version": "1.0"}).model_dump(mode="json")
    legacy_tools = {record["tool_call_id"]: record for record in legacy["records"] if record["kind"] == "tool_call"}
    assert legacy_tools["tool-fast"]["duration_ms"] == pytest.approx(1000)
    assert all(tool["timing_source"] == "artifact" for tool in legacy_tools.values())
    assert not bundle.gaps
    assert {tool.timings[0].clock_domain for tool in timings.values()} == {"claude-code-wall:a"}
    assert {tool.agent_instance for tool in timings.values()} == {"claude-code-agent:a"}
    assert all(tool.timings[0].boundary == "transcript" for tool in timings.values())
    assert timings["tool-grandchild"].parent_span_id == timings["tool-child"].span_id
    assert timings["tool-fast"].model_call_ref.response_id == "msg-root"


def test_extracts_explicit_compaction_markers(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _assistant("root", "2026-07-22T09:59:59Z", "msg-before", {"type": "text", "text": "before"}),
        _event(
            "root",
            "system",
            "2026-07-22T10:00:01Z",
            "",
            subtype="compact_boundary",
        ),
        _event(
            "root",
            "user",
            "bad-timestamp",
            "summary",
            isCompactSummary=True,
            compactMetadata={"tokensBefore": 1000, "tokensAfter": 200, "trigger": "auto"},
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
    assert compaction.observed_at == pytest.approx(1784714401)
    assert compaction.before_model_call.response_id == "msg-before"
    assert compaction.after_model_call.response_id == "msg-after"
    assert compaction.model_calls == []
    codes = {gap.code for gap in bundle.gaps}
    assert "compaction_model_call_reference_unavailable" in codes
    assert "compaction_timestamp_missing" not in codes


def test_compaction_metadata_does_not_assume_success(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _event(
            "root",
            "system",
            "2026-07-22T10:00:00Z",
            "",
            compact_metadata={"tokensBefore": 100, "tokensAfter": 80},
        ),
    )

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    [compaction] = _records(bundle, ContextCompactionObservation)
    assert compaction.outcome == "unknown"
    assert "compaction_outcome_unavailable" in {gap.code for gap in bundle.gaps}


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
        _assistant(
            "root",
            "2026-07-22T10:00:05Z",
            "msg-later",
            {"type": "tool_use", "id": "reversed", "name": "Bash", "input": {}},
        ),
        _tool_result("root", "2026-07-22T09:59:59Z", "reversed"),
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
        "tool_clock_discontinuity",
    } <= codes
    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}
    assert all(not invocation.model_calls for invocation in invocations.values())
    assert [item.type for item in invocations["root"].conversation] == [
        "function_call",
        "function_call",
        "function_call_output",
        "function_call_output",
    ]
    assert sentinel not in bundle.model_dump_json()


def test_rejects_cyclic_subagent_parents(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        _tool_result("root", "2026-07-22T10:00:00Z", "self", child_id="root"),
        _tool_result("root", "2026-07-22T10:00:01Z", "spawn", child_id="agent-a"),
        _tool_result(
            "root",
            "2026-07-22T10:00:02Z",
            "back-edge",
            agent="agent-a",
            child_id="root",
        ),
    )

    bundle = extract_claude_code_observations(tmp_path)
    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}

    assert invocations["root"].parent_invocation_id is None
    assert invocations["agent-a"].parent_invocation_id == "root"
    assert [gap.code for gap in bundle.gaps].count("cyclic_subagent_parent") == 2


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


def _compaction_bundle() -> AgentObservationBundle:
    before = ModelCallRef(model_ref=MODEL_REF, response_id="msg-before")
    after = ModelCallRef(model_ref=MODEL_REF, response_id="msg-after")
    return AgentObservationBundle(
        source="claude_code",
        records=[
            AgentInvocation(invocation_id="root", model_calls=[before, after]),
            ContextCompactionObservation(
                invocation_id="root",
                outcome="completed",
                summary=(
                    "This session is being continued from a previous conversation that ran out of context. "
                    "The summary below covers the earlier portion of the conversation.\n\n"
                    "Summary:\nKeep this.\n\nRecent messages are preserved verbatim."
                ),
                before_model_call=before,
                after_model_call=after,
            ),
        ],
        gaps=[ObservationGap(code="compaction_model_call_reference_unavailable", invocation_id="root")],
    )


def _captured_call(call_id: str, response_id: str, *, compact: bool = False) -> dict:
    return {
        "model_call_id": call_id,
        "response_id": response_id,
        "dialect": "messages",
        "status_code": 200,
        "model_ref": MODEL_REF.model_dump(mode="json"),
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Your task is to create a detailed summary of this conversation." if compact else "continue"
                    ),
                }
            ]
        },
        "response": {
            "id": response_id,
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "content": [
                {
                    "type": "text",
                    "text": ("<analysis>private</analysis><summary>Keep this.</summary>" if compact else "ok"),
                }
            ],
        },
    }


def _rollout_record(tmp_path: Path, *calls: dict) -> dict:
    store = CaptureStore(tmp_path)
    for call in calls:
        store.record("0-0", call)
    return {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": _compaction_bundle().model_dump(mode="json"),
    }


def _agent(capture_dir: Path) -> ClaudeCodeAgent:
    return ClaudeCodeAgent(
        config=ClaudeCodeAgentConfig(
            name="claude", host="127.0.0.1", port=8080, entrypoint="app.py", model_server=MODEL_REF
        ),
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"observability_enabled": True, "model_call_capture_dir": str(capture_dir)},
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("execution", ["local", "sandbox"])
async def test_adapter_associates_compaction_before_collection(tmp_path: Path, monkeypatch, execution: str) -> None:
    record = _rollout_record(
        tmp_path,
        _captured_call("call-before", "msg-before"),
        _captured_call("call-compact", "msg-compact", compact=True),
        _captured_call("call-after", "msg-after"),
    )
    summary = _records(_compaction_bundle(), ContextCompactionObservation)[0].summary
    transcript = tmp_path / "projects" / "work" / "session.jsonl"
    _write(
        transcript,
        _assistant("root", "2026-07-22T09:59:59Z", "msg-before", {"type": "text", "text": "before"}),
        _event("root", "system", "2026-07-22T10:00:00Z", "", subtype="compact_boundary"),
        _event("root", "user", "2026-07-22T10:00:01Z", summary, isCompactSummary=True),
        _assistant("root", "2026-07-22T10:00:02Z", "msg-after", {"type": "text", "text": "after"}),
    )
    instance = _agent(tmp_path)
    body = NeMoGymResponseCreateParamsNonStreaming(input="Continue")
    if execution == "local":

        async def run(self, prompt, *, observation_collector, **kwargs):
            observation_collector(tmp_path, {"status": "completed"})
            return [], "fixture", {}

        monkeypatch.setattr(ClaudeCodeAgent, "_run_claude_code", run)
        observations = (await instance._create_episode(body, rollout_id="0-0")).observations
    else:
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            archive.add(transcript, arcname="projects/work/session.jsonl")
        artifacts = ClaudeCodeArtifacts(
            '{"type":"result","subtype":"success","is_error":false}', archive_bytes.getvalue()
        )
        state = ClaudeCodeSandboxSession(
            request=AgentSeedSessionRequest(
                agent_session_id="session",
                episode_id={"rollout_id": "0-0"},
                task_id={"taskset": "fixture", "task_id": "task"},
            ),
            runtime="/runtime",
            session=SimpleNamespace(session_dir="/tmp/session", workdir="/app", cleanup=None),
        )
        monkeypatch.setattr(ClaudeCodeSandboxSession, "execute", AsyncMock(return_value=artifacts))
        monkeypatch.setattr(ClaudeCodeAgent, "_resolve_call_base_url", lambda *args: None)
        await instance._sandbox_response(state, body, prompt="Continue", system=None)
        observations = state.observations

    [compaction] = _records(observations, ContextCompactionObservation)
    assert [reference.model_call_id for reference in compaction.model_calls] == ["call-compact"]
    record["ng_agent_observations"] = observations.model_dump(mode="json")

    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)

    observations = AgentObservationBundle.model_validate(record["ng_agent_observations"])
    [invocation] = _records(observations, AgentInvocation)
    [compaction] = _records(observations, ContextCompactionObservation)
    assert [reference.model_call_id for reference in invocation.model_calls] == [
        "call-before",
        "call-compact",
        "call-after",
    ]
    assert [reference.model_call_id for reference in compaction.model_calls] == ["call-compact"]
    assert "compaction_model_call_reference_unavailable" not in {gap.code for gap in observations.gaps}
    trajectory = _build_trajectory_record(record, record)
    assert [reference.model_call_id for turn in trajectory.turns for reference in turn.model_calls] == [
        "call-before",
        "call-after",
    ]
    assert record["ng_model_call_capture"]["metrics"]["tokens_in"] == 30


@pytest.mark.asyncio
async def test_compaction_resolver_failure_preserves_generic_model_call_join(tmp_path: Path) -> None:
    record = _rollout_record(
        tmp_path,
        _captured_call("call-before", "msg-before"),
        _captured_call("call-after", "msg-after"),
    )

    with patch(
        "responses_api_agents.claude_code_agent.app.read_available_model_call_records",
        side_effect=RuntimeError,
    ):
        observations = await _agent(tmp_path)._associate_compaction_calls(_compaction_bundle(), rollout_id="0-0")
    record["ng_agent_observations"] = observations.model_dump(mode="json")
    merge_model_call_capture_into_record(record, [tmp_path])

    observations = AgentObservationBundle.model_validate(record["ng_agent_observations"])
    [invocation] = _records(observations, AgentInvocation)
    assert [reference.model_call_id for reference in invocation.model_calls] == ["call-before", "call-after"]
    assert "compaction_model_call_join_failed" in {gap.code for gap in observations.gaps}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "missing", "other_rollout", "other_model"])
async def test_adapter_requires_its_own_capture(tmp_path: Path, case: str) -> None:
    instance = _agent(tmp_path)
    if case == "disabled":
        instance.server_client.global_config_dict = {}
    call = _captured_call("call-compact", "msg-compact", compact=True)
    if case == "other_model":
        call["model_ref"] = {"type": "responses_api_models", "name": "other"}
    if case != "missing":
        store = CaptureStore(tmp_path)
        for captured in (_captured_call("call-before", "msg-before"), call, _captured_call("call-after", "msg-after")):
            store.record("other" if case == "other_rollout" else "0-0", captured)
    bundle = _compaction_bundle()
    associated = await instance._associate_compaction_calls(bundle, rollout_id="0-0")
    assert associated == bundle


@pytest.mark.asyncio
async def test_adapter_waits_for_claimed_capture_write(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path)
    store.mark_pending("0-0", "call-compact")

    async def finish_capture():
        await asyncio.sleep(0.01)
        for call in (
            _captured_call("call-before", "msg-before"),
            _captured_call("call-compact", "msg-compact", compact=True),
            _captured_call("call-after", "msg-after"),
        ):
            store.record("0-0", call)
        store.clear_pending("0-0", "call-compact")

    associated, _ = await asyncio.gather(
        _agent(tmp_path)._associate_compaction_calls(_compaction_bundle(), rollout_id="0-0"), finish_capture()
    )
    [compaction] = _records(associated, ContextCompactionObservation)
    assert [reference.model_call_id for reference in compaction.model_calls] == ["call-compact"]


@pytest.mark.parametrize("separator", ["\n", "\n\n", "\nunexpected content\n"])
def test_compaction_matches_native_continuation_suffix(separator: str) -> None:
    bundle = _compaction_bundle()
    [compaction] = _records(bundle, ContextCompactionObservation)
    compaction.summary = (
        "This session is being continued from a previous conversation that ran out of context. "
        "The summary below covers the earlier portion of the conversation.\n\nSummary:\nKeep this."
        "\n\nIf you need specific details from before compaction "
        "(like exact code snippets, error messages, or content you generated), "
        "read the full transcript at: /config/projects/work/session.jsonl"
        + separator
        + "Continue the conversation from where it left off without asking the user "
        "any further questions. Resume directly — do not acknowledge the summary, do not recap "
        'what was happening, do not preface with "I\'ll continue" or similar. '
        "Pick up the last task as if the break never happened."
    )
    calls = [
        ModelCallRecord.model_validate(call | {"call_index": index})
        for index, call in enumerate(
            [
                _captured_call("call-before", "msg-before"),
                _captured_call("call-compact", "msg-compact", compact=True),
                _captured_call("call-after", "msg-after"),
            ]
        )
    ]
    associated = associate_claude_code_compaction_calls(bundle, calls)
    [compaction] = _records(associated, ContextCompactionObservation)
    assert [ref.model_call_id for ref in compaction.model_calls] == (
        [] if "unexpected content" in separator else ["call-compact"]
    )


def test_compaction_call_correlation_rejects_ambiguous_matches() -> None:
    calls = [
        ModelCallRecord.model_validate(_captured_call("call-before", "msg-before") | {"call_index": 0}),
        ModelCallRecord.model_validate(_captured_call("call-1", "msg-1", compact=True) | {"call_index": 1}),
        ModelCallRecord.model_validate(_captured_call("call-2", "msg-2", compact=True) | {"call_index": 2}),
        ModelCallRecord.model_validate(_captured_call("call-after", "msg-after") | {"call_index": 3}),
    ]

    associated = associate_claude_code_compaction_calls(_compaction_bundle(), calls)

    [compaction] = _records(associated, ContextCompactionObservation)
    assert compaction.model_calls == []
    assert "compaction_model_call_reference_unavailable" in {gap.code for gap in associated.gaps}
    assert "compaction_model_call_match_ambiguous" in {gap.code for gap in associated.gaps}


@pytest.mark.parametrize("case", ["marker_in_history", "failed_call", "outside_boundaries"])
def test_compaction_call_correlation_rejects_inexact_matches(case: str) -> None:
    calls = [
        ModelCallRecord.model_validate(_captured_call("call-before", "msg-before") | {"call_index": 0}),
        ModelCallRecord.model_validate(
            _captured_call("call-compact", "msg-compact", compact=True) | {"call_index": 1}
        ),
        ModelCallRecord.model_validate(_captured_call("call-after", "msg-after") | {"call_index": 2}),
    ]
    compact_call = calls[1]
    if case == "marker_in_history":
        compact_call.request["messages"].append({"role": "user", "content": "continue"})
    elif case == "failed_call":
        compact_call.status_code = 500
    else:
        compact_call.call_index = 3

    associated = associate_claude_code_compaction_calls(_compaction_bundle(), calls)

    [compaction] = _records(associated, ContextCompactionObservation)
    assert compaction.model_calls == []
    assert "compaction_model_call_reference_unavailable" in {gap.code for gap in associated.gaps}


def test_reused_ids_keep_exact_response_attempts_and_replay(tmp_path: Path) -> None:
    from nemo_gym.trajectory import TrajectoryModelCall, normalize_tool_calls

    events = []
    for index, failed in enumerate((False, True)):
        response_id = f"response-{index}"
        events.append(
            _assistant(
                "root",
                str(10 + index * 2),
                response_id,
                {"type": "tool_use", "id": "reused", "name": "Bash", "input": {}},
            )
        )
        result = _tool_result("root", str(11 + index * 2), "reused", is_error=failed)
        result.update(uuid=f"result-{index}", sourceToolAssistantUUID=f"{response_id}-event")
        events.append(result)
    # Numeric epoch values are legitimate transcript timestamps.
    for event in events:
        event["timestamp"] = float(event["timestamp"])
    path = tmp_path / "projects" / "work" / "session.jsonl"
    _write(path, *events)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    tools = _records(bundle, ToolCallObservation)
    assert len(tools) == 2
    assert len({tool.span_id for tool in tools}) == 2
    assert [tool.model_call_ref.response_id for tool in tools] == ["response-0", "response-1"]
    assert [tool.status for tool in tools] == ["completed", "failed"]
    assert [tool.timings[0].duration_ms for tool in tools] == [1000, 1000]
    calls = [
        TrajectoryModelCall(
            model_call_id=f"capture-{index}",
            response_metadata={"model_ref": MODEL_REF, "response_id": f"response-{index}", "status_code": 200},
            response={"content": [{"type": "tool_use", "id": "reused", "name": "Bash", "input": {}}]},
        )
        for index in range(2)
    ]
    normalized, gaps = normalize_tool_calls(
        tools,
        capture_key="capture",
        producer_id="claude_code",
        model_calls=calls,
        invocations=_records(bundle, AgentInvocation),
    )
    assert [tool.model_call_id for tool in normalized] == ["capture-0", "capture-1"]
    assert "tool_failure_detail_unavailable" in {gap.code for gap in gaps}
    replay = AgentObservationBundle.model_validate_json(bundle.model_dump_json())
    assert replay.model_dump() == bundle.model_dump()
    repeated, _ = normalize_tool_calls(
        [*normalized, *normalized],
        capture_key="capture",
        producer_id="claude_code",
        model_calls=calls,
        invocations=_records(bundle, AgentInvocation),
    )
    # Canonical trajectories omit input-only join evidence such as model_call_ref.
    assert [tool.model_dump(exclude={"model_call_ref"}) for tool in repeated] == [
        tool.model_dump(exclude={"model_call_ref"}) for tool in normalized
    ]
    # Copying the same persisted transcript does not create additional tool attempts.
    _write(path.with_name("copy.jsonl"), *events)
    assert _records(extract_claude_code_observations(tmp_path, model_ref=MODEL_REF), ToolCallObservation) == tools


def test_ambiguous_reused_ids_keep_starts_and_failed_result(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _assistant(
            "root", "2026-07-22T10:00:01Z", "first", {"type": "tool_use", "id": "reused", "name": "Bash", "input": {}}
        ),
        _assistant(
            "root", "2026-07-22T10:00:02Z", "second", {"type": "tool_use", "id": "reused", "name": "Bash", "input": {}}
        ),
        _tool_result("root", "2026-07-22T10:00:03Z", "reused", is_error=True),
    )
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    tools = _records(bundle, ToolCallObservation)
    assert len(tools) == 3
    assert [tool.status for tool in tools] == ["incomplete", "incomplete", "failed"]
    assert all(tool.timings[0].completed_at is None for tool in tools[:2])
    assert tools[2].timings[0].started_at is None
    assert tools[2].model_call_ref is None
    assert {gap.span_id for gap in bundle.gaps if gap.code == "ambiguous_tool_artifact"} == {
        tool.span_id for tool in tools
    }


def test_parent_completion_does_not_complete_child_and_partial_identity_is_stable(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "work" / "root.jsonl"
    child = root.parent / "root" / "subagents" / "child.jsonl"
    _write(
        root,
        _assistant(
            "root",
            "2026-07-22T10:00:01Z",
            "parent-response",
            {"type": "tool_use", "id": "spawn", "name": "Agent", "input": {}},
        ),
        _tool_result("root", "2026-07-22T10:00:03Z", "spawn", child_id="child"),
    )
    start = _assistant(
        "root",
        "2026-07-22T10:00:02Z",
        "child-response",
        {"type": "tool_use", "id": "pending", "name": "Bash", "input": {}},
        agent="child",
    )
    _write(child, start, '{"truncated":')
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF, root_status="completed")
    tools = {tool.tool_call_id: tool for tool in _records(bundle, ToolCallObservation)}
    assert tools["spawn"].status == "completed"
    assert tools["pending"].status == "incomplete"
    assert tools["pending"].timings[0].completed_at is None
    assert tools["pending"].parent_span_id == tools["spawn"].span_id
    _write(child, start, _tool_result("root", "2026-07-22T10:00:04Z", "pending", agent="child"))
    complete = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    completed = {tool.tool_call_id: tool for tool in _records(complete, ToolCallObservation)}
    assert completed["pending"].span_id == tools["pending"].span_id
    assert completed["pending"].timings[0].duration_ms == 2000


def test_records_without_uuid_use_file_position_and_unknown_model(tmp_path: Path) -> None:
    start = _event(
        "root",
        "assistant",
        "2026-07-22T10:00:01Z",
        [{"type": "tool_use", "id": "pending", "name": "Bash", "input": {}}],
    )
    path = tmp_path / "projects" / "work" / "root.jsonl"
    _write(path, start)
    first = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    [tool] = _records(first, ToolCallObservation)
    assert tool.span_id
    assert tool.model_call_ref is None
    assert tool.status == "incomplete"
    _write(path, start, _tool_result("root", "2026-07-22T10:00:03Z", "pending"))
    [complete] = _records(extract_claude_code_observations(tmp_path, model_ref=MODEL_REF), ToolCallObservation)
    assert complete.span_id == tool.span_id


def test_result_with_different_exact_assistant_reference_stays_unmatched(tmp_path: Path) -> None:
    result = _tool_result("root", "2026-07-22T10:00:02Z", "reused", is_error=True)
    result["sourceToolAssistantUUID"] = "missing-assistant"
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        _assistant(
            "root",
            "2026-07-22T10:00:01Z",
            "observed",
            {"type": "tool_use", "id": "reused", "name": "Bash", "input": {}},
        ),
        result,
    )
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    requested, orphan = _records(bundle, ToolCallObservation)
    assert requested.status == "incomplete"
    assert requested.timings[0].completed_at is None
    assert orphan.status == "failed"
    assert orphan.timings[0].started_at is None
    assert orphan.model_call_ref is None


async def test_default_v1_export_keeps_claude_ids_and_is_replay_stable(tmp_path: Path) -> None:
    from nemo_gym.rollouts import enrich_run_result

    events = []
    for index in range(2):
        tool_id = f"tool-{index}"
        events.append(
            _assistant(
                "root",
                f"2026-07-22T10:00:0{index * 2}Z",
                f"response-{index}",
                {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}},
            )
        )
        result = _tool_result("root", f"2026-07-22T10:00:0{index * 2 + 1}Z", tool_id)
        result.update(uuid=f"result-{index}", sourceToolAssistantUUID=f"response-{index}-event")
        result["message"]["content"][0]["content"] = f"output-{index}"
        events.append(result)
    _write(tmp_path / "config" / "projects" / "work" / "root.jsonl", *events)
    bundle = extract_claude_code_observations(tmp_path / "config", root_status="completed")
    captures = tmp_path / "captures"
    captures.mkdir()
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    record = {"ng_agent_observations": bundle.model_dump(mode="json")}
    first = None
    for _ in range(3):
        await enrich_run_result(row, record, capture_dirs=[captures], observability_enabled=True)
        trajectory = record["ng_trajectory"]
        assert trajectory["schema_version"] == "1.0"
        tools = trajectory["tool_calls"]
        conversation = trajectory["invocations"][0]["conversation"]
        assert [tool["output"] for tool in tools] == ["output-0", "output-1"]
        assert all("span_id" not in tool and "timings" not in tool for tool in tools)
        assert all(tool["duration_ms"] == 1000 for tool in tools)
        assert [tool["tool_call_id"] for tool in tools] == ["tool-0", "tool-1"]
        assert [item["call_id"] for item in conversation if "call_id" in item] == [
            "tool-0",
            "tool-0",
            "tool-1",
            "tool-1",
        ]
        if first is None:
            first = copy.deepcopy(trajectory)
        assert trajectory == first


def _bash(tool_id: str) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {}}


async def _enriched_trajectory(tmp_path: Path, bundle: AgentObservationBundle, schema: str) -> dict:
    from nemo_gym.rollouts import enrich_run_result

    captures = tmp_path / f"captures-{schema}"
    captures.mkdir()
    record = {"ng_agent_observations": bundle.model_dump(mode="json")}
    await enrich_run_result(
        {"_ng_task_index": 0, "_ng_rollout_index": 0},
        record,
        capture_dirs=[captures],
        observability_enabled=True,
        trajectory_schema_version=schema,
    )
    return record["ng_trajectory"]


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
async def test_split_parallel_records_without_uuid_stay_separate_attempts(tmp_path: Path, schema: str) -> None:
    # Claude persists parallel tool_use blocks as one record each, all sharing the response ID.
    _write(
        tmp_path / "config" / "projects" / "work" / "root.jsonl",
        _event("root", "assistant", "2026-07-22T10:00:01Z", [_bash("t1")], message_id="msg-1"),
        _event("root", "assistant", "2026-07-22T10:00:01Z", [_bash("t2")], message_id="msg-1"),
        _event("root", "user", "2026-07-22T10:00:02Z", [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}]),
        _event("root", "user", "2026-07-22T10:00:03Z", [{"type": "tool_result", "tool_use_id": "t2", "content": "b"}]),
    )
    bundle = extract_claude_code_observations(tmp_path / "config", model_ref=MODEL_REF, root_status="completed")
    tools = _records(bundle, ToolCallObservation)
    assert len({tool.span_id for tool in tools}) == 2
    trajectory = await _enriched_trajectory(tmp_path, bundle, schema)
    assert sorted(tool["tool_call_id"] for tool in trajectory["tool_calls"]) == ["t1", "t2"]
    assert sorted(tool["output"] for tool in trajectory["tool_calls"]) == ["a", "b"]
    assert "tool_attempt_identity_conflict" not in {gap["code"] for gap in trajectory["gaps"]}


def test_split_parallel_records_join_exact_assistant_references(tmp_path: Path) -> None:
    events = [
        _event("root", "assistant", "2026-07-22T10:00:01Z", [_bash("t1")], message_id="msg-1", uuid="a1"),
        _event("root", "assistant", "2026-07-22T10:00:01Z", [_bash("t2")], message_id="msg-1", uuid="a2"),
    ]
    # Results arrive in reverse order; each names its own requesting record.
    for tool_id, assistant, timestamp in (("t2", "a2", "2026-07-22T10:00:02Z"), ("t1", "a1", "2026-07-22T10:00:04Z")):
        result = _tool_result("root", timestamp, tool_id)
        result["sourceToolAssistantUUID"] = assistant
        events.append(result)
    _write(tmp_path / "projects" / "work" / "root.jsonl", *events)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    tools = {tool.tool_call_id: tool for tool in _records(bundle, ToolCallObservation)}
    assert [tools[name].timings[0].duration_ms for name in ("t1", "t2")] == [3000, 1000]
    assert {tool.source_message_id for tool in tools.values()} == {"a1", "a2"}
    assert {tool.model_call_ref.response_id for tool in tools.values()} == {"msg-1"}
    assert {gap.code for gap in bundle.gaps if gap.span_id} == {"tool_clock_domain_unavailable"}


def test_parent_uuid_attributes_a_result_without_an_exact_reference(tmp_path: Path) -> None:
    result = _tool_result("root", "2026-07-22T10:00:04Z", "reused")
    result["parentUuid"] = "second-event"
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        _assistant("root", "2026-07-22T10:00:01Z", "first", _bash("reused")),
        _assistant("root", "2026-07-22T10:00:03Z", "second", _bash("reused")),
        result,
    )
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    first, second = _records(bundle, ToolCallObservation)
    assert (first.status, second.status) == ("incomplete", "completed")
    assert second.model_call_ref.response_id == "second"
    assert second.timings[0].duration_ms == 1000
    assert first.timings[0].completed_at is None
    assert {gap.code for gap in bundle.gaps if gap.span_id == first.span_id} == {
        "tool_result_missing",
        "tool_clock_domain_unavailable",
    }
    assert "ambiguous_tool_artifact" not in {gap.code for gap in bundle.gaps}


def test_unfinished_reused_start_is_missing_not_ambiguous_when_results_are_attributed(tmp_path: Path) -> None:
    result = _tool_result("root", "2026-07-22T10:00:02Z", "reused")
    result["sourceToolAssistantUUID"] = "first-event"
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        _assistant("root", "2026-07-22T10:00:01Z", "first", _bash("reused")),
        result,
        # The run stopped before this second request returned.
        _assistant("root", "2026-07-22T10:00:03Z", "second", _bash("reused")),
    )
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    first, second = _records(bundle, ToolCallObservation)
    assert (first.status, second.status) == ("completed", "incomplete")
    assert {gap.code for gap in bundle.gaps if gap.span_id == second.span_id} == {
        "tool_result_missing",
        "tool_clock_domain_unavailable",
    }
    assert "ambiguous_tool_artifact" not in {gap.code for gap in bundle.gaps}


def test_timestamp_without_offset_is_reported_missing(tmp_path: Path) -> None:
    _write(
        tmp_path / "projects" / "work" / "root.jsonl",
        _assistant("root", "2026-07-22T10:00:01", "first", _bash("naive")),
        _tool_result("root", "2026-07-22T10:00:02Z", "naive"),
    )
    bundle = extract_claude_code_observations(
        tmp_path, model_ref=MODEL_REF, agent_instance="claude-code-agent:a", clock_domain="claude-code-wall:a"
    )
    [tool] = _records(bundle, ToolCallObservation)
    assert tool.timings[0].started_at is None
    assert tool.timings[0].duration_ms is None
    assert tool.timings[0].completed_at is not None
    assert "tool_start_timestamp_missing" in {gap.code for gap in bundle.gaps if gap.span_id == tool.span_id}


async def test_default_v1_keeps_reused_id_attempts_with_explicit_gaps(tmp_path: Path) -> None:
    events = []
    for index in range(2):
        events.append(_assistant("root", f"2026-07-22T10:00:0{index * 2}Z", f"response-{index}", _bash("reused")))
        result = _tool_result("root", f"2026-07-22T10:00:0{index * 2 + 1}Z", "reused")
        result.update(uuid=f"result-{index}", sourceToolAssistantUUID=f"response-{index}-event")
        events.append(result)
    _write(tmp_path / "config" / "projects" / "work" / "root.jsonl", *events)
    bundle = extract_claude_code_observations(tmp_path / "config", model_ref=MODEL_REF, root_status="completed")
    trajectory = await _enriched_trajectory(tmp_path, bundle, "1.0")
    tools = trajectory["tool_calls"]
    # V1 has one call ID per invocation, so both attempts keep Claude's ID without an output.
    assert [tool["tool_call_id"] for tool in tools] == ["reused", "reused"]
    assert [tool["status"] for tool in tools] == ["completed", "completed"]
    assert all(tool["output"] is None and tool["duration_ms"] == 1000 for tool in tools)
    assert {"tool_native_identity_unrepresentable", "tool_output_ambiguous"} <= {
        gap["code"] for gap in trajectory["gaps"]
    }


@pytest.mark.parametrize("schema", ["1.0", "2.0"])
async def test_copied_transcript_keeps_one_attempt_with_its_output(tmp_path: Path, schema: str) -> None:
    result = _tool_result("root", "2026-07-22T10:00:02Z", "toolu")
    result["sourceToolAssistantUUID"] = "response-event"
    result["message"]["content"][0]["content"] = "ok"
    events = [_assistant("root", "2026-07-22T10:00:01Z", "response", _bash("toolu")), result]
    path = tmp_path / "config" / "projects" / "work" / "root.jsonl"
    _write(path, *events)
    single = extract_claude_code_observations(tmp_path / "config", model_ref=MODEL_REF, root_status="completed")
    _write(path.with_name("copy.jsonl"), *events)
    copied = extract_claude_code_observations(tmp_path / "config", model_ref=MODEL_REF, root_status="completed")
    assert copied.model_dump() == single.model_dump()
    trajectory = await _enriched_trajectory(tmp_path, copied, schema)
    [tool] = trajectory["tool_calls"]
    assert tool["output"] == "ok"
    assert tool["tool_call_id"] == "toolu"
    [invocation] = trajectory["invocations"]
    assert [item["type"] for item in invocation["conversation"]] == ["function_call", "function_call_output"]
    if schema == "2.0":
        assert tool["timings"][0]["duration_ms"] == 1000
    else:
        assert tool["duration_ms"] == 1000


def _spawn_result(timestamp: str, record: str, assistant: str) -> dict:
    result = _tool_result("root", timestamp, "spawn", child_id="child")
    result.update(uuid=record, sourceToolAssistantUUID=assistant)
    return result


def _child_tool() -> dict:
    return _assistant("root", "2026-07-22T10:00:02Z", "child-response", _bash("child-tool"), agent="child")


def test_competing_result_records_leave_the_subagent_parent_unset(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "work" / "root.jsonl"
    _write(
        root,
        _assistant("root", "2026-07-22T10:00:01Z", "a1", {**_bash("spawn"), "name": "Agent"}),
        _spawn_result("2026-07-22T10:00:03Z", "r1", "a1-event"),
        _assistant("root", "2026-07-22T10:00:04Z", "a2", {**_bash("spawn"), "name": "Agent"}),
        _spawn_result("2026-07-22T10:00:05Z", "r2", "a2-event"),
    )
    _write(root.parent / "root" / "subagents" / "agent-child.jsonl", _child_tool())
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    tools = _records(bundle, ToolCallObservation)
    spawns = [tool for tool in tools if tool.native_tool_call_id == "spawn"]
    [child_tool] = [tool for tool in tools if tool.invocation_id == "child"]
    assert len({tool.span_id for tool in spawns}) == 2
    assert [tool.status for tool in spawns] == ["completed", "completed"]
    assert child_tool.parent_span_id is None
    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}
    assert invocations["child"].parent_invocation_id is None
    assert invocations["child"].spawned_by_tool_call_id is None
    assert "conflicting_subagent_parent" in {gap.code for gap in bundle.gaps if gap.invocation_id == "child"}


def test_one_result_record_links_the_subagent_even_when_replayed(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "work" / "root.jsonl"
    events = [
        _assistant("root", "2026-07-22T10:00:01Z", "a1", {**_bash("spawn"), "name": "Agent"}),
        _spawn_result("2026-07-22T10:00:03Z", "r1", "a1-event"),
    ]
    _write(root, *events)
    _write(root.with_name("copy.jsonl"), *events)
    _write(root.parent / "root" / "subagents" / "agent-child.jsonl", _child_tool())
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    tools = _records(bundle, ToolCallObservation)
    [spawn] = [tool for tool in tools if tool.native_tool_call_id == "spawn"]
    [child_tool] = [tool for tool in tools if tool.invocation_id == "child"]
    assert child_tool.parent_span_id == spawn.span_id
    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}
    assert invocations["child"].parent_invocation_id == "root"
    assert invocations["child"].spawned_by_tool_call_id == "spawn"
    assert "conflicting_subagent_parent" not in {gap.code for gap in bundle.gaps}
