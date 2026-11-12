# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
    captured_claude_code_usage,
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
    )

    invocations = {invocation.invocation_id: invocation for invocation in _records(bundle, AgentInvocation)}
    assert set(invocations) == {session, child, grandchild}
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
    assert timings["tool-fast"].duration_ms == pytest.approx(1000)
    assert timings["tool-child"].duration_ms == pytest.approx(3000)
    assert timings["tool-grandchild"].duration_ms == pytest.approx(1000)
    assert timings["tool-grandchild"].status == "timeout"
    assert all(tool.timing_source == "artifact" for tool in timings.values())
    assert bundle.gaps == []


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
    assert compaction.trigger == "automatic"
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
        "tool_timing_invalid",
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
        episode = await instance._create_episode(body, rollout_id="0-0")
        observations, response = episode.observations, episode.response
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
        response = await instance._sandbox_response(state, body, prompt="Continue", system=None)
        observations = state.observations

    assert (response.usage.input_tokens, response.usage.output_tokens) == (30, 6)
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
        observations = await _agent(tmp_path)._reconcile_captured_calls(_compaction_bundle(), rollout_id="0-0")
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
    associated = await instance._reconcile_captured_calls(bundle, rollout_id="0-0")
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
        _agent(tmp_path)._reconcile_captured_calls(_compaction_bundle(), rollout_id="0-0"), finish_capture()
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


@pytest.mark.parametrize("broken", [None, "missing", "ambiguous", "unknown_usage"])
def test_captured_usage_counts_owned_calls_once_including_cache(broken):
    from nemo_gym.trajectory import build_model_call_record

    exchanges = [
        _captured_call("call-before", "msg-before"),
        _captured_call("call-compact", "msg-compact", compact=True),
        _captured_call("call-after", "msg-after"),
        _captured_call("unrelated", "other"),
    ]
    for exchange in exchanges:
        exchange["response"]["usage"].update(cache_read_input_tokens=3, cache_creation_input_tokens=2)
    calls = [build_model_call_record(e, call_index=i) for i, e in enumerate(exchanges)]
    bundle = associate_claude_code_compaction_calls(_compaction_bundle(), calls)
    invocation = _records(bundle, AgentInvocation)[0]
    invocation.model_calls.append(invocation.model_calls[0].model_copy())
    if broken == "missing":
        calls.pop(0)
    elif broken == "ambiguous":
        calls.append(calls[0].model_copy())
    elif broken == "unknown_usage":
        calls[1].tokens_in = None
    usage = captured_claude_code_usage(bundle, calls)
    if broken:
        assert usage is None
    else:
        assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (45, 6, 51)
        assert usage.input_tokens_details.cached_tokens == 9
        assert usage.output_tokens_details.reasoning_tokens is None
