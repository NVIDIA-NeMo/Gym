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
    read_transcripts,
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
            observation_collector(read_transcripts(tmp_path), {"status": "completed"})
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


# The text of a summary record of Claude Code 2.1.312, which kept the latest messages of the conversation.
SUMMARY_OF_2_1_312 = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
    "<system-reminder>\n<tools-used-before-summary>Bash</tools-used-before-summary>\n</system-reminder>\n"
    "Summary:\nKeep this."
    "\n\nIf you need specific details from before compaction "
    "(like exact code snippets, error messages, or content you generated), "
    "read the full transcript at: /config/projects/work/session.jsonl"
    "\nContinue the conversation from where it left off without asking the user "
    "any further questions. Resume directly — do not acknowledge the summary, do not recap "
    'what was happening, do not preface with "I\'ll continue" or similar. '
    "Pick up the last task as if the break never happened."
    "\n\nThe messages after this summary are the most recent messages from before compaction, kept verbatim. "
    "The summary was written without seeing them, so something it says has not happened yet may already have "
    "happened in them."
)


@pytest.mark.parametrize("tools", ["Bash", "Bash, Read", "unexpected\ncontent"])
def test_compaction_matches_summary_record_of_claude_code_2_1_312(tools: str) -> None:
    """Claude Code 2.1.312 keeps the latest messages and summarizes the earlier ones. Its record names the tools
    used before the summary in front of the summary and ends with a sentence on the kept messages."""
    bundle = _compaction_bundle()
    [compaction] = _records(bundle, ContextCompactionObservation)
    compaction.summary = SUMMARY_OF_2_1_312.replace("Bash", tools)
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
    assert [ref.model_call_id for ref in compaction.model_calls] == ([] if "\n" in tools else ["call-compact"])


def _kept_messages_transcript(tmp_path: Path, turns: int, session: str = "session") -> None:
    """Write the transcript of a conversation that Claude Code 2.1.312 compacted before its last model call.

    ``turns`` model calls precede the compaction. The ``compact_boundary`` record states the token counts. Two
    attachment records follow it, and then the summary record, each the child of the record before it. The ids
    of the records and of the replies of a session other than ``session`` end with its name.
    """
    end = "" if session == "session" else f"-{session}"
    link = {"sessionId": session, "timestamp": "2026-07-22T10:00:10Z"}
    kept = {"anchorUuid": f"s{end}"}
    metadata = {"trigger": "auto", "preTokens": 197000, "postTokens": 2400, "preservedSegment": kept}
    _write(
        tmp_path / "projects" / "work" / f"{session}.jsonl",
        _event(session, "user", "2026-07-22T10:00:00Z", "prompt", uuid=f"prompt{end}"),
        *(
            _assistant(session, f"2026-07-22T10:00:0{turn}Z", f"msg-{turn}{end}", {"type": "text", "text": "ok"})
            for turn in range(1, turns + 1)
        ),
        {"type": "system", "subtype": "compact_boundary", "uuid": f"b{end}", "compactMetadata": metadata} | link,
        {"type": "attachment", "uuid": f"a1{end}", "parentUuid": f"b{end}", "attachment": {"type": "session_context"}}
        | link,
        {"type": "attachment", "uuid": f"a2{end}", "parentUuid": f"a1{end}", "attachment": {"type": "date"}} | link,
        _event(session, "user", "2026-07-22T10:00:10Z", SUMMARY_OF_2_1_312, uuid=f"s{end}", parentUuid=f"a2{end}")
        | {"isCompactSummary": True},
        _assistant(session, "2026-07-22T10:00:20Z", f"msg-last{end}", {"type": "text", "text": "Done."}),
    )


def _capture(*calls: dict) -> list[ModelCallRecord]:
    """Return the records of captured calls in their order, each with 10 input tokens and 2 output tokens."""
    tokens = {"tokens_in": 10, "tokens_out": 2, "tokens_total": 12}
    return [ModelCallRecord.model_validate(call | tokens | {"call_index": index}) for index, call in enumerate(calls)]


def _unused_summary_call(call_id: str, block: dict) -> dict:
    """Return a captured summary call whose answer is ``block``, which is not the persisted summary."""
    return _captured_call(call_id, f"msg-{call_id}", compact=True) | {
        "response": {"id": f"msg-{call_id}", "content": [block]}
    }


def test_summary_record_behind_attachments_joins_its_boundary_record(tmp_path: Path) -> None:
    """One compaction is one observation: the boundary record states its token counts and the summary record its
    summary, and the usage counts the summary call."""
    _kept_messages_transcript(tmp_path, turns=1)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    [compaction] = _records(bundle, ContextCompactionObservation)
    assert (compaction.tokens_before, compaction.tokens_after, compaction.trigger) == (197000, 2400, "automatic")
    assert compaction.summary == SUMMARY_OF_2_1_312 and compaction.outcome == "completed"
    assert (compaction.before_model_call.response_id, compaction.after_model_call.response_id) == ("msg-1", "msg-last")

    calls = _capture(
        _captured_call("call-1", "msg-1"),
        _captured_call("call-summary", "msg-summary", compact=True),
        _captured_call("call-last", "msg-last"),
    )
    associated = associate_claude_code_compaction_calls(bundle, calls)

    [compaction] = _records(associated, ContextCompactionObservation)
    assert [reference.model_call_id for reference in compaction.model_calls] == ["call-summary"]
    assert "compaction_model_call_reference_unavailable" not in {gap.code for gap in associated.gaps}
    usage = captured_claude_code_usage(associated, calls)
    assert (usage.input_tokens, usage.output_tokens) == (30, 6)


REASONING_ALONE = {"type": "thinking", "thinking": "<summary>Keep this.</summary>"}
NO_SUMMARY_BLOCK = {"type": "text", "text": "The conversation so far, in plain words."}


@pytest.mark.parametrize(
    ("unused", "attempts", "block"),
    [(2, 2, REASONING_ALONE), (1, 2, REASONING_ALONE), (3, 2, REASONING_ALONE), (1, 0, NO_SUMMARY_BLOCK)],
    ids=["as many calls as failed attempts", "fewer calls", "more calls", "an answer without a summary block"],
)
def test_summary_calls_without_used_answer_are_invocation_calls(
    tmp_path: Path, unused: int, attempts: int, block: dict
) -> None:
    """Claude Code asks for a summary again after an answer of reasoning alone, which its events state as a failed
    attempt, and after an answer without a summary block. It writes no record for a call whose answer it did not
    use. The invocation names every such call, so the usage counts it, and the failed attempts take the calls
    where there are as many of both."""
    _kept_messages_transcript(tmp_path, turns=3)
    failed_attempts = [{"invocation_id": "session", "outcome": "failed"}] * attempts
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF, compaction_attempts=failed_attempts)
    calls = _capture(
        _captured_call("call-1", "msg-1"),
        *(_unused_summary_call(f"unused-{number}", block) for number in range(unused)),
        _captured_call("call-2", "msg-2"),
        _captured_call("call-3", "msg-3"),
        _captured_call("call-summary", "msg-summary", compact=True),
        _captured_call("call-last", "msg-last"),
    )

    associated = associate_claude_code_compaction_calls(bundle, calls)

    compactions = _records(associated, ContextCompactionObservation)
    of_the_failed = [
        [reference.model_call_id for reference in record.model_calls]
        for record in compactions
        if record.outcome == "failed"
    ]
    [completed] = [record for record in compactions if record.outcome == "completed"]
    [invocation] = _records(associated, AgentInvocation)
    names = [f"unused-{number}" for number in range(unused)]
    assert of_the_failed == ([[name] for name in names] if unused == attempts else [[]] * attempts)
    assert [reference.model_call_id for reference in completed.model_calls] == ["call-summary"]
    # In the order of the capture.
    assert [reference.model_call_id or reference.response_id for reference in invocation.model_calls] == [
        "msg-1",
        *names,
        "msg-2",
        "msg-3",
        "call-summary",
        "msg-last",
    ]
    usage = captured_claude_code_usage(associated, calls)
    assert (usage.input_tokens, usage.output_tokens) == (10 * len(calls), 2 * len(calls))


def test_summary_record_behind_message_joins_no_earlier_boundary_record(tmp_path: Path) -> None:
    """The search for the boundary record of a summary ends at a record that holds a message. A summary record
    without a boundary record of its own, behind a later message, is a compaction of its own."""
    _kept_messages_transcript(tmp_path, turns=1)
    transcript = tmp_path / "projects" / "work" / "session.jsonl"
    later_summary = SUMMARY_OF_2_1_312.replace("Keep this.", "Keep that.")
    last = _assistant("session", "2026-07-22T10:00:20Z", "msg-last", {"type": "text", "text": "Done."})
    _write(
        transcript,
        *transcript.read_text().splitlines()[:-1],
        last | {"parentUuid": "s"},
        _event("session", "user", "2026-07-22T10:00:30Z", later_summary, uuid="s2", parentUuid=last["uuid"])
        | {"isCompactSummary": True},
    )

    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)

    first, second = _records(bundle, ContextCompactionObservation)
    assert (first.summary, first.tokens_before) == (SUMMARY_OF_2_1_312, 197000)
    assert (second.summary, second.tokens_before) == (later_summary, None)
    assert second.before_model_call.response_id == "msg-last"


def test_unused_summary_call_joins_neither_of_two_invocations(tmp_path: Path) -> None:
    """With a compaction in each of two invocations, nothing says which of them made a request for a summary whose
    answer was not used. No invocation names it, and the usage is not stated as a sum without it."""
    _kept_messages_transcript(tmp_path, turns=1)
    _kept_messages_transcript(tmp_path, turns=1, session="other")
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    calls = _capture(
        _captured_call("call-1", "msg-1"),
        _unused_summary_call("unused", REASONING_ALONE),
        _captured_call("call-summary", "msg-summary", compact=True),
        _captured_call("call-last", "msg-last"),
        _captured_call("call-1-other", "msg-1-other"),
        _captured_call("call-summary-other", "msg-summary-other", compact=True),
        _captured_call("call-last-other", "msg-last-other"),
    )

    associated = associate_claude_code_compaction_calls(bundle, calls)

    named = {
        invocation.invocation_id: [
            reference.model_call_id or reference.response_id for reference in invocation.model_calls
        ]
        for invocation in _records(associated, AgentInvocation)
    }
    assert named == {
        "session": ["msg-1", "call-summary", "msg-last"],
        "other": ["msg-1-other", "call-summary-other", "msg-last-other"],
    }
    assert captured_claude_code_usage(associated, calls) is None


def test_summary_request_without_compaction_leaves_usage_unreconciled(tmp_path: Path) -> None:
    """A capture can hold a request for a summary where the transcript and the events state no compaction. No
    invocation names the call, so the sum over the named calls is not the usage."""
    _write(
        tmp_path / "projects" / "work" / "session.jsonl",
        _event("session", "user", "2026-07-22T10:00:00Z", "prompt", uuid="prompt"),
        _assistant("session", "2026-07-22T10:00:01Z", "msg-1", {"type": "text", "text": "ok"}),
    )
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    calls = _capture(_captured_call("call-1", "msg-1"), _unused_summary_call("unused", NO_SUMMARY_BLOCK))

    associated = associate_claude_code_compaction_calls(bundle, calls)

    [invocation] = _records(associated, AgentInvocation)
    assert [reference.response_id for reference in invocation.model_calls] == ["msg-1"]
    assert captured_claude_code_usage(associated, calls) is None
    assert captured_claude_code_usage(associated, calls[:1]).input_tokens == 10


def test_tool_result_quoting_summary_instruction_is_no_summary_request(tmp_path: Path) -> None:
    """Claude Code appends its instruction to the last user message as a text block, beside the tool results of
    that message. A tool result that quotes the instruction makes no summary call of an unowned call."""
    _kept_messages_transcript(tmp_path, turns=1)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    instruction = "Your task is to create a detailed summary of this conversation."
    result = {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": instruction}]}
    quoting = _captured_call("call-quoting", "msg-quoting") | {
        "request": {"messages": [{"role": "user", "content": [result, {"type": "text", "text": "Go on."}]}]}
    }
    appended = _captured_call("call-summary", "msg-summary", compact=True) | {
        "request": {"messages": [{"role": "user", "content": [result, {"type": "text", "text": instruction}]}]}
    }
    calls = _capture(_captured_call("call-1", "msg-1"), quoting, appended, _captured_call("call-last", "msg-last"))

    associated = associate_claude_code_compaction_calls(bundle, calls)

    [compaction] = _records(associated, ContextCompactionObservation)
    [invocation] = _records(associated, AgentInvocation)
    assert [reference.model_call_id for reference in compaction.model_calls] == ["call-summary"]
    assert [reference.model_call_id or reference.response_id for reference in invocation.model_calls] == [
        "msg-1",
        "call-summary",
        "msg-last",
    ]
    usage = captured_claude_code_usage(associated, calls)
    assert (usage.input_tokens, usage.output_tokens) == (30, 6)


def test_compaction_without_summary_call_leaves_usage_unreconciled(tmp_path: Path) -> None:
    _kept_messages_transcript(tmp_path, turns=1)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF)
    calls = _capture(_captured_call("call-1", "msg-1"), _captured_call("call-last", "msg-last"))

    associated = associate_claude_code_compaction_calls(bundle, calls)

    assert captured_claude_code_usage(associated, calls) is None
    assert "compaction_model_call_reference_unavailable" in {gap.code for gap in associated.gaps}


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
