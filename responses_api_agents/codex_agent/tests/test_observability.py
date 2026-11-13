# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.trajectory import AgentInvocation, ToolCallObservation
from responses_api_agents.codex_agent.observability import (
    parse_codex_observations,
    parse_codex_usage,
    read_codex_observations,
)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_native_call_ids_arguments_and_full_output_are_preserved(tmp_path: Path, exit_code: int) -> None:
    folder = tmp_path / "sessions"
    folder.mkdir()
    output = f"Chunk ID: abc\nProcess exited with code {exit_code}\nOutput:\nProcess exited with code 999"
    records = [
        {"type": "function_call", "call_id": "native-call", "name": "exec_command", "arguments": '{"cmd":"false"}'},
        {"type": "function_call_output", "call_id": "native-call", "output": output},
    ]
    (folder / "episode.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": f"2026-10-05T10:00:0{index}Z",
                    "payload": payload,
                }
            )
            for index, payload in enumerate(records)
        )
    )
    bundle = read_codex_observations(tmp_path, "invocation", returncode=0, timed_out=False)
    invocation, tool = bundle.records
    assert isinstance(invocation, AgentInvocation) and isinstance(tool, ToolCallObservation)
    assert invocation.conversation[0].call_id == tool.tool_call_id == "native-call"
    assert invocation.conversation[0].arguments == '{"cmd":"false"}'
    assert invocation.conversation[1].output == output
    assert tool.status == ("failed" if exit_code else "completed")
    assert tool.duration_ms == 1000
    assert not bundle.gaps


def test_missing_codex_session_record_is_explicit_and_preserves_failed_invocation(tmp_path: Path) -> None:
    bundle = read_codex_observations(tmp_path, "invocation", returncode=1, timed_out=False)
    assert bundle.records[0].status == "failed"
    assert bundle.records[0].invocation_id == "invocation"
    assert bundle.gaps[0].code == "agent_artifact_unavailable"


@pytest.mark.parametrize(
    ("record", "tool_timing_known"),
    [
        (None, False),
        ([], False),
        ("unexpected", False),
        ({"type": "response_item", "payload": {}}, False),
        ({"type": "event_msg", "payload": ["invalid"]}, True),
    ],
)
def test_malformed_native_record_reports_gap_without_losing_invocation(
    tmp_path: Path, record: object, tool_timing_known: bool
) -> None:
    """A record that is not a native event may have been a tool call, so the tool timing is unknown too."""
    folder = tmp_path / "sessions"
    folder.mkdir()
    (folder / "episode.jsonl").write_text(json.dumps(record) + "\n")
    bundle = read_codex_observations(tmp_path, "invocation", returncode=1, timed_out=False)
    assert bundle.records[0].invocation_id == "invocation"
    assert bundle.records[0].status == "failed"
    assert [gap.code for gap in bundle.gaps] == ["agent_artifact_record_unparseable"] + (
        [] if tool_timing_known else ["tool_timing_unavailable"]
    )


def test_multiple_native_session_files_preserve_record_boundaries(tmp_path: Path) -> None:
    folder = tmp_path / "sessions"
    folder.mkdir()
    for index in range(2):
        record = {
            "type": "response_item",
            "timestamp": f"2026-10-19T12:00:0{index}Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"message {index}"}],
            },
        }
        (folder / f"episode-{index}.jsonl").write_text(json.dumps(record) + "\n")
    bundle = read_codex_observations(tmp_path, "invocation", returncode=0, timed_out=False)
    assert len(bundle.records[0].conversation) == 2
    assert not bundle.gaps


@pytest.mark.parametrize("separator", ["\u0085", " ", " "])
def test_unicode_line_separators_inside_records_keep_the_tool_output(tmp_path: Path, separator: str) -> None:
    output = f"Chunk ID: abc\nProcess exited with code 0\nOutput:\nfirst{separator}second"
    records = [
        {"type": "function_call", "call_id": "call-1", "name": "exec_command", "arguments": '{"cmd":"cat f"}'},
        {"type": "function_call_output", "call_id": "call-1", "output": output},
    ]
    # Codex leaves these characters unescaped; the sandbox snapshot joins its files with "\n".
    transcript = "\n".join(
        json.dumps(
            {"type": "response_item", "timestamp": f"2026-10-05T10:00:0{index}Z", "payload": payload},
            ensure_ascii=False,
        )
        for index, payload in enumerate(records)
    )
    folder = tmp_path / "sessions"
    folder.mkdir()
    (folder / "episode.jsonl").write_text(transcript + "\n", encoding="utf-8")
    for bundle in (
        parse_codex_observations(transcript, "invocation", returncode=0, timed_out=False),
        read_codex_observations(tmp_path, "invocation", returncode=0, timed_out=False),
    ):
        invocation, tool = bundle.records
        assert invocation.conversation[1].output == output
        assert tool.status == "completed"
        assert not bundle.gaps


def test_native_usage_uses_latest_cumulative_total_without_adding_details() -> None:
    records = []
    for count in (20, 41):
        records.append(
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "total_token_usage": {
                                "input_tokens": count,
                                "output_tokens": 8,
                                "total_tokens": count + 8,
                                "cached_input_tokens": 0,
                                "reasoning_output_tokens": 0,
                            }
                        },
                    },
                }
            )
        )
    records.extend(["malformed", json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": None}})])
    usage = parse_codex_usage("\n".join(records))
    assert usage.input_tokens == 41
    assert usage.output_tokens == 8
    assert usage.total_tokens == 49
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None


@pytest.mark.parametrize("usage", [None, {}, {"input_tokens": True, "output_tokens": 8, "total_tokens": 9}])
def test_missing_or_invalid_native_usage_is_unknown(usage: object) -> None:
    record = {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}}
    assert parse_codex_usage(json.dumps(record)) is None


@pytest.mark.parametrize("info", [{"renamed_totals": {}}, {"total_token_usage": {"input_tokens": "7"}}])
def test_changed_usage_schema_does_not_fall_back_to_stale_totals(info, caplog) -> None:
    previous = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    transcript = "\n".join(
        json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": snapshot}})
        for snapshot in ({"total_token_usage": previous}, info)
    )
    assert parse_codex_usage(transcript) is None
    assert "unsupported token_count record" in caplog.text
    bundle = parse_codex_observations(transcript, "invocation", returncode=1, timed_out=False)
    assert [gap.code for gap in bundle.gaps] == ["agent_artifact_record_unparseable"]


def test_unrecognized_transcript_record_reports_gap() -> None:
    bundle = parse_codex_observations(
        json.dumps({"type": "renamed_response_item", "payload": {}}),
        "invocation",
        returncode=1,
        timed_out=False,
    )
    assert [gap.code for gap in bundle.gaps] == ["agent_artifact_record_unparseable", "tool_timing_unavailable"]


def test_unreadable_records_report_the_tool_timing_gap_once() -> None:
    bundle = parse_codex_observations("not json\nnot json either", "invocation", returncode=1, timed_out=False)
    assert [gap.code for gap in bundle.gaps] == [
        "agent_artifact_record_unparseable",
        "agent_artifact_record_unparseable",
        "tool_timing_unavailable",
    ]


@pytest.mark.parametrize("success", [True, False, None])
def test_native_patch_retains_call_and_output_with_execution_outcome(success: bool | None) -> None:
    items = [
        {
            "type": "custom_tool_call",
            "call_id": "patch-1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        },
        {"type": "custom_tool_call_output", "call_id": "patch-1", "output": "Untrusted output claiming success"},
    ]
    events = [
        {"type": "response_item", "timestamp": f"2026-10-21T10:00:0{index}Z", "payload": item}
        for index, item in enumerate(items)
    ]
    if success is not None:
        events.insert(
            1, {"type": "event_msg", "payload": {"type": "patch_apply_end", "call_id": "patch-1", "success": success}}
        )
    bundle = parse_codex_observations("\n".join(map(json.dumps, events)), "invocation", returncode=0, timed_out=False)
    invocation, tool = bundle.records
    assert tool.tool_call_id == invocation.conversation[0].call_id == "patch-1"
    assert invocation.conversation[0].input == items[0]["input"]
    assert invocation.conversation[1].output == items[1]["output"]
    assert tool.tool_name == "apply_patch"
    assert tool.status == ("unknown" if success is None else "completed" if success else "failed")
    assert tool.error_type == ("tool_error" if success is False else None)
    assert tool.duration_ms == 1000
    assert [gap.code for gap in bundle.gaps] == (["tool_outcome_unavailable"] if success is None else [])


@pytest.mark.parametrize("status", ["completed", "failed", "searching"])
def test_native_web_search_retains_identity_action_and_status(status: str) -> None:
    item = {
        "type": "web_search_call",
        "id": "search-1",
        "status": status,
        "action": {"type": "search", "query": "Gym"},
    }
    event = {"type": "response_item", "timestamp": "2026-10-21T10:00:00Z", "payload": item}
    bundle = parse_codex_observations(json.dumps(event), "invocation", returncode=0, timed_out=False)
    invocation, tool = bundle.records
    assert tool.tool_call_id == invocation.conversation[0].id == "search-1"
    assert invocation.conversation[0].action.query == "Gym"
    assert tool.tool_name == "web_search"
    assert tool.status == ("incomplete" if status == "searching" else status)
    assert tool.started_at is None
    assert (tool.completed_at is None) == (status == "searching")
    assert not bundle.gaps


def _time_split(observations) -> tuple:
    """The run-health timeouts section's policy, tool and untracked shares for a 100-second timed-out rollout whose
    only model call ran for its first 40 seconds, with these observations attached as collection attaches them."""
    from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
    from nemo_gym.rollouts import ELAPSED_S_KEY, read_rollout_timing, summarize_timeouts
    from nemo_gym.rollouts.trajectory_builder import _attach_trajectory_record

    started = 1_000_000.0
    call = {
        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
        "started_at": started,
        "completed_at": started + 40,
        "latency_total_ms": 40_000,
        "tokens_out": 100,
    }
    row = {
        "agent_timed_out": True,
        ELAPSED_S_KEY: 100.0,
        "reward": 0.0,
        "ng_model_call_capture": {"calls": [call]},
        "ng_agent_observations": observations.model_dump(mode="json"),
    }
    _attach_trajectory_record({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}, row)
    split = summarize_timeouts([read_rollout_timing(row)]).model_dump()["time_split"]
    return split["policy_model"], split["tool"], split["untracked"]


def test_a_transcript_that_could_not_be_read_leaves_tool_time_unknown() -> None:
    """Codex reports `agent_artifact_unavailable` without its transcript: that is not an empty tool history."""
    assert _time_split(parse_codex_observations(None, "0-0", returncode=-9, timed_out=True)) == (0.4, None, None)


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("not json", (0.4, None, None)),
        (
            json.dumps(
                {"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": {"x": "7"}}}}
            ),
            (0.4, 0.0, 0.6),
        ),
    ],
    ids=["unreadable-record", "unreadable-usage"],
)
def test_a_record_it_could_not_read_leaves_tool_time_unknown(transcript: str, expected: tuple) -> None:
    """An unreadable record may have been a tool call; a token usage record it could not read was not."""
    assert _time_split(parse_codex_observations(transcript, "0-0", returncode=-9, timed_out=True)) == expected


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "missing_model",
        "policy_request",
        "different_summary",
        "duplicate",
        "shared_event",
        "failed",
        "transport",
        "missing_id",
    ],
)
def test_compaction_requires_native_event_and_unique_session_helper(damage):
    from copy import deepcopy

    from nemo_gym.config import ModelServerRef
    from nemo_gym.trajectory import ContextCompactionObservation
    from responses_api_agents.codex_agent.observability import (
        _COMPACTION_PREFIX,
        associate_codex_compaction_calls,
    )

    transcript = json.dumps(
        {
            "type": "compacted",
            "timestamp": "2026-10-29T10:00:00Z",
            "payload": {"message": _COMPACTION_PREFIX + "Keep the changes."},
        }
    )
    bundle = parse_codex_observations(transcript, "root", returncode=0, timed_out=False)
    model_ref = ModelServerRef(type="responses_api_models", name="policy")
    attempt = {
        "status": 200,
        "compaction_response": {"response_id": "response-helper", "summary": "Keep the changes."},
    }
    attempts = [attempt]
    if damage == "missing_model":
        model_ref = None
    elif damage == "policy_request":
        del attempt["compaction_response"]
    elif damage == "different_summary":
        attempt["compaction_response"]["summary"] = "Different"
    elif damage == "duplicate":
        attempts.append(deepcopy(attempt))
    elif damage == "shared_event":
        bundle.records.append(bundle.records[-1].model_copy(deep=True))
    elif damage == "failed":
        attempt["status"] = 500
    elif damage == "transport":
        attempt["transport_error"] = "stream closed"
    elif damage == "missing_id":
        del attempt["compaction_response"]["response_id"]
    joined = associate_codex_compaction_calls(bundle, attempts, model_ref=model_ref)
    event = next(r for r in joined.records if isinstance(r, ContextCompactionObservation))
    assert event.summary == "Keep the changes."
    assert event.outcome == "completed"
    assert [ref.response_id for ref in event.model_calls] == ([] if damage else ["response-helper"])
    assert joined.records[0].model_calls == event.model_calls
    assert bool(joined.gaps) == bool(damage)
    assert bundle.records[-1].model_calls == []
    if damage is None:
        repeated = associate_codex_compaction_calls(joined, attempts, model_ref=model_ref)
        assert repeated == joined
        repeated = associate_codex_compaction_calls(joined, attempts * 2, model_ref=model_ref)
        event = next(r for r in repeated.records if isinstance(r, ContextCompactionObservation))
        assert not event.model_calls and repeated.gaps
        assert not repeated.records[0].model_calls


def test_codex_child_contexts_and_status_are_not_flattened_into_parent():
    def session(identity, source, items, terminal):
        events = [{"type": "session_meta", "payload": {"id": identity, "source": source}}]
        events += [{"type": "response_item", "timestamp": "2026-11-09T12:00:00Z", "payload": item} for item in items]
        events.append({"type": "event_msg", "payload": {"type": terminal}})
        return "\n".join(json.dumps(event, ensure_ascii=False) for event in events)

    parent = session(
        "parent",
        "exec",
        [
            {
                "type": "function_call",
                "name": "spawn_agent",
                "call_id": "spawn",
                "arguments": '{ "message": "child task" }',
            },
            {"type": "function_call_output", "call_id": "spawn", "output": '{"agent_id":"child"}'},
            {
                "type": "message",
                "role": "assistant",
                "id": "final",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Parent final", "annotations": []}],
            },
        ],
        "task_complete",
    )
    child = session(
        "child",
        {"subagent": {"thread_spawn": {"parent_thread_id": "parent"}}},
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Child-only\n α\u2028β"}]},
        ],
        "task_started",
    )
    bundle = parse_codex_observations(child + "\n" + parent, "episode", returncode=0, timed_out=False)
    owners = {r.invocation_id: r for r in bundle.records if isinstance(r, AgentInvocation)}
    assert set(owners) == {"episode", "episode.child"}
    assert owners["episode"].status == "completed"
    assert owners["episode.child"].status == "incomplete"
    assert owners["episode.child"].parent_invocation_id == "episode"
    assert owners["episode.child"].spawned_by_tool_call_id == "spawn"
    assert len(owners["episode"].conversation) == 3
    assert owners["episode.child"].conversation[0].content[0]["text"] == "Child-only\n α\u2028β"
    assert not any(g.code == "subagent_hierarchy_unavailable" for g in bundle.gaps)


@pytest.mark.parametrize("parent", ["root", "missing", "child"])
def test_child_compaction_keeps_ownership_and_invalid_parent_is_explicit(parent):
    from nemo_gym.config import ModelServerRef
    from nemo_gym.trajectory import ContextCompactionObservation
    from responses_api_agents.codex_agent.observability import _COMPACTION_PREFIX, associate_codex_compaction_calls

    transcript = "\n".join(
        json.dumps(event)
        for event in [
            {"type": "session_meta", "payload": {"id": "root", "source": "exec"}},
            {
                "type": "session_meta",
                "payload": {"id": "child", "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent}}}},
            },
            {
                "type": "compacted",
                "timestamp": "2026-11-09T12:00:00Z",
                "payload": {"message": _COMPACTION_PREFIX + "Child summary"},
            },
            {"type": "event_msg", "payload": {"type": "task_complete"}},
        ]
    )
    bundle = parse_codex_observations(transcript, "episode", returncode=0, timed_out=False)
    bundle = associate_codex_compaction_calls(
        bundle,
        [{"status": 200, "compaction_response": {"response_id": "summary", "summary": "Child summary"}}],
        model_ref=ModelServerRef(type="responses_api_models", name="policy"),
    )
    owners = {r.invocation_id: r for r in bundle.records if isinstance(r, AgentInvocation)}
    event = next(r for r in bundle.records if isinstance(r, ContextCompactionObservation))
    assert not owners["episode"].conversation and not owners["episode"].model_calls
    assert event.invocation_id == "episode.child"
    assert owners["episode.child"].model_calls == event.model_calls
    assert event.model_calls[0].response_id == "summary"
    assert any(g.code == "subagent_hierarchy_unavailable" for g in bundle.gaps) == (parent != "root")


@pytest.mark.parametrize("metadata", [None, {}, {"id": ""}, {"id": 42}])
def test_local_reader_marks_malformed_session_boundaries(tmp_path, metadata):
    directory = tmp_path / "sessions"
    directory.mkdir()
    events = [
        {"type": "session_meta", "payload": {"id": "root", "source": "exec"}},
        {"type": "session_meta", "payload": metadata},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        },
    ]
    (directory / "native.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    bundle = read_codex_observations(tmp_path, "invocation", returncode=0, timed_out=False)
    assert any(gap.code == "subagent_hierarchy_unavailable" for gap in bundle.gaps)
