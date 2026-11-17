# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.models.token_capture import ParentResolutionStatus, TokenEntry, resolve_terminal, stamp_lineage
from nemo_gym.trajectory import AgentInvocation, ToolCallObservation, transcript_to_atif
from responses_api_agents.codex_agent.app import parse_exec_jsonl
from responses_api_agents.codex_agent.observability import (
    CodexOutputUnavailable,
    parse_codex_observations,
    parse_codex_output,
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


def test_native_prompt_images_are_preserved() -> None:
    content = [
        {"type": "input_text", "text": "image follows"},
        {"type": "input_image", "image_url": "data:image/png;base64,image-bytes", "detail": "high"},
    ]
    transcript = json.dumps(
        {
            "type": "response_item",
            "timestamp": "2026-11-10T12:00:00Z",
            "payload": {"type": "message", "role": "user", "content": content},
        }
    )
    bundle = parse_codex_observations(transcript, "images", returncode=0, timed_out=False)
    observed = bundle.records[0].conversation[0].model_dump(mode="json")["content"]
    assert observed[0]["text"] == "image follows"
    assert observed[1]["image_url"] == content[1]["image_url"]
    assert observed[1]["detail"] == "high"
    assert not bundle.gaps


def _session(*payloads: dict, extra: list[dict] | None = None) -> str:
    """A Codex session record: its header, the given ``response_item`` payloads in order, then ``extra`` records."""
    records = [
        {"type": "session_meta", "payload": {"id": "session-1", "cli_version": "0.144.4"}},
        {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
        *({"type": "response_item", "payload": payload} for payload in payloads),
        *(extra or []),
    ]
    return "\n".join(json.dumps(record) for record in records)


def _call(call_id: str, name: str, arguments: dict | str, **fields: object) -> dict:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": text, **fields}


def _result(call_id: str, output: object, kind: str = "function_call_output") -> dict:
    return {"type": kind, "call_id": call_id, "output": output}


def _user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _shape(items) -> list[tuple]:
    return [(item.type, getattr(item, "call_id", None)) for item in items]


def test_output_keeps_the_models_command_arguments_and_ids_verbatim() -> None:
    """The CLI shows `/bin/bash -lc ...` with only the command; the record has what the model sent."""
    arguments = '{"cmd":"sed -n \'1,120p\' "$SKILLS/demo/SKILL.md"","workdir":"/workspace","yield_time_ms":10000}'
    transcript = _session(
        _user("task"),
        _call("call_9f3", "exec_command", arguments),
        _result("call_9f3", "Chunk ID: a\nProcess exited with code 0\nOutput:\nskill text"),
    )
    call, result = parse_codex_output(transcript)
    assert (call.type, call.name, call.call_id) == ("function_call", "exec_command", "call_9f3")
    assert call.arguments == arguments
    assert result.output == "Chunk ID: a\nProcess exited with code 0\nOutput:\nskill text"


def test_output_keeps_parallel_calls_in_model_order_and_write_stdin() -> None:
    """Both calls of a parallel pair precede their results, which arrive in completion order; a command that keeps
    running is polled with write_stdin, which the CLI merges into the first command's result."""
    transcript = _session(
        _user("task"),
        _call("slow", "exec_command", {"cmd": "sleep 5 && echo slow", "yield_time_ms": 1000}),
        _call("fast", "exec_command", {"cmd": "echo fast"}),
        _result("fast", "Process exited with code 0\nOutput:\nfast"),
        _result("slow", "Process running with session ID 7\nOutput:\n"),
        _call("poll", "write_stdin", {"session_id": 7, "chars": "", "yield_time_ms": 30000}),
        _result("poll", "Process exited with code 0\nOutput:\nslow"),
    )
    items = parse_codex_output(transcript)
    assert _shape(items) == [
        ("function_call", "slow"),
        ("function_call", "fast"),
        ("function_call_output", "fast"),
        ("function_call_output", "slow"),
        ("function_call", "poll"),
        ("function_call_output", "poll"),
    ]
    assert json.loads(items[0].arguments) == {"cmd": "sleep 5 && echo slow", "yield_time_ms": 1000}
    assert items[4].name == "write_stdin"


def test_output_keeps_each_plan_update_and_mcp_namespace() -> None:
    transcript = _session(
        _call("plan-1", "update_plan", {"plan": [{"step": "read", "status": "in_progress"}]}),
        _result("plan-1", "Plan updated"),
        _call("mcp-1", "search_docs", {"query": "pdf"}, namespace="mcp__gym_docs__"),
        _result("mcp-1", "hit"),
        _call("plan-2", "update_plan", {"plan": [{"step": "read", "status": "completed"}]}),
        _result("plan-2", "Plan updated"),
    )
    items = parse_codex_output(transcript)
    assert [item.name for item in items if item.type == "function_call"] == [
        "update_plan",
        "search_docs",
        "update_plan",
    ]
    assert items[2].namespace == "mcp__gym_docs__"
    assert items[0].namespace is None


def test_output_keeps_a_free_form_patch_call() -> None:
    patch = "*** Begin Patch\n*** Add File: notes.txt\n+hello\n*** End Patch"
    transcript = _session(
        {"type": "custom_tool_call", "call_id": "patch-1", "name": "apply_patch", "input": patch},
        _result("patch-1", "Success. Updated the following files:\nA notes.txt\n", "custom_tool_call_output"),
    )
    call, result = parse_codex_output(transcript)
    assert (call.type, call.name, call.input) == ("custom_tool_call", "apply_patch", patch)
    assert (result.type, result.call_id, result.output.splitlines()[0]) == (
        "custom_tool_call_output",
        "patch-1",
        "Success. Updated the following files:",
    )


def test_output_reads_a_content_item_result_as_text() -> None:
    items = [
        {"type": "input_text", "text": "viewing"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    transcript = _session(_call("view-1", "view_image", {"path": "/workspace/plot.png"}), _result("view-1", items))
    assert parse_codex_output(transcript)[1].output == "viewing\n[image]"


def test_output_keeps_what_the_model_said_and_not_what_it_was_told() -> None:
    transcript = _session(
        _user("task"),
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "instructions"}]},
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "check the file"}]},
        {"type": "reasoning", "summary": [], "content": [{"type": "text", "text": "then run it"}]},
        _call("c1", "exec_command", {"cmd": "ls"}),
        _result("c1", "a"),
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_served",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Done."}],
        },
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Without an ID."}]},
        extra=[{"type": "event_msg", "payload": {"type": "token_count", "info": None}}],
    )
    reasoning, content_reasoning, call, result, answer, anonymous = parse_codex_output(transcript)
    assert [item.type for item in (reasoning, content_reasoning, call, result, answer, anonymous)] == [
        "reasoning",
        "reasoning",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert reasoning.summary[0].text == "check the file" and reasoning.encrypted_content is None
    assert content_reasoning.content[0].text == "then run it"
    assert (answer.id, answer.phase, answer.content[0].text) == ("msg_served", "final_answer", "Done.")
    assert anonymous.id and anonymous.content[0].text == "Without an ID."


def test_output_turns_a_web_search_into_a_call_with_no_result() -> None:
    action = {"type": "search", "query": "Gym"}
    transcript = _session({"type": "web_search_call", "id": "ws-1", "status": "completed", "action": action})
    call, result = parse_codex_output(transcript)
    assert (call.name, call.call_id, json.loads(call.arguments)) == ("web_search", "ws-1", action)
    assert (result.call_id, result.output) == ("ws-1", "")


def test_a_call_without_its_result_stays_open() -> None:
    """The run was stopped while the tool ran: the record has the call and nothing more."""
    items = parse_codex_output(_session(_call("long", "exec_command", {"cmd": "sleep 1000"})))
    assert _shape(items) == [("function_call", "long")]


@pytest.mark.parametrize(
    ("transcript", "reason"),
    [
        (None, "unavailable"),
        ("", "no model output"),
        (_session(_user("task")), "no model output"),
        ("not json", "not JSON"),
        ("[]", "not an object"),
        (json.dumps({"type": "renamed", "payload": {}}), "unsupported session record"),
        (_session({"type": "ghost_snapshot"}), "unsupported session item 'ghost_snapshot'"),
        (_session(_call("c1", "t", {}), _call("c1", "t", {})), "call c1 is made more than once"),
        (_session(_result("c404", "x")), "matches no open call"),
        (_session(_call("c1", "t", {}), _result("c1", "x"), _result("c1", "y")), "matches no open call"),
        (_session(_call("c1", "t", {}), _result("c1", [{"type": "input_file"}])), "unsupported content item"),
        (_session(_call("c1", "t", {}), _result("c1", 7)), "neither text nor content items"),
        (_session({"type": "function_call", "call_id": "c1", "name": "t"}), "lacks its call_id, name or arguments"),
        (_session(_call("c1", "", {}), _result("c1", "x")), "lacks its call_id, name or arguments"),
        (_session(_call("c1", "  ", {})), "lacks its call_id, name or arguments"),
        (_session(_call(" ", "t", {})), "lacks its call_id, name or arguments"),
        (
            _session({"type": "custom_tool_call", "call_id": "c1", "name": "", "input": "text"}),
            "lacks its call_id, name or input",
        ),
        (_session({"type": "message", "role": "assistant", "content": 5}), "unexpected shape: TypeError"),
        (_session({"type": "reasoning", "summary": 5}), "unexpected shape: TypeError"),
        (_session({"type": "reasoning", "summary": [], "content": 5}), "unexpected shape: TypeError"),
        (json.dumps({"type": ["response_item"], "payload": {}}), "unexpected shape: TypeError"),
        (
            _session({"type": "message", "role": "assistant", "content": [{"type": "input_image"}]}),
            "unsupported content part",
        ),
        (_session(_call("c1", "t", {})) + "\n" + _session(_call("c2", "t", {})), "several sessions"),
    ],
)
def test_output_is_unavailable_where_the_record_would_misstate_or_break_the_run(transcript, reason) -> None:
    with pytest.raises(CodexOutputUnavailable, match=reason):
        parse_codex_output(transcript)


def test_a_record_nested_too_deeply_to_decode_is_unavailable() -> None:
    """The decoder raises RecursionError, not ValueError, for a valid record nested past the recursion limit."""
    with pytest.raises(CodexOutputUnavailable, match="not JSON"):
        parse_codex_output("[" * 100_000 + "]" * 100_000)


@pytest.mark.parametrize("search_id", [None, "", "  ", ["ws-1"], 7])
def test_a_web_search_without_a_usable_id_is_given_one_the_export_accepts(search_id) -> None:
    transcript = _session({"type": "web_search_call", "id": search_id, "action": {"type": "search"}})
    call, result = parse_codex_output(transcript)
    assert call.call_id == result.call_id == "ws_0"
    transcript_to_atif(
        "task",
        [item.model_dump(mode="json") for item in (call, result)],
        session_id="s",
        agent_name="codex_agent",
        agent_version="test",
    )


def _paths(node: object, prefix: tuple = ()):
    """Every path to a value inside nested dicts and lists, the node itself first."""
    yield prefix
    for key, child in node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ():
        yield from _paths(child, (*prefix, key))


def _replaced(node: object, path: tuple, value: object) -> object:
    if not path:
        return value
    copy = dict(node) if isinstance(node, dict) else list(node)
    copy[path[0]] = _replaced(node[path[0]], path[1:], value)
    return copy


def test_a_record_of_any_shape_is_either_unavailable_or_output_the_export_accepts() -> None:
    """Replace each value of each record, in a session that holds every kind of item the mapping reads, with a value
    of another shape. The agent catches only CodexOutputUnavailable, so nothing else may escape; and output that is
    returned is what the Harbor Environment Server exports, so it must convert, or the server masks the turn."""
    records = [
        json.loads(line)
        for line in _session(
            _user("task"),
            {
                "type": "reasoning",
                "id": "rs-1",
                "summary": [{"type": "summary_text", "text": "plan"}],
                "content": [{"type": "text", "text": "think"}],
            },
            _call("c1", "exec_command", {"cmd": "ls"}, id="fc-1", namespace="ns"),
            _result("c1", "a"),
            {"type": "custom_tool_call", "id": "ctc-1", "call_id": "c2", "name": "apply_patch", "input": "patch"},
            _result("c2", [{"type": "input_text", "text": "ok"}], "custom_tool_call_output"),
            {"type": "web_search_call", "id": "ws-1", "action": {"type": "search"}},
            {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ).split("\n")
    ]
    unavailable = exported = 0
    for index, record in enumerate(records):
        for path in _paths(record):
            for value in ([], {}, 5, None, "", " ", ["a"], [[]], {"a": 1}):
                mutated = [*records[:index], _replaced(record, path, value), *records[index + 1 :]]
                try:
                    output = parse_codex_output("\n".join(json.dumps(r) for r in mutated))
                except CodexOutputUnavailable:
                    unavailable += 1
                    continue
                transcript_to_atif(
                    "task",
                    [item.model_dump(mode="json") for item in output],
                    session_id="s",
                    agent_name="codex_agent",
                    agent_version="test",
                )
                exported += 1
    assert unavailable > 100 and exported > 100


def test_output_joins_the_models_captured_call_where_the_cli_display_cannot() -> None:
    """Training credits a rollout to the captured model call its last model-authored item names, by the call's ID and
    content. A run stopped on a tool call matched nothing in the CLI's display: another ID, another command."""
    arguments = json.dumps({"cmd": "sed -n '1,120p' SKILL.md", "workdir": "/workspace"})
    captured = TokenEntry(
        capture_key="rollout",
        model_call_id="call-1",
        prompt_token_ids=[1, 2],
        generation_token_ids=[3, 4],
        generation_log_probs=[-0.1, -0.1],
        output_items=[
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_abc",
                "name": "exec_command",
                "arguments": arguments,
                "status": "completed",
            }
        ],
        token_item_index=0,
    )
    stamp_lineage(captured, None, parent_resolution=ParentResolutionStatus.ROOT)
    stopped_on_the_call = _session(_call("call_abc", "exec_command", arguments))
    from_record = [item.model_dump(mode="json") for item in parse_codex_output(stopped_on_the_call)]
    command = {
        "id": "item_0",
        "type": "command_execution",
        "command": "/bin/bash -lc \"sed -n '1,120p' SKILL.md\"",
        "status": "in_progress",
    }
    from_events, _ = parse_exec_jsonl(json.dumps({"type": "item.started", "item": command}), include_partial=True)
    from_events = [item.model_dump(mode="json") for item in from_events]

    joined = resolve_terminal([captured], {"id": "response", "output": from_record})
    assert (joined.model_call_id, joined.method) == ("call-1", "item_id")
    assert resolve_terminal([captured], {"id": "response", "output": from_events}).model_call_id is None
