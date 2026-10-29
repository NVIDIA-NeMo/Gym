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
    from nemo_gym.rollouts.health.timeouts import ELAPSED_S_KEY, read_rollout_timing, summarize_timeouts
    from nemo_gym.rollouts.rollout_collection import _attach_trajectory_record

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
