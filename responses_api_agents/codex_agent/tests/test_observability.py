# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.rollout_observability import AgentInvocation, ToolCallObservation
from responses_api_agents.codex_agent.observability import parse_codex_usage, read_codex_observations


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


@pytest.mark.parametrize("record", [None, [], "unexpected", {"type": "response_item", "payload": {}}])
def test_malformed_native_record_reports_gap_without_losing_invocation(tmp_path: Path, record: object) -> None:
    folder = tmp_path / "sessions"
    folder.mkdir()
    (folder / "episode.jsonl").write_text(json.dumps(record) + "\n")
    bundle = read_codex_observations(tmp_path, "invocation", returncode=1, timed_out=False)
    assert bundle.records[0].invocation_id == "invocation"
    assert bundle.records[0].status == "failed"
    assert [gap.code for gap in bundle.gaps] == ["agent_artifact_record_unparseable"]


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
