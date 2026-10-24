# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise the real pinned Harbor converters with complete and broken traces."""

import hashlib
import json

import pytest
from harbor.models.trajectories import Trajectory

from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.trajectory import convert_trace


def trace(harness: str) -> list[dict]:
    if harness == "pi":
        return [
            {"type": "message_end", "message": {"role": "user", "content": "Use the relevant skill"}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Reading instructions."},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "read",
                            "arguments": {"path": "/skills/target/SKILL.md"},
                        },
                    ],
                },
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "call-1",
                "toolName": "read",
                "isError": False,
                "result": {"content": [{"type": "text", "text": "Actual returned instructions"}]},
            },
        ]
    if harness == "codex":
        return [
            {"type": "session_meta", "payload": {"id": "session-1", "cli_version": "0.156.1", "source": "exec"}},
            {"type": "turn_context", "payload": {"model": "test-model"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Use the relevant skill"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call-1",
                    "arguments": '{"cmd":"cat /skills/target/SKILL.md"}',
                },
            },
            {"type": "event_msg", "payload": {"type": "exec_command_end", "call_id": "call-1", "exit_code": 0}},
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "Actual returned instructions",
                },
            },
        ]
    return [
        {"role": "user", "content": "Use the relevant skill"},
        {
            "role": "assistant",
            "content": "Reading instructions.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "skill_view", "arguments": '{"name":"target"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "skill_view", "content": "Actual returned instructions"},
    ]


@pytest.mark.parametrize("harness,version", [("pi", "1.3.0"), ("codex", "0.156.1"), ("hermes", "0.22.2")])
def test_existing_harbor_converter_preserves_call_output_and_identity(tmp_path, harness, version):
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in trace(harness)) + "\n")
    raw = source.read_bytes()
    result = convert_trace(
        source,
        harness=harness,
        session_id="session-1",
        model="test-model",
        system_prompt="Recorded system instructions",
    )
    roundtrip = Trajectory.model_validate_json(result.model_dump_json())
    assert roundtrip.session_id == "session-1"
    assert roundtrip.trajectory_id == "session-1"
    assert roundtrip.agent.version == version
    assert roundtrip.steps[0].source == "system"
    assert roundtrip.steps[0].message == "Recorded system instructions"
    assert any(s.source == "user" and s.message == "Use the relevant skill" for s in roundtrip.steps)
    observations = [r for s in result.steps if s.observation for r in s.observation.results]
    assert len(observations) == 1
    assert observations[0].source_call_id == "call-1"
    assert observations[0].content == "Actual returned instructions"
    assert source.read_bytes() == raw
    assert result.extra["canonical_trace_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("harness", ["pi", "codex", "hermes"])
def test_unmatched_tool_result_is_not_silently_omitted(tmp_path, harness):
    events = trace(harness)
    if harness == "pi":
        events[-1]["toolCallId"] = "unmatched"
    elif harness == "codex":
        events[-1]["payload"]["call_id"] = "unmatched"
    else:
        events[-1]["tool_call_id"] = "unmatched"
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError):
        convert_trace(source, harness=harness, session_id="session-1", model="test-model")


def test_codex_root_is_explicit_not_chosen_from_latest_child_filename(tmp_path):
    source = tmp_path / "root.jsonl"
    source.write_text("\n".join(json.dumps(e) for e in trace("codex")))
    (tmp_path / "zz-child.jsonl").write_text('{"type":"session_meta","payload":{"id":"unrelated"}}')
    assert convert_trace(source, harness="codex", session_id="session-1", model="test-model").session_id == "session-1"
    with pytest.raises(EvidenceError, match="owning session"):
        convert_trace(source, harness="codex", session_id="another-session", model="test-model")


def test_truncated_trace_is_an_error_not_a_partial_success(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(trace("pi")[0]) + '\n{"unfinished":')
    with pytest.raises(ValueError):
        convert_trace(source, harness="pi", session_id="session-1", model="test-model")
