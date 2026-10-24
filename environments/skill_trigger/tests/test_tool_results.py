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
"""Canonical returned text, success proof, and exact-file resolution."""

import json

import pytest

from environments.skill_trigger.scoring import EvidenceError
from environments.skill_trigger.tool_results import extract_tool_results


PATHS = {"target": "/skills/target/SKILL.md", "other": "/skills/other/SKILL.md"}


def extract(events, harness, times=None):
    return extract_tool_results(
        events,
        harness=harness,
        session_id="root",
        completed_at={"c1": 2.0} if times is None else times,
        skill_paths=PATHS,
    )


def pi(*, tool="read", path=PATHS["target"], error=False, exit_code=0):
    return [
        {"type": "tool_execution_start", "toolCallId": "c1", "args": {"path": path}},
        {
            "type": "tool_execution_end",
            "toolCallId": "c1",
            "toolName": tool,
            "isError": error,
            "result": {
                "content": [
                    {"type": "text", "text": "returned body"},
                    {"type": "image", "data": "not model-readable text"},
                ],
                "structuredContent": {"exit_code": exit_code},
            },
        },
    ]


def codex(*, exit_code=0, item_completed=False):
    completion = {
        "type": "exec_command_end",
        "call_id": "c1",
        "exit_code": exit_code,
        "aggregated_output": "display-only body",
    }
    if item_completed:
        completion = {
            "type": "item_completed",
            "item": {
                "id": "c1",
                "type": "CommandExecution",
                "exit_code": exit_code,
                "status": "completed",
                "output": "display-only body",
            },
        }
    return [
        {"type": "event_msg", "payload": completion},
        {
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "c1", "output": "canonical returned body"},
        },
    ]


def hermes(*, tool="skill_view", args=None, result=None, nested=False):
    args = {"name": "target"} if args is None else args
    result = {"success": True, "name": "target", "content": "returned body"} if result is None else result
    name = tool
    if nested:
        name = "tool_call"
        args = {"calls": [{"name": tool, "arguments": json.dumps(args)}]}
    return [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": "c1", "name": name, "content": json.dumps(result)},
    ]


def test_pi_exact_file_read_resolves_and_ignores_image_payload():
    result = extract(pi(), "pi")[0]
    assert result.text == "returned body"
    assert result.resolved_skill == "target"
    assert result.at == 2.0


@pytest.mark.parametrize("tool,path", [("bash", PATHS["target"]), ("read", "/skills/target/references.md")])
def test_pi_shell_or_support_file_does_not_claim_exact_instruction_read(tool, path):
    assert extract(pi(tool=tool, path=path), "pi")[0].resolved_skill is None


@pytest.mark.parametrize("options", [{"error": True}, {"exit_code": 1}, {"tool": "write"}])
def test_pi_failed_or_non_read_tool_is_not_a_candidate(options):
    assert extract(pi(**options), "pi") == []


@pytest.mark.parametrize("item_completed", [False, True])
def test_codex_uses_canonical_returned_text_not_terminal_display(item_completed):
    result = extract(codex(item_completed=item_completed), "codex")
    assert len(result) == 1
    assert result[0].text == "canonical returned body"
    assert result[0].tool == "terminal"
    assert result[0].resolved_skill is None


def test_codex_requires_matching_successful_execution():
    assert extract(codex(exit_code=1), "codex") == []
    events = codex()
    events[0]["payload"]["call_id"] = "another-call"
    assert extract(events, "codex") == []
    assert extract(codex()[1:], "codex") == []


@pytest.mark.parametrize("nested", [False, True])
def test_hermes_skill_view_resolves_requested_and_returned_name(nested):
    result = extract(hermes(nested=nested), "hermes")[0]
    assert result.tool == "skill_view"
    assert result.resolved_skill == "target"
    assert result.text == "returned body"


@pytest.mark.parametrize(
    "args,result",
    [
        ({"name": "target", "file_path": "references.md"}, {"success": True, "name": "target", "content": "body"}),
        ({"name": "target"}, {"success": True, "name": "other", "content": "body"}),
    ],
)
def test_hermes_support_file_and_name_mismatch_do_not_bypass_shared_text_rule(args, result):
    assert extract(hermes(args=args, result=result), "hermes")[0].resolved_skill is None


@pytest.mark.parametrize(
    "result",
    [
        {"success": False, "content": "body"},
        {"error": "failed", "success": True, "content": "body"},
        {"content": "body"},
        {"success": True, "content": None},
    ],
)
def test_hermes_skill_view_requires_positive_success_and_returned_text(result):
    assert extract(hermes(result=result), "hermes") == []


def test_hermes_prefers_api_content_over_display_content_and_accepts_session_export():
    events = hermes()
    events[-1]["content"] = "display-only body"
    events[-1]["api_content"] = {"success": True, "name": "target", "content": "model-visible body"}
    assert extract([{"messages": events}], "hermes")[0].text == "model-visible body"


def test_hermes_read_file_exact_path_and_terminal_exit_code():
    assert (
        extract(hermes(tool="read_file", args={"path": PATHS["target"]}, result={"content": "body"}), "hermes")[
            0
        ].resolved_skill
        == "target"
    )
    assert extract(hermes(tool="terminal", result={"exit_code": 0, "output": "body"}), "hermes")[0].text == "body"
    assert extract(hermes(tool="terminal", result={"output": "body"}), "hermes") == []
    assert extract(hermes(tool="terminal", result={"exit_code": 1, "output": "body"}), "hermes") == []


@pytest.mark.parametrize("harness,factory", [("pi", pi), ("codex", codex), ("hermes", hermes)])
def test_successful_candidate_cannot_invent_missing_host_timestamp(harness, factory):
    with pytest.raises(EvidenceError, match="host completion timestamp"):
        extract(factory(), harness, times={})
