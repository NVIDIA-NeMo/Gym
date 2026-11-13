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
"""Behavioral tests for the version 0.2 root-trajectory score."""

import json

import pytest
from harbor.models.trajectories import Trajectory

from benchmarks.skill_trigger.scoring import EvidenceError, has_excerpt, score_trajectory


BODY = "Use the example parser to load the input. Validate its columns, reject missing values, and report the counts before saving the normalized table."
SKILL = "---\nname: target\ndescription: Parse tables\n---\n" + BODY


def trajectory(tool="read", content=BODY, arguments=None, extra=None):
    return Trajectory.model_validate(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": "test",
            "agent": {"name": "pi", "version": "1.3.0", "model_name": "fixture"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "Parse a table"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Reading instructions",
                    "tool_calls": [
                        {
                            "tool_call_id": "read-1",
                            "function_name": tool,
                            "arguments": arguments or {"path": "/root/.agents/skills/target/SKILL.md"},
                        }
                    ],
                    "observation": {"results": [{"source_call_id": "read-1", "content": content, "extra": extra}]},
                },
            ],
        }
    )


def score(value, harness="pi", sources=None):
    return score_trajectory(value, sources=sources or {"target": SKILL}, target="target", harness=harness)


@pytest.mark.parametrize(
    "harness,tool,args",
    [
        ("pi", "read", {"path": "/home/agent/.agents/skills/target/SKILL.md"}),
        ("codex", "exec_command", {"cmd": "cat /skills/target/SKILL.md"}),
        ("codex_guidance", "functions.exec_command", {"cmd": "cat /skills/target/SKILL.md"}),
        ("hermes", "skill_view", {"name": "target"}),
        ("hermes_guidance", "read_file", {"path": "/tmp/hermes/skills/target/SKILL.md"}),
    ],
)
def test_actual_tool_results_across_profiles(harness, tool, args):
    assert score(trajectory(tool, arguments=args), harness) == 1


@pytest.mark.parametrize(
    "change", ["claim", "name", "front_matter", "wrong_file", "write", "failed", "exit", "json_error"]
)
def test_claims_errors_and_inadequate_content_do_not_count(change):
    value = trajectory()
    step = value.steps[1]
    result = step.observation.results[0]
    if change == "claim":
        step.message = BODY
        result.content = "Done"
    elif change == "name":
        result.content = "I used target"
    elif change == "front_matter":
        result.content = SKILL.split("---", 2)[1]
    elif change == "wrong_file":
        step.tool_calls[0].arguments["path"] = "/skills/target/reference.md"
    elif change == "write":
        step.tool_calls[0].function_name = "write"
    elif change == "failed":
        result.extra = {"is_error": True}
    elif change == "exit":
        result.content = "Process exited with code 1\n" + BODY
    elif change == "json_error":
        result.content = json.dumps({"success": False, "content": BODY})
    assert score(value) == 0


def test_hermes_json_result_and_support_file():
    value = trajectory("skill_view", json.dumps({"success": True, "content": BODY}), {"name": "target"})
    assert score(value, "hermes") == 1
    value.steps[1].tool_calls[0].arguments["file_path"] = "reference.md"
    assert score(value, "hermes") == 0


def test_shared_boilerplate_requires_direct_read():
    sources = {"target": SKILL, "other": SKILL}
    assert score(trajectory(), sources=sources) == 1
    assert score(trajectory("bash"), sources=sources) == 0


def test_short_body_line_numbers_and_utf16():
    assert has_excerpt("12| small body", "---\nname: x\n---\nsmall body", competing=[])
    assert not has_excerpt("small", "small body", competing=[])
    assert has_excerpt("😀" * 60, "😀" * 70, competing=[], javascript=True)
    assert not has_excerpt("😀" * 60, "😀" * 70, competing=[], javascript=False)


@pytest.mark.parametrize("change", ["missing", "unpaired", "duplicate", "copied"])
def test_bad_evidence_is_not_scored_as_a_miss(change):
    value = trajectory()
    step = value.steps[1]
    if change == "missing":
        step.observation.results.clear()
    elif change == "unpaired":
        step.observation.results[0].source_call_id = "absent"
    elif change == "duplicate":
        step.observation.results.append(step.observation.results[0])
    elif change == "copied":
        step.is_copied_context = True
    with pytest.raises(EvidenceError):
        score(value)


def test_completed_no_tool_answer_is_a_valid_miss():
    value = trajectory()
    value.steps[1].tool_calls = None
    value.steps[1].observation = None
    assert score(value) == 0


@pytest.mark.parametrize("code", [1, 137, -1, -9])
def test_signed_nonzero_exit_cannot_establish_read(code):
    assert score(trajectory(tool="exec_command", content=f"Process exited with code {code}\n" + BODY)) == 0


def test_zero_exit_retains_read():
    assert score(trajectory(tool="exec_command", content="Process exited with code 0\n" + BODY)) == 1


def test_missing_successful_content_is_unscored():
    with pytest.raises(EvidenceError, match="Missing tool result content"):
        score(trajectory(content=None))
    assert score(trajectory(content="")) == 0
    assert score(trajectory(content=None, extra={"is_error": True})) == 0


def test_negative_controls_reject_bulk_reads_and_preserve_uncertainty():
    from benchmarks.skill_trigger.scoring import score_negative_trajectory

    sources = {"target": SKILL, "other": "Different instructions: answer carefully and check all inputs."}
    value = trajectory(tool="exec_command", content=BODY + "\n" + sources["other"])
    assert score_negative_trajectory(value, sources=sources, harness="pi") == 0
    value.steps[1].tool_calls = None
    value.steps[1].observation = None
    assert score_negative_trajectory(value, sources=sources, harness="pi") == 1
    with pytest.raises(EvidenceError):
        score_negative_trajectory(trajectory(content=None), sources=sources, harness="pi")


def test_session_start_exemption_preserves_read_counts():
    from benchmarks.skill_trigger.scoring import negative_openings, score_negative_trajectory

    sources = {"target": SKILL, "other": "Different instructions: answer carefully and check all inputs."}
    kwargs = dict(sources=sources, harness="pi", allowed_session_start_skills=("target",))
    assert score_negative_trajectory(trajectory(), **kwargs) == 1
    assert negative_openings(trajectory(), **kwargs) == {
        "all_skill_openings": 1,
        "session_start_openings": 1,
        "unnecessary_skill_openings": 0,
    }
    bulk = trajectory(tool="exec_command", content=BODY + "\n" + sources["other"])
    assert score_negative_trajectory(bulk, **kwargs) == 0
    with pytest.raises(EvidenceError, match="absent from catalog"):
        negative_openings(trajectory(), sources=sources, harness="pi", allowed_session_start_skills=("absent",))
    with pytest.raises(EvidenceError):
        negative_openings(trajectory(content=None), **kwargs)
