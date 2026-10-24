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
"""Behavioral fixtures for positive skill opening, error masking, and shared limits."""

import pytest

from environments.skill_trigger.scoring import (
    Access,
    EvidenceError,
    Identity,
    Request,
    Session,
    Spawn,
    ToolResult,
    TrialEvidence,
    has_excerpt,
    normalize,
    score_trial,
)


BODY = "Use the example parser to load the input. Validate its columns, reject missing values, and report the counts before saving the normalized table."
SKILL = "---\nname: target\ndescription: Parse tables\n---\n" + BODY


@pytest.fixture
def evidence() -> TrialEvidence:
    return TrialEvidence(
        identity=Identity(
            task_id="case", prompt_sha256="prompt", catalog_sha256="catalog", model="model", harness="pi"
        ),
        root_session_id="root",
        sessions=[Session(id="root", model="model")],
        requests=[Request(session_id="root", model="model", sent_at=0.0, max_output_tokens=8192)],
        catalog_staged_at=-2.0,
        observer_ready_at=-1.0,
        observer_stopped_at=5.0,
        finished_at=5.0,
        active_work_stopped=True,
        outcome="completed",
        accesses=[Access(skill="target", at=2.0)],
        outputs=[
            ToolResult(
                session_id="root",
                call_id="read-1",
                tool="read",
                text=BODY,
                success=True,
                at=3.0,
                resolved_skill="target",
            )
        ],
    )


def score(evidence: TrialEvidence, *, sources: dict[str, str] | None = None):
    return score_trial(
        evidence,
        expected=evidence.identity.model_copy(deep=True),
        sources=sources or {"target": SKILL},
        target="target",
    )


@pytest.mark.parametrize(
    "harness,tool",
    [
        ("pi", "read"),
        ("codex", "terminal"),
        ("codex_guidance", "terminal"),
        ("hermes", "skill_view"),
        ("hermes_guidance", "skill_view"),
    ],
)
def test_success_across_all_five_profiles(evidence, harness, tool):
    evidence.identity.harness = harness
    evidence.outputs[0].tool = tool
    result = score(evidence)
    assert result.reward == 1
    assert result.loaded_skills == ["target"]
    assert result.reads[0].call_id == "read-1"


@pytest.mark.parametrize(
    "change", ["name_only", "front_matter", "failed_read", "no_access", "wrong_skill", "write_tool"]
)
def test_missing_or_insufficient_opening_is_a_valid_miss(evidence, change):
    if change == "name_only":
        evidence.outputs[0].text = "I will use target"
    if change == "front_matter":
        evidence.outputs[0].text = SKILL.split("---\n", 2)[1]
    if change == "failed_read":
        evidence.outputs[0].success = False
    if change == "no_access":
        evidence.accesses = []
    if change == "wrong_skill":
        evidence.outputs[0].resolved_skill = "other"
    if change == "write_tool":
        evidence.outputs[0].tool = "write"
    assert score(evidence).reward == 0


def test_shared_text_requires_direct_read_proof(evidence):
    sources = {"target": SKILL, "other": SKILL}
    assert score(evidence, sources=sources).reward == 1
    evidence.outputs[0].tool = "bash"
    evidence.outputs[0].resolved_skill = None
    assert score(evidence, sources=sources).reward == 0


def test_short_bodies_require_whole_body_and_line_numbers_are_normalized():
    assert has_excerpt("12| Small body\n", "---\nname: x\n---\nSmall body", competing=[])
    assert not has_excerpt("Small", "---\nname: x\n---\nSmall body", competing=[])
    assert not has_excerpt("name: x", "---\nname: x\n---\n", competing=[])


def test_pi_utf16_threshold_preserves_original_scorer():
    source = "😀" * 70
    output = "😀" * 60
    assert has_excerpt(output, source, competing=[], javascript=True)
    assert not has_excerpt(output, source, competing=[], javascript=False)


def test_pi_whitespace_and_line_number_rules_preserve_ecmascript_semantics():
    assert normalize("\ufeff12| body\ufeff", javascript=True) == "body"
    assert normalize("12| body\u0085text", javascript=True) == "body\u0085text"
    assert normalize("12| body\u0085text") == "body text"
    assert normalize("١٢| body", javascript=True) == "١٢| body"
    assert normalize("١٢| body") == "body"


@pytest.mark.parametrize("outcome", ["timeout", "routing_horizon", "output_limit", "routing_boundary"])
def test_success_survives_later_model_budget_outcomes(evidence, outcome):
    evidence.outcome = outcome
    if outcome == "timeout":
        evidence.finished_at = evidence.observer_stopped_at = 121.0
    assert score(evidence).reward == 1
    evidence.outputs = []
    assert score(evidence).reward == 0


def test_output_after_deadline_does_not_count(evidence):
    evidence.outcome = "timeout"
    evidence.finished_at = evidence.observer_stopped_at = 121.0
    evidence.outputs[0].at = 120.5
    assert score(evidence).reward == 0


@pytest.mark.parametrize("outcome", ["provider_error", "setup_error", "observation_error", "trace_error", "cancelled"])
def test_errors_do_not_become_zero_even_if_an_earlier_read_exists(evidence, outcome):
    evidence.outcome = outcome
    with pytest.raises(EvidenceError, match="Unscored"):
        score(evidence)


@pytest.mark.parametrize(
    "change",
    [
        "observer_late",
        "observer_early",
        "staging_late",
        "running",
        "four_requests",
        "wrong_model",
        "wrong_tokens",
        "duplicate_output",
        "unknown_session",
        "request_late",
        "identity",
    ],
)
def test_invalid_observation_or_request_evidence_is_unscored(evidence, change):
    expected = evidence.identity.model_copy(deep=True)
    if change == "observer_late":
        evidence.observer_ready_at = 0.1
    if change == "observer_early":
        evidence.observer_stopped_at = 4.0
    if change == "staging_late":
        evidence.catalog_staged_at = 1.0
    if change == "running":
        evidence.active_work_stopped = False
    if change == "four_requests":
        evidence.requests *= 4
    if change == "wrong_model":
        evidence.requests[0].model = "another-model"
    if change == "wrong_tokens":
        evidence.requests[0].max_output_tokens = 16384
    if change == "duplicate_output":
        evidence.outputs *= 2
    if change == "unknown_session":
        evidence.outputs[0].session_id = "unknown"
    if change == "request_late":
        evidence.requests[0].sent_at = 121.0
    if change == "identity":
        expected.catalog_sha256 = "other-catalog"
    with pytest.raises(EvidenceError):
        score_trial(evidence, expected=expected, sources={"target": SKILL}, target="target")


def delegate(evidence: TrialEvidence) -> None:
    evidence.sessions.append(Session(id="child", model="model", parent_id="root", spawn_call_id="spawn-1"))
    evidence.spawns.append(Spawn(parent_id="root", child_id="child", call_id="spawn-1", at=1.0))
    evidence.requests.append(Request(session_id="child", model="model", sent_at=1.5, max_output_tokens=8192))
    evidence.outputs[0].session_id = "child"


def test_same_model_child_read_counts_with_lineage_and_shared_budget(evidence):
    delegate(evidence)
    assert score(evidence).reward == 1
    evidence.sessions[-1].inherited_call_ids = ["read-1"]
    assert score(evidence).reward == 0


@pytest.mark.parametrize(
    "change", ["missing_spawn", "wrong_model", "cyclic", "child_request_before_spawn", "missing_child_request"]
)
def test_child_read_requires_real_own_session_and_budget_evidence(evidence, change):
    delegate(evidence)
    if change == "missing_spawn":
        evidence.spawns = []
    if change == "wrong_model":
        evidence.sessions[-1].model = "other-model"
    if change == "cyclic":
        evidence.sessions[-1].parent_id = "child"
    if change == "child_request_before_spawn":
        evidence.requests[-1].sent_at = 0.5
    if change == "missing_child_request":
        evidence.requests.pop()
    with pytest.raises(EvidenceError):
        score(evidence)
