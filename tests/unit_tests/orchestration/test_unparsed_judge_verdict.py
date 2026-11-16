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
"""A saved judge evaluation without a verdict is a grading fault, never a healthy grade.

A scored row whose judge evaluation recorded no verdict (a null label, or a parse/judge error label such as XSTest's
``judge_parsing_error``) is counted as ``judge_unparseable`` and selected for re-judging: recovery plans the re-judge, the gate reads PLANNED,
and evaluation health counts the row as an evaluation-side fault until the re-judge settles it. On the artifact path the
verifier's captured judge reply must also have completed; a truncated reply leaves the trial unobserved.
"""

import pytest

from nemo_gym.orchestration.recovery import (
    RecoveryBudget,
    decide_recovery,
    grading_artifact,
    reverification_cause,
    saved_row_evidence,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import collect_with_score, ident, write_job_dir
from tests.unit_tests.orchestration.test_evaluation_health import _run
from tests.unit_tests.orchestration.test_grading_artifacts import _judge_call, _observed, _ungraded


COMPLETED = {"id": "resp-answer", "status": "completed", "output": [{"type": "message", "role": "assistant"}]}


def _xstest_row(label):
    """A scored row as XSTest saves it: one judge evaluation with this verdict label, unmasked, reward 0."""
    evaluation = {"responses_create_params": {"input": "kept on the run host"}, "verdict_label": label}
    return ident(0, reward=0.0, response=COMPLETED, judge_evaluations=[evaluation])


def _collected(tmp_path, row: dict) -> dict:
    job = write_job_dir(tmp_path / "job", n=1, results=[row], failures=[])
    [collected] = collect_with_score(job, "[]").results
    return collected


@pytest.mark.parametrize("label", ["judge_parsing_error", "JUDGE_ERROR", None])
def test_an_unparsed_judge_verdict_is_no_grade_and_is_selected_for_rejudging(tmp_path, label):
    row = _xstest_row(label)
    collected = _collected(tmp_path, row)
    assert collected["_ng_collected_grading_facts"]["judge_unparsed"] is True
    assert grading_artifact(collected) is None
    assert reverification_cause(collected) == "judge"
    assert reverification_cause(saved_row_evidence(row)) == "judge"


@pytest.mark.parametrize("label", ["full_compliance", "[[A!=B]]"])
def test_a_parsed_judge_verdict_is_unchanged(tmp_path, label):
    row = _xstest_row(label)
    collected = _collected(tmp_path, row)
    assert grading_artifact(collected) == "judge_record"
    assert reverification_cause(collected) is None
    assert reverification_cause(saved_row_evidence(row)) is None


def _unparsed(row: dict) -> None:
    _ungraded(row, judge_evaluations=True, judge_verdicts=[None], judge_unparsed=True)
    row["reward"] = 0.0


def test_the_gate_counts_an_unparsed_judge_verdict_as_a_grading_fault_while_a_rejudge_is_planned():
    # Health counts the row as an evaluation-side fault of a scored trial; the gate itself reads PLANNED, because
    # recovery plans a re-judge of its saved answer.
    collected = _run(100)
    _unparsed(collected.results[0])
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["evaluation"]["kinds"] == {"judge_unparseable": 1}
    assert task["trial_counts"]["unhealthy"] == 1 and task["trial_counts"]["observed_healthy"] == 99
    assert task["status"] == "PLANNED"
    # The row's reward is in the metric while the re-judge is pending: coverage and failure facts count it as scored.
    assert task["failures"]["metric_coverage"]["trials"] == 100


def test_unparsed_judge_verdicts_at_the_limit_fail_evaluation_health():
    collected = _run(100)
    for row in collected.results[:2]:
        _unparsed(row)
    passed, reason, report, task = _observed(collected)
    assert passed is False and report["status"] == "FAIL", reason
    assert report["evaluation"]["kinds"] == {"judge_unparseable": 2}
    assert task["status"] == "PLANNED"


@pytest.mark.parametrize("truncated", [False, True])
def test_a_judge_record_needs_its_captured_judge_reply_to_have_completed(truncated):
    collected = _run(100)
    for task, row in enumerate(collected.results[:10]):
        # A label-less judge evaluation (the math judge's shape): its verdict is in the judge reply itself.
        _ungraded(row, judge_evaluations=True, judge_verdicts=[None])
        row["reward"] = 0.0
        _judge_call(row, task)
        if truncated:
            meta = row["ng_trial_observations"]["model_calls"][-1]["response_metadata"]
            meta |= {"response_status": "incomplete", "finish_reason": "length", "error_category": "incomplete"}
    passed, reason, report, _ = _observed(collected)
    if truncated:
        assert passed is None and report["status"] == "UNVERIFIED", reason
    else:
        assert passed is True and report["exact"] is True, reason


def _verifier(outcome):
    """The verifier invocation a judged row saves in ``ng_trajectory``; its one grading operation has this outcome."""
    grade = {"operation_id": "grade", "kind": "llm_judge", "outcome": outcome, "result_binding_sha256": None}
    return {
        "invocation_id": "verify-0",
        "status": "completed",
        "failure_kind": None,
        "model_calls": [],
        "grading_operations": [grade | {"dependencies": []}],
        "grading_record": None,
        "sandbox_observations": [],
    }


def _single_judge_row(outcome):
    """A scored row as a single-judge benchmark saves it: a reward, no failure kind, and the verifier record's grading
    outcome; its judge reply is not kept as a labelled evaluation."""
    return ident(0, reward=0.0, response=COMPLETED, ng_trajectory={"verifier_invocations": [_verifier(outcome)]})


@pytest.mark.parametrize(
    "outcome,kind,cause",
    [
        ("unknown", "llm_judge", "judge"),
        ("completed", "llm_judge", None),
        ("unknown", "programmatic_calculation", None),
    ],
)
def test_a_judge_grading_whose_verifier_record_outcome_is_unknown_is_selected_for_rejudging(
    tmp_path, outcome, kind, cause
):
    row = _single_judge_row(outcome)
    row["ng_trajectory"]["verifier_invocations"][0]["grading_operations"][0]["kind"] = kind
    assert reverification_cause(_collected(tmp_path, row)) == cause
    assert reverification_cause(saved_row_evidence(row)) == cause


def test_the_gate_counts_a_grading_with_an_unknown_outcome_as_judge_unparseable():
    from nemo_gym.orchestration.recovery import decide_from_collected, gate_summary_from_collected, trial_counts
    from tests.unit_tests.orchestration.test_llm_judge_trial_coverage import _recovered_fixture

    collected, _, grade = _recovered_fixture()
    grade["outcome"] = "unknown"
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert decision.action.value == "reverify" and decision.counts == {"judge_recoverable": 1}
    expected = {"observed_healthy": 0, "unhealthy": 1, "unobserved": 0, "verifier_recorded": 0}
    assert trial_counts(decision, collected) == expected
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert task["trial_counts"] == expected
    assert task["evaluation_health"]["evaluation"]["kinds"] == {"judge_unparseable": 1}


def _game(verdict):
    """One judge game as the pairwise resources server saves it."""
    return {
        "responses_create_params": {"input": "kept on the run host"},
        "response": {"id": "judge-reply", "status": "completed", "output": []},
        "verdict": verdict,
    }


def _pairwise_row(*verdicts, reward):
    return ident(0, reward=reward, response=COMPLETED, games=[_game(verdict) for verdict in verdicts])


@pytest.mark.parametrize(
    "verdicts,reward,cause",
    [
        (("A>B", None), 0.5, "judge"),
        ((None, "A>B"), 0.5, "judge"),
        ((None, None), 0.5, "judge"),
        (("A>B", "B>A"), 1.0, None),
        (("A=B", "A=B"), 0.5, None),
    ],
)
def test_a_pairwise_row_with_a_game_lacking_its_verdict_is_selected_for_rejudging(tmp_path, verdicts, reward, cause):
    row = _pairwise_row(*verdicts, reward=reward)
    collected = _collected(tmp_path, row)
    facts = collected.get("_ng_collected_grading_facts", {})
    assert facts.get("games_unjudged", False) is (cause is not None)
    assert reverification_cause(collected) == cause
    assert reverification_cause(saved_row_evidence(row)) == cause


def test_the_gate_counts_a_pairwise_row_with_a_game_lacking_its_verdict_as_judge_unparseable():
    collected = _run(100)
    _ungraded(collected.results[0], games_unjudged=True)
    collected.results[0]["reward"] = 0.5
    passed, reason, report, task = _observed(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["evaluation"]["kinds"] == {"judge_unparseable": 1}
    assert task["trial_counts"]["unhealthy"] == 1 and task["trial_counts"]["observed_healthy"] == 99
    assert task["status"] == "PLANNED"


def _decide_one(row: dict):
    budget = RecoveryBudget(max_infra_retries=2, infra_retries_used=0, walltime_used_s=0.0, live_job=False)
    return decide_recovery(
        materialized=[{"_ng_task_index": 0, "_ng_rollout_index": 0}], results=[row], failures=[], budget=budget
    )


def test_recovery_plans_a_rejudge_for_a_scored_row_whose_judge_gave_no_verdict(tmp_path):
    decision = _decide_one(_collected(tmp_path, _xstest_row("judge_parsing_error")))
    assert decision.action.value == "reverify", decision.reason
    assert decision.counts == {"judge_recoverable": 1}
    assert [e.failure_kind for e in decision.evidence] == ["judge_unparseable"]


def test_a_scored_row_with_a_parsed_verdict_needs_no_recovery(tmp_path):
    decision = _decide_one(_collected(tmp_path, _xstest_row("full_compliance")))
    assert decision.action.value == "noop", decision.reason
    assert decision.counts == {"scored": 1}
