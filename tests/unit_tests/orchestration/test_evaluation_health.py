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
"""Evaluation health fails only on evaluation-side faults; model faults and response and trial failures are reported.

Runs are built from the one-trial programmatic-grading fixture, repeated with fresh operation identities.
"""

import json
from pathlib import Path

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
    gate_summary_from_collected,
    trial_counts,
    trial_verdict,
)
from tests.unit_tests.orchestration.test_eval_resubmit_decision import final_model_zero


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"
IDS = (
    "resp_scripted_1",
    "cc9b71700e2e47bcb280552c053e1ba9",  # pragma: allowlist secret
    "db88801997f34c978b960e86a4abc542",  # pragma: allowlist secret
    "verify-a188f19c8d794e77bbae490042027a9c",
    "grade-bbef44c2491649afb24ebd10d87ad6d8",
)


def _run(scored: int, *, settled: dict[str, int] | None = None) -> CollectedEvidence:
    """``scored`` graded trials (reward 1) and, after them, settled counts-as-zero failures by kind."""
    base = json.loads(FIXTURE.read_text())
    template = json.dumps(base["results"][0])
    rows, materialized, selection, failures = [], [], [], []
    for task in range(scored):
        text = template.replace('"task_id": "0"', f'"task_id": "{task}"').replace('"0-0"', f'"{task}-0"')
        for value in IDS:
            text = text.replace(value, f"{value}-{task}")
        row = json.loads(text)
        row["_ng_task_index"], row["_line"] = task, task
        rows.append(row)
    task = scored
    for kind, count in (settled or {}).items():
        for _ in range(count):
            failure = final_model_zero(task, 0)
            failure["failure_kind"] = kind
            failure["_ng_failure_events"][f"f-{task}-0"]["failure_kind"] = kind
            failure["_line"] = len(failures)
            failures.append(failure)
            task += 1
    for index in range(task):
        materialized.append({"_ng_task_index": index, "_ng_rollout_index": 0, "_ng_collected_task_id": str(index)})
        selection.append(
            {"identity": [index, 0], "attempts": 0, "max_attempts": 3, "pending": False, "reserved": True}
        )
    base |= {"results": rows, "materialized": materialized, "managed_selection": selection, "failures": failures}
    base["quality"]["artifacts"]["expected"] = task
    return CollectedEvidence.model_validate(base)


def _truncate(row: dict) -> None:
    """The model ran into its output cap; the benchmark's rule scored the cut-off answer zero without grading it."""
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["reward"] = 0.0
    row["response"]["status"] = "incomplete"
    row["response"]["output"] = [{"type": "reasoning", "status": "incomplete"}]
    [call] = row["ng_trial_observations"]["model_calls"]
    call["response_metadata"] |= {
        "response_status": "incomplete",
        "error_category": "incomplete",
        "finish_reason": "length",
        "dialect": "responses",
    }
    call["output_items"] = [{"type": "reasoning", "status": "incomplete"}]


def _empty(row: dict) -> None:
    """The answer holds no assistant message; the benchmark's rule scored it zero without grading it."""
    row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    row["reward"] = 0.0
    row["response"]["output"] = [{"type": "reasoning", "status": "completed"}]
    [call] = row["ng_trial_observations"]["model_calls"]
    call["output_items"] = [{"type": "reasoning", "status": "completed"}, *call["output_items"]]


def _gate(collected: CollectedEvidence):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    return decision, trial_verdict(decision, collected), task


def _health_errors(task: dict) -> list[str]:
    return [e for e in task["errors"] if e.startswith("response/trial health")]


def test_a_run_of_answers_truncated_at_the_output_cap_passes_evaluation_health_with_the_response_failures():
    collected = _run(1000)
    for row in collected.results[:123]:
        _truncate(row)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True, reason
    assert report["status"] == "PASS" and report["exact"] is True
    assert report["evaluation"] == {"count": 0, "fraction": 0.0, "kinds": {}, "server": 0}
    assert report["model"] == {"count": 123, "fraction": 0.123, "kinds": {"truncated_answer": 123}}
    assert "model faults: 123 of 1000 (12.30%) (truncated_answer 123)" in reason
    assert "lower bound" not in reason and "UNVERIFIED" not in reason
    assert not _health_errors(task)
    assert task["failures"]["response_failures"]["by_cause"]["truncated"] == {"count": 123, "rate": 0.123}
    assert task["health_verdict"] == (
        "evaluation health: PASS; response failures: 12.30% (truncated 123 of 1000); "
        "trial failures: 0.00% (none of 1000); metric over 1000 of 1000 trials"
    )
    assert "model_health" not in task


def test_three_percent_transport_errors_fail_evaluation_health():
    _, (passed, reason, report), task = _gate(_run(97, settled={"model_transport_error": 3}))
    assert passed is False and report["status"] == "FAIL"
    assert report["evaluation"]["kinds"] == {"model_transport_error": 3} and report["evaluation"]["server"] == 3
    assert reason == (
        "trial: evaluation-side faults: 3 of 100 (3.00%) is not below 2% (server 3: model_transport_error 3); "
        "model faults: 0 of 100 (0.00%)"
    )
    assert _health_errors(task)[0].startswith("response/trial health failed")
    assert task["status"] == "FAILED"
    assert task["health_verdict"].startswith("evaluation health: FAIL; ")
    assert task["failures"]["trial_failures"]["by_cause"]["request"] == {"count": 3, "rate": 0.03}


def test_a_mixed_run_fails_on_its_evaluation_share_and_reports_both_classes():
    collected = _run(90, settled={"model_transport_error": 3, "model_context_limit": 7})
    for row in collected.results[:5]:
        _truncate(row)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is False
    assert report["model"]["kinds"] == {"model_context_limit": 7, "truncated_answer": 5}
    assert reason == (
        "trial: evaluation-side faults: 3 of 100 (3.00%) is not below 2% (server 3: model_transport_error 3); "
        "model faults: 12 of 100 (12.00%) (model_context_limit 7, truncated_answer 5)"
    )
    assert task["health_verdict"] == (
        "evaluation health: FAIL; response failures: 5.00% (truncated 5 of 100); "
        "trial failures: 10.00% (request 10 of 100); metric over 100 of 100 trials"
    )


def test_a_mixed_run_under_the_evaluation_limit_passes_with_many_model_faults():
    collected = _run(170, settled={"model_transport_error": 1, "model_context_limit": 29})
    for row in collected.results[:20]:
        _truncate(row)
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True, reason
    assert reason.startswith(
        "trial: evaluation-side faults: 1 of 200 (0.50%) is below 2% (server 1: model_transport_error 1); "
        "model faults: 49 of 200 (24.50%) (model_context_limit 29, truncated_answer 20); exact count:"
    )
    assert not _health_errors(task)


def test_unverified_coverage_withholds_a_pass_without_a_health_failure():
    collected = _run(100)
    for row in collected.results[:10]:
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    _, (passed, reason, report), task = _gate(collected)
    assert passed is None and report["status"] == "UNVERIFIED" and report["exact"] is False
    assert reason.startswith("trial: UNOBSERVED 10 of 100 expected trials")
    assert "unverified: this withholds a pass and is no health failure" in reason
    assert not _health_errors(task)
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    assert health.startswith("response/trial health unknown")
    # FAILED only for this fixture's missing score and MLflow evidence; health alone is UNVERIFIED, not a failure.
    assert task["status"] == "FAILED" and task["health_verdict"].startswith("evaluation health: UNVERIFIED")


def test_records_that_do_not_cover_every_trial_keep_the_lower_bound_wording():
    collected = _run(99, settled={"model_transport_error": 1})
    # A completed grading operation that is not bound to the saved answer: that trial is unobserved, never healthy.
    grade = collected.results[0]["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"][0]
    grade["result_binding_sha256"] = "0" * 64
    _, (passed, reason, report), _ = _gate(collected)
    assert passed is None and report["exact"] is False
    assert reason.startswith("trial: UNOBSERVED 1 of 100 expected trials")
    assert "observed healthy 98, unhealthy 1; known so far: evaluation-side faults: 1 of 100" in reason


def test_a_policy_capture_that_does_not_join_the_bound_answer_is_reported_not_a_coverage_condition():
    collected = _run(99, settled={"model_transport_error": 1})
    # The grading is bound to the saved answer; only the policy capture does not join it.
    collected.results[0]["response"]["id"] = "another-response"
    _, (passed, reason, report), _ = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["evaluation"]["kinds"] == {"model_transport_error": 1}
    assert report["trajectory_evidence"]["incomplete"] == 1
    assert "trajectory evidence incomplete: 1 of 100" in reason


def test_full_coverage_gives_the_exact_count_with_kinds_and_classes():
    collected = _run(97, settled={"model_transport_error": 1, "model_request_error": 2})
    _, (passed, reason, report), task = _gate(collected)
    assert passed is True and report["exact"] is True
    assert "lower bound" not in reason and "exact count:" in reason
    assert report["evaluation"]["kinds"] == {"model_transport_error": 1}
    assert report["model"]["kinds"] == {"model_request_error": 2}
    assert task["trial_counts"] == {"observed_healthy": 97, "unhealthy": 3, "unobserved": 0, "verifier_recorded": 0}


def test_response_failures_are_reported_and_never_a_second_verdict():
    collected = _run(1000)
    for row in collected.results[:41]:
        _truncate(row)
    for row in collected.results[41:47]:
        _empty(row)
    _, (passed, reason, _), task = _gate(collected)
    assert passed is True, reason
    assert task["health_verdict"] == (
        "evaluation health: PASS; response failures: 4.70% (truncated 41, empty 6 of 1000); "
        "trial failures: 0.00% (none of 1000); metric over 1000 of 1000 trials"
    )
    assert task["failures"]["response_failures"]["rate"] == 0.047
    assert not _health_errors(task) and "PASS" not in str(task["failures"])
    assert task["criteria"]["response"]["applied"] is False and task["criteria"]["trial"]["applied"] is False


def test_trial_counts_follow_the_gates_classification_of_a_grading_without_a_verdict():
    collected = _run(1)
    grade = collected.results[0]["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"][0]
    grade["outcome"] = "unknown"
    decision, (passed, _, _), task = _gate(collected)
    # A programmatic grading with no verdict is not joined, so it is unresolved, never healthy.
    assert passed is None
    assert trial_counts(decision, collected) == task["trial_counts"]
    assert task["trial_counts"]["observed_healthy"] == 0 and task["trial_counts"]["unobserved"] == 1


def test_an_accepted_omission_leaves_a_fully_observed_run_exact():
    from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION

    run = _run(1).model_dump()
    omitted = {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_collected_task_id": "1"}
    omitted[ACCEPTED_STAGE_OMISSION] = {"stage_index": 1, "tolerate_unresolved": True}
    run["materialized"].append(omitted)
    run["managed_selection"].append(
        {"identity": [1, 0], "attempts": 0, "max_attempts": 3, "pending": False, "reserved": True}
    )
    run["quality"]["artifacts"]["expected"] = 2
    collected = CollectedEvidence.model_validate(run)
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.complete and decision.counts == {"scored": 1, "omitted": 1}, decision.reason
    # The omission is no trial: the one observed healthy trial is the whole population, so health passes exactly.
    assert passed is True and report["status"] == "PASS" and report["exact"] is True, reason
    assert task["trial_counts"] == {"observed_healthy": 1, "unhealthy": 0, "unobserved": 0, "verifier_recorded": 0}


def test_trial_counts_leave_a_grading_bound_to_another_answer_unobserved():
    collected = _run(1)
    # The completed grading operation no longer binds the saved answer and score.
    collected.results[0]["_ng_collected_grading_binding"] = "0" * 64
    decision, (passed, _, _), task = _gate(collected)
    assert passed is None
    assert task["trial_counts"] == trial_counts(decision, collected)
    assert task["trial_counts"]["observed_healthy"] == 0 and task["trial_counts"]["unobserved"] == 1


def test_trial_counts_leave_an_unbound_grading_artifact_unobserved():
    from tests.unit_tests.orchestration.test_grading_artifacts import _ungraded

    collected = _run(1)
    _ungraded(collected.results[0], sentence_metrics=["sentence_chrf"])
    # A saved grading artifact counts only with the collector's binding of the saved answer and score.
    del collected.results[0]["_ng_collected_grading_binding"]
    decision, (passed, _, _), task = _gate(collected)
    assert passed is None
    assert task["trial_counts"]["observed_healthy"] == 0 and task["trial_counts"]["unobserved"] == 1


def test_a_scored_harbor_agent_timeout_counts_its_own_failure_kind_once():
    collected = _run(100)
    # Harbor's terminal graded timeout: a main result with its own failure kind, graded and not masked.
    for row in collected.results[:30]:
        row |= {"failure_kind": "agent_timeout", "evaluation_completed": True, "mask_sample": False}
    decision, (passed, reason, report), task = _gate(collected)
    assert decision.counts == {"scored": 100}
    assert report["model"]["kinds"] == {"agent_timeout": 30} and report["evaluation"]["count"] == 0, reason
    # A model fault never fails health; every other trial is observed, so the count is exact.
    assert passed is True and report["exact"] is True, reason
    assert task["failures"]["trial_failures"]["by_cause"]["timeout"]["count"] == 30
    assert task["trial_counts"] == {"observed_healthy": 70, "unhealthy": 30, "unobserved": 0, "verifier_recorded": 0}
