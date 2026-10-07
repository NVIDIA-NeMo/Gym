# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy

import pytest

from benchmarks.swe_together.recipes.check_health import audit, healthy_record
from resources_servers.swe_together.coverage import compute_scores
from resources_servers.swe_together.evaluation import derive_score


def measured_zero():
    # Use the real scoring functions so fixture shapes track Resources output.
    verdict = derive_score(
        {"completeness_goals": [{"id": "g", "weight": 1.0}]},
        {"goal_results": [{"id": "g", "met": False}], "judge_score": 0.0},
    ) | {"judge_runtime": "2.1.108 (Claude Code)"}
    activations = [{"activation_id": 0, "response": {"metadata": {"opencode_version": "1.15.13"}}}]
    return {
        "task_id": "example",
        "reward": 0.0,
        "judge_score": 0.0,
        "verdict": verdict,
        "ng_activations": activations,
        "ng_steps": [{"activation_id": 0, "continue_episode": False, "responses_create_params": None}],
        "ng_agent_close": {"activations": activations, "cleanup_confirmed": True},
        "user_correction": 0.0,
        "intent_coverage": compute_scores({"per_intent": []}, 0, 0),
    }


def test_missing_grade_is_not_measured_zero():
    row = measured_zero()
    assert healthy_record(row) == []
    row.update(judge_score=None, mask_sample=True)
    assert "missing_or_invalid_grade" in healthy_record(row)
    assert "failed_or_masked" in healthy_record(row)


def test_retry_retains_failure_and_does_not_cover_missing_task(tmp_path):
    path = tmp_path / "attempts.jsonl"
    row = measured_zero()
    failed = row | {"mask_sample": True, "judge_score": None}
    path.write_text(json.dumps(failed) + "\n" + json.dumps(row) + "\n")
    report = audit([path], {"example", "missing"})
    assert not report["passed"]
    assert report["attempts"] == 2
    assert report["healthy_graded_tasks"] == 1
    assert report["tasks_without_healthy_grade"] == ["missing"]
    assert report["problem_counts"]["failed_or_masked"] == 1


def test_close_must_match_cumulative_activation_evidence():
    row = measured_zero()
    row["ng_agent_close"] = {"cleanup_confirmed": True, "activations": []}
    assert "unconfirmed_candidate_close" in healthy_record(row)


@pytest.mark.parametrize("metric", ["coverage_rate", "weighted_coverage", "scope_precision", "overall_score"])
@pytest.mark.parametrize("invalid", [None, True, -0.1, 1.1, float("nan"), float("inf")])
def test_coverage_requires_all_four_finite_fraction_scores(metric, invalid):
    row = measured_zero()
    row["intent_coverage"][metric] = invalid
    assert "missing_intent_coverage" in healthy_record(row)


def test_coverage_auxiliary_failure_cannot_hide_behind_numeric_defaults():
    row = measured_zero()
    row["coverage_error"] = "provider unavailable"
    assert "missing_intent_coverage" in healthy_record(row)
    row = measured_zero()
    row["tagger_error"] = "missing message tags"
    assert "missing_user_correction" in healthy_record(row)


def test_coverage_optional_evidence_and_gradable_partial_work_are_accepted():
    row = measured_zero()
    row["intent_coverage"].update(match_table={"per_intent": []}, schema_warnings=[])
    row["ng_activations"][0].update(turn_complete=False, stop_reason="session_budget_exhausted")
    row["ng_activations"][0]["response"]["status"] = "incomplete"
    assert healthy_record(row) == []


def budget_checkpoint():
    row = measured_zero()
    row["ng_activations"][0]["response"]["metadata"]["opencode_session_id"] = "native-session"
    row["ng_activations"].append(
        {
            "activation_id": 1,
            "response": {
                "status": "incomplete",
                "output": [],
                "metadata": {"native_input_dispatched": "false", "opencode_session_id": "native-session"},
            },
            "observation": {"harness_steps": 0, "events": []},
            "turn_complete": False,
            "stop_reason": "session_budget_exhausted",
        }
    )
    row["ng_steps"][0].update(continue_episode=True, responses_create_params={"input": "continue"})
    row["ng_steps"].append(
        {
            "activation_id": 1,
            "continue_episode": False,
            "responses_create_params": None,
            "stop_reason": "session_budget_exhausted",
        }
    )
    return row


def test_final_undispatched_budget_checkpoint_retains_prior_runtime_qualification():
    assert healthy_record(budget_checkpoint()) == []


@pytest.mark.parametrize("damage", ["identity", "output", "events", "steps", "completion", "resource_reason"])
def test_undispatched_checkpoint_requires_empty_terminal_same_session_evidence(damage):
    row = budget_checkpoint()
    last = row["ng_activations"][-1]
    if damage == "identity":
        last["response"]["metadata"]["opencode_session_id"] = "different-session"
    elif damage == "output":
        last["response"]["output"] = [{"type": "message"}]
    elif damage == "events":
        last["observation"]["events"] = [{"kind": "text"}]
    elif damage == "steps":
        last["observation"]["harness_steps"] = 1
    elif damage == "completion":
        last["turn_complete"] = True
    else:
        row["ng_steps"][-1]["stop_reason"] = "no_op_limit"
    assert "invalid_undispatched_budget_checkpoint" in healthy_record(row)


def test_budget_checkpoint_without_native_execution_is_not_a_healthy_task():
    row = budget_checkpoint()
    del row["ng_activations"][0]
    del row["ng_steps"][0]
    row["ng_activations"][0]["activation_id"] = 0
    row["ng_steps"][0]["activation_id"] = 0
    assert "invalid_undispatched_budget_checkpoint" in healthy_record(row)


@pytest.mark.parametrize("goals", ["present", [{}], [{"id": "g", "met": "false"}], [{"id": "g", "met": False}] * 2])
def test_malformed_or_duplicate_goal_results_do_not_count_as_a_grade(goals):
    row = measured_zero()
    row["verdict"]["goal_results"] = goals
    assert "missing_frozen_goal_evidence" in healthy_record(row)


def test_top_level_grade_matches_derived_verdict():
    row = measured_zero()
    row["verdict"]["judge_score"] = 0.4
    assert "verdict_disagrees_with_grade" in healthy_record(row)


def test_steps_match_activation_ids_and_end_the_episode():
    row = measured_zero()
    row["ng_steps"][0]["activation_id"] = 1
    assert "invalid_resource_steps" in healthy_record(row)
    row = measured_zero()
    row["ng_steps"][0].update(continue_episode=True, responses_create_params={"input": "continue"})
    assert "invalid_resource_steps" in healthy_record(row)


def test_valid_continuation_matches_two_ordered_steps():
    row = measured_zero()
    resumed = deepcopy(row["ng_activations"][0])
    resumed["activation_id"] = 1
    row["ng_activations"].append(resumed)
    row["ng_steps"] = [
        {"activation_id": 0, "continue_episode": True, "responses_create_params": {"input": "continue"}},
        {"activation_id": 1, "continue_episode": False},
    ]
    assert healthy_record(row) == []


@pytest.mark.parametrize("field,value", [("ng_activations", [None]), ("ng_steps", [None]), ("ng_agent_close", "ok")])
def test_malformed_lifecycle_records_report_problems_without_crashing(field, value):
    row = measured_zero()
    row[field] = value
    assert healthy_record(row)


def test_collector_identity_must_match_resource_result(tmp_path):
    path = tmp_path / "result.jsonl"
    path.write_text(json.dumps(measured_zero() | {"_ng_task_id": {"task_id": "different"}}) + "\n")
    report = audit([path], {"different"})
    assert not report["passed"]
    assert report["problem_counts"]["task_identity_mismatch"] == 1


def test_empty_task_manifest_is_not_a_success():
    assert not audit([], set())["passed"]
