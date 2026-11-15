# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata fixtures are projected from actual local HTTP model/agent/tool/verifier/Collection executions."""

import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    _observed_trial_operations,
    decide_from_collected,
    trial_health,
)


def fixture(case="healthy"):
    return CollectedEvidence.model_validate_json(
        (Path(__file__).parent / "fixtures" / f"cpu-tool-trial-{case}.json").read_text()
    )


def health(collected):
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    return decision, trial_health(decision, collected)


def trajectory_reported(collected):
    """The trial verdict and its reported trajectory evidence: the grading bound to the answer decides."""
    from nemo_gym.orchestration.recovery import trial_verdict

    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    passed, reason, report = trial_verdict(decision, collected)
    return passed, reason, report


def assert_reported(collected, incomplete=1):
    """The run passes on its bound grading, and the trajectory evidence it lacks is reported."""
    passed, reason, report = trajectory_reported(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["trajectory_evidence"]["incomplete"] == incomplete, report
    assert "trajectory evidence incomplete:" in reason


@pytest.mark.parametrize(
    "case,reward,expected,action",
    [
        ("healthy", 1, True, "noop"),
        ("wrong", 0, True, "noop"),
        ("tool-error", 1, True, "noop"),
        ("verifier-error", None, None, "stop"),
        ("hosted-tool", 1, True, "noop"),
    ],
)
def test_actual_tool_producer_outcomes(case, reward, expected, action):
    collected = fixture(case)
    decision, (passed, _) = health(collected)
    assert passed is expected
    # A tool that failed or ran hosted is trajectory evidence; only the verifier's error leaves grading unbound.
    assert _observed_trial_operations(decision, collected) is (case != "verifier-error")
    assert decision.action.value == action
    assert decision.expected == 1
    if reward is not None:
        assert collected.results[0]["reward"] == reward


# Mutations that leave a trial's grading unobserved, or its failures unaccounted for; the rest are trajectory evidence.
COVERAGE_MUTATIONS = {
    "duplicate_trial",
    "missing_expected_trial",
    "missing_verifier",
    "unobserved_grader",
    "failed_verifier",
    "unselected_failure",
    "omitted_failures",
    "stale_source",
}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_tools",
        "removed_tool",
        "truncated_calls",
        "truncated_turns",
        "wrong_tool_id",
        "wrong_tool_owner",
        "wrong_tool_name",
        "duplicate_tool",
        "unknown_tool",
        "failed_tool",
        "missing_tool_result",
        "duplicate_tool_result",
        "foreign_tool_result",
        "missing_terminal_response",
        "wrong_turn",
        "duplicate_operation",
        "duplicate_attempt",
        "duplicate_capture",
        "missing_verifier",
        "failed_verifier",
        "unknown_runtime",
        "nested_invocation",
        "unobserved_grader",
        "missing_expected_trial",
        "duplicate_trial",
        "unselected_failure",
        "omitted_failures",
        "stale_source",
        "missing_raw_output",
    ],
)
def test_incomplete_or_contradictory_tool_evidence_is_reported_not_a_coverage_condition(mutation):
    c = fixture()
    row = c.results[0]
    t = row["ng_trial_observations"]
    if mutation == "missing_tools":
        t.pop("tool_calls")
    elif mutation == "removed_tool":
        t["tool_calls"] = []
        t["tool_call_count"] = 0
    elif mutation == "truncated_calls":
        t["model_calls"].pop()
    elif mutation == "truncated_turns":
        t["turns"].pop()
    elif mutation == "wrong_tool_id":
        t["tool_calls"][0]["tool_call_id"] = "foreign"
    elif mutation == "wrong_tool_owner":
        t["tool_calls"][0]["invocation_id"] = "foreign"
    elif mutation == "wrong_tool_name":
        t["tool_calls"][0]["tool_name"] = "foreign"
    elif mutation == "duplicate_tool":
        t["tool_calls"].append(dict(t["tool_calls"][0]))
        t["tool_call_count"] += 1
    elif mutation in ("unknown_tool", "failed_tool"):
        t["tool_calls"][0]["status"] = "unknown" if mutation == "unknown_tool" else "failed"
    elif mutation == "missing_tool_result":
        t["model_calls"][1]["input_items"] = []
    elif mutation == "duplicate_tool_result":
        t["model_calls"][1]["input_items"].append({"type": "function_call_output", "call_id": "call-1"})
    elif mutation == "foreign_tool_result":
        t["model_calls"][1]["input_items"].append({"type": "function_call_output", "call_id": "foreign"})
    elif mutation == "missing_terminal_response":
        t["model_calls"][1]["output_items"] = []
    elif mutation == "wrong_turn":
        t["turns"][1]["model_calls"] = t["turns"][0]["model_calls"]
    elif mutation in ("duplicate_operation", "duplicate_attempt"):
        key = "operation_id" if mutation == "duplicate_operation" else "final_attempt_id"
        t["model_calls"][1]["response_metadata"]["model_operations"][0][key] = t["model_calls"][0][
            "response_metadata"
        ]["model_operations"][0][key]
    elif mutation == "duplicate_capture":
        t["model_calls"].append(t["model_calls"][0])
    elif mutation == "missing_verifier":
        t["verifier_invocations"] = []
    elif mutation == "failed_verifier":
        t["verifier_invocations"][0]["status"] = "failed"
    elif mutation == "unknown_runtime":
        t["invocations"][0]["in_session"] = None
    elif mutation == "nested_invocation":
        t["invocations"][0]["parent_invocation_id"] = "parent"
    elif mutation == "unobserved_grader":
        t["verifier_invocations"][0]["model_calls"] = [{"model_call_id": "foreign"}]
    elif mutation == "missing_expected_trial":
        c.materialized.append({"_ng_task_index": 1, "_ng_rollout_index": 0})
    elif mutation == "duplicate_trial":
        c.results.append(dict(row))
    elif mutation == "unselected_failure":
        row["_ng_failure_events"] = {"event": {"boundary": "resources_server"}}
    elif mutation == "omitted_failures":
        row["_ng_failure_events_omitted"] = True
    elif mutation == "stale_source":
        c.quality["source_matches"] = False
    elif mutation == "missing_raw_output":
        t["model_calls"][0]["output_items"] = None
    if mutation in COVERAGE_MUTATIONS:
        # Missing or unbound grading, a missing or duplicated trial, a failure no recovery accounts for, or an
        # unreadable source: the run is not observed.
        decision, (passed, _) = health(c)
        assert passed is None
        assert _observed_trial_operations(decision, c) is False
    else:
        # Tool and turn evidence is trajectory evidence: reported, never a coverage condition.
        assert_reported(c)


def test_tool_coverage_uses_observed_operations_not_benchmark_names():
    c = fixture()
    for server in c.quality["provenance"]["servers"].values():
        if server["kind"] in ("responses_api_agents", "resources_servers"):
            server["implementation"] = "another_adapter_with_the_same_observed_contract"
    decision, (passed, _) = health(c)
    assert _observed_trial_operations(decision, c) is True
    assert passed is True


def test_recovered_model_counters_remain_distinct_and_unknown():
    c = fixture()
    counters = {"recovered_attempt_errors": 2, "recovered_operations": 1}
    c.quality["observed_upstream_recovery"] = counters.copy()
    decision, (passed, reason) = health(c)
    assert passed is None and "recovered attempts not joined" in reason
    assert c.quality["observed_upstream_recovery"] == counters
    assert decision.action.value == "noop"


def test_selected_tool_metadata_contains_no_model_or_tool_payloads():
    data = fixture().model_dump_json()
    for forbidden in ("arguments", "content", "text", "question", "answer", "conversation"):
        assert json.dumps(forbidden) not in data


def test_actual_repeated_tool_trials_keep_the_full_inventory():
    c = fixture("repeats")
    decision, (passed, _) = health(c)
    assert passed is True and decision.expected == 2
    assert _observed_trial_operations(decision, c) is True
    assert {(r["_ng_task_index"], r["_ng_rollout_index"]) for r in c.results} == {(0, 0), (0, 1)}
    c.results.pop()
    decision, (passed, _) = health(c)
    assert passed is None
    assert _observed_trial_operations(decision, c) is False


@pytest.mark.parametrize("change", ["id", "status"])
def test_a_graded_response_that_does_not_match_the_final_capture_is_reported(change):
    c = fixture()
    c.results[0]["response"][change] = {"id": "foreign", "status": "incomplete"}[change]
    assert_reported(c)


def test_a_graded_response_that_saved_an_error_cannot_accept():
    c = fixture()
    c.results[0]["response"]["error"] = {}
    decision, (passed, _) = health(c)
    assert passed is None
    assert _observed_trial_operations(decision, c) is False


@pytest.mark.parametrize(
    "location,item",
    [
        ("output_items", {"type": "mcp_call", "status": "completed"}),
        ("output_items", {"type": "custom_tool_call", "call_id": "foreign", "name": "x"}),
        ("input_items", {"type": "custom_tool_call_output", "call_id": "foreign"}),
    ],
)
def test_unjoined_runtime_operations_beside_supported_items_are_reported(location, item):
    c = fixture()
    c.results[0]["ng_trial_observations"]["model_calls"][-1][location].append(item)
    assert_reported(c)


@pytest.mark.parametrize(
    "counters",
    [
        {},
        {"recovered_attempt_errors": "1", "recovered_operations": 0},
        {"recovered_attempt_errors": 0, "recovered_operations": True},
        {"recovered_attempt_errors": -1, "recovered_operations": 0},
        False,
    ],
)
def test_malformed_recovery_counters_do_not_become_zero(counters):
    c = CollectedEvidence.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/cpu-mcqa-trial-operations.json").read_text()
    )
    c.quality["observed_upstream_recovery"] = counters
    decision, (passed, _) = health(c)
    assert passed is None
    assert c.quality["observed_upstream_recovery"] == counters


@pytest.mark.parametrize("section", ["tool_calls", "turns"])
@pytest.mark.parametrize("malformed", [None, False, 7, "unknown", []])
def test_malformed_runtime_entries_are_reported(section, malformed):
    c = fixture()
    c.results[0]["ng_trial_observations"][section][0] = malformed
    assert_reported(c)


def test_actual_judge_backed_verifier_without_captures_cannot_be_called_programmatic():
    c = fixture("unobserved-judge")
    decision, (passed, reason) = health(c)
    observed = c.results[0]["ng_trial_observations"]
    assert observed["verifier_invocations"][0]["status"] == "completed"
    assert observed["verifier_invocations"][0]["model_calls"] == []
    assert c.quality["provenance"]["servers"]["mcqa"]["implementation"] == "equivalence_llm_judge"
    assert _observed_trial_operations(decision, c) is False
    assert passed is None
    assert c.results[0]["reward"] == 1


def test_step_counter_is_not_reinterpreted_as_post_tool_timing():
    c = fixture()
    c.results[0]["ng_trial_observations"]["turns"][0]["step_count"] = 0
    decision, (passed, _) = health(c)
    assert _observed_trial_operations(decision, c) is True
    assert passed is True


@pytest.mark.parametrize("change", ["absent", "empty", "unknown", "dependency", "binding", "duplicate"])
def test_unobserved_or_unbound_grading_cannot_accept(change):
    c = fixture()
    verifier = c.results[0]["ng_trial_observations"]["verifier_invocations"][0]
    operations = verifier["grading_operations"]
    if change == "absent":
        verifier.pop("grading_operations")
    elif change == "empty":
        verifier["grading_operations"] = []
    elif change == "unknown":
        operations[0]["outcome"] = "unknown"
    elif change == "dependency":
        operations[0]["dependencies"] = [{"model_call_id": "unjoined"}]
    elif change == "binding":
        operations[0]["result_binding_sha256"] = "0" * 64
    elif change == "duplicate":
        operations.append(dict(operations[0]))
    decision, (passed, _) = health(c)
    assert passed is None and decision.action.value == "noop"


@pytest.mark.parametrize("case,reward", [("healthy", 1), ("wrong", 0)])
def test_actual_mcqa_calculation_keeps_ordinary_scores(case, reward):
    c = CollectedEvidence.model_validate_json(
        (Path(__file__).parent / "fixtures" / f"cpu-grading-mcqa-{case}.json").read_text()
    )
    decision, (passed, _) = health(c)
    assert passed is True and decision.action.value == "noop"
    assert c.results[0]["reward"] == reward


def test_historical_mcqa_without_grading_observation_is_unknown_not_rescored():
    c = CollectedEvidence.model_validate_json(
        (Path(__file__).parents[1] / "fixtures/cpu-mcqa-trial-operations.json").read_text()
    )
    before = [r.get("reward") for r in c.results]
    decision, (passed, _) = health(c)
    assert passed is None and decision.action.value == "noop"
    assert [r.get("reward") for r in c.results] == before


@pytest.mark.parametrize("location", ["materialized", "trajectory", "turn"])
def test_a_trajectory_with_another_task_identity_is_reported(location):
    c = fixture()
    assert health(c)[1][0] is True
    if location == "materialized":
        c.materialized[0]["_ng_collected_task_id"] = "foreign"
    elif location == "trajectory":
        c.results[0]["ng_trial_observations"]["task_id"] = "foreign"
    else:
        c.results[0]["ng_trial_observations"]["turns"][0]["task_id"] = "foreign"
    assert_reported(c)


@pytest.mark.parametrize("bad", [None, False, 7, "unknown", [], {}])
def test_malformed_grading_entry_cannot_accept(bad):
    c = fixture()
    c.results[0]["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = [bad]
    assert health(c)[1][0] is None


def test_reused_grading_operation_cannot_cover_two_expected_trials():
    c = fixture("repeats")
    assert health(c)[1][0] is True
    operations = [
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"][0] for row in c.results
    ]
    operations[1]["operation_id"] = operations[0]["operation_id"]
    assert health(c)[1][0] is None
