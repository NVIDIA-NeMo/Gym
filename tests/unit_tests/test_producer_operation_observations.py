# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_gym.rollout_health import _observed_upstream_recovery


def test_terminal_observations_do_not_certify_unobserved_attempts():
    assert _observed_upstream_recovery([]) is None
    assert _observed_upstream_recovery([{"model_operations": [{"operation_id": "o", "outcome": "completed"}]}]) is None


def test_recovered_attempts_join_by_operation_and_deduplicate():
    call = {
        "status_code": 200,
        "failure_report": {
            "events": {
                "e": {"boundary": "model", "operation": "model_request", "operation_id": "o", "attempt_id": "o-1"}
            }
        },
        "model_operations": [{"operation_id": "o", "outcome": "completed"}],
    }
    result = _observed_upstream_recovery([call, call])
    assert result == {
        "observed_attempt_errors": 1,
        "observed_failed_operations": 1,
        "recovered_attempt_errors": 1,
        "recovered_operations": 1,
        "operations_without_terminal_observation": 0,
    }
    result = _observed_upstream_recovery([call, {"model_operations": [{"operation_id": "o", "outcome": "failed"}]}])
    assert result["recovered_operations"] == 0
    assert result["recovered_attempt_errors"] == 0
    assert result["operations_without_terminal_observation"] == 1


def test_failed_capture_does_not_call_a_retried_header_a_recovery():
    call = {
        "status_code": 500,
        "failure_report": {
            "events": {
                "e": {"boundary": "model", "operation": "model_request", "operation_id": "o", "attempt_id": "o-1"}
            }
        },
        "model_operations": [{"operation_id": "o", "outcome": "completed"}],
    }
    result = _observed_upstream_recovery([call])
    assert result["recovered_operations"] == 0
    assert result["operations_without_terminal_observation"] == 1


def test_failure_payload_preserves_only_valid_typed_slots():
    from nemo_gym.trajectory.trajectory_record import failure_observations

    assert (
        failure_observations({"response": {"id": "not-a-result"}, "reward": 1, "ng_trajectory": {"bad": True}}) == {}
    )
    assert (
        failure_observations(
            {"ng_verifier_invocations": [{"invocation_id": "v", "status": "failed", "failure_kind": "verifier_error"}]}
        )["ng_verifier_invocations"][0]["invocation_id"]
        == "v"
    )
    assert failure_observations({"ng_verifier_invocations": [{"invocation_id": "v", "invented": True}]}) == {}


def test_trial_boundary_consumes_runtime_observations_without_claiming_completeness():
    from nemo_gym.orchestration.recovery import trial_boundary

    quality = {
        "observed_runtime": {
            "invocation_observations": 1,
            "in_session": {"true": 0, "false": 1, "absent": 0},
            "verification_skipped": {"true": 0, "false": 1, "absent": 0},
        }
    }
    message = trial_boundary(quality)
    assert "observed invocation in_session" in message
    assert "do not certify the full trial population" in message
    assert "runtime state that no report records" not in message
    quality["observed_runtime"]["in_session"]["false"] = True
    assert "absent or malformed" in trial_boundary(quality)


def test_runtime_and_cost_observations_reject_coercion_and_nonfinite_values():
    import pytest
    from pydantic import ValidationError

    from nemo_gym.failure_transport import ModelOperationObservation
    from nemo_gym.trajectory.trajectory_record import AgentInvocation

    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="a", in_session=1)
    with pytest.raises(ValidationError):
        AgentInvocation(invocation_id="a", verification_skipped="false")
    with pytest.raises(ValidationError):
        ModelOperationObservation(operation_id="o", outcome="completed", retry_wait_s=float("inf"))


def test_trial_coverage_from_actual_cpu_collector_metadata_fails_closed_when_evidence_is_removed():
    import json
    from copy import deepcopy
    from pathlib import Path

    from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_health

    fixture = json.loads((Path(__file__).parent / "fixtures/cpu-mcqa-programmatic-operations.json").read_text())
    collected = CollectedEvidence.model_validate(deepcopy(fixture))
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert trial_health(decision, collected)[0] is True
    for key in ("invocations", "verifier_invocations", "model_calls", "turns"):
        negative = CollectedEvidence.model_validate(deepcopy(fixture))
        negative.results[0]["ng_trial_observations"].pop(key)
        assert trial_health(decision, negative)[0] is None
    for flag, value in (("in_session", None), ("verification_skipped", True)):
        negative = CollectedEvidence.model_validate(deepcopy(fixture))
        negative.results[0]["ng_trial_observations"]["invocations"][0][flag] = value
        assert trial_health(decision, negative)[0] is None
    negative = CollectedEvidence.model_validate(deepcopy(fixture))
    negative.results[0]["ng_trial_observations"]["tool_call_count"] = False
    assert trial_health(decision, negative)[0] is None

    negative = CollectedEvidence.model_validate(deepcopy(fixture))
    observed = negative.results[0]["ng_trial_observations"]
    observed["invocations"][0]["model_calls"][0]["model_call_id"] = observed["model_calls"][0]["model_call_id"]
    observed["invocations"][0]["model_calls"][0]["response_id"] = "contradictory-response"
    assert trial_health(decision, negative)[0] is None


def test_trial_coverage_requires_whole_row_report_and_matching_turn():
    import json
    from pathlib import Path

    from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_health

    fixture = json.loads((Path(__file__).parent / "fixtures/cpu-mcqa-programmatic-operations.json").read_text())
    baseline = CollectedEvidence.model_validate(fixture)
    decision = decide_from_collected(baseline, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert trial_health(decision, baseline)[0] is True
    row_mutations = (
        ("_ng_failure_events_omitted", True),
        ("_ng_failure_events_omitted", "false"),
        ("_ng_failure_id", "selected-event"),
        ("_ng_failure_events", {"unaccounted": {"boundary": "resources_server"}}),
        ("_ng_failure_events", []),
    )
    for key, value in row_mutations:
        negative = baseline.model_copy(deep=True)
        negative.results[0][key] = value
        assert trial_health(decision, negative)[0] is None
    for value in (None, "", " "):
        negative = baseline.model_copy(deep=True)
        negative.results[0]["ng_trial_observations"]["invocations"][0]["invocation_id"] = value
        assert trial_health(decision, negative)[0] is None
    for key in ("invocation_id", "task_id", "rollout_id"):
        negative = baseline.model_copy(deep=True)
        negative.results[0]["ng_trial_observations"]["turns"][0][key] = "contradictory"
        assert trial_health(decision, negative)[0] is None
    for value in (None, False, "true"):
        negative = baseline.model_copy(deep=True)
        negative.quality["source_matches"] = value
        assert trial_health(decision, negative)[0] is None


def test_recovered_observations_remain_unknown_and_keep_both_counters():
    import json
    from pathlib import Path

    from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_health

    fixture = json.loads((Path(__file__).parent / "fixtures/cpu-mcqa-trial-operations.json").read_text())
    collected = CollectedEvidence.model_validate(fixture)
    counters = {"recovered_attempt_errors": 2, "recovered_operations": 1}
    collected.quality["observed_upstream_recovery"] = counters.copy()
    decision = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    passed, reason = trial_health(decision, collected)
    assert passed is None and "not defined" in reason
    assert collected.quality["observed_upstream_recovery"] == counters
    assert decision.action.value == "noop"


def test_historical_mcqa_without_grading_observation_stays_unknown():
    import json
    from pathlib import Path

    from nemo_gym.orchestration.recovery import CollectedEvidence, decide_from_collected, trial_health

    fixtures = Path(__file__).parent / "fixtures"
    historical = CollectedEvidence.model_validate_json((fixtures / "cpu-mcqa-trial-operations.json").read_text())
    decision = decide_from_collected(historical, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert decision.action.value == "noop"
    assert trial_health(decision, historical)[0] is None
    assert all(
        "grading_operations" not in verifier
        for verifier in historical.results[0]["ng_trial_observations"]["verifier_invocations"]
    )
    # Fresh positive metadata came from the real programmatic scorer and collector, not a historical backfill.
    current_text = (fixtures / "cpu-mcqa-programmatic-operations.json").read_text()
    current = json.loads(current_text)
    verifier = current["results"][0]["ng_trial_observations"]["verifier_invocations"][0]
    assert (
        verifier["grading_operations"][0]["result_binding_sha256"]
        == current["results"][0]["_ng_collected_grading_binding"]
    )
    for raw_field in ('"text":', '"content":', '"question":', '"answer":'):
        assert raw_field not in current_text
