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
"""The trial gate's treatment of recovered policy requests and of rollouts that ended on a failed policy request.

A policy request retried inside one completed operation (a timeout, then an answer) is a recovered attempt, not a
trial failure, when each failed attempt joins that operation. A scored row whose rollout ended on a failed policy
request and was scored zero is a settled trial failure, counted once in the trial family.
"""

from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
    final_zero_trials,
    gate_summary_from_collected,
    trial_health,
)


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"
RECOVERED = {
    "observed_attempt_errors": 1,
    "observed_failed_operations": 1,
    "recovered_attempt_errors": 1,
    "recovered_operations": 1,
    "operations_without_terminal_observation": 0,
}


def _collected() -> CollectedEvidence:
    return CollectedEvidence.model_validate_json(FIXTURE.read_text())


def _policy_call(collected: CollectedEvidence) -> dict:
    [call] = collected.results[0]["ng_trial_observations"]["model_calls"]
    return call


def _recover_one_attempt(collected: CollectedEvidence, *, http_status: int | None = None) -> dict:
    """The policy request failed once (a timeout, or an HTTP error) and its retry answered in the same operation."""
    meta = _policy_call(collected)["response_metadata"]
    [operation] = meta["model_operations"]
    op = operation["operation_id"]
    if http_status is None:
        operation["transport_failures"] = 1
        attempt = f"{op}-transport-1"
    else:
        operation["http_attempts"], operation["final_attempt_id"] = 2, f"{op}-2"
        attempt = f"{op}-1"
    event = {
        "event_id": "attempt-failure-1",
        "boundary": "model",
        "operation": "model_request",
        "operation_id": op,
        "attempt_id": attempt,
        "cause_id": None,
        "failure_kind": "model_request_error",
        "http_status": http_status,
    }
    meta["failure_report"]["events"] = {event["event_id"]: event}
    saved = {k: v for k, v in event.items() if k != "cause_id"} | {"failure_reason": "timed out waiting"}
    collected.results[0]["_ng_failure_events"] = {event["event_id"]: saved}
    collected.quality["observed_upstream_recovery"] = dict(RECOVERED)
    return event


def _decide(collected: CollectedEvidence):
    return decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)


def test_the_unchanged_fixture_passes_the_trial_gate():
    collected = _collected()
    assert trial_health(_decide(collected), collected)[0] is True


@pytest.mark.parametrize("http_status", [None, 503])
def test_a_policy_attempt_retried_and_completed_in_the_same_operation_is_no_trial_failure(http_status):
    collected = _collected()
    _recover_one_attempt(collected, http_status=http_status)
    decision = _decide(collected)
    passed, reason = trial_health(decision, collected)
    assert passed is True, reason
    assert reason.startswith("trial: evaluation-side faults: 0 of 1 (0.00%) is below 2%; model faults: 0 of 1")
    assert "1 failed policy attempt(s) were retried and completed in the same operation" in reason
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert not [e for e in task["errors"] if "health" in e], task["errors"]
    assert any(
        w.startswith("health gate within the bounded failure policy: trial: evaluation-side faults: 0 of 1")
        for w in task["warnings"]
    )


def _counted_twice(collected, event):
    collected.quality["observed_upstream_recovery"]["recovered_attempt_errors"] = 2


def _foreign_operation(collected, event):
    event["operation_id"] = "another-operation"


def _unrecorded_attempt(collected, event):
    _policy_call(collected)["response_metadata"]["model_operations"][0]["transport_failures"] = 0


def _the_answer_is_the_failed_attempt(collected, event):
    event["attempt_id"] = _policy_call(collected)["response_metadata"]["model_operations"][0]["final_attempt_id"]


def _another_kind(collected, event):
    event["failure_kind"] = "model_context_limit"


def _the_row_saved_another_event(collected, event):
    collected.results[0]["_ng_failure_events"][event["event_id"]]["operation_id"] = "another-operation"


def _the_answer_failed(collected, event):
    _policy_call(collected)["response_metadata"]["model_operations"][0]["outcome"] = "failed"


@pytest.mark.parametrize(
    "break_join",
    [
        _counted_twice,
        _foreign_operation,
        _unrecorded_attempt,
        _the_answer_is_the_failed_attempt,
        _another_kind,
        _the_row_saved_another_event,
        _the_answer_failed,
    ],
)
def test_a_recovery_that_does_not_join_one_completed_policy_operation_stays_unknown(break_join):
    collected = _collected()
    event = _recover_one_attempt(collected)
    break_join(collected, event)
    passed, reason = trial_health(_decide(collected), collected)
    assert passed is None, reason
    assert "recovered attempts not joined" in reason


def _end_on_a_failed_request(collected: CollectedEvidence, *, reward: float = 0.0, recorded: bool = True) -> None:
    """The only policy request failed (HTTP 400) and the rollout ended there; the verifier scored the row."""
    meta = _policy_call(collected)["response_metadata"]
    [operation] = meta["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", "request-failure-1"
    meta["response_status"], meta["error_category"] = "incomplete", "context_length_exceeded"
    event = {
        "event_id": "request-failure-1",
        "boundary": "model",
        "operation": "model_request",
        "operation_id": operation["operation_id"],
        "attempt_id": operation["final_attempt_id"],
        "cause_id": None,
        "failure_kind": "model_request_error",
        "http_status": 400,
    }
    meta["failure_report"]["events"] = {event["event_id"]: event} if recorded else {}
    collected.results[0]["reward"] = reward
    collected.quality["observed_upstream_recovery"] = {
        "observed_attempt_errors": 1,
        "observed_failed_operations": 1,
        "recovered_attempt_errors": 0,
        "recovered_operations": 0,
        "operations_without_terminal_observation": 0,
    }


def test_a_rollout_that_ended_on_a_failed_policy_request_and_scored_zero_is_a_settled_trial_failure():
    collected = _collected()
    _end_on_a_failed_request(collected)
    decision = _decide(collected)
    assert decision.counts == {"scored": 1}  # the decision and the score are unchanged
    assert final_zero_trials(decision, collected) == {(0, 0): "model_context_limit"}
    passed, reason = trial_health(decision, collected)
    # A policy request the server refused at the context limit is the model's fault: counted and named, never a health
    # failure.
    assert passed is True
    assert reason.startswith(
        "trial: evaluation-side faults: 0 of 1 (0.00%) is below 2%; model faults: 1 of 1 (100.00%) "
        "(model_context_limit 1); exact count:"
    )
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert not any(e.startswith("response/trial health") for e in task["errors"]), task["errors"]
    assert task["failures"]["trial_failures"]["by_cause"]["request"] == {"count": 1, "rate": 1.0}
    events = _policy_call(collected)["response_metadata"]["failure_report"]["events"]
    events["request-failure-1"]["failure_kind"] = "model_transport_error"
    passed, reason = trial_health(_decide(collected), collected)
    assert passed is False and "(server 1: model_transport_error 1)" in reason


def test_a_failed_last_request_without_its_event_is_not_counted():
    collected = _collected()
    _end_on_a_failed_request(collected, recorded=False)
    decision = _decide(collected)
    assert final_zero_trials(decision, collected) == {}
    passed, reason = trial_health(decision, collected)
    assert passed is None
    assert "known failures are a lower bound: evaluation-side faults: 0 of 1" in reason


def test_a_graded_row_that_ended_on_the_models_own_failed_request_is_a_model_result():
    # The verifier graded the state the model reached before its own context limit ended the rollout: a model result
    # at any reward, not a final zero and not an unaccounted failure.
    collected = _collected()
    _end_on_a_failed_request(collected, reward=1.0)
    decision = _decide(collected)
    assert final_zero_trials(decision, collected) == {}
    passed, reason = trial_health(decision, collected)
    assert passed is True, reason


def test_a_graded_row_that_ended_on_an_evaluation_side_failed_request_stays_unverified():
    collected = _collected()
    _end_on_a_failed_request(collected, reward=1.0)
    events = _policy_call(collected)["response_metadata"]["failure_report"]["events"]
    events["request-failure-1"]["failure_kind"] = "model_transport_error"
    passed, reason = trial_health(_decide(collected), collected)
    assert passed is None
    assert "known failures are a lower bound" in reason
