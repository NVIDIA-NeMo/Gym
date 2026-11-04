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
"""Recovered model attempts are joined per attempt, whether or not the run's grading coverage is complete.

A model request that failed and was retried and completed in the same operation is a recovered attempt, not a trial
failure: the trial gate counts it as recovered and reports it. Its join does not depend on the coverage join, so a
run whose grading coverage is incomplete is judged on that coverage (unobserved or unverified, never passed), not
reported as having recovered failures with no defined treatment. A recovered attempt the run counted that no scored
trial's call joins is reported as such and leaves the run unverified; it never fails evaluation health.
"""

from pathlib import Path

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
    gate_summary_from_collected,
    recovered_attempt_joins,
    trial_verdict,
)


FIXTURE = Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json"


def _collected() -> CollectedEvidence:
    return CollectedEvidence.model_validate_json(FIXTURE.read_text())


def _decide(collected: CollectedEvidence):
    return decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)


def _row(collected: CollectedEvidence) -> dict:
    return collected.results[0]


def _counted(collected: CollectedEvidence, *, attempts: int, operations: int) -> None:
    """The run's observed upstream recovery, as the health summary counts it over every saved row."""
    collected.quality["observed_upstream_recovery"] = {
        "observed_attempt_errors": attempts,
        "observed_failed_operations": operations,
        "recovered_attempt_errors": attempts,
        "recovered_operations": operations,
        "operations_without_terminal_observation": 0,
    }


def _retried(meta: dict, *, statuses: list[int | None], saved: dict | None = None) -> None:
    """The call's one operation failed once per status (None: a transport failure) and its last attempt answered."""
    [operation] = meta["model_operations"]
    op = operation["operation_id"]
    http = [status for status in statuses if status is not None]
    operation["http_attempts"] = len(http) + 1
    operation["transport_failures"] = len(statuses) - len(http)
    operation["final_attempt_id"] = f"{op}-{len(http) + 1}"
    events = {}
    for number, status in enumerate(statuses, 1):
        attempt = f"{op}-{number}" if status is not None else f"{op}-transport-{number}"
        event = {
            "event_id": f"{op}-event-{number}",
            "boundary": "model",
            "operation": "model_request",
            "operation_id": op,
            "attempt_id": attempt,
            "cause_id": None,
            "failure_kind": "model_request_error",
            "http_status": status,
        }
        events[event["event_id"]] = event
        if saved is not None:
            saved[event["event_id"]] = {k: v for k, v in event.items() if k != "cause_id"} | {
                "failure_reason": "server disconnected" if status is None else f"HTTP {status} from the model server"
            }
    meta["failure_report"]["events"] = events


def _ungraded(collected: CollectedEvidence) -> None:
    """The verifier completed but recorded no grading operation and no grading record of its own."""
    [verifier] = _row(collected)["ng_trial_observations"]["verifier_invocations"]
    verifier["grading_operations"] = None


def _policy_retried_once(collected: CollectedEvidence) -> None:
    row = _row(collected)
    [call] = row["ng_trial_observations"]["model_calls"]
    row["_ng_failure_events"] = {}
    _retried(call["response_metadata"], statuses=[None], saved=row["_ng_failure_events"])


def _simulated_user_retried(collected: CollectedEvidence) -> None:
    """Another model the trial called (a simulated user) answered after three HTTP 503 attempts in one operation."""
    row = _row(collected)
    meta = {
        "response_id": "resp_simulated_user_1",
        "status_code": 200,
        "response_status": None,
        "error_category": None,
        "model_ref": {"type": "responses_api_models", "name": "simulated_user"},
        "model_operations": [
            {
                "operation_id": "op-simulated-user-1",
                "outcome": "completed",
                "final_attempt_id": "op-simulated-user-1-1",
                "failure_id": None,
                "http_attempts": 1,
                "transport_failures": 0,
            }
        ],
        "failure_report": {"failure_id": None, "omitted": False, "events": {}},
    }
    row["_ng_failure_events"] = {}
    _retried(meta, statuses=[503, 503, 503], saved=row["_ng_failure_events"])
    row["ng_trial_observations"]["model_calls"].append(
        {"model_call_id": "call-simulated-user-1", "response_metadata": meta, "output_items": [], "input_items": []}
    )


def test_a_policy_retry_in_a_run_without_recorded_grading_is_recovered_and_the_run_is_unobserved():
    collected = _collected()
    _ungraded(collected)
    _policy_retried_once(collected)
    _counted(collected, attempts=1, operations=1)
    decision = _decide(collected)
    assert recovered_attempt_joins(decision, collected) == {"attempts": 1, "operations": 1}
    passed, reason, evaluation = trial_verdict(decision, collected)
    assert passed is None, reason
    assert reason.startswith("trial: UNOBSERVED 1 of 1 expected trials: no completed grading operation"), reason
    assert (
        "1 failed model attempt(s) in 1 operation(s) were retried and completed in the same operation "
        "(recovered, no trial failure)"
    ) in reason
    assert "treatment is not defined" not in reason
    assert evaluation["status"] == "UNVERIFIED"
    assert evaluation["evaluation"]["count"] == 0 and evaluation["model"]["count"] == 0
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert task["trial_counts"]["unobserved"] == 1


def test_a_retried_call_of_another_model_the_trial_called_is_recovered():
    collected = _collected()
    _ungraded(collected)
    _simulated_user_retried(collected)
    _counted(collected, attempts=3, operations=1)
    decision = _decide(collected)
    assert recovered_attempt_joins(decision, collected) == {"attempts": 3, "operations": 1}
    passed, reason, _ = trial_verdict(decision, collected)
    assert passed is None, reason
    assert reason.startswith("trial: UNOBSERVED 1 of 1 expected trials"), reason
    assert "3 failed model attempt(s) in 1 operation(s) were retried and completed in the same operation" in reason


def test_a_recovered_attempt_never_stands_in_for_the_answers_grading_binding():
    collected = _collected()
    _policy_retried_once(collected)
    _counted(collected, attempts=1, operations=1)
    # The grading operation completed, but its binding is not the saved answer's.
    [verifier] = _row(collected)["ng_trial_observations"]["verifier_invocations"]
    verifier["grading_operations"][0]["result_binding_sha256"] = "0" * 64
    decision = _decide(collected)
    assert recovered_attempt_joins(decision, collected) == {"attempts": 1, "operations": 1}
    passed, reason, evaluation = trial_verdict(decision, collected)
    assert passed is None, reason
    # The unbound grading leaves the trial unobserved, never healthy.
    assert reason.startswith("trial: UNOBSERVED 1 of 1 expected trials"), reason
    assert "1 failed model attempt(s) in 1 operation(s) were retried" in reason
    assert evaluation["status"] == "UNVERIFIED" and evaluation["exact"] is False


def test_a_counted_recovered_attempt_no_scored_trial_joins_is_reported_and_leaves_the_run_unverified():
    collected = _collected()
    _ungraded(collected)
    _policy_retried_once(collected)
    # The run counted one more recovered attempt, in an operation no scored trial's call holds.
    _counted(collected, attempts=2, operations=2)
    decision = _decide(collected)
    passed, reason, evaluation = trial_verdict(decision, collected)
    assert passed is None, reason
    assert reason.startswith(
        "trial: UNVERIFIED, recovered attempts not joined: the run counted 2 recovered model attempt error(s) in 2 "
        "operation(s), and 1 in 1 join the completed operation of a scored trial's own call"
    ), reason
    assert reason.endswith("unverified: this withholds a pass and is no health failure")
    assert evaluation["status"] == "UNVERIFIED" and evaluation["evaluation"]["count"] == 0


def test_the_covered_run_with_a_recovered_policy_retry_still_passes():
    collected = _collected()
    _policy_retried_once(collected)
    _counted(collected, attempts=1, operations=1)
    passed, reason, evaluation = trial_verdict(_decide(collected), collected)
    assert passed is True, reason
    assert evaluation["exact"] is True
