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
"""The trial gate's treatment of recovered policy requests.

A policy request retried inside one completed operation (a timeout, then an answer) is a recovered attempt, not a
trial failure, when each failed attempt joins that operation.
"""

from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    CollectedEvidence,
    decide_from_collected,
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
    assert reason.startswith("trial: 0 failed of 1 expected trials (0.00%)")
    assert "1 failed policy attempt(s) were retried and completed in the same operation" in reason
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert not [e for e in task["errors"] if "health" in e], task["errors"]
    assert any(
        w.startswith("health gate within the bounded failure policy: trial: 0 failed") for w in task["warnings"]
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
    assert "treatment is not defined" in reason
