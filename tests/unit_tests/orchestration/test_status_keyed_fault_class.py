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
"""A counted failure's class is keyed on the HTTP status it recorded, not on its kind alone.

A model server that replied 5xx is the server's fault (evaluation-side) whatever the kind says; a 4xx rejection of
the evaluated policy's own request stays the model's fault; another model's 4xx rejection is evaluation-side; without
a recorded status the kind decides. An agent that references none of its calls ended on a failed request only when its
last captured call failed. A trial whose agent run failed but whose answer was graded is reported as a fact.
"""

import pytest

from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run
from tests.unit_tests.orchestration.test_trajectory_evidence_reporting import _simulated_user_shape


def _fail_last_request(call: dict, *, status: int | None, event_id: str = "rejected-1") -> None:
    """The call's final operation failed, with its own recorded model-request event."""
    [operation] = call["response_metadata"]["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", event_id
    call["response_metadata"]["failure_report"]["events"] = {
        event_id: {
            "event_id": event_id,
            "boundary": "model",
            "operation": "model_request",
            "operation_id": operation["operation_id"],
            "attempt_id": operation["final_attempt_id"],
            "cause_id": None,
            "failure_kind": "model_request_error",
            "http_status": status,
        }
    }


def _ended_on_a_failed_policy_request(status: int | None):
    """One trial's only policy request failed and the rollout scored zero; 99 others are graded."""
    collected = _run(100)
    row = collected.results[0]
    row["reward"] = 0.0
    [call] = row["ng_trial_observations"]["model_calls"]
    _fail_last_request(call, status=status)
    return collected


@pytest.mark.parametrize(
    "status,evaluation,model",
    [
        (503, {"model_server_error": 1}, {}),
        (400, {}, {"model_request_error": 1}),
        (None, {}, {"model_request_error": 1}),
    ],
    ids=["server-5xx", "policy-4xx", "no-status"],
)
def test_a_rollout_that_ended_on_a_failed_request_is_classed_by_its_status(status, evaluation, model):
    _, (passed, reason, report), _ = _gate(_ended_on_a_failed_policy_request(status))
    assert report["exact"] is True, reason
    assert report["evaluation"]["kinds"] == evaluation
    assert report["model"]["kinds"] == model
    assert report["evaluation"]["server"] == (1 if status == 503 else 0)
    if status == 503:
        assert "(server 1: model_server_error 1)" in reason


def test_a_settled_model_failure_with_a_5xx_status_is_the_servers_fault():
    collected = _run(99, settled={"model_request_error": 1})
    [failure] = collected.failures
    failure["_ng_failure_http_status"] = 503
    _, (passed, reason, report), _ = _gate(collected)
    assert report["evaluation"]["kinds"] == {"model_server_error": 1}, reason
    assert report["model"]["count"] == 0


def _unreferenced_run():
    collected = _run(20)
    for task, row in enumerate(collected.results):
        _simulated_user_shape(row, task)
        row["ng_policy_model_server"] = "model"
    return collected


def _second_call(row: dict, *, started_at: float) -> dict:
    """A later captured call that answered: a copy of the first with its own ids."""
    import copy

    [first] = row["ng_trial_observations"]["model_calls"]
    first["started_at"] = 1.0
    later = copy.deepcopy(first)
    later["model_call_id"] = "later-call"
    later["started_at"] = started_at
    later["response_metadata"]["response_id"] = "resp-later"
    [operation] = later["response_metadata"]["model_operations"]
    operation["operation_id"], operation["final_attempt_id"] = "later-op", "later-op-1"
    row["ng_trial_observations"]["model_calls"].append(later)
    return first


def test_an_unreferenced_agent_that_answered_after_a_failed_call_is_not_settled_as_a_failure():
    collected = _unreferenced_run()
    row = collected.results[0]
    row["reward"] = 0.0
    row["_ng_collected_grading_facts"]["environment_reward"] = 0.0
    first = _second_call(row, started_at=2.0)
    _fail_last_request(first, status=400)
    _, (passed, reason, report), _ = _gate(collected)
    # The failed call is no end: it stays unaccounted, so the run is unverified, never settled as a model failure.
    assert passed is None and report["exact"] is False, reason
    assert report["model"]["count"] == 0


def test_an_unreferenced_agent_whose_last_call_failed_is_settled_with_that_calls_kind():
    collected = _unreferenced_run()
    row = collected.results[0]
    row["reward"] = 0.0
    row["_ng_collected_grading_facts"]["environment_reward"] = 0.0
    _second_call(row, started_at=2.0)
    _fail_last_request(row["ng_trial_observations"]["model_calls"][-1], status=400)
    _, (passed, reason, report), _ = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"model_request_error": 1}


def test_another_models_rejection_that_ended_an_unreferenced_rollout_is_evaluation_side():
    collected = _unreferenced_run()
    row = collected.results[0]
    row["reward"] = 0.0
    row["_ng_collected_grading_facts"]["environment_reward"] = 0.0
    [call] = row["ng_trial_observations"]["model_calls"]
    call["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": "simulated_user"}
    _fail_last_request(call, status=400)
    _, (_, reason, report), _ = _gate(collected)
    assert report["evaluation"]["kinds"] == {"non_policy_model_rejection": 1}, reason
    assert report["model"]["count"] == 0


def test_trials_whose_agent_run_failed_but_were_graded_are_reported_as_a_fact():
    collected = _run(20)
    for row in collected.results[:3]:
        row["ng_trial_observations"]["invocations"][0]["status"] = "failed"
    _, (passed, reason, report), _ = _gate(collected)
    assert passed is True and report["exact"] is True, reason
    assert report["facts"] == {"agent_invocation_failed_grading_recorded": 3}
    assert "agent invocation failed, grading recorded: 3" in reason
