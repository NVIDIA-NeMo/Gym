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
"""A policy call the model server rejected for exceeding the context window is the model's ``model_context_limit``.

A scored zero whose policy's last call was rejected with HTTP 4xx and a context-length reason (the call's
``error_category`` or its ``local_response_reason``) is a counted model failure of that kind, read from the saved
call: the trial gate is exact and the zero stays. Another model's 4xx stays evaluation-side, a 5xx stays the
server's, and a policy 4xx without a context reason stays ``model_request_error``.
"""

import json

import pytest

from nemo_gym.orchestration.recovery import decide_from_collected, final_zero_trials
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run
from tests.unit_tests.orchestration.test_recovery_health_binding import collect, save_rows, setup
from tests.unit_tests.orchestration.test_trajectory_evidence_reporting import _simulated_user_shape


def _reject(call: dict, *, status: int | None, kind: str | None = "model_request_error", event_id: str = "ctx-1"):
    """The call's final operation failed with its own model-request event; returns the event."""
    meta = call["response_metadata"]
    [operation] = meta["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", event_id
    event = {
        "event_id": event_id,
        "boundary": "model",
        "operation": "model_request",
        "operation_id": operation["operation_id"],
        "attempt_id": operation["final_attempt_id"],
        "cause_id": None,
        "failure_kind": kind,
        "http_status": status,
    }
    meta["failure_report"]["events"] = {event_id: event}
    meta["status_code"], meta["response_status"] = status, None
    call["output_items"] = []
    return event


def _tau_style_run(*, status: int = 400, model: str = "model", merged: bool = True):
    """A tool-transcript agent and a simulated user: the policy's last call was rejected and the trial scored zero.

    With ``merged``, the simulated user's earlier call was captured after the policy's (each server's capture is
    appended in turn). The row saved the rejected call's failure event.
    """
    import copy

    collected = _run(20)
    for task, row in enumerate(collected.results):
        _simulated_user_shape(row, task)
        row["ng_policy_model_server"] = "model"
    row = collected.results[0]
    row["reward"] = 0.0
    row["_ng_collected_grading_facts"]["environment_reward"] = 0.0
    [policy] = row["ng_trial_observations"]["model_calls"]
    user = copy.deepcopy(policy)
    policy["started_at"], user["started_at"] = 2.0, 1.0
    policy["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": model}
    user["model_call_id"] = "user-call"
    user["response_metadata"]["response_id"] = "resp-user"
    user["response_metadata"]["model_ref"] = {"type": "responses_api_models", "name": "simulated_user"}
    [operation] = user["response_metadata"]["model_operations"]
    operation["operation_id"], operation["final_attempt_id"] = "user-op", "user-op-1"
    event = _reject(policy, status=status)
    policy["response_metadata"]["error_category"] = "context_length_exceeded"
    row["_ng_failure_events"] = {event["event_id"]: dict(event)}
    if merged:
        row["ng_trial_observations"]["model_calls"].append(user)
    return collected


def _lmarena_style_run(
    *, upstream: int = 400, reason: str | None = "context_length_exceeded", kind: str | None = None
):
    """One policy call, rejected upstream: by default its event names no kind; the call kept its upstream status and reason."""
    collected = _run(100)
    row = collected.results[0]
    row["reward"] = 0.0
    [call] = row["ng_trial_observations"]["model_calls"]
    event = _reject(call, status=upstream, kind=kind)
    call["response_metadata"]["upstream_status_code"] = upstream
    call["response_metadata"]["local_response_reason"] = reason
    row["_ng_failure_events"] = {event["event_id"]: dict(event)}
    return collected


def test_a_tau_style_trial_that_ended_on_the_policys_context_limit_is_an_exact_model_failure():
    _, (passed, reason, report), task = _gate(_tau_style_run())
    assert passed is True and report["exact"] is True, reason
    assert "lower bound" not in reason and "UNVERIFIED" not in reason
    assert report["model"] == {"count": 1, "fraction": 0.05, "kinds": {"model_context_limit": 1}}
    assert report["evaluation"] == {"count": 0, "fraction": 0.0, "kinds": {}, "server": 0}
    assert task["failures"]["trial_failures"]["by_cause"]["request"] == {"count": 1, "rate": 0.05}


def test_an_lmarena_style_trial_rejected_upstream_for_its_context_is_an_exact_model_failure():
    _, (passed, reason, report), _ = _gate(_lmarena_style_run())
    assert passed is True and report["exact"] is True, reason
    assert report["model"]["kinds"] == {"model_context_limit": 1}
    assert report["evaluation"]["count"] == 0


def test_a_policy_rejection_without_a_context_reason_stays_a_request_error():
    _, (_, reason, report), _ = _gate(_lmarena_style_run(reason=None, kind="model_request_error"))
    assert report["exact"] is True, reason
    assert report["model"]["kinds"] == {"model_request_error": 1}


def test_another_models_context_rejection_stays_evaluation_side():
    _, (_, reason, report), _ = _gate(_tau_style_run(model="simulated_user", merged=False))
    assert report["model"]["count"] == 0, reason
    assert report["evaluation"]["kinds"] == {"non_policy_model_rejection": 1}


def test_a_server_error_with_a_context_reason_stays_the_servers():
    _, (_, reason, report), _ = _gate(_lmarena_style_run(upstream=503, kind="model_request_error"))
    assert report["model"]["count"] == 0, reason
    assert report["evaluation"]["kinds"] == {"model_server_error": 1}
    assert report["evaluation"]["server"] == 1


def test_the_collector_keeps_a_calls_upstream_status_and_local_reason(tmp_path):
    a, rows = setup(tmp_path)
    meta = rows[0]["ng_trajectory"]["model_calls"][0]["response_metadata"]
    meta["upstream_status_code"], meta["local_response_reason"] = 400, "context_length_exceeded"
    save_rows(a, rows)
    collected = collect(a)
    [call] = collected.results[0]["ng_trial_observations"]["model_calls"]
    kept = call["response_metadata"]
    assert (kept["upstream_status_code"], kept["local_response_reason"]) == (400, "context_length_exceeded")
    assert "synthetic" not in json.dumps(kept)


@pytest.mark.parametrize("field", ["error_category", "local_response_reason"])
def test_an_output_budget_the_window_cannot_hold_is_an_evaluation_side_setup_failure(field):
    """The request's output budget alone exceeds the context window: the deployment cannot serve it."""
    collected = _lmarena_style_run(reason=None)
    [call] = collected.results[0]["ng_trial_observations"]["model_calls"]
    call["response_metadata"][field] = "invalid_output_token_parameter"
    _, (passed, reason, report), _ = _gate(collected)
    assert report["exact"] is True, reason
    assert report["model"]["count"] == 0
    assert report["evaluation"]["kinds"] == {"invalid_output_budget": 1}
    assert report["evaluation"]["server"] == 0


def test_a_saved_upstream_context_rejection_is_typed_when_the_run_is_read_again(tmp_path):
    """The saved shape of an upstream rejection: its event names no kind and records the failure reason and status;
    the call keeps the upstream status and the local response reason, and no error category."""
    a, rows = setup(tmp_path)
    meta = rows[0]["ng_trajectory"]["model_calls"][0]["response_metadata"]
    [operation] = meta["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", "rejected-1"
    event = {
        "event_id": "rejected-1",
        "boundary": "model",
        "operation": "model_request",
        "operation_id": operation["operation_id"],
        "attempt_id": operation["final_attempt_id"],
        "cause_id": None,
        "failure_kind": None,
        "failure_reason": "HTTP 400 from the model server",
        "http_status": 400,
    }
    meta["failure_report"]["events"] = {"rejected-1": event}
    meta |= {"status_code": None, "response_status": None, "error_category": None}
    meta |= {"upstream_status_code": 400, "local_response_reason": "context_length_exceeded"}
    rows[0]["_ng_failure_events"] = {"rejected-1": dict(event)}
    save_rows(a, rows)
    collected = collect(a)
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert final_zero_trials(decision, collected) == {(0, 0): "model_context_limit"}


def _parameter_value_refusals(count: int):
    """The policy's only call refused with HTTP 400 because a request parameter carries a value the deployment does not
    support (vllm_model records ``invalid_request_parameter``), its event typed ``model_request_error``, the row scored
    zero: the shape of a baseline whose every policy call was refused for its reasoning effort."""
    collected = _run(100)
    for task, row in enumerate(collected.results[:count]):
        row["reward"] = 0.0
        [call] = row["ng_trial_observations"]["model_calls"]
        event = _reject(call, status=400, event_id=f"param-{task}")
        call["response_metadata"]["error_category"] = "invalid_request_parameter"
        row["_ng_failure_events"] = {event["event_id"]: dict(event)}
    return collected


@pytest.mark.parametrize(("refused", "status"), [(1, "PASS"), (2, "FAIL")])
def test_a_policy_call_refused_for_an_unsupported_parameter_value_is_an_evaluation_side_setup_failure(refused, status):
    """Not the model's request error: the trial counts toward evaluation health, which fails at the 2% limit."""
    _, (passed, reason, report), _ = _gate(_parameter_value_refusals(refused))
    assert report["status"] == status and passed is (status == "PASS"), reason
    # A failure at the limit is decided on the known faults before coverage is proven, so only a pass is exact.
    assert report["exact"] is (status == "PASS"), reason
    assert report["model"]["count"] == 0
    assert report["evaluation"]["kinds"] == {"invalid_request_parameter": refused}
    assert report["evaluation"]["server"] == 0


def _tau3_style_saved_row(a, rows, invocations):
    """A saved tool-transcript trial whose only policy call was rejected at the context limit (HTTP 400, no later
    call) and scored zero; the call kept its error category and a failure event of a model request error."""
    trajectory = rows[0]["ng_trajectory"]
    meta = trajectory["model_calls"][0]["response_metadata"]
    [operation] = meta["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", "rejected-1"
    event = {
        "event_id": "rejected-1",
        "boundary": "model",
        "operation": "model_request",
        "operation_id": operation["operation_id"],
        "attempt_id": operation["final_attempt_id"],
        "cause_id": None,
        "failure_kind": "model_request_error",
        "http_status": 400,
    }
    meta["failure_report"]["events"] = {"rejected-1": event}
    meta |= {"status_code": 400, "response_status": None, "error_category": "context_length_exceeded"}
    trajectory["invocations"], trajectory["turns"] = invocations, []
    rows[0]["ng_policy_model_server"] = "policy"
    rows[0]["_ng_failure_events"] = {"rejected-1": dict(event)}
    save_rows(a, rows)
    return collect(a)


@pytest.mark.parametrize(
    "invocations",
    [
        [],
        [{"invocation_id": "agent", "status": "unknown", "in_session": None, "model_calls": []}],
    ],
    ids=["no-invocation", "unreferenced-invocation"],
)
def test_a_saved_tool_transcript_trial_that_ended_on_the_context_limit_is_typed_when_read(tmp_path, invocations):
    """The agent recorded a tool exchange for none of its calls, or for some: the policy's last captured call ended
    the rollout either way."""
    a, rows = setup(tmp_path)
    collected = _tau3_style_saved_row(a, rows, invocations)
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert final_zero_trials(decision, collected) == {(0, 0): "model_context_limit"}


def _local_context_rejection(a, rows, invocations, *, events=1, error_category=None):
    """A saved trial whose last policy call the model server answered locally (status 200) after its upstream
    rejected the request with 400 for the context window: the failure report holds model request errors only."""
    trajectory = rows[0]["ng_trajectory"]
    meta = trajectory["model_calls"][0]["response_metadata"]
    [operation] = meta["model_operations"]
    operation["outcome"], operation["failure_id"] = "failed", f"rejected-{events}"
    saved = {}
    for number in range(1, events + 1):
        saved[f"rejected-{number}"] = {
            "event_id": f"rejected-{number}",
            "boundary": "model",
            "operation": "model_request",
            "operation_id": operation["operation_id"],
            "attempt_id": f"{operation['final_attempt_id']}-{number}",
            "cause_id": None,
            "failure_kind": "model_request_error",
            "http_status": 400,
        }
    meta["failure_report"]["events"] = saved
    meta |= {
        "status_code": 200,
        "response_status": None,
        "error_category": error_category,
        "finish_reason": "length",
        "response_source": "local",
        "upstream_attempted": True,
        "upstream_status_code": 400,
        "local_response_reason": "context_length_exceeded",
    }
    trajectory["invocations"], trajectory["turns"] = invocations, []
    rows[0]["ng_policy_model_server"] = "policy"
    save_rows(a, rows)
    return collect(a)


_TOOL_INVOCATION = {"invocation_id": "agent", "status": "unknown", "in_session": None, "model_calls": []}


@pytest.mark.parametrize(
    "invocations,events",
    [([], 1), ([_TOOL_INVOCATION], 1), ([], 2)],
    ids=["no-invocation", "one-unreferenced-invocation", "no-invocation-two-request-errors"],
)
@pytest.mark.parametrize("error_category", [None, "context_length_exceeded"])
def test_a_local_reply_to_an_upstream_context_rejection_is_typed_when_read(
    tmp_path, invocations, events, error_category
):
    a, rows = setup(tmp_path)
    collected = _local_context_rejection(a, rows, invocations, events=events, error_category=error_category)
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert final_zero_trials(decision, collected) == {(0, 0): "model_context_limit"}
