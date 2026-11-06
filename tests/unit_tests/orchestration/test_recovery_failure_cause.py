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
"""A failure whose producer recorded no kind, or an unregistered one, is typed by its recorded cause.

The rows are injected controls in the shapes producers persist: an environment's episode deadline with no kind, and
an unregistered producer class whose saved failure events are failed model requests, with the model call capture
that says whose call each one was. The remote collector reads them from the run directory as the gate does.
"""

import json
import subprocess
import sys

import pytest

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    IdentityState,
    RecoveryAction,
    decide_from_collected,
    final_zero_trials,
    select_managed_evidence,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore


POLICY = {"type": "responses_api_models", "name": "policy_model"}
SIMULATOR = {"type": "responses_api_models", "name": "user_simulator_model"}


def _event(event_id: str, status: int, kind: str = "model_request_error") -> dict:
    return {
        "event_id": event_id,
        "boundary": "model",
        "failure_reason": f"HTTP {status} from the model server",
        "failure_kind": kind,
        "operation": "model_request",
        "operation_id": f"op-{event_id}",
        "attempt_id": f"op-{event_id}-1",
        "http_status": status,
    }


def _call(model_ref: dict | None, events: dict, **extra) -> dict:
    return {
        "model_ref": model_ref,
        "status_code": 400,
        "failure_report": {"failure_id": None, "events": events},
    } | extra


def _producer_failure(task: int, events: dict, calls: list, *, kept: bool = False) -> dict:
    """An episode that a producer reported as failed under its own, unregistered class."""
    row = {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "example_agent_error",
        "_ng_failure_message": "Agent reported a no-result failure",
        "_ng_failure_max_attempts": 3,
        "_ng_failure_record": {
            "source": "environment",
            "delivery": "delivered",
            "failure": {"failure_reason": "Agent reported a no-result failure", "failure_kind": "example_agent_error"},
        },
        "_ng_failure_events": events,
        "ng_model_call_capture": {"calls": calls},
    }
    if kept:
        row["_ng_failure_response"] = {"id": f"failure-{task}", "object": "response", "output": [{"type": "message"}]}
    return row


def _episode_deadline(task: int) -> dict:
    """An episode the environment's own deadline ended while the policy call was still outstanding; no kind."""
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_message": "Episode timed out",
        "_ng_failure_terminal": False,
        "_ng_failure_max_attempts": 3,
        "_ng_failure_record": {
            "source": "environment",
            "delivery": "delivered",
            "failure": {"failure_reason": "Episode timed out", "terminal": False},
        },
        "ng_model_call_capture": {
            "calls": [{"model_ref": POLICY, "status_code": None, "error_category": "cancelled"}]
        },
    }


def _decide(job, failures: list[dict], *, n: int):
    artifacts = job / "artifacts"
    artifacts.mkdir(parents=True)
    inventory = [{"_ng_task_index": t, "_ng_rollout_index": 0} for t in range(n)]
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory))
    (artifacts / "rollouts.jsonl").write_text(
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0}) + "\n"
    )
    (artifacts / "rollouts_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in failures))
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    assert "Agent reported" not in raw.stdout  # Only facts of the cause leave the run host.
    return decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)


def test_an_episode_deadline_without_a_kind_is_retried_under_the_cap(tmp_path):
    d = _decide(tmp_path, [_episode_deadline(1), _episode_deadline(2)], n=4)
    assert d.counts == {"scored": 1, "retryable": 2, "missing": 1}
    assert {e.failure_kind for e in d.evidence if e.state == IdentityState.RETRYABLE} == {"environment_episode_failed"}
    assert d.action == RecoveryAction.RESUBMIT


def _two_requests(owner: dict, *, status: int = 400, **extra) -> tuple[dict, list]:
    """Two refused model requests of one caller under two saved kinds, so the events name no one kind."""
    events = {"e1": _event("e1", status), "e2": _event("e2", status, "model_policy_response_timeout")}
    return events, [_call(owner, {"e1": events["e1"]}, **extra), _call(owner, {"e2": events["e2"]}, **extra)]


def test_a_harness_model_call_refused_is_evaluation_side_and_retried(tmp_path):
    events, calls = _two_requests(SIMULATOR)
    d = _decide(tmp_path, [_producer_failure(1, events, calls)], n=2)
    assert d.counts == {"scored": 1, "retryable": 1}
    assert d.evidence[1].failure_kind == "environment_episode_failed"
    assert d.action == RecoveryAction.RESUBMIT


def test_unselected_model_failures_of_a_row_that_kept_a_response_are_not_its_cause(tmp_path):
    # A user simulator's refused requests, beside the producer's no-result response: attempts, not the cause.
    events = {"e1": _event("e1", 400), "e2": _event("e2", 400)}
    calls = [_call(SIMULATOR, {"e1": events["e1"]}), _call(SIMULATOR, {"e2": events["e2"]})]
    d = _decide(tmp_path, [_producer_failure(1, events, calls, kept=True)], n=2)
    assert d.counts == {"scored": 1, "unknown": 1}
    assert d.evidence[1].reason == (
        "unregistered kind example_agent_error: it kept a response, so its saved model failures are attempts, not "
        "its cause"
    )


@pytest.mark.parametrize(
    "extra,kind", [({"error_category": "context_length_exceeded"}, "model_context_limit"), ({}, "model_request_error")]
)
def test_the_policys_refused_request_is_the_models_final_result(tmp_path, extra, kind):
    events, calls = _two_requests(POLICY, **extra)
    row = _producer_failure(1, events, calls) | {"_ng_failure_terminal": True}
    d = _decide(tmp_path, [row], n=2)
    assert d.counts == {"scored": 1, "model_final": 1}
    assert d.evidence[1].failure_kind == kind
    assert d.action == RecoveryAction.NOOP


def test_the_policys_refused_request_that_a_resume_would_rerun_stays_unknown(tmp_path):
    events, calls = _two_requests(POLICY)
    d = _decide(tmp_path, [_producer_failure(1, events, calls)], n=2)
    assert d.counts == {"scored": 1, "unknown": 1}
    assert d.evidence[1].reason == (
        "the policy's request failed (model_request_error), a final model result, but a resume would run it again"
    )


@pytest.mark.parametrize(
    "second,second_kind,kind",
    [
        (503, "model_policy_response_timeout", "model_request_error"),
        (None, "model_transport_error", "model_transport_error"),
    ],
)
def test_a_server_side_or_transport_model_failure_is_infrastructure(tmp_path, second, second_kind, kind):
    events = {"e1": _event("e1", 503), "e2": _event("e2", second, second_kind)}
    calls = [_call(POLICY, {"e1": events["e1"]}), _call(POLICY, {"e2": events["e2"]})]
    d = _decide(tmp_path, [_producer_failure(1, events, calls)], n=2)
    assert d.counts == {"scored": 1, "retryable": 1}
    assert d.evidence[1].failure_kind == kind


def test_a_refused_request_whose_caller_is_not_recorded_stays_unknown_with_a_reason(tmp_path):
    events, _ = _two_requests(POLICY)
    d = _decide(tmp_path, [_producer_failure(1, events, [])], n=2)
    assert d.counts == {"scored": 1, "unknown": 1}
    assert d.evidence[1].reason == (
        "unregistered kind example_agent_error: the model call that failed is not recorded, so whose call it was is "
        "unknown"
    )


def test_a_store_resume_and_the_decision_agree_on_an_episode_deadline(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(2)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        inputs = json.loads(json.dumps(rows))
        return inputs, RunManifest.create(source, inputs, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SLURM_JOB_ID", "100")
        with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
            first = dict(rows[0])
            store.record_dispatch(first)
            store.record_outcome(first | {"reward": 1.0})
            second = dict(rows[1])
            store.record_dispatch(second)
            failure = _episode_deadline(1)
            failure["_ng_failure_record"] |= {
                "schema_version": 1,
                "run_id": store.manifest.run_id,
                "episode_id": {"rollout_id": "1-0", "attempt": 0},
            }
            store.record_outcome(second | failure)
    (tmp_path / ".gym_job_chain").write_text("100\n")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), tmp_path)
    assert collected.managed_reader_error is None
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    # The Store runs it again under its cap, and the decision now says so instead of stopping on an unowned row.
    assert [row["_ng_task_index"] for row in RolloutStore.read(artifacts / "rollouts.jsonl").pending()] == [1]
    assert d.counts == {"scored": 1, "retryable": 1} and d.action == RecoveryAction.RESUBMIT


def _scored_zero(task: int, calls: list[dict]) -> dict:
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": 0,
        "reward": 0.0,
        "ng_policy_model_server": "policy_model",
        "ng_trial_observations": {
            "invocations": [{"invocation_id": "agent", "model_calls": []}],
            "model_calls": [{"model_call_id": f"c{i}", "response_metadata": meta} for i, meta in enumerate(calls)],
        },
    }


def _metadata(model_ref: dict, *, failed: bool, status: int = 400, **extra) -> dict:
    operation = {
        "operation_id": "op",
        "outcome": "failed" if failed else "completed",
        "failure_id": "e" if failed else None,
    }
    events = {"e": _event("e", status) | {"operation_id": "op"}} if failed else {}
    return {"model_ref": model_ref, "model_operations": [operation], "failure_report": {"events": events}} | extra


def test_a_scored_zero_that_ended_on_the_policys_context_limit_is_named_so():
    from nemo_gym.orchestration.recovery import decide_recovery
    from tests.unit_tests.orchestration.test_eval_recovery_gate import OK_BUDGET, inventory

    limit = _metadata(POLICY, failed=True, error_category="context_length_exceeded")
    rows = [
        _scored_zero(0, [_metadata(POLICY, failed=False), limit]),
        _scored_zero(1, [_metadata(POLICY, failed=True, status=400)]),
        # A server error the policy's retries recovered from: the rollout ended normally and its zero is its result.
        _scored_zero(2, [_metadata(POLICY, failed=True, status=503), _metadata(POLICY, failed=False)]),
        # A harness call ended the rollout: its refused request is evaluation-side.
        _scored_zero(3, [_metadata(POLICY, failed=False), _metadata(SIMULATOR, failed=True)]),
    ]
    decision = decide_recovery(materialized=inventory(4), results=rows, failures=[], budget=OK_BUDGET)
    collected = CollectedEvidence(materialized=inventory(4), results=rows, failures=[])
    assert final_zero_trials(decision, collected) == {
        (0, 0): "model_context_limit",
        (1, 0): "model_request_error",
        (3, 0): "non_policy_model_rejection",
    }
