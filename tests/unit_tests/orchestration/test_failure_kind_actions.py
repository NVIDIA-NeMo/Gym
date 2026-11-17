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
"""Failure kinds that had no recovery action: a fresh retry when no answer was kept, else a stop that says why."""

import json
import subprocess
import sys

import pytest

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    STORE_DOES_NOT_REPLAY,
    CollectedEvidence,
    ManagedRecoveryFact,
    RecoveryBudget,
    RecoveryOutcomePolicyConfig,
    decide_from_collected,
    decide_recovery,
    select_managed_evidence,
)
from nemo_gym.rollouts.rollout_recovery import RunManifest
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.unit_tests.rollouts.test_possibly_delivered_replay import _ANSWER, _route_failure_row


_BUDGET = RecoveryBudget(max_infra_retries=2, infra_retries_used=0, walltime_used_s=0.0, live_job=False)
_KEPT = "the row kept an answer that a rerun would replace"


def _decide(*failures, scored=1, budget=_BUDGET, **kwargs):
    """One scored row, then one standing sidecar row per failure, as saved without a rollout store."""
    total = scored + len(failures)
    materialized = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(total)]
    results = [{"_ng_task_index": i, "_ng_rollout_index": 0, "reward": 1.0} for i in range(scored)]
    rows = [{"_ng_task_index": scored + i, "_ng_rollout_index": 0, "_line": i} | f for i, f in enumerate(failures)]
    return decide_recovery(materialized=materialized, results=results, failures=rows, budget=budget, **kwargs)


def _harbor_failed(**extra):
    """The Harbor agent's failed trial: a producer label as kind, an empty failed response, no event report."""
    return {
        "_ng_failure_class": "harbor_failed",
        "failure_kind": "harbor_failed",
        "_ng_failure_message": "Agent reported a no-result failure",
        "_ng_failure_record": {"source": "environment", "delivery": "delivered", "failure": {"stage": None}},
        "_ng_failure_response": {"object": "response", "id": "harbor-error", "status": "failed", "output": []},
        **extra,
    }


def _sandbox_create_rejected(**extra):
    """A terminal-bench legacy row: the sandbox create call refused its credentials before the agent ran."""
    return {
        "_ng_failure_class": "infrastructure_error",
        "failure_kind": None,
        "_ng_failure_message": "SandboxApiException",
        "_ng_failure_record": {
            "source": "environment",
            "delivery": "delivered",
            "failure": {"stage": None, "failure_kind": "infrastructure_error"},
        },
        **extra,
    }


def _harness_unconfirmed(**extra):
    events = {
        "stop": {
            "boundary": "harness",
            "failure_kind": "harness_termination_unconfirmed",
            "operation": "harness.terminate",
        }
    }
    return {"_ng_failure_class": "agent_run_error", "_ng_failure_events": events, "_ng_failure_id": "stop", **extra}


@pytest.mark.parametrize(
    "row",
    [
        _route_failure_row(),
        _harbor_failed(),
        _sandbox_create_rejected(),
        # Saved before the collector dropped it: the agent's empty response object is not a kept answer.
        _sandbox_create_rejected(
            _ng_failure_response={"object": "response", "id": "resp-empty", "status": None, "output": []}
        ),
        _harness_unconfirmed(),
        # A legacy row that saved events but selected none: every event names the same kind.
        {
            "_ng_failure_class": "agent_run_error",
            "_ng_failure_events": {"e": {"boundary": "agent", "failure_kind": "agent_route_failed"}},
        },
    ],
    ids=[
        "legacy-run-5xx",
        "harbor-failed",
        "sandbox-credentials",
        "sandbox-credentials-empty-response",
        "harness-unconfirmed",
        "unselected-events",
    ],
)
def test_a_failure_that_kept_no_answer_is_retried_within_its_budget(row):
    decision = _decide(row)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 1, "retryable": 1}


@pytest.mark.parametrize(
    "row,label",
    [
        (_route_failure_row(_ng_failure_response=_ANSWER), f"collection_dispatch_failed: {_KEPT}"),
        (_harness_unconfirmed(response=_ANSWER), f"harness_termination_unconfirmed: {_KEPT}"),
    ],
    ids=["legacy-run-5xx", "harness-unconfirmed"],
)
def test_a_failure_that_kept_an_answer_stops_and_names_the_rule(row, label):
    decision = _decide(row)
    assert decision.action.value == "stop"
    assert decision.reason.endswith(
        f"retryable 0 (resumed), unknown 1 (1 {label}) (reported, not retried), retained 1 (saved rows kept)"
    )


@pytest.mark.parametrize(
    "graded,action,counts",
    [(False, "resubmit", {"scored": 1, "retryable": 1}), (True, "stop", {"scored": 1, "unknown": 1})],
    ids=["no-answer-no-grade-retried", "no-answer-graded-kept"],
)
def test_a_failure_with_no_answer_is_retried_only_when_no_grade_was_recorded(graded, action, counts):
    decision = _decide(_harness_unconfirmed(evaluation_completed=graded))
    assert decision.action.value == action, decision.reason
    assert decision.counts == counts
    if graded:
        label = "harness_termination_unconfirmed: the verifier recorded a grade that a rerun would replace"
        assert decision.reason.endswith(
            f"retryable 0 (resumed), unknown 1 (1 {label}) (reported, not retried), retained 1 (saved rows kept)"
        )


_EMPTY_PARTIAL = {"object": "response", "id": "resp-partial", "status": None, "output": []}


@pytest.mark.parametrize(
    "row,action,label",
    [
        # 1. A completed grade: the verifier graded the attempt, so it is a graded result and is kept.
        (
            _harness_unconfirmed(evaluation_completed=True),
            "stop",
            "harness_termination_unconfirmed: the verifier recorded a grade that a rerun would replace",
        ),
        # 2. A partial native response from the cleanup, even without output items: the agent ran; it is kept.
        (
            _harness_unconfirmed(_ng_failure_partial_response=_EMPTY_PARTIAL),
            "stop",
            f"harness_termination_unconfirmed: {_KEPT}",
        ),
        # 3. A placeholder zero (no completed evaluation, masked) with no kept answer is retried within the cap.
        (_harness_unconfirmed(reward=0.0, evaluation_completed=False, mask_sample=True), "resubmit", None),
        (_sandbox_create_rejected(reward=0.0, evaluation_completed=False, mask_sample=True), "resubmit", None),
    ],
    ids=[
        "completed-grade-kept",
        "partial-native-response-kept",
        "placeholder-zero-retried",
        "placeholder-zero-sandbox-retried",
    ],
)
def test_the_three_protection_cases(row, action, label):
    decision = _decide(row)
    assert decision.action.value == action, decision.reason
    if label is None:
        assert decision.counts == {"scored": 1, "retryable": 1}
    else:
        assert decision.reason.endswith(
            f"retryable 0 (resumed), unknown 1 (1 {label}) (reported, not retried), retained 1 (saved rows kept)"
        )


_INFRA_RETRY = {
    "collection_dispatch_failed": _route_failure_row(),
    "harness_termination_unconfirmed": _harness_unconfirmed(),
}


@pytest.mark.parametrize("kind", _INFRA_RETRY)
def test_an_infrastructure_kind_retries_under_a_failing_default_and_spends_the_infra_retry_cap(kind):
    # A partial override with a failing default: only the kind's own entry can still retry it.
    policy = RecoveryOutcomePolicyConfig.model_validate(
        {"default_action": "fail", "by_failure_kind": {"agent_timeout": "fail"}}
    )
    under = _decide(_INFRA_RETRY[kind], outcome_policy=policy)
    assert under.action.value == "resubmit", under.reason
    assert under.counts == {"scored": 1, "retryable": 1}
    spent = _decide(
        _INFRA_RETRY[kind], outcome_policy=policy, budget=_BUDGET.model_copy(update={"infra_retries_used": 2})
    )
    assert (spent.action.value, spent.reason) == ("stop", "the stored infra-retry budget is used (2 of 2)")


@pytest.mark.parametrize("kind", _INFRA_RETRY)
def test_a_retry_the_store_does_not_plan_stops_and_names_the_kind(kind):
    # The Store reader of a run pinned at an older revision may refuse a replay this revision's policy allows.
    fact = ManagedRecoveryFact(identity=(1, 0), attempts=1, max_attempts=3, pending=False, reserved=False)
    decision = _decide(_INFRA_RETRY[kind], managed_selection=[fact])
    assert decision.action.value == "stop"
    assert decision.reason == (
        f"1 failure(s) ({kind}) have no registered kind, no supported action or no rerun in the Store, so nobody owns "
        f"them; retryable 0 (resumed), unknown 1 (1 {kind}: {STORE_DOES_NOT_REPLAY}) (reported, not retried), "
        "retained 1 (saved rows kept)"
    )


def test_a_failure_with_no_kind_anywhere_stops_and_says_so():
    decision = _decide({"_ng_failure_class": "unowned_class"})
    assert decision.action.value == "stop"
    assert "unknown 1 (1 no failure kind)" in decision.reason


def test_the_store_and_the_plan_agree_on_a_legacy_run_5xx_route_failure(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    rows = [{"_ng_task_index": i, "_ng_rollout_index": 0} for i in range(2)]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare():
        return rows, RunManifest.create(source, rows, {}, {}).model_copy(update={"max_rollout_attempts": 3})

    with RolloutStore.start_or_resume(artifacts / "rollouts.jsonl", prepare, resume=False) as store:
        scored = dict(rows[0])
        store.allocate_attempt(scored)
        store.record_outcome(scored | {"reward": 1.0})
        failed = dict(rows[1])
        store.allocate_attempt(failed)
        failure = _route_failure_row()
        failure["_ng_failure_record"] = {
            "schema_version": 1,
            "episode_id": {"rollout_id": "1-0", "attempt": 0},
            "run_id": store.manifest.run_id,
            "source": "collector",
            "delivery": "delivered",
            "failure": {
                "failure_reason": "500 Internal Server Error",
                "terminal": False,
                "failure_kind": "agent_run_error",
            },
            "http_status": 500,
            "exception_type": "ClientResponseError",
        }
        store.record_outcome(failed | failure)
        assert [row["_ng_task_index"] for row in store.pending()] == [1]
    (tmp_path / ".gym_job_chain").write_text("100\n")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = select_managed_evidence(CollectedEvidence.model_validate_json(raw.stdout), tmp_path)
    decision = decide_from_collected(collected, max_infra_retries=2, max_walltime_s=None, live_job=False)
    assert decision.action.value == "resubmit", decision.reason
    assert decision.counts == {"scored": 1, "retryable": 1}


@pytest.mark.parametrize(
    "walltime_used_s,named",
    [(2727.0, True), (2000.0, False)],
    ids=["walltime-also-spent", "walltime-left"],
)
def test_a_spent_infra_retry_budget_also_names_a_spent_walltime(walltime_used_s, named):
    budget = _BUDGET.model_copy(
        update={
            "infra_retries_used": 2,
            "walltime_capped": True,
            "max_walltime_s": 2745.0,
            "walltime_used_s": walltime_used_s,
        }
    )
    spent = _decide(_INFRA_RETRY["collection_dispatch_failed"], budget=budget)

    assert spent.action.value == "stop" and spent.reason.startswith("the stored infra-retry budget is used (2 of 2)")
    assert ("the stored walltime budget is used (2727 of 2745 s" in spent.reason) is named
    assert ("raise both in one call" in spent.reason) is named
