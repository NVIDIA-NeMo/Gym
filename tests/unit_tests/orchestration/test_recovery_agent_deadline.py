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
"""An agent that ran out of its own deadline and kept its answer: the model's final result, never run again.

The rows are injected controls in the shape a staged environment persists for an agent wall-time stop: an
``agent_timeout`` failures-sidecar row that keeps the agent's response, a placeholder zero and a mask, with no
terminal stamp, no counted-zero marker and no recorded grade.
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
    decide_recovery,
    failure_facts,
    gate_summary_from_collected,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import (
    OK_BUDGET,
    ident,
    infra_failure,
    inventory,
    model_final,
    scored,
)
from tests.unit_tests.orchestration.test_recovery_saved_generation import persist_producer_result, response


def answer(task: int) -> dict:
    return {
        "id": f"agent-{task}-0",
        "object": "response",
        "error": None,
        "metadata": {"stop_reason": "wall_time"},
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "draft"}]}],
    }


def deadline(task: int, *, kept: bool = True, kind: str = "agent_timeout") -> dict:
    row = ident(task, _ng_failure_class=kind, failure_kind=kind, reward=0.0, mask_sample=True)
    if kept:
        row["response"] = answer(task)
    return row


def test_the_shared_predicate_needs_the_kind_and_a_kept_answer() -> None:
    from nemo_gym.rollouts.rollout_recovery import is_answered_agent_deadline

    assert is_answered_agent_deadline(deadline(0))
    assert not is_answered_agent_deadline(deadline(0, kept=False))
    assert not is_answered_agent_deadline(deadline(0, kind="agent_limit"))
    # A selected event decides the kind; a selected event that is missing names none.
    event = {"failure_kind": "model_transport_error"}
    assert not is_answered_agent_deadline(deadline(0) | {"_ng_failure_id": "e", "_ng_failure_events": {"e": event}})
    assert not is_answered_agent_deadline(deadline(0) | {"_ng_failure_id": "e"})
    assert not is_answered_agent_deadline(deadline(0) | {"_ng_failure_id": {"id": "e"}, "_ng_failure_events": {}})


def test_an_answered_agent_deadline_is_the_models_result_and_does_not_stop_recovery() -> None:
    # One scored, one answered deadline, one never started, one retryable infrastructure failure.
    d = decide_recovery(
        materialized=inventory(4),
        results=[scored(0)],
        failures=[deadline(1), infra_failure(3, attempt=0)],
        budget=OK_BUDGET,
    )
    assert d.identities[IdentityState.MODEL_FINAL] == [(1, 0)]
    assert d.evidence[1].failure_kind == "agent_timeout"
    assert IdentityState.UNKNOWN not in d.counts
    assert d.action == RecoveryAction.RESUBMIT


def test_answered_agent_deadlines_alone_leave_nothing_to_run() -> None:
    d = decide_recovery(materialized=inventory(2), results=[scored(0)], failures=[deadline(1)], budget=OK_BUDGET)
    assert d.action == RecoveryAction.NOOP and d.counts == {"scored": 1, "model_final": 1}


def test_an_agent_deadline_without_an_answer_keeps_its_retry() -> None:
    d = decide_recovery(materialized=inventory(1), results=[], failures=[deadline(0, kept=False)], budget=OK_BUDGET)
    assert d.counts == {IdentityState.RETRYABLE: 1} and d.action == RecoveryAction.RESUBMIT


def test_another_kind_that_kept_an_answer_is_still_refused() -> None:
    d = decide_recovery(
        materialized=inventory(2), results=[], failures=[deadline(0, kind="agent_limit")], budget=OK_BUDGET
    )
    assert d.counts.get(IdentityState.UNKNOWN) == 1 and d.action == RecoveryAction.STOP
    assert d.evidence[0].reason == "the row kept an answer that a rerun would replace"


def test_the_cluster_projection_of_a_legacy_run_keeps_the_answered_deadline_final(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(3)))
    (artifacts / "rollouts.jsonl").write_text(json.dumps(scored(0)) + "\n")
    (artifacts / "rollouts_failures.jsonl").write_text(json.dumps(deadline(1)) + "\n")
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    assert "draft" not in raw.stdout
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities[IdentityState.MODEL_FINAL] == [(1, 0)]
    assert d.identities[IdentityState.MISSING] == [(2, 0)]
    assert IdentityState.UNKNOWN not in d.counts
    # No coverage report: the metric covered the one scored result, not the kept answer of the agent deadline.
    assert failure_facts(d, collected)["metric_coverage"] == {"trials": 1, "expected": 3, "source": "saved_rows"}


def test_metric_coverage_counts_a_producer_marked_zero_and_leaves_out_an_answered_deadline() -> None:
    collected = CollectedEvidence(
        materialized=inventory(4), results=[scored(0)], failures=[model_final(1), deadline(2), deadline(3, kept=False)]
    )
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities[IdentityState.MODEL_FINAL] == [(1, 0), (2, 0)]
    assert failure_facts(d, collected)["metric_coverage"] == {"trials": 2, "expected": 4, "source": "saved_rows"}
    # Collection's own count, when its coverage report is for this inventory, is the source.
    report = collected.model_copy(update={"coverage_scored": (3, 4)})
    assert failure_facts(d, report)["metric_coverage"] == {"trials": 3, "expected": 4, "source": "coverage_report"}
    stale = collected.model_copy(update={"coverage_scored": (3, 5)})
    assert failure_facts(d, stale)["metric_coverage"]["source"] == "saved_rows"


def test_a_benchmark_declared_metric_coverage_is_preferred_and_named(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(3)))
    (artifacts / "rollouts.jsonl").write_text(json.dumps(scored(0)) + "\n")
    (artifacts / "rollouts_failures.jsonl").write_text(json.dumps(deadline(1)) + "\n")
    (artifacts / "rollouts_coverage.json").write_text(json.dumps({"scored": 1, "expected": 3}))

    def gate(metrics):
        aggregate = [{"agent_ref": {"name": "a"}, "agent_metrics": {"mean/reward": 1.0, **metrics}, "key_metrics": {}}]
        (artifacts / "rollouts_aggregate_metrics.json").write_text(json.dumps(aggregate))
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
        return gate_summary_from_collected(d, collected, max_unhealthy_fraction=None)

    task = gate({"metric_coverage/trials": 212, "metric_coverage/expected": 220})
    assert task["failures"]["metric_coverage"] == {
        "trials": 212,
        "expected": 220,
        "source": "benchmark",
        "name": "metric",
    }
    assert task["health_verdict"].endswith("; metric over 212 of 220 trials")
    named = gate({"metric_coverage/trials": 212, "metric_coverage/expected": 220, "metric_coverage/name": "ELO"})
    assert named["health_verdict"].endswith("; ELO over 212 of 220 trials")
    # A malformed declaration is not one: collection's report is the source.
    for bad in ({"metric_coverage/trials": 221, "metric_coverage/expected": 220}, {"metric_coverage/trials": 1.5}):
        assert gate(bad)["failures"]["metric_coverage"] == {"trials": 1, "expected": 3, "source": "coverage_report"}


async def test_collection_keeps_an_answered_agent_deadline_and_a_resume_does_not_dispatch_it(
    tmp_path, monkeypatch
) -> None:
    result = {
        "_ng_failure_class": "agent_timeout",
        "failure_kind": "agent_timeout",
        "reward": 0.0,
        "mask_sample": True,
        "response": response() | {"metadata": {"stop_reason": "wall_time"}},
    }
    # The helper resumes collection when the Store selects nothing and asserts that no request was sent.
    decision, store, _ = await persist_producer_result(tmp_path, monkeypatch, result)
    assert store.pending() == []
    assert decision.counts == {IdentityState.MODEL_FINAL: 1}
    assert decision.action == RecoveryAction.NOOP
    # The Store settles it where recovery does: a final model result, so the run is complete.
    coverage = store.coverage()
    assert coverage["final_zeros"] == 1 and coverage["retryable"] == 0
    assert coverage["complete"] is True and coverage["completion_fraction"] == 1.0
    # Collection's report, written when the resume ended, says the metric covered none of the one expected trial.
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    assert collected.coverage_scored == (0, 1)
    task = gate_summary_from_collected(decision, collected, max_unhealthy_fraction=None)
    assert task["failures"]["metric_coverage"] == {"trials": 0, "expected": 1, "source": "coverage_report"}
    assert task["health_verdict"].endswith("; metric over 0 of 1 trials")


@pytest.mark.parametrize("marked", [False, True])
def test_an_answered_deadline_is_a_scored_sample_only_when_the_metric_scores_it(marked) -> None:
    from nemo_gym.orchestration.recovery import gate_task_summary
    from nemo_gym.rollouts.rollout_collection import _counted_failure_rows

    timeout = deadline(1) | ({"_ng_failure_counts_as_zero": True} if marked else {})
    collected = CollectedEvidence(materialized=inventory(2), results=[scored(0)], failures=[timeout])
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    # Settled either way: the kept answer is the model's result and no resume runs it again.
    assert d.action == RecoveryAction.NOOP and d.identities[IdentityState.MODEL_FINAL] == [(1, 0)]
    # The metric input: collection's default policy counts the row only when its producer marked it a zero.
    metric_input = _counted_failure_rows([timeout], [])
    assert len(metric_input) == int(marked)
    task = gate_task_summary(
        d, score=1.0, summary_valid=True, summary_reasons=[], health_passed=True, health_reasons=[]
    )
    coverage = failure_facts(d, collected)["metric_coverage"]
    assert task["scored_samples"] == coverage["trials"] == 1 + len(metric_input)
    if not marked:
        assert any("not scored by the metric" in warning for warning in task["warnings"]), task["warnings"]
