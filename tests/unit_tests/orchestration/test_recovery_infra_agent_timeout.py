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
"""An agent timeout whose window held a recorded server-side fault is the evaluation's failure, not the model's.

The rows are mocked in the shapes collection saves: a failures-sidecar ``agent_timeout`` row (GDPVal-style) or a
Harbor graded zero in the results, each with its captured model calls under ``ng_model_call_capture.calls``.
"""

import json
import subprocess
import sys

import pytest

from nemo_gym.core.failure_kinds import AGENT_TIMEOUT_INFRASTRUCTURE, is_gate_derived
from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    IdentityState,
    RecoveryAction,
    decide_from_collected,
    decide_recovery,
    failure_facts,
    fault_report,
    trial_faults,
)
from nemo_gym.rollouts.rollout_recovery import (
    is_answered_agent_deadline,
    is_infrastructure_agent_timeout,
    is_server_fault_call,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import OK_BUDGET, ident, inventory
from tests.unit_tests.orchestration.test_recovery_saved_generation import persist_producer_result, response


WINDOW_S = 600.0


def call(start: float, seconds: float, status: int | None = 200, **extra) -> dict:
    """One captured model call as the capture record saves it (payloads already stripped)."""
    return {
        "model_call_id": f"call-{start:g}",
        "started_at": 1000.0 + start,
        "completed_at": 1000.0 + start + seconds,
        "status_code": status,
        "latency_total_ms": seconds * 1000.0,
    } | extra


def answer() -> dict:
    message = {"id": "msg_draft", "type": "message", "role": "assistant", "status": "completed"}
    return response([message | {"content": [{"type": "output_text", "text": "draft", "annotations": []}]}])


def timeout_row(task: int, calls: list[dict], *, kept: bool = False, **extra) -> dict:
    row = ident(
        task,
        _ng_failure_class="agent_timeout",
        failure_kind="agent_timeout",
        _ng_failure_max_attempts=3,
        _ng_elapsed_s=WINDOW_S,
        ng_model_call_capture={"calls": calls},
        **extra,
    )
    if kept:
        row["response"] = answer()
    return row


def harbor_zero(task: int, calls: list[dict]) -> dict:
    """Harbor's graded outcome of an agent timeout: a result row in the pass-rate denominator."""
    return ident(
        task,
        reward=0.0,
        failure_kind="agent_timeout",
        evaluation_completed=True,
        mask_sample=False,
        response=answer(),
        ng_model_call_capture={"calls": calls},
    )


CLEAN = [call(0, 100), call(100, 400)]
LAST_503 = [call(0, 100), call(100, 30, 503, error_category="upstream_error")]
UPSTREAM_502 = [call(0, 100), call(100, 30, 200, upstream_status_code=502)]
CONNECTION = [call(0, 100), call(100, 30, None, error_category="connection")]
PEER_DROP = [
    call(0, 100),
    call(
        100,
        30,
        None,
        error_category="exception",
        failure_report={"events": {"e": {"failure_kind": "transport_peer_drop", "http_status": None}}},
    ),
]
# A 5xx early in the window that a retry recovered: one second of a ten-minute window.
EARLY_RECOVERED = [call(0, 1, 503, error_category="upstream_error"), call(1, 500)]
# Faulted calls that took 200 of 600 seconds, though the agent's last call succeeded.
LONG_FAULTS = [call(0, 200, 500, error_category="upstream_error"), call(200, 300)]


@pytest.mark.parametrize(
    "calls,infra",
    [
        (CLEAN, False),
        (LAST_503, True),
        (UPSTREAM_502, True),
        (CONNECTION, True),
        (PEER_DROP, True),
        (EARLY_RECOVERED, False),
        (LONG_FAULTS, True),
    ],
)
def test_the_predicate_reads_the_captured_calls_of_the_window(calls, infra) -> None:
    assert is_infrastructure_agent_timeout(timeout_row(0, calls)) is infra
    # Another kind is never reclassified, whatever its calls show.
    assert not is_infrastructure_agent_timeout(timeout_row(0, calls) | {"failure_kind": "agent_limit"})


@pytest.mark.parametrize(
    "excluded",
    [
        {"status_code": None, "error_category": "timeout"},
        {"status_code": None, "error_category": "cancelled"},
        {"status_code": 200, "error_category": "stream_truncated"},
        {"status_code": 429, "error_category": "rate_limit"},
        {"status_code": None, "failure_report": {"events": {"e": {"failure_kind": "transport_timeout"}}}},
    ],
)
def test_a_timeout_cancel_or_truncated_stream_is_no_server_fault(excluded) -> None:
    last = call(100, 30) | excluded
    assert not is_server_fault_call(last)
    assert not is_infrastructure_agent_timeout(timeout_row(0, [call(0, 100), last]))


def test_a_sandbox_or_tool_server_failure_event_in_the_row_is_infrastructure() -> None:
    events = {
        "t": {"event_id": "t", "boundary": "agent", "failure_kind": "agent_timeout", "cause_id": None},
        "s": {"event_id": "s", "boundary": "harness", "failure_kind": "provider_unavailable", "cause_id": None},
    }
    row = timeout_row(0, CLEAN, _ng_failure_id="t", _ng_failure_events=events)
    assert is_infrastructure_agent_timeout(row)
    tool = {
        "event_id": "s",
        "boundary": "resources_server",
        "operation": "tool",
        "failure_kind": "x",
        "http_status": 503,
    }
    assert is_infrastructure_agent_timeout(row | {"_ng_failure_events": events | {"s": tool}})
    # The model's own command exhausting the sandbox's memory, or a judge after the window, is no such fault.
    oom = events | {"s": events["s"] | {"failure_kind": "provider_oom_killed"}}
    assert not is_infrastructure_agent_timeout(row | {"_ng_failure_events": oom})
    judge = events | {"s": tool | {"operation": "verify"}}
    assert not is_infrastructure_agent_timeout(row | {"_ng_failure_events": judge})


def test_the_kind_is_gate_derived_and_evaluation_side() -> None:
    assert is_gate_derived(AGENT_TIMEOUT_INFRASTRUCTURE)


@pytest.mark.parametrize("calls", [LAST_503, CONNECTION, PEER_DROP, LONG_FAULTS])
def test_an_unanswered_timeout_with_server_faults_is_retried_as_infrastructure(calls) -> None:
    d = decide_recovery(materialized=inventory(1), results=[], failures=[timeout_row(0, calls)], budget=OK_BUDGET)
    assert d.counts == {IdentityState.RETRYABLE: 1} and d.action == RecoveryAction.RESUBMIT
    assert d.evidence[0].failure_kind == AGENT_TIMEOUT_INFRASTRUCTURE


@pytest.mark.parametrize("calls", [CLEAN, EARLY_RECOVERED])
def test_a_clean_answered_timeout_stays_the_models_final_result(calls) -> None:
    row = timeout_row(0, calls, kept=True)
    assert is_answered_agent_deadline(row)
    d = decide_recovery(materialized=inventory(1), results=[], failures=[row], budget=OK_BUDGET)
    assert d.counts == {IdentityState.MODEL_FINAL: 1} and d.action == RecoveryAction.NOOP
    assert d.evidence[0].failure_kind == "agent_timeout"
    # Unanswered, it keeps its kind (and the default retry), as before.
    d = decide_recovery(materialized=inventory(1), results=[], failures=[timeout_row(0, calls)], budget=OK_BUDGET)
    assert d.evidence[0].failure_kind == "agent_timeout"


def test_an_answered_timeout_with_server_faults_is_retryable_not_final() -> None:
    row = timeout_row(0, LAST_503, kept=True)
    assert not is_answered_agent_deadline(row)
    d = decide_recovery(materialized=inventory(1), results=[], failures=[row], budget=OK_BUDGET)
    assert d.counts == {IdentityState.RETRYABLE: 1} and d.action == RecoveryAction.RESUBMIT
    assert d.evidence[0].failure_kind == AGENT_TIMEOUT_INFRASTRUCTURE
    # Under the attempt cap only.
    capped = [timeout_row(0, LAST_503, kept=True, _line=i) for i in range(3)]
    d = decide_recovery(materialized=inventory(1), results=[], failures=capped, budget=OK_BUDGET)
    assert d.counts == {IdentityState.EXHAUSTED: 1}


def test_a_legacy_harbor_graded_zero_is_kept_and_its_server_faults_are_counted_evaluation_side() -> None:
    # Without a Store a resume cannot reopen a saved result: it stays scored, and the gate counts its fault.
    d = decide_recovery(
        materialized=inventory(2),
        results=[harbor_zero(0, LAST_503), harbor_zero(1, CLEAN)],
        failures=[],
        budget=OK_BUDGET,
    )
    assert d.counts == {IdentityState.SCORED: 2}
    collected = CollectedEvidence(
        materialized=inventory(2), results=[harbor_zero(0, LAST_503), harbor_zero(1, CLEAN)], failures=[]
    )
    kinds = trial_faults(d, collected)
    # The clean graded zero still counts its own agent_timeout once, as the model's (``_scored_fault``).
    assert kinds == {AGENT_TIMEOUT_INFRASTRUCTURE: 1, "agent_timeout": 1}
    report = fault_report(kinds, 2)
    assert report["evaluation"]["count"] == 1 and report["evaluation"]["server"] == 1
    assert report["model"]["count"] == 1 and report["model"]["kinds"] == {"agent_timeout": 1}
    facts = failure_facts(d, collected)
    assert facts["trial_failures"]["by_cause"]["timeout"] == {"count": 2, "rate": 1.0}


def test_the_collector_projects_the_failure_rows_calls_and_recovery_reclassifies_a_saved_run(tmp_path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory(3)))
    (artifacts / "rollouts.jsonl").write_text("")
    rows = [timeout_row(0, LAST_503, kept=True), timeout_row(1, CLEAN, kept=True), timeout_row(2, EARLY_RECOVERED)]
    (artifacts / "rollouts_failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    raw = subprocess.run(
        [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(tmp_path)], capture_output=True, text=True, check=True
    )
    assert "draft" not in raw.stdout
    collected = CollectedEvidence.model_validate_json(raw.stdout)
    projected = collected.failures[0]["_ng_collected_model_calls"]
    assert projected[-1] == {
        "status_code": 503,
        "upstream_status_code": None,
        "error_category": "upstream_error",
        "failure_events": [],
        "started_at": 1100.0,
        "duration_ms": 30000.0,
    }
    d = decide_from_collected(collected, max_infra_retries=3, max_walltime_s=None, live_job=False)
    assert d.identities[IdentityState.RETRYABLE] == [(0, 0), (2, 0)]
    assert d.identities[IdentityState.MODEL_FINAL] == [(1, 0)]
    assert [e.failure_kind for e in d.evidence] == [AGENT_TIMEOUT_INFRASTRUCTURE, "agent_timeout", "agent_timeout"]
    assert d.action == RecoveryAction.RESUBMIT


async def test_the_store_reruns_an_answered_timeout_with_server_faults(tmp_path, monkeypatch) -> None:
    result = {
        "_ng_failure_class": "agent_timeout",
        "failure_kind": "agent_timeout",
        "_ng_elapsed_s": WINDOW_S,
        "response": answer(),
        "ng_model_call_capture": {"calls": LAST_503},
    }
    decision, store, _ = await persist_producer_result(tmp_path, monkeypatch, result)
    assert len(store.pending()) == 1
    assert decision.counts == {IdentityState.RETRYABLE: 1} and decision.action == RecoveryAction.RESUBMIT
    assert decision.evidence[0].failure_kind == AGENT_TIMEOUT_INFRASTRUCTURE


@pytest.mark.parametrize("calls,reopened", [(LAST_503, True), (CLEAN, False)])
async def test_the_store_reopens_a_harbor_graded_zero_only_with_server_faults(
    tmp_path, monkeypatch, calls, reopened
) -> None:
    result = {
        "reward": 0.0,
        "failure_kind": "agent_timeout",
        "evaluation_completed": True,
        "mask_sample": False,
        "response": answer(),
        "ng_model_call_capture": {"calls": calls},
    }
    decision, store, _ = await persist_producer_result(tmp_path, monkeypatch, result)
    if reopened:
        # Not the model's result: out of the scored results, run again within the attempt cap.
        assert len(store.pending()) == 1 and store.selected("success") == []
        assert store.attempt_count(store.pending()[0]) == 1
        assert decision.counts == {IdentityState.RETRYABLE: 1} and decision.action == RecoveryAction.RESUBMIT
        assert decision.evidence[0].failure_kind == AGENT_TIMEOUT_INFRASTRUCTURE
    else:
        assert store.pending() == [] and len(store.selected("success")) == 1
        assert decision.counts == {IdentityState.SCORED: 1} and decision.action == RecoveryAction.NOOP


@pytest.mark.parametrize("stage,expected", [("agent", False), ("seed", True), ("verification", False), (None, False)])
def test_timeout_materiality_uses_shared_stage_class_for_sandbox_evidence(stage, expected):
    event = {"event_id": "oom", "boundary": "sandbox", "failure_kind": "provider_oom_killed", "stage": stage}
    row = timeout_row(0, CLEAN, _ng_failure_events={"oom": event})
    assert is_infrastructure_agent_timeout(row) is expected
    assert event["failure_kind"] == "provider_oom_killed"
    assert event["stage"] == stage


def test_unregistered_provider_prefix_is_not_infrastructure_evidence():
    event = {"event_id": "unknown", "boundary": "sandbox", "failure_kind": "provider_not_registered"}
    row = timeout_row(0, CLEAN, _ng_failure_events={"unknown": event})
    assert not is_infrastructure_agent_timeout(row)


def test_materiality_uses_configured_class_without_widening_timeout_eligibility():
    from nemo_gym.core.failure_kinds import FailureClassificationConfig

    config = FailureClassificationConfig(
        by_failure_kind={"model_server_error": "model", "model_policy_response_timeout": "evaluation"}
    )
    row = timeout_row(0, [call(0, 600, 503)])
    assert is_infrastructure_agent_timeout(row)
    assert not is_infrastructure_agent_timeout(row, classification=config)
    timeout = {"failure_kind": "model_policy_response_timeout", "stage": "agent"}
    assert not is_server_fault_call({"failure_events": [timeout]}, classification=config)


@pytest.mark.parametrize("action", ["count_as_zero", "fail"])
def test_applied_final_action_prevents_timeout_replay(action):
    row = timeout_row(0, [call(0, 600, 503)])
    row.update(failure_outcome_action=action, failure_outcome_attribution="evaluation")
    assert not is_infrastructure_agent_timeout(row)
