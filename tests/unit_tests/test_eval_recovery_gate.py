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
"""The recovery decision `gym eval resubmit` takes from a run's persisted evidence.

All rows are injected controls in the shape rollout collection persists (identity keys, failures-sidecar flags,
saved failure events); none is provider evidence.
"""

import hashlib
import json
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import IdentityState, RecoveryAction, RecoveryBudget, decide_recovery


OK_BUDGET = RecoveryBudget(
    max_infra_retries=3,
    infra_retries_used=0,
    max_walltime_s=57600.0,
    walltime_capped=True,
    walltime_used_s=3600.0,
    live_job=False,
)


def ident(task: int, rollout: int = 0, **extra) -> dict:
    return {"_ng_task_index": task, "_ng_rollout_index": rollout, **extra}


def inventory(n: int, repeats: int = 1) -> list[dict]:
    return [ident(t, r, responses_create_params={"input": f"q{t}"}) for t in range(n) for r in range(repeats)]


def infra_failure(task: int, *, attempt: int, terminal: bool = False, kind: str = "provider_unavailable") -> dict:
    event = {
        "event_id": f"f{task}-{attempt}",
        "boundary": "model",
        "failure_kind": kind,
        "operation": "chat_completions",
        "operation_id": f"op{task}",
        "attempt_id": f"a{task}-{attempt}",
        "cause_id": None,
    }
    return ident(
        task,
        _ng_attempt_index=attempt,
        failure_kind=kind,
        _ng_failure_terminal=terminal,
        _ng_failure_max_attempts=3,
        _ng_failure_id=f"f{task}-{attempt}",
        _ng_failure_events={event["event_id"]: event},
    )


def model_final(task: int) -> dict:
    return ident(
        task,
        failure_kind="model_context_limit",
        _ng_failure_terminal=True,
        _ng_failure_counts_as_zero=True,
        _ng_failure_max_attempts=3,
    )


def judge_failed(task: int) -> dict:
    return ident(
        task,
        reward=0.0,
        mask_sample=True,
        failure_kind="judge_failed",
        _ng_failure_class="judge_failed",
        response={
            "id": f"saved-{task}",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"saved answer {task}"}],
                }
            ],
        },
    )


def scored(task: int, reward: float = 1.0) -> dict:
    return ident(task, reward=reward, response={"output_text": f"answer {task}"})


def test_all_healthy_is_a_noop_and_complete() -> None:
    d = decide_recovery(
        materialized=inventory(3), results=[scored(0), scored(1, 0.0), scored(2)], failures=[], budget=OK_BUDGET
    )
    assert d.action == RecoveryAction.NOOP
    assert d.complete and d.expected == 3 and d.counts == {"scored": 3}


def test_model_final_failure_is_a_scored_zero_never_retried() -> None:
    d = decide_recovery(materialized=inventory(2), results=[scored(0)], failures=[model_final(1)], budget=OK_BUDGET)
    assert d.action == RecoveryAction.NOOP
    assert d.identities["model_final"] == [(1, 0)]
    assert d.complete


def test_recoverable_infra_failure_resubmits_with_its_attempt_and_operation_identity() -> None:
    d = decide_recovery(
        materialized=inventory(2), results=[scored(0)], failures=[infra_failure(1, attempt=0)], budget=OK_BUDGET
    )
    assert d.action == RecoveryAction.RESUBMIT
    (item,) = [e for e in d.evidence if e.identity == (1, 0)]
    assert item.state == IdentityState.RETRYABLE and item.attempts == 1 and item.max_attempts == 3
    assert item.failure_id == "f1-0"
    assert item.events == [("model", "chat_completions", "op1", "a1-0", None)]
    assert not d.complete


def test_judge_only_failures_reverify() -> None:
    d = decide_recovery(
        materialized=inventory(3), results=[scored(0), judge_failed(1), judge_failed(2)], failures=[], budget=OK_BUDGET
    )
    assert d.action == RecoveryAction.REVERIFY
    assert d.identities["judge_recoverable"] == [(1, 0), (2, 0)]


def test_exhausted_attempts_alone_stop_because_a_resume_would_run_nothing() -> None:
    rows = [infra_failure(0, attempt=a) for a in range(3)]
    d = decide_recovery(materialized=inventory(1), results=[], failures=rows, budget=OK_BUDGET)
    assert d.identities["exhausted"] == [(0, 0)] and d.action == RecoveryAction.STOP
    assert "used every attempt" in d.reason
    mixed = decide_recovery(materialized=inventory(2), results=[], failures=rows, budget=OK_BUDGET)
    assert mixed.action == RecoveryAction.RESUBMIT and mixed.identities["missing"] == [(1, 0)]
    spent = RecoveryBudget(max_infra_retries=3, infra_retries_used=3, walltime_used_s=0.0, live_job=False)
    assert decide_recovery(materialized=inventory(1), results=[], failures=rows, budget=spent).action == "stop"
    late = RecoveryBudget(
        max_infra_retries=3,
        infra_retries_used=0,
        max_walltime_s=57600.0,
        walltime_capped=True,
        walltime_used_s=57600.0,
        live_job=False,
    )
    assert decide_recovery(materialized=inventory(1), results=[], failures=rows, budget=late).action == "stop"


def test_missing_and_partial_rows_restart_their_inputs_with_an_exact_denominator() -> None:
    d = decide_recovery(
        materialized=inventory(4, repeats=2),
        results=[scored(0), ident(0, 1, reward=1.0)],
        failures=[],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.RESUBMIT
    assert d.expected == 8 and d.counts == {"scored": 2, "missing": 6}


def test_mixed_cache_keeps_every_class_apart() -> None:
    d = decide_recovery(
        materialized=inventory(6),
        results=[scored(0), judge_failed(1)],
        failures=[model_final(2), infra_failure(3, attempt=0), infra_failure(4, attempt=0, terminal=True)],
        budget=OK_BUDGET,
    )
    assert d.counts == {
        "scored": 1,
        "judge_recoverable": 1,
        "model_final": 1,
        "retryable": 1,
        "terminal": 1,
        "missing": 1,
    }
    assert d.action == RecoveryAction.STOP and not d.complete
    assert sum(d.counts.values()) == d.expected == 6


def test_terminal_unscored_failure_alone_stops_and_is_never_complete() -> None:
    d = decide_recovery(
        materialized=inventory(2),
        results=[scored(0)],
        failures=[infra_failure(1, attempt=0, terminal=True)],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.STOP and not d.complete


def test_unknown_ownership_stops_with_a_reason() -> None:
    d = decide_recovery(
        materialized=inventory(2),
        results=[scored(0)],
        failures=[infra_failure(1, attempt=0, kind="something odd")],
        budget=OK_BUDGET,
    )
    assert d.action == RecoveryAction.STOP and "registered kind" in d.reason
    assert d.identities["unknown"] == [(1, 0)]


def test_unavailable_budget_evidence_stops_instead_of_resubmitting() -> None:
    d = decide_recovery(materialized=inventory(1), results=[], failures=[], budget=RecoveryBudget())
    assert d.action == RecoveryAction.STOP and "unavailable" in d.reason


def test_a_live_job_stops_a_second_queue() -> None:
    live = RecoveryBudget(max_infra_retries=3, infra_retries_used=0, live_job=True)
    assert decide_recovery(materialized=inventory(1), results=[], failures=[], budget=live).action == "stop"


def test_duplicate_or_foreign_result_rows_stop_rather_than_inflate_the_denominator() -> None:
    dup = decide_recovery(
        materialized=inventory(2), results=[scored(0), scored(0), scored(1)], failures=[], budget=OK_BUDGET
    )
    assert dup.action == RecoveryAction.STOP and dup.duplicate_results == [(0, 0)]
    foreign = decide_recovery(materialized=inventory(1), results=[scored(0), scored(9)], failures=[], budget=OK_BUDGET)
    assert foreign.action == RecoveryAction.STOP and foreign.expected == 1


def test_the_decision_reads_saved_files_without_changing_a_byte(tmp_path: Path) -> None:
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("materialized", "results", "failures")}
    rows = {
        "materialized": inventory(3),
        "results": [scored(0), judge_failed(1)],
        "failures": [infra_failure(2, attempt=0)],
    }
    for name, path in paths.items():
        path.write_text("".join(json.dumps(r) + "\n" for r in rows[name]))
    before = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()}
    loaded = {n: [json.loads(line) for line in p.read_text().splitlines()] for n, p in paths.items()}
    d = decide_recovery(
        materialized=loaded["materialized"], results=loaded["results"], failures=loaded["failures"], budget=OK_BUDGET
    )
    assert d.action == RecoveryAction.STOP
    assert {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()} == before
    assert loaded["results"][1]["response"] == rows["results"][1]["response"]


def _selected_event_failure(*, kind: str, selected: bool) -> dict:
    # A collection row carries its failure kind only on the event its failure_id selects, not at the top level.
    event = {"event_id": "e0", "boundary": "agent", "failure_kind": kind, "operation": "collection.dispatch"}
    return ident(
        0,
        _ng_attempt_index=0,
        _ng_failure_terminal=False,
        _ng_failure_max_attempts=3,
        _ng_failure_id="e0" if selected else "other",
        _ng_failure_events={"e0": event},
    )


def test_a_kind_named_only_by_the_selected_event_is_known() -> None:
    row = _selected_event_failure(kind="collection_dispatch_failed", selected=True)
    d = decide_recovery(materialized=inventory(1), results=[], failures=[row], budget=OK_BUDGET)
    assert d.counts.get(IdentityState.RETRYABLE) == 1 and d.action == RecoveryAction.RESUBMIT


def test_an_unregistered_or_unselected_event_kind_stays_unknown() -> None:
    for row in (
        _selected_event_failure(kind="not_a_registered_kind", selected=True),
        _selected_event_failure(kind="collection_dispatch_failed", selected=False),
    ):
        d = decide_recovery(materialized=inventory(1), results=[], failures=[row], budget=OK_BUDGET)
        assert d.counts.get(IdentityState.UNKNOWN) == 1 and d.action == RecoveryAction.STOP


def _verifier_failure_after_answer(
    *, attempt: int = 0, terminal: bool = False, trajectory: object = "answered"
) -> dict:
    # The saved shape of a rollout whose agent answered and whose verification then failed: no top-level kind or
    # response, a completed agent trajectory, and the selected chain dispatch -> agent route -> verify.
    events = {
        "e0": {
            "event_id": "e0",
            "boundary": "collection",
            "failure_kind": "collection_dispatch_failed",
            "operation": "collection.run",
            "cause_id": "e1",
        },
        "e1": {
            "event_id": "e1",
            "boundary": "agent",
            "failure_kind": "agent_route_failed",
            "operation": "agent.route",
            "cause_id": "e2",
        },
        "e2": {
            "event_id": "e2",
            "boundary": "resources_server",
            "failure_kind": "verifier_error",
            "operation": "verify",
            "cause_id": None,
        },
    }
    row = ident(
        0,
        _ng_attempt_index=attempt,
        _ng_failure_terminal=terminal,
        _ng_failure_max_attempts=3,
        _ng_failure_id="e0",
        _ng_failure_events=events,
    )
    if trajectory == "answered":
        row["ng_trajectory"] = {
            "invocations": [{"invocation_id": "root", "status": "completed"}],
            "turns": [
                {
                    "invocation_id": "root",
                    "answer": [
                        {
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "saved answer"}],
                        }
                    ],
                }
            ],
        }
    elif trajectory is not None:
        row["ng_trajectory"] = trajectory
    return row


def test_a_failed_verification_after_an_agent_answer_stops_instead_of_a_new_model_chance() -> None:
    for trajectory in ("answered", None, "malformed"):
        row = _verifier_failure_after_answer(trajectory=trajectory)
        d = decide_recovery(materialized=inventory(1), results=[], failures=[row], budget=OK_BUDGET)
        assert d.counts.get(IdentityState.UNKNOWN) == 1 and d.action == RecoveryAction.STOP
        assert d.counts.get(IdentityState.RETRYABLE) is None


def test_terminal_and_exhausted_verification_failures_keep_their_state() -> None:
    terminal = _verifier_failure_after_answer(terminal=True)
    d = decide_recovery(materialized=inventory(1), results=[], failures=[terminal], budget=OK_BUDGET)
    assert d.counts.get(IdentityState.TERMINAL) == 1
    spent = [_verifier_failure_after_answer(attempt=a) for a in range(3)]
    d = decide_recovery(materialized=inventory(1), results=[], failures=spent, budget=OK_BUDGET)
    assert d.action == RecoveryAction.STOP and d.counts.get(IdentityState.RETRYABLE) is None


def test_an_explicit_fail_policy_still_stops_a_failed_verification_as_settled() -> None:
    from nemo_gym.agent_outcome import OutcomeAction
    from nemo_gym.orchestration.recovery import RecoveryOutcomePolicyConfig

    policy = RecoveryOutcomePolicyConfig(by_failure_kind={"collection_dispatch_failed": OutcomeAction.FAIL})
    d = decide_recovery(
        materialized=inventory(1),
        results=[],
        failures=[_verifier_failure_after_answer()],
        budget=OK_BUDGET,
        outcome_policy=policy,
    )
    assert d.counts.get(IdentityState.TERMINAL) == 1 and d.action == RecoveryAction.STOP


def test_an_infrastructure_failure_before_any_answer_still_retries_within_its_budget() -> None:
    first = _selected_event_failure(kind="collection_dispatch_failed", selected=True)
    d = decide_recovery(materialized=inventory(1), results=[], failures=[first], budget=OK_BUDGET)
    assert d.counts.get(IdentityState.RETRYABLE) == 1 and d.action == RecoveryAction.RESUBMIT
    spent = [dict(first, _ng_attempt_index=a) for a in range(3)]
    d = decide_recovery(materialized=inventory(1), results=[], failures=spent, budget=OK_BUDGET)
    assert d.counts.get(IdentityState.EXHAUSTED) == 1


@pytest.mark.parametrize(
    "extra,unsupported",
    [
        (
            {
                "response": {
                    "id": "saved",
                    "object": "response",
                    "output": [{"type": "message", "role": "assistant"}],
                },
                "evaluation_completed": True,
            },
            True,
        ),
        (
            {
                "response": {
                    "id": "saved",
                    "object": "response",
                    "output": [{"type": "message", "role": "assistant"}],
                },
                "evaluation_completed": False,
            },
            True,
        ),
        ({"_ng_collected_saved_answer": True}, True),
        ({"_ng_collected_saved_answer": False, "response": {"output": []}}, False),
        ({}, False),
    ],
)
def test_unconfirmed_harness_termination_replays_only_without_a_saved_answer(extra, unsupported):
    from nemo_gym.orchestration.recovery import _policy_replay_is_unsupported

    events = {
        "h": {
            "boundary": "harness",
            "operation": "harness.terminate",
            "failure_kind": "harness_termination_unconfirmed",
        }
    }
    assert _policy_replay_is_unsupported({"_ng_failure_id": "h", "_ng_failure_events": events, **extra}) is unsupported


def test_other_harness_failures_and_malformed_causes_keep_their_replay_decision():
    from nemo_gym.orchestration.recovery import _policy_replay_is_unsupported

    other = {"h": {"boundary": "harness", "operation": "harness.run"}}
    assert not _policy_replay_is_unsupported(
        {"_ng_failure_id": "h", "_ng_failure_events": other, "response": {"a": 1}}
    )
    assert _policy_replay_is_unsupported({"_ng_failure_id": "missing", "_ng_failure_events": other})
