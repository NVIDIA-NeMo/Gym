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

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    IdentityState,
    RecoveryAction,
    decide_from_collected,
    decide_recovery,
)
from tests.unit_tests.test_eval_recovery_gate import OK_BUDGET, ident, infra_failure, inventory, scored
from tests.unit_tests.test_recovery_saved_generation import persist_producer_result, response


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
    from nemo_gym.rollout_recovery import is_answered_agent_deadline

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
