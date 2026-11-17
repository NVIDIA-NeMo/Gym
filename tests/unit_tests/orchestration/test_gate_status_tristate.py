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
"""The acceptance gate's status is SUCCESS, UNVERIFIED or FAILED: unknown is never a pass and never a failure."""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from nemo_gym.cli.main import _eval_resubmit
from nemo_gym.orchestration import submit
from nemo_gym.orchestration.recovery import CollectedEvidence, gate_task_summary
from tests.unit_tests.orchestration.test_evaluation_health import _gate, _run


def _accountable(collected: CollectedEvidence) -> CollectedEvidence:
    """A read score and a saved config that requires no MLflow export, so only health and validity decide."""
    collected.score = 0.9
    collected.quality["provenance"]["mlflow"]["tracking_required"] = False
    return collected


def _uncovered(collected: CollectedEvidence) -> CollectedEvidence:
    for row in collected.results[:10]:
        row["ng_trial_observations"]["verifier_invocations"][0]["grading_operations"] = None
    return collected


def test_a_clean_run_is_success():
    _, (passed, _, report), task = _gate(_accountable(_run(100)))
    assert passed is True and report["status"] == "PASS"
    assert task["status"] == "SUCCESS" and task["errors"] == []


def test_unverified_coverage_is_unverified_with_the_reason_in_warnings():
    _, (passed, reason, report), task = _gate(_accountable(_uncovered(_run(100))))
    assert passed is None and report["status"] == "UNVERIFIED"
    assert task["status"] == "UNVERIFIED"
    assert task["errors"] == []
    [health] = [w for w in task["warnings"] if w.startswith("response/trial health")]
    health += " " + ((task["evaluation_health"] or {}).get("detail") or "")
    assert health.startswith("response/trial health unknown") and "trial: UNOBSERVED 10 of 100" in health
    assert task["health_verdict"].startswith("evaluation health: UNVERIFIED")


def test_an_evaluation_side_failure_is_failed():
    _, (passed, _, report), task = _gate(_accountable(_run(97, settled={"model_transport_error": 3})))
    assert passed is False and report["status"] == "FAIL"
    assert task["status"] == "FAILED"
    assert any(e.startswith("response/trial health failed") for e in task["errors"])


def test_a_real_error_beside_unverified_health_is_failed():
    collected = _uncovered(_run(100))
    collected.score = None
    _, _, task = _gate(collected)
    assert task["status"] == "FAILED"
    assert (
        "score unavailable: no requested metric was read (pass --score-key, and --score-agent or --score-environment for a Gym aggregate)"
        in task["errors"]
    )
    assert any(w.startswith("response/trial health unknown") for w in task["warnings"])


def test_an_unread_score_of_an_incomplete_run_names_the_run_not_the_score_options():
    from nemo_gym.orchestration.recovery import IdentityState, RecoveryAction, RecoveryDecision

    # Both score options were passed; the run is half done, so there is no aggregate to read yet.
    decision = RecoveryDecision(
        action=RecoveryAction.RESUBMIT,
        reason="78 identities need the policy again",
        expected=160,
        counts={IdentityState.SCORED.value: 82, IdentityState.MISSING.value: 78},
        identities={IdentityState.MISSING.value: [(task, 0) for task in range(82, 160)]},
        evidence=[],
    )
    task = gate_task_summary(
        decision, score=None, summary_valid=True, summary_reasons=[], health_passed=None, health_reasons=[]
    )

    assert (
        "score unavailable: the run is not complete (82/160 rollouts settled); "
        "the score is read once every rollout is settled"
    ) in task["errors"]
    assert not any("--score-key" in error for error in task["errors"])


def test_missing_mlflow_delivery_beside_unverified_health_is_a_fact_not_a_failure():
    collected = _uncovered(_run(100))
    collected.score = 0.9
    # Tracking is required and there is no MLflow evidence: a pending delivery fact, not an error.
    _, _, task = _gate(collected)
    assert task["status"] == "UNVERIFIED" and not any("MLflow" in e for e in task["errors"])
    assert task["mlflow_delivery"]["state"] == "pending"
    assert any(w.startswith("response/trial health unknown") for w in task["warnings"])


@pytest.mark.parametrize(
    "summary_valid,health_passed,score,status",
    [
        (True, True, 0.9, "SUCCESS"),
        (None, True, 0.9, "UNVERIFIED"),
        (True, None, 0.9, "UNVERIFIED"),
        (None, None, 0.9, "UNVERIFIED"),
        (False, None, 0.9, "FAILED"),
        (None, False, 0.9, "FAILED"),
        (None, None, None, "FAILED"),
    ],
)
def test_status_precedence_of_a_complete_noop(summary_valid, health_passed, score, status):
    decision, _, _ = _gate(_run(10))
    task = gate_task_summary(
        decision,
        score=score,
        summary_valid=summary_valid,
        summary_reasons=["why"],
        health_passed=health_passed,
        health_reasons=["why"],
    )
    assert task["status"] == status
    unknown = [
        p for p, v in (("summary validity", summary_valid), ("response/trial health", health_passed)) if v is None
    ]
    assert all(any(w.startswith(f"{p} unknown") for w in task["warnings"]) for p in unknown)
    assert not any("unknown" in e for e in task["errors"])


def _exit_code(monkeypatch, gate_task: dict, **flags) -> int:
    outcome = SimpleNamespace(ok=True, gate_task=gate_task, operator_verdict=None, model_dump_json=lambda **kw: "{}")
    monkeypatch.setattr(submit, "recover", lambda *a, **kw: outcome)
    monkeypatch.setattr(submit, "inspect_recovery", lambda *a, **kw: outcome)
    try:
        _eval_resubmit(Namespace(gym_job_id="saved", benchmark="judge", recover=True, **flags), [])
    except SystemExit as error:
        return error.code
    return 0


@pytest.mark.parametrize("flags", [{"score_key": '["reward"]'}, {"inspect": True}])
def test_an_unverified_gate_exits_non_zero_and_success_exits_zero(monkeypatch, flags):
    _, _, unverified = _gate(_accountable(_uncovered(_run(100))))
    _, _, success = _gate(_accountable(_run(100)))
    assert (unverified["status"], success["status"]) == ("UNVERIFIED", "SUCCESS")
    assert _exit_code(monkeypatch, unverified, **flags) != 0
    assert _exit_code(monkeypatch, success, **flags) == 0
