# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.orchestration.recovery import (
    ManagedRecoveryFact,
    RecoveryOutcomePolicyConfig,
    decide_recovery,
    saved_row_evidence,
)
from tests.unit_tests.orchestration.test_eval_recovery_gate import OK_BUDGET, inventory
from tests.unit_tests.orchestration.test_recovery_infra_agent_timeout import LAST_503, harbor_zero
from tests.unit_tests.orchestration.test_unparsed_judge_verdict import _xstest_row


def decide(row, policy, **kwargs):
    return decide_recovery(
        materialized=inventory(1),
        results=[row],
        failures=[],
        budget=OK_BUDGET,
        outcome_policy=RecoveryOutcomePolicyConfig(**policy),
        **kwargs,
    )


@pytest.mark.parametrize(
    "policy,state",
    [
        ({}, "judge_recoverable"),
        ({"saved_answer_default_action": "fail"}, "terminal"),
        ({"by_failure_kind": {"judge_unparseable": "fail"}}, "terminal"),
        ({"by_failure_kind": {"judge_unparseable": "retry"}}, "unknown"),
        ({"by_failure_class": {"evaluation": "fail"}}, "terminal"),
    ],
)
def test_saved_unparsed_answer_obeys_policy_without_resampling(policy, state):
    row = saved_row_evidence(_xstest_row(None))
    decision = decide(row, policy)
    assert decision.evidence[0].state.value == state
    assert decision.action.value != "resubmit"


def test_saved_decision_is_not_overridden_by_recovery():
    row = saved_row_evidence(_xstest_row(None))
    row["failure_outcome_action"] = "fail"
    assert decide(row, {"by_failure_kind": {"judge_unparseable": "verify"}}).evidence[0].state == "terminal"


def test_masked_judge_status_kind_class_precedence():
    row = saved_row_evidence(_xstest_row(None)) | {
        "mask_sample": True,
        "failure_kind": "judge_unparseable",
        "_ng_failure_http_status": 503,
    }
    policy = {
        "by_status_code": {503: "verify"},
        "by_failure_kind": {"judge_unparseable": "fail"},
        "by_failure_class": {"evaluation": "retry"},
    }
    assert decide(row, policy).evidence[0].state == "judge_recoverable"
    del policy["by_status_code"]
    assert decide(row, policy).evidence[0].state == "terminal"


@pytest.mark.parametrize("action,state", [("retry", "retryable"), ("fail", "terminal"), ("verify", "unknown")])
def test_managed_infrastructure_timeout_obeys_policy(action, state):
    row = harbor_zero(0, LAST_503)
    managed = [ManagedRecoveryFact(identity=(0, 0), attempts=1, max_attempts=3, pending=True, reserved=False)]
    result = decide(row, {"by_failure_class": {"evaluation": action}}, managed_selection=managed)
    assert result.evidence[0].state == state


def test_explicit_zero_stays_scored_despite_completion_and_recovery_overrides():
    row = harbor_zero(0, LAST_503) | {
        "evaluation_completed": False,
        "failure_outcome_action": "count_as_zero",
        "failure_outcome_attribution": "model",
    }
    assert decide(row, {"default_action": "fail"}).evidence[0].state == "scored"


def test_clean_timeout_override_does_not_stop_infrastructure_timeout():
    managed = [ManagedRecoveryFact(identity=(0, 0), attempts=1, max_attempts=3, pending=True, reserved=False)]
    result = decide(
        harbor_zero(0, LAST_503), {"by_failure_kind": {"agent_timeout": "fail"}}, managed_selection=managed
    )
    assert result.evidence[0].state == "retryable"
    assert result.evidence[0].failure_kind == "agent_timeout_infrastructure"


@pytest.mark.parametrize("kind", ["provider_quota_exhausted", "provider_credentials_rejected"])
@pytest.mark.parametrize("action,state", [("verify", "judge_recoverable"), ("fail", "terminal"), ("retry", "unknown")])
def test_provider_stopped_judge_obeys_policy_without_resampling(kind, action, state):
    row = saved_row_evidence(_xstest_row(None)) | {
        "mask_sample": True,
        "failure_kind": kind,
        "failure_reason": "provider unavailable",
        "_ng_failure_class": "judge_failed",
    }
    result = decide(row, {"by_failure_kind": {kind: action}})
    assert result.evidence[0].state == state
    assert result.action.value != "resubmit"
    if action == "fail":
        assert "policy" in result.reason
        assert "After the reset" not in result.reason


@pytest.mark.parametrize("source", ["result", "failure"])
@pytest.mark.parametrize("saved_fail", [False, True])
def test_final_quota_policy_stop_is_not_a_future_reset_wait(source, saved_fail, monkeypatch):
    monkeypatch.setattr("nemo_gym.orchestration.recovery._now", lambda: 0)
    row = saved_row_evidence(_xstest_row(None)) | {
        "mask_sample": True,
        "failure_kind": "provider_quota_exhausted",
        "failure_reason": "resets at 2099-01-01 00:00 UTC",
        "_ng_failure_class": "judge_failed",
    }
    if saved_fail:
        row["failure_outcome_action"] = "fail"
    policy = RecoveryOutcomePolicyConfig(
        by_failure_kind={"provider_quota_exhausted": "verify" if saved_fail else "fail"}
    )
    result = decide_recovery(
        materialized=inventory(1),
        results=[row] if source == "result" else [],
        failures=[row] if source == "failure" else [],
        budget=OK_BUDGET,
        outcome_policy=policy,
    )
    assert result.evidence[0].state == "terminal"
    assert result.evidence[0].reason is None
    assert "policy" in result.reason
    assert "After the reset" not in result.reason


@pytest.mark.parametrize("source", ["result", "failure"])
@pytest.mark.parametrize(
    "reset,waiting", [("2099-01-01 00:00 UTC", True), ("1970-01-01 00:00 UTC", False), (None, False)]
)
def test_permitted_quota_regrading_waits_only_for_a_future_reset(source, reset, waiting, monkeypatch):
    monkeypatch.setattr("nemo_gym.orchestration.recovery._now", lambda: 100)
    row = saved_row_evidence(_xstest_row(None)) | {
        "mask_sample": True,
        "failure_kind": "provider_quota_exhausted",
        "failure_reason": f"resets at {reset}" if reset else "provider unavailable",
        "_ng_failure_class": "judge_failed",
    }
    result = decide_recovery(
        materialized=inventory(1),
        results=[row] if source == "result" else [],
        failures=[row] if source == "failure" else [],
        budget=OK_BUDGET,
        outcome_policy=RecoveryOutcomePolicyConfig(by_failure_kind={"provider_quota_exhausted": "verify"}),
    )
    assert result.action.value == ("stop" if waiting else "reverify")
    if waiting:
        assert "After the reset" in result.reason
        assert reset in result.reason
    else:
        assert "After the reset" not in result.reason
