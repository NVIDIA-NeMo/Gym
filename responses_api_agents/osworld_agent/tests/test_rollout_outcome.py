# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from responses_api_agents.osworld_agent.rollout_outcome import (
    RUNTIME_ADMISSION_POLICY_ID,
    RolloutOutcomeFacts,
    classify_rollout_outcome,
)


@pytest.mark.parametrize(
    ("facts", "termination_reason", "horizon_reached"),
    [
        (
            RolloutOutcomeFacts(evaluation_completed=True, terminal_action="DONE"),
            "agent_done",
            False,
        ),
        (
            RolloutOutcomeFacts(evaluation_completed=True, terminal_action="FAIL"),
            "agent_fail",
            False,
        ),
        (
            RolloutOutcomeFacts(evaluation_completed=True, environment_done=True),
            "environment_done",
            False,
        ),
        (
            RolloutOutcomeFacts(evaluation_completed=True, horizon_reached=True),
            "max_steps",
            True,
        ),
        (
            RolloutOutcomeFacts(
                evaluation_completed=True,
                policy_stop_reason="model_response_invalid",
            ),
            "model_response_invalid",
            False,
        ),
    ],
)
def test_evaluated_policy_and_environment_outcomes_remain_runtime_eligible(
    facts: RolloutOutcomeFacts,
    termination_reason: str,
    horizon_reached: bool,
) -> None:
    outcome = classify_rollout_outcome(facts)

    assert outcome.termination_reason == termination_reason
    assert outcome.horizon_reached is horizon_reached
    assert outcome.evaluation_completed is True
    assert outcome.runtime_eligible is True
    assert outcome.mask_sample is False
    assert outcome.runtime_admission_reason == "valid_evaluated_outcome"
    assert outcome.runtime_admission_policy_id == RUNTIME_ADMISSION_POLICY_ID


@pytest.mark.parametrize(
    "reason",
    [
        "timeout",
        "task_timeout",
        "model_call_failed",
        "rollout_error",
        "evaluator_error",
        "proxy_setup_error",
        "proxy_configuration_error",
    ],
)
def test_infrastructure_failures_are_runtime_ineligible(reason: str) -> None:
    outcome = classify_rollout_outcome(
        RolloutOutcomeFacts(
            evaluation_completed=reason not in {"task_timeout", "proxy_configuration_error"},
            infrastructure_failure_reason=reason,
        )
    )

    assert outcome.termination_reason == reason
    assert outcome.runtime_eligible is False
    assert outcome.mask_sample is True
    assert outcome.runtime_admission_reason == reason


def test_unclassified_or_unevaluated_rollouts_fail_closed() -> None:
    unevaluated = classify_rollout_outcome(RolloutOutcomeFacts(evaluation_completed=False, horizon_reached=True))
    unclassified = classify_rollout_outcome(RolloutOutcomeFacts(evaluation_completed=True))

    assert unevaluated.termination_reason == "evaluation_incomplete"
    assert unevaluated.mask_sample is True
    assert unclassified.termination_reason == "rollout_state_incomplete"
    assert unclassified.mask_sample is True


def test_outcome_facts_reject_conflicting_or_invalid_states() -> None:
    with pytest.raises(TypeError, match="evaluation_completed must be boolean"):
        RolloutOutcomeFacts(evaluation_completed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported terminal_action"):
        RolloutOutcomeFacts(evaluation_completed=True, terminal_action="WAIT")
    with pytest.raises(ValueError, match="horizon_reached cannot also carry"):
        RolloutOutcomeFacts(
            evaluation_completed=True,
            horizon_reached=True,
            terminal_action="DONE",
        )
