# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OSWorld rollout outcomes and runtime-admission decisions.

This module deliberately owns only the runtime boundary.  It decides whether
the VM/evaluator result is trustworthy enough to hand to a training consumer;
it does not decide whether exact token evidence is complete or which tokens a
trainer should include in a loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


RUNTIME_ADMISSION_POLICY_ID = "osworld-runtime-admission-v1"
LEGACY_RUNTIME_ADMISSION_POLICY_ID = "legacy-mask-sample-v1"


@dataclass(frozen=True)
class RolloutOutcomeFacts:
    """Lossless runner/evaluator facts consumed by runtime admission."""

    evaluation_completed: bool
    infrastructure_failure_reason: Optional[str] = None
    terminal_action: Optional[str] = None
    environment_done: bool = False
    horizon_reached: bool = False
    policy_stop_reason: Optional[str] = None

    def __post_init__(self) -> None:
        for name in (
            "evaluation_completed",
            "environment_done",
            "horizon_reached",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        for name in (
            "infrastructure_failure_reason",
            "terminal_action",
            "policy_stop_reason",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        if self.terminal_action is not None and self.terminal_action.upper() not in {"DONE", "FAIL"}:
            raise ValueError(f"unsupported terminal_action: {self.terminal_action!r}")
        if self.horizon_reached and any((self.terminal_action, self.environment_done, self.policy_stop_reason)):
            raise ValueError("horizon_reached cannot also carry a terminal or policy-stop fact")
        if self.policy_stop_reason is not None and any((self.terminal_action, self.environment_done)):
            raise ValueError("policy_stop_reason cannot also carry an environment terminal fact")


@dataclass(frozen=True)
class RolloutOutcome:
    """Derived outcome plus the backward-compatible runtime mask."""

    termination_reason: str
    horizon_reached: bool
    evaluation_completed: bool
    runtime_eligible: bool
    runtime_admission_reason: str
    runtime_admission_policy_id: str = RUNTIME_ADMISSION_POLICY_ID

    @property
    def mask_sample(self) -> bool:
        """Legacy NeMo-RL carrier for runtime ineligibility."""

        return not self.runtime_eligible


def classify_rollout_outcome(facts: RolloutOutcomeFacts) -> RolloutOutcome:
    """Classify one rollout without changing its evaluator score or actions.

    Normal policy/environment outcomes remain runtime-eligible after a
    successful evaluation, including horizon exhaustion, explicit DONE/FAIL,
    environment termination, and a sampled response that the adapter could not
    parse.  Infrastructure and evaluator failures remain masked.
    """

    if facts.infrastructure_failure_reason is not None:
        reason = facts.infrastructure_failure_reason.strip()
        return RolloutOutcome(
            termination_reason=reason,
            horizon_reached=facts.horizon_reached,
            evaluation_completed=facts.evaluation_completed,
            runtime_eligible=False,
            runtime_admission_reason=reason,
        )

    if not facts.evaluation_completed:
        return RolloutOutcome(
            termination_reason="evaluation_incomplete",
            horizon_reached=facts.horizon_reached,
            evaluation_completed=False,
            runtime_eligible=False,
            runtime_admission_reason="evaluation_incomplete",
        )

    if facts.policy_stop_reason is not None:
        termination_reason = facts.policy_stop_reason.strip()
    elif facts.terminal_action is not None:
        termination_reason = f"agent_{facts.terminal_action.lower()}"
    elif facts.environment_done:
        termination_reason = "environment_done"
    elif facts.horizon_reached:
        termination_reason = "max_steps"
    else:
        return RolloutOutcome(
            termination_reason="rollout_state_incomplete",
            horizon_reached=False,
            evaluation_completed=True,
            runtime_eligible=False,
            runtime_admission_reason="rollout_state_incomplete",
        )

    return RolloutOutcome(
        termination_reason=termination_reason,
        horizon_reached=facts.horizon_reached,
        evaluation_completed=True,
        runtime_eligible=True,
        runtime_admission_reason="valid_evaluated_outcome",
    )
