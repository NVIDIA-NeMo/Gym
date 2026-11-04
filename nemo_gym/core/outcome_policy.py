# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configurable outcome selection; each boundary executes its own actions."""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from nemo_gym.core.failure_kinds import FAILURE_KINDS


if TYPE_CHECKING:
    from nemo_gym.core.typed_exceptions import NgFailureEvent


class OutcomeAction(StrEnum):
    """Action selected for one boundary decision.

    Each boundary restricts supported actions and enforces their prerequisites.
    """

    VERIFY = "verify"
    RETRY = "retry"
    FAIL = "fail"
    COUNT_AS_ZERO = "count_as_zero"


class OutcomePolicyConfig(BaseModel):
    """Select an action from explicit failure evidence; partial overrides retain boundary defaults."""

    model_config = ConfigDict(extra="forbid")

    by_failure_kind: dict[str, OutcomeAction] = Field(default_factory=dict)
    by_status_code: dict[Annotated[int, Field(ge=400, le=599)], OutcomeAction] = Field(default_factory=dict)
    default_action: OutcomeAction

    @field_validator("by_failure_kind", "by_status_code")
    @classmethod
    def merge_default_rules(
        cls, value: dict[str | int, OutcomeAction], info: ValidationInfo
    ) -> dict[str | int, OutcomeAction]:
        """Merge validated overrides so JSON status keys are normalized before replacing defaults."""
        return cls.model_fields[info.field_name].get_default(call_default_factory=True) | value

    @field_validator("by_failure_kind")
    @classmethod
    def require_registered_kinds(cls, value: dict[str, OutcomeAction]) -> dict[str, OutcomeAction]:
        """Reject misspelled failure names at configuration load, before starting an evaluation."""
        invalid = value.keys() - FAILURE_KINDS - {"unknown"}
        if invalid:
            raise ValueError(f"Unregistered failure kinds: {sorted(invalid)}")
        return value


def choose_action(*, failure_kind: str, http_status: int | None, config: OutcomePolicyConfig) -> OutcomeAction:
    """Select by status, then kind, then default; parsing and execution belong to the boundary."""
    return config.by_status_code.get(http_status, config.by_failure_kind.get(failure_kind, config.default_action))


def failure_outcome_policy(failure: "NgFailureEvent", *, config: OutcomePolicyConfig) -> OutcomeAction:
    """Adapt only this event's selected kind and status, without searching its causal chain."""
    return choose_action(failure_kind=failure.failure_kind, http_status=failure.http_status, config=config)
