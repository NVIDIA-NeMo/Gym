# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configurable outcome selection; each boundary executes its own actions."""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from nemo_gym.core.failure_kinds import (
    CANCELLED,
    FAILURE_KINDS,
    KILL_SHAPED,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_OOM_KILLED,
    SHUTDOWN,
    FailureClass,
    FailureClassificationConfig,
    FailureStage,
    _merge_stage_overrides,
    classify_failure,
)


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


def _require_registered_kinds[T](value: dict[str, T]) -> dict[str, T]:
    invalid = value.keys() - FAILURE_KINDS - {"unknown"}
    if invalid:
        raise ValueError(f"Unregistered failure kinds: {sorted(invalid)}")
    return value


class OutcomePolicyConfig(BaseModel):
    """Select an action from explicit failure evidence; partial overrides retain boundary defaults."""

    model_config = ConfigDict(extra="forbid")
    supported_actions: ClassVar[frozenset[OutcomeAction]] = frozenset(OutcomeAction)

    by_failure_kind: dict[str, OutcomeAction] = Field(default_factory=dict)
    by_status_code: dict[Annotated[int, Field(ge=400, le=599)], OutcomeAction] = Field(default_factory=dict)
    by_failure_class: dict[FailureClass, OutcomeAction] = Field(default_factory=dict)
    default_action: OutcomeAction

    @field_validator("by_failure_kind", "by_status_code", "by_failure_class")
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
        return _require_registered_kinds(value)

    @model_validator(mode="after")
    def require_supported_actions(self) -> "OutcomePolicyConfig":
        selected = {
            self.default_action,
            *self.by_status_code.values(),
            *self.by_failure_kind.values(),
            *self.by_failure_class.values(),
        }
        invalid = selected - self.supported_actions
        if invalid:
            raise ValueError(f"{type(self).__name__} does not support {', '.join(sorted(invalid))}")
        return self


def choose_action(
    *,
    failure_kind: str,
    http_status: int | None,
    config: OutcomePolicyConfig,
    failure_class: FailureClass | None = None,
) -> OutcomeAction:
    """Select by status, kind, class, then default; execution belongs to the boundary."""
    return config.by_status_code.get(
        http_status,
        config.by_failure_kind.get(
            failure_kind,
            config.by_failure_class.get(failure_class, config.default_action)
            if failure_class is not None
            else config.default_action,
        ),
    )


def failure_outcome_policy(
    failure: "NgFailureEvent",
    *,
    config: OutcomePolicyConfig,
    classification: FailureClassificationConfig | None = None,
) -> OutcomeAction:
    """Adapt only this event's evidence, without searching its causal chain."""
    return choose_action(
        failure_kind=failure.failure_kind,
        http_status=failure.http_status,
        config=config,
        failure_class=classify_failure(
            failure_kind=failure.failure_kind,
            stage=failure.stage,
            http_status=failure.http_status,
            config=classification,
        ),
    )


class OutcomeDecision(BaseModel):
    """One policy decision for scoring and health consumers to preserve together."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: OutcomeAction
    attribution: FailureClass


class StagePolicyConfig(OutcomePolicyConfig):
    """Action rules for one stage; classification is configured independently."""

    default_action: OutcomeAction = OutcomeAction.FAIL


def _default_stage_policies() -> dict[FailureStage, StagePolicyConfig]:
    return {
        "agent": StagePolicyConfig(
            by_failure_kind={PROVIDER_OOM_KILLED: OutcomeAction.COUNT_AS_ZERO},
        ),
        "seed": StagePolicyConfig(
            by_failure_kind={PROVIDER_OOM_KILLED: OutcomeAction.RETRY},
        ),
        "verification": StagePolicyConfig(
            by_failure_kind={
                CANCELLED: OutcomeAction.FAIL,
                SHUTDOWN: OutcomeAction.FAIL,
                KILL_SHAPED: OutcomeAction.FAIL,
                PROVIDER_CREDENTIALS_REJECTED: OutcomeAction.FAIL,
            },
            default_action=OutcomeAction.RETRY,
        ),
    }


class StageOutcomePolicyConfig(BaseModel):
    """Shared episode policies selected by the failure's originating stage.

    Callers establish ownership and evidence before selecting a stage. A retry
    decision permits only safe retries within the caller's existing budget; once
    that budget is exhausted, the result stays ungraded. Provider-operation retry
    policies remain separate from these episode decisions. Verification defaults
    apply to admitted inconclusive attempts, never ordinary verifier scores.
    """

    model_config = ConfigDict(extra="forbid")

    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)
    by_stage: dict[FailureStage, StagePolicyConfig] = Field(default_factory=_default_stage_policies)
    default_policy: StagePolicyConfig = Field(default_factory=StagePolicyConfig)

    @field_validator("by_stage", mode="before")
    @classmethod
    def merge_stage_defaults(cls, value: object) -> object:
        """Merge partial stage and rule overrides without dropping other defaults."""
        return _merge_stage_overrides(
            _default_stage_policies(),
            value,
            dict_fields=("by_failure_kind", "by_status_code", "by_failure_class"),
        )


def choose_stage_outcome(
    *,
    failure_kind: str,
    http_status: int | None,
    stage: FailureStage | None,
    config: StageOutcomePolicyConfig,
) -> OutcomeDecision:
    """Select one decision from established context, without examining causes or SDK details.

    Missing stages use the conservative default policy. Preserve a known origin
    when a failure propagates: catching an agent failure during verification does
    not turn it into a verifier failure.
    """
    policy = config.by_stage.get(stage, config.default_policy)
    classification = classify_failure(
        failure_kind=failure_kind,
        stage=stage,
        http_status=http_status,
        config=config.classification,
    )
    return OutcomeDecision(
        action=choose_action(
            failure_kind=failure_kind,
            http_status=http_status,
            config=policy,
            failure_class=classification,
        ),
        attribution=classification,
    )
