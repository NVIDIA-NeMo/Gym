# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared configurable outcome selection; each boundary executes its own actions."""

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

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
    is_namespaced,
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


def is_policy_counted_zero(row: Mapping[str, object]) -> bool:
    """Consume a saved explicit-zero decision, independently of completion or diagnostic evidence.

    The owning boundary admits the failure and applies its policy before persisting ``failure_outcome_action``.
    Readers check only consistency with the measured result; counting does not establish fault attribution.
    """
    return (
        row.get("failure_outcome_action") == OutcomeAction.COUNT_AS_ZERO
        and type(row.get("reward")) in (int, float)
        and row["reward"] == 0
        and row.get("mask_sample", False) is False
    )


def _require_registered_kinds[T](value: dict[str, T]) -> dict[str, T]:
    invalid = {kind for kind in value.keys() - FAILURE_KINDS - {"unknown"} if not is_namespaced(kind)}
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

    @classmethod
    def overlay(cls, defaults: "StageOutcomePolicyConfig", override: object) -> object:
        """Merge an explicit override over a boundary's own defaults instead of the shared ones.

        Rules the override does not mention keep the boundary's default, in the classification as in the stage
        policies. Anything that is not a policy mapping or model is returned unchanged for validation to reject.
        """
        if isinstance(override, StageOutcomePolicyConfig):
            override = override.model_dump(exclude_unset=True)
        if not isinstance(override, dict):
            return override
        classification = override.get("classification", {})
        if isinstance(classification, FailureClassificationConfig):
            classification = classification.model_dump(exclude_unset=True)
        return {
            **override,
            "classification": {
                **classification,
                "by_stage": _merge_stage_overrides(
                    defaults.classification.by_stage,
                    classification.get("by_stage", {}),
                    dict_fields=("by_failure_kind",),
                ),
            }
            if isinstance(classification, dict)
            else classification,
            "by_stage": _merge_stage_overrides(
                defaults.by_stage,
                override.get("by_stage", {}),
                dict_fields=("by_failure_kind", "by_status_code", "by_failure_class"),
            ),
        }


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


def environment_fault_policy(stage: FailureStage, failure_kinds: tuple[str, ...]) -> StageOutcomePolicyConfig:
    """A boundary's defaults for faults of the environment itself that it names (its own ``<server>:<kind>`` kinds).

    The kinds are classified as the evaluation's, and the stage's default action is ``retry``. Both are
    configuration: a deployment can classify a kind differently or select ``fail`` as the stage's default.
    """
    return StageOutcomePolicyConfig(
        classification={
            "by_stage": {stage: {"by_failure_kind": dict.fromkeys(failure_kinds, FailureClass.EVALUATION)}}
        },
        by_stage={stage: {"default_action": OutcomeAction.RETRY}},
    )


def require_retry_or_fail(
    policy: StageOutcomePolicyConfig, stage: FailureStage, failure_kinds: tuple[str, ...]
) -> StageOutcomePolicyConfig:
    """Reject a configuration whose decision for one of the boundary's kinds is not one it can perform.

    These boundaries mask the row and either let the harness run it again (``retry``) or leave it ungraded
    (``fail``). ``count_as_zero`` would save a masked row that is not a counted zero, so it is refused at load.
    Recovery never reruns a row attributed to the model, so ``retry`` for a kind classified as ``model`` would
    leave the masked row final under a decision that says otherwise: that pairing is refused too.
    """
    for kind in failure_kinds:
        decision = choose_stage_outcome(failure_kind=kind, http_status=None, stage=stage, config=policy)
        if decision.action not in (OutcomeAction.RETRY, OutcomeAction.FAIL):
            raise ValueError(f"{kind} supports only retry or fail at the {stage} stage, not {decision.action}")
        if decision.action == OutcomeAction.RETRY and decision.attribution == FailureClass.MODEL:
            raise ValueError(
                f"{kind} cannot be retried at the {stage} stage while it is classified as {decision.attribution}"
            )
    return policy


def applied_outcome_fields(
    *, failure_kind: str, stage: FailureStage, config: StageOutcomePolicyConfig
) -> dict[str, Any]:
    """The decision the shared policy takes for an observed failure, as the fields a row saves it in.

    The action, its attribution and the originating stage are saved together; the observed failure kind stays the
    caller's own field, independent of the action.
    """
    decision = choose_stage_outcome(failure_kind=failure_kind, http_status=None, stage=stage, config=config)
    return {
        "failure_outcome_action": decision.action,
        "failure_outcome_attribution": decision.attribution,
        "_ng_failure_stage": stage,
    }
