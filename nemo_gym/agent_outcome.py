# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy."""

from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from nemo_gym.base_environment_server import is_retryable_dependency_status
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    FAILURE_KINDS,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
)


if TYPE_CHECKING:
    from nemo_gym.core.responses_api_types import NeMoGymResponse


AGENT_OUTCOME_METADATA_KEY = "ng_agent_outcome"


class AgentOutcome(BaseModel):
    """Additional evidence distinguishing causes of a failed agent response.

    Ordinary completion and limits already use response.status. HTTP status here
    belongs to the upstream model request, not the agent endpoint's HTTP 200.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    failure_kind: str
    status_code: int | None = Field(default=None, ge=400, le=599, strict=True)

    @field_validator("failure_kind")
    @classmethod
    def require_registered_kind(cls, value: str) -> str:
        """Accept the shared vocabulary plus an explicitly undetermined cause."""
        if value != "unknown" and value not in FAILURE_KINDS:
            raise ValueError(f"Unregistered failure kind: {value}")
        return value

    @model_validator(mode="after")
    def require_request_status(self) -> "AgentOutcome":
        if (self.failure_kind == MODEL_REQUEST_ERROR) != (self.status_code is not None):
            raise ValueError("Only model_request_error requires an upstream HTTP status")
        return self


class OutcomeAction(StrEnum):
    """Mutually exclusive grading or episode-failure action."""

    VERIFY = "verify"
    RETRY = "retry"
    FAIL = "fail"


class AgentOutcomePolicyConfig(BaseModel):
    """Route classified failed responses; HTTP status rules take precedence over failure-kind rules.

    Supplied entries override matching defaults; omitted entries keep their defaults.
    Unmatched failures use default_action.
    Retry means an ungraded episode failure eligible for the collector's bounded retries.
    """

    model_config = ConfigDict(extra="forbid")

    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            AGENT_INVALID_OUTPUT: OutcomeAction.VERIFY,
            AGENT_LIMIT: OutcomeAction.VERIFY,
            MODEL_TRANSPORT_ERROR: OutcomeAction.RETRY,
            "unknown": OutcomeAction.RETRY,
        }
    )
    by_status_code: dict[Annotated[int, Field(ge=400, le=599)], OutcomeAction] = Field(
        default_factory=lambda: {
            **dict.fromkeys((400, 413, 422), OutcomeAction.VERIFY),
            **dict.fromkeys(filter(is_retryable_dependency_status, range(400, 600)), OutcomeAction.RETRY),
        }
    )
    default_action: OutcomeAction = OutcomeAction.FAIL

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


def agent_outcome_policy(
    response: "NeMoGymResponse", *, config: AgentOutcomePolicyConfig | None = None
) -> OutcomeAction:
    """Apply configured grading/retry rules to a classified failed response.

    Unclassified legacy responses retain their existing grading behavior; malformed
    metadata is an ungraded, nonretryable failure regardless of configuration.
    """
    raw = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    if response.status != "failed" or raw is None:
        return OutcomeAction.VERIFY
    try:
        outcome = AgentOutcome.model_validate_json(raw)
    except ValidationError:
        # Keep malformed classification in the partial response for diagnosis, not scoring.
        return OutcomeAction.FAIL
    config = config or AgentOutcomePolicyConfig()
    return config.by_status_code.get(
        outcome.status_code, config.by_failure_kind.get(outcome.failure_kind, config.default_action)
    )
