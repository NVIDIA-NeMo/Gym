# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy."""

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    FAILURE_KINDS,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_OOM_KILLED,
    FailureClassificationConfig,
    classify_failure,
)
from nemo_gym.core.failure_records import FailureBoundary, FailureReport, ModelFailureDetails

# Compatibility imports for existing agent/environment policy consumers.
from nemo_gym.core.outcome_policy import OutcomeAction as OutcomeAction
from nemo_gym.core.outcome_policy import OutcomePolicyConfig as OutcomePolicyConfig
from nemo_gym.core.outcome_policy import choose_action as choose_action
from nemo_gym.core.outcome_policy import failure_outcome_policy as failure_outcome_policy
from nemo_gym.server_runtime.base_environment_server import is_retryable_dependency_status


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


class AgentOutcomePolicyConfig(OutcomePolicyConfig):
    """Existing Environment Server defaults for classified failed agent responses."""

    supported_actions: ClassVar[frozenset[OutcomeAction]] = frozenset(
        {OutcomeAction.VERIFY, OutcomeAction.RETRY, OutcomeAction.FAIL}
    )
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)

    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            AGENT_INVALID_OUTPUT: OutcomeAction.VERIFY,
            AGENT_LIMIT: OutcomeAction.VERIFY,
            MODEL_TRANSPORT_ERROR: OutcomeAction.RETRY,
            PROVIDER_OOM_KILLED: OutcomeAction.VERIFY,
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


def agent_outcome_policy(
    response: "NeMoGymResponse",
    *,
    config: AgentOutcomePolicyConfig | None = None,
    classification: FailureClassificationConfig | None = None,
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
    return choose_action(
        failure_kind=outcome.failure_kind,
        http_status=outcome.status_code,
        config=config,
        failure_class=classify_failure(
            failure_kind=outcome.failure_kind,
            http_status=outcome.status_code,
            config=classification or config.classification,
        ),
    )


def confirmed_policy_response_timeout(report: FailureReport | None) -> bool:
    """Whether the selected model operation timed out after upload on every observed attempt.

    Unidentified, mixed, omitted or connecting attempts do not establish a final response timeout.
    """
    if report is None or report.omitted or report.failure_id is None:
        return False
    final = report.events.get(report.failure_id)
    if (
        final is None
        or final.boundary != FailureBoundary.MODEL
        or final.operation != "model_request"
        or not final.operation_id
        or not final.attempt_id
    ):
        return False
    observed = [
        event
        for event in report.events.values()
        if event.boundary == FailureBoundary.MODEL and event.operation == "model_request"
    ]
    # An unidentified attempt cannot be proven to belong to an earlier operation instead of the final one.
    if any(not event.operation_id or not event.attempt_id for event in observed):
        return False
    attempts = [event for event in observed if event.operation_id == final.operation_id]
    return (
        bool(attempts)
        and len({event.attempt_id for event in attempts}) == len(attempts)
        and all(
            event.attempt_id
            and isinstance(event.details, ModelFailureDetails)
            and event.details.request_upload_complete is True
            and event.failure_reason == MODEL_RESPONSE_TIMEOUT_REASON
            for event in attempts
        )
    )
