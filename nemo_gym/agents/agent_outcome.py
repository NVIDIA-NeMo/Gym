# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy."""

from typing import TYPE_CHECKING, Annotated, ClassVar

from aiohttp import ClientResponseError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    FAILURE_KINDS,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_SERVER_QUEUE_TIMEOUT,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
    PROVIDER_INTERNAL_ERROR,
    PROVIDER_IO_TIMEOUT,
    PROVIDER_OOM_KILLED,
    PROVIDER_PROTOCOL_ERROR,
    PROVIDER_PTY_UNAVAILABLE,
    PROVIDER_REQUEST_FAILED,
    PROVIDER_STATUS_UNAVAILABLE,
    PROVIDER_UNAVAILABLE,
    FailureClassificationConfig,
    classify_failure,
)
from nemo_gym.core.failure_records import FailureBoundary, FailureReport, ModelFailureDetails
from nemo_gym.core.failure_transport import FailureHeader

# Compatibility imports for existing agent/environment policy consumers.
from nemo_gym.core.outcome_policy import OutcomeAction as OutcomeAction
from nemo_gym.core.outcome_policy import OutcomePolicyConfig as OutcomePolicyConfig
from nemo_gym.core.outcome_policy import choose_action as choose_action
from nemo_gym.core.outcome_policy import failure_outcome_policy as failure_outcome_policy
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.server_runtime.base_environment_server import is_retryable_dependency_status


if TYPE_CHECKING:
    from nemo_gym.core.responses_api_types import NeMoGymResponse


AGENT_OUTCOME_METADATA_KEY = "ng_agent_outcome"
# Response metadata a harness adapter sets to "true" when the last prompt made at least as many model calls as the
# harness's own step limit allows. The harness may only ask for a final answer at that limit (OpenCode adds a note and
# still offers the tools), so the mark says the limit was reached, not how the model answered. The environment decides
# whether a marked answer is scored.
HARNESS_STEP_LIMIT_METADATA_KEY = "harness_step_limit_reached"


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
            MODEL_SERVER_QUEUE_TIMEOUT: OutcomeAction.RETRY,
            # Transient sandbox-provider failures reported by an agent: the episode runs again, which is not a
            # replay of the uncertain operation inside the provider.
            **dict.fromkeys(
                (
                    PROVIDER_UNAVAILABLE,
                    PROVIDER_IO_TIMEOUT,
                    PROVIDER_STATUS_UNAVAILABLE,
                    PROVIDER_COMMAND_OUTCOME_UNKNOWN,
                    PROVIDER_PROTOCOL_ERROR,
                    PROVIDER_REQUEST_FAILED,
                    PROVIDER_INTERNAL_ERROR,
                    PROVIDER_PTY_UNAVAILABLE,
                ),
                OutcomeAction.RETRY,
            ),
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


def queued_policy_response_timeout(report: FailureReport | None) -> bool:
    """Whether a confirmed policy response timeout (``confirmed_policy_response_timeout``) waited in the model
    server's queue: every attempt recorded the server's queue at its timeout, with requests waiting.

    An attempt without a queue reading, or with an empty queue, keeps the timeout the model's.
    """
    if not confirmed_policy_response_timeout(report):
        return False
    final = report.events[report.failure_id]
    attempts = [
        event
        for event in report.events.values()
        if event.boundary == FailureBoundary.MODEL
        and event.operation == "model_request"
        and event.operation_id == final.operation_id
    ]
    return all(
        isinstance(event.details, ModelFailureDetails)
        and event.details.server_requests_waiting is not None
        and event.details.server_requests_waiting > 0
        for event in attempts
    )


# The classified policy-request timeout whose outcome an agent boundary selects.
POLICY_TIMEOUT_OPERATION = "agent.policy_request"


def policy_timeout_failure(error: ClientResponseError) -> NgFailureEvent | None:
    """The agent's policy-timeout failure for a failed model reply, or ``None``.

    The kind is ``MODEL_SERVER_QUEUE_TIMEOUT`` when every attempt waited in the model server's queue
    (``queued_policy_response_timeout``), else ``MODEL_POLICY_RESPONSE_TIMEOUT``.

    The reply's report must show that the final model call timed out after its upload on every attempt
    (``confirmed_policy_response_timeout``). The failure is caused by ``error``, so its reply keeps the model's
    attempts; raise it ``from error``.
    """
    report = FailureHeader.decode(error.headers)
    if not confirmed_policy_response_timeout(report):
        return None
    # Waiting in the model server's queue is that server's capacity, not the model's answer.
    queued = queued_policy_response_timeout(report)
    failure = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=MODEL_SERVER_QUEUE_TIMEOUT if queued else MODEL_POLICY_RESPONSE_TIMEOUT,
        failure_reason=(
            "Policy model request timed out on every attempt while the model server queued requests"
            if queued
            else "Policy model request timed out on every attempt"
        ),
        operation=POLICY_TIMEOUT_OPERATION,
        operation_id=report.events[report.failure_id].operation_id,
    )
    failure.__cause__ = error
    return failure
