# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-reported causes, independent of environment grading and retry policy.

The outcome ladder (``RunEvidence`` -> ``decide_outcome``) turns what an adapter observed about one harness run into
the response's status and failure kind, the same way for every harness. ``SimpleResponsesAPIAgent.finish_response``
applies it to a response. The Environment Servers then choose verify, retry or fail from that response with
``agent_response_action``.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from aiohttp import ClientConnectorError, ClientResponseError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nemo_gym.client.openai_client import MODEL_RESPONSE_TIMEOUT_REASON
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    AGENT_RUN_ERROR,
    FAILURE_KINDS,
    MODEL_CONTEXT_LIMIT,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_REQUEST_REJECTED,
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
from nemo_gym.core.typed_exceptions import NgFailureEvent, rebuild_failure
from nemo_gym.models.token_capture.records import CONTEXT_LENGTH_EXCEEDED_ERROR_CODE
from nemo_gym.server_runtime.base_environment_server import is_retryable_dependency_status


if TYPE_CHECKING:
    from nemo_gym.core.responses_api_types import NeMoGymResponse


AGENT_OUTCOME_METADATA_KEY = "ng_agent_outcome"
# Response metadata naming the outcome ladder's row that decided the response (``RunDecision.stop_reason``).
AGENT_STOP_REASON_METADATA_KEY = "ng_agent_stop_reason"


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
            # The ladder reports a context limit as incomplete; a failed response that names one is graded the same.
            MODEL_CONTEXT_LIMIT: OutcomeAction.VERIFY,
            # No usable result is not the model's answer, and the harness may succeed on a fresh sandbox.
            AGENT_RUN_ERROR: OutcomeAction.RETRY,
            # The model never answered a request the run's setup made invalid: not graded, and a retry fails the same.
            MODEL_REQUEST_REJECTED: OutcomeAction.FAIL,
            MODEL_TRANSPORT_ERROR: OutcomeAction.RETRY,
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


# The classified policy-request timeout whose outcome an agent boundary selects.
POLICY_TIMEOUT_OPERATION = "agent.policy_request"


def policy_timeout_failure(error: ClientResponseError) -> NgFailureEvent | None:
    """The agent's policy-timeout failure for a failed model reply, or ``None``.

    The reply's report must show that the final model call timed out after its upload on every attempt
    (``confirmed_policy_response_timeout``). The failure is caused by ``error``, so its reply keeps the model's
    attempts; raise it ``from error``.
    """
    report = FailureHeader.decode(error.headers)
    if not confirmed_policy_response_timeout(report):
        return None
    failure = NgFailureEvent(
        boundary=FailureBoundary.AGENT,
        failure_kind=MODEL_POLICY_RESPONSE_TIMEOUT,
        failure_reason="Policy model request timed out on every attempt",
        operation=POLICY_TIMEOUT_OPERATION,
        operation_id=report.events[report.failure_id].operation_id,
    )
    failure.__cause__ = error
    return failure


# --- The outcome ladder: one harness run's evidence -> the response's status and failure kind ------------------- #

StopReason = Literal[
    "completed",
    "stopped",
    "timeout",
    "context_limit",
    "output_limit",
    "agent_limit",
    "refusal",
    "error",
    "infra",
    "no_result",
]
"""Which row of the ladder decided a run."""


@dataclass(frozen=True, kw_only=True)
class RunEvidence:
    """What an adapter observed about one harness run, read from the harness's own records.

    Each adapter fills this from its parser; ``decide_outcome`` then decides the response the same way for every
    harness. Use structured evidence (the supervisor's receipt, the harness's event log or status fields). Text
    matching belongs only in ``classify_outcome``, tested against a pinned runtime version.
    """

    has_result: bool
    """The harness produced a usable result: it exited cleanly and left its answer or its work. A nonzero exit or a
    missing result file is not one."""

    timed_out: bool
    """The run's deadline ended it, as the supervisor's receipt or a structured deadline reports. Never set from
    message text: a "timed out" message is ``model_transport_error`` evidence at most."""

    model_replied: bool | None
    """Whether any reply of the policy model was observed in this run; ``None`` when the adapter cannot tell."""

    stopped: bool = False
    """A rollout stop (``POST /rollouts/{rollout_id}/stop``) or the session's close ended the run before it finished.
    Only a rollout stop gains the ``ng_rollout_stop`` marker (``finish_response``)."""

    context_limit: bool = False
    """The harness or the model endpoint explicitly reported that the model's context window was exceeded. An HTTP
    400 or 413 alone is not this evidence."""

    model_limit: Literal["output", "turns"] | None = None
    """The run stopped at the model's output limit or at the harness's configured turn or step limit."""

    refusal: bool = False
    """The model refused the task. The refusal is the model's answer and is graded."""

    error: AgentOutcome | None = None
    """``classify_outcome()`` of the error the run ended on, if it ended on one. An error the harness recovered from
    (a retried model call, a subagent's error) is not this."""

    infra_fault: NgFailureEvent | None = None
    """A recorded server or provider fault in the run's window, such as a model call that failed with a 5xx, a
    transport error or a queue timeout, or a sandbox provider failure. It decides only a timed-out run: a deadline
    spent on our servers is retried instead of graded."""

    retry_deadline_before_reply: bool = False
    """Keeps today's retry for a deadline that ended the run before any model reply, for an adapter whose harness
    retried such a run until now (its harness retries failed model calls without reporting them, so an outage cannot
    be told from a slow model). It goes away once every model call carries ``x-session-id`` and capture can supply
    ``infra_fault`` for every harness: a deadline is then graded unless ``infra_fault`` is set."""


@dataclass(frozen=True)
class RunDecision:
    """The response the ladder decided for one run."""

    status: Literal["completed", "incomplete", "failed"]
    stop_reason: StopReason
    outcome: AgentOutcome | None = None
    """Written to ``ng_agent_outcome``; set exactly when ``status`` is ``failed``."""


def _fault_outcome(fault: NgFailureEvent) -> AgentOutcome:
    """The outcome naming a recorded infrastructure fault, or ``unknown`` when its kind cannot be one."""
    try:
        return AgentOutcome(
            failure_kind=fault.failure_kind,
            status_code=fault.http_status if fault.failure_kind == MODEL_REQUEST_ERROR else None,
        )
    except ValidationError:
        return AgentOutcome(failure_kind="unknown")


def decide_outcome(evidence: RunEvidence) -> RunDecision:
    """Apply the outcome ladder: the first row that matches ``evidence`` decides.

    1. stopped -> incomplete
    2. context limit -> incomplete
    3. deadline with a recorded infrastructure fault -> failed with the fault's kind
    4. deadline -> incomplete (until the flip, a deadline before any reply that the adapter keeps retried -> failed
       ``unknown``)
    5. error -> failed with the classified outcome; a 4xx other than 408, 425 and 429 before any model reply ->
       ``model_request_rejected``
    6. output or turn limit -> incomplete
    7. refusal -> completed, graded as the model's answer
    8. a usable result -> completed
    9. otherwise -> failed ``agent_run_error`` (no usable result)

    The Environment Server's ``agent_outcome_policy`` then grades, retries or fails a failed response by its
    outcome; completed and incomplete responses are graded.
    """
    if evidence.stopped:
        return RunDecision("incomplete", "stopped")
    if evidence.context_limit or (evidence.error is not None and evidence.error.failure_kind == MODEL_CONTEXT_LIMIT):
        return RunDecision("incomplete", "context_limit")
    if evidence.timed_out:
        if evidence.infra_fault is not None:
            return RunDecision("failed", "infra", _fault_outcome(evidence.infra_fault))
        if evidence.retry_deadline_before_reply and evidence.model_replied is not True:
            return RunDecision("failed", "timeout", AgentOutcome(failure_kind="unknown"))
        return RunDecision("incomplete", "timeout")
    if evidence.error is not None:
        error = evidence.error
        if (
            error.failure_kind == MODEL_REQUEST_ERROR
            and error.status_code < 500
            and not is_retryable_dependency_status(error.status_code)
            and evidence.model_replied is False
        ):
            return RunDecision("failed", "error", AgentOutcome(failure_kind=MODEL_REQUEST_REJECTED))
        return RunDecision("failed", "error", error)
    if evidence.model_limit == "output":
        return RunDecision("incomplete", "output_limit")
    if evidence.model_limit == "turns":
        return RunDecision("incomplete", "agent_limit")
    if evidence.refusal:
        return RunDecision("completed", "refusal")
    if evidence.has_result:
        return RunDecision("completed", "completed")
    return RunDecision("failed", "no_result", AgentOutcome(failure_kind=AGENT_RUN_ERROR))


def model_http_error_outcome(status: int, *, error_body: bytes | str | None = None) -> AgentOutcome:
    """Classify a model request that failed with HTTP ``status``, for ``RunEvidence.error``.

    ``error_body`` is the endpoint's error body when the harness exposes it: an HTTP client's response content, or
    the raw JSON a CLI prints verbatim. A 400 whose body is OpenAI's error envelope with code
    ``context_length_exceeded`` is ``model_context_limit``: Gym's model servers answer a context overflow they
    propagate that way, and the ladder grades it as incomplete, before or after the first reply. Any other failure is
    ``model_request_error`` with its status.
    """
    if status == 400 and _openai_error_code(error_body) == CONTEXT_LENGTH_EXCEEDED_ERROR_CODE:
        return AgentOutcome(failure_kind=MODEL_CONTEXT_LIMIT)
    return AgentOutcome(failure_kind=MODEL_REQUEST_ERROR, status_code=status)


def _openai_error_code(body: bytes | str | None) -> str | None:
    """The ``error.code`` string of an OpenAI error envelope (``{"error": {"code": ...}}``), else ``None``."""
    try:
        error = json.loads(body or b"").get("error")
    except (TypeError, ValueError, AttributeError):
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None


# --- Environment Server side: what to do with an agent's response before grading ---------------------------------- #


def reported_agent_failure(response: "NeMoGymResponse") -> NgFailureEvent | None:
    """The failure a failed response's ``failure_report`` selects, rebuilt, when it matches the response's
    ``ng_agent_outcome``; else ``None``."""
    if response.status != "failed":
        return None
    try:
        outcome = AgentOutcome.model_validate_json((response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY, ""))
        report = FailureReport.model_validate((response.model_extra or {}).get("failure_report"))
    except (ValidationError, TypeError):
        return None
    if report.omitted or report.failure_id is None:
        return None
    selected = report.events[report.failure_id]
    if selected.failure_kind != outcome.failure_kind:
        return None
    if selected.http_status is not None and selected.http_status != outcome.status_code:
        return None
    if selected.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT:
        if (
            selected.boundary != FailureBoundary.AGENT
            or selected.operation != POLICY_TIMEOUT_OPERATION
            or not selected.operation_id
            or selected.cause_id is None
            or report.events[selected.cause_id].operation_id != selected.operation_id
            or not confirmed_policy_response_timeout(report.model_copy(update={"failure_id": selected.cause_id}))
        ):
            return None
    return rebuild_failure(report)


def model_request_not_sent(response: "NeMoGymResponse") -> bool:
    """Whether every attempt of the model request a response's report selects failed to connect, so the request was
    never sent."""
    report = FailureReport.model_validate(response.model_extra["failure_report"])
    selected = report.events[report.failure_id]
    if (
        selected.boundary != FailureBoundary.MODEL
        or selected.operation != "model_request"
        or not selected.operation_id
        or not selected.attempt_id
    ):
        return False
    observed = [
        event
        for event in report.events.values()
        if event.boundary == FailureBoundary.MODEL and event.operation == "model_request"
    ]
    if any(not event.operation_id or not event.attempt_id for event in observed):
        return False
    attempts = [event for event in observed if event.operation_id == selected.operation_id]
    connector_type = f"{ClientConnectorError.__module__}.{ClientConnectorError.__qualname__}"
    return len({event.attempt_id for event in attempts}) == len(attempts) and all(
        isinstance(event.details, ModelFailureDetails)
        and event.details.request_upload_complete is False
        and event.foreign_error is not None
        and event.foreign_error.exception_type == connector_type
        for event in attempts
    )


def agent_response_action(
    response: "NeMoGymResponse",
    *,
    config: AgentOutcomePolicyConfig,
    timeout_action: Callable[[OutcomeAction], OutcomeAction] | None = None,
) -> tuple[OutcomeAction, NgFailureEvent | None]:
    """The action an Environment Server takes on an agent's response before grading, and the failure its report selects.

    ``agent_outcome_policy`` decides from the response's ``ng_agent_outcome``. A failed response that also carries a
    ``failure_report`` must agree with it:

    - a report that selects a failure not matching the outcome fails the episode (inconsistent evidence);
    - a confirmed policy-response timeout takes ``timeout_action(action)`` when given;
    - a ``model_request_error`` whose selected event has no HTTP status (an enclosing 500, not a provider verdict)
      fails the episode unless every attempt failed to connect.

    The selected failure is the cause of the episode failure the caller raises, so the failure row keeps its kind.
    """
    action = agent_outcome_policy(response, config=config)
    cause = reported_agent_failure(response)
    if response.status == "failed" and AGENT_OUTCOME_METADATA_KEY in (response.metadata or {}):
        extra = response.model_extra or {}
        report = extra.get("failure_report")
        # A report that selects nothing (e.g. a provider 400 without a Gym failure) keeps the policy action.
        unselected = isinstance(report, dict) and report.get("failure_id") is None and not report.get("omitted")
        if cause is None and "failure_report" in extra and not unselected:
            action = OutcomeAction.FAIL
        elif cause is not None and cause.failure_kind == MODEL_POLICY_RESPONSE_TIMEOUT:
            if timeout_action is not None:
                action = timeout_action(action)
        elif cause is not None and cause.failure_kind == MODEL_REQUEST_ERROR and cause.http_status is None:
            # An enclosing HTTP 500 is not a provider verdict or evidence of an unexecuted request.
            action = action if model_request_not_sent(response) else OutcomeAction.FAIL
    return action, cause
