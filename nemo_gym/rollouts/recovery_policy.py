# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode recovery decisions shared by collection and orchestration.

Failure evidence is interpreted once; the Store enforces result ownership and attempt
budgets without inventing a second retry policy.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from nemo_gym.agents.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.core.failure_kinds import (
    COLLECTION_DISPATCH_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    JUDGE_FAILED,
    JUDGE_FAILURE_KINDS,
    JUDGE_UNPARSEABLE,
    KILL_SHAPED,
    MODEL_REQUEST_ERROR,
    FailureClassificationConfig,
    classify_failure,
    is_registered,
)
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import FailureReportRow
from nemo_gym.core.outcome_policy import OutcomeAction, OutcomePolicyConfig, choose_action
from nemo_gym.rollouts.rollout_recovery import (
    applied_failure_class as _applied_class,
)
from nemo_gym.rollouts.rollout_recovery import (
    failure_stage as _policy_stage,
)
from nemo_gym.rollouts.rollout_recovery import (
    is_terminal_failure,
    matches_failure_class,
)


FAILURE_EVENTS = "_ng_failure_events"
FAILURE_ID = "_ng_failure_id"


class RecoveryOutcomePolicyConfig(OutcomePolicyConfig):
    """Shared episode recovery policy. Retry permits another episode within the attempt budget;
    verify re-judges a saved judge answer; fail leaves the failure ungraded. Settled outcomes are immutable.
    Unsupported context/action combinations stop, and explicit-zero scoring is not supported here.
    """

    # Infrastructure failures retry whatever the default action is: overrides merge into these entries, and each
    # retry still spends the identity's attempt cap and the run's infra-retry cap.
    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            JUDGE_FAILED: OutcomeAction.VERIFY,
            JUDGE_UNPARSEABLE: OutcomeAction.VERIFY,
            KILL_SHAPED: OutcomeAction.FAIL,
            COLLECTION_DISPATCH_FAILED: OutcomeAction.RETRY,
            HARNESS_TERMINATION_UNCONFIRMED: OutcomeAction.RETRY,
        }
    )
    default_action: OutcomeAction = OutcomeAction.RETRY
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)

    supported_actions = frozenset({OutcomeAction.VERIFY, OutcomeAction.RETRY, OutcomeAction.FAIL})


def _legacy_agent_outcome(row: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """Read the final agent classification without selecting an unbound diagnostic event.

    Older environments returned a failed agent response but did not select its report's cause. Support only a
    delivered agent-stage outcome with one matching observed model rejection. An internal wrapper's 500, mixed
    history, incomplete reports and saved completed answers cannot establish an upstream HTTP rejection.
    """
    record = row.get("_ng_failure_record")
    if (
        not isinstance(record, Mapping)
        or record.get("source") != "environment"
        or record.get("delivery") != "delivered"
    ):
        return None, None
    failure = record.get("failure")
    if not isinstance(failure, Mapping) or failure.get("stage") != "agent":
        return None, None
    answer = row.get("_ng_failure_partial_response")
    if not isinstance(answer, Mapping) or answer.get("object") != "response" or answer.get("status") != "failed":
        return None, None
    metadata = answer.get("metadata")
    raw = metadata.get(AGENT_OUTCOME_METADATA_KEY) if isinstance(metadata, Mapping) else None
    if not isinstance(raw, str):
        return None, None
    try:
        outcome = AgentOutcome.model_validate_json(raw)
    except ValidationError:
        return None, None
    report = FailureReportRow.report(row)
    if report.omitted or report.failure_id is not None or len(report.events) != 1:
        return None, None
    event = next(iter(report.events.values()))
    if (
        outcome.failure_kind != MODEL_REQUEST_ERROR
        or event.failure_kind != outcome.failure_kind
        or event.http_status != outcome.status_code
        or event.boundary != FailureBoundary.MODEL
        or event.operation != "model_request"
        or not event.operation_id
        or not event.attempt_id
        or event.cause_id is not None
    ):
        return None, None
    return outcome.failure_kind, outcome.status_code


def _policy_evidence(row: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """The failure kind and HTTP status the policy selects on: the event ``failure_id`` selects first, then the
    row's own producer fields, then a validated final legacy agent outcome. Unselected events are not causes."""
    events = row.get(FAILURE_EVENTS)
    selected_id = row.get(FAILURE_ID)
    selected = events.get(selected_id) if isinstance(events, Mapping) and isinstance(selected_id, str) else None
    selected = selected if isinstance(selected, Mapping) else {}
    kind = selected.get("failure_kind") if isinstance(selected.get("failure_kind"), str) else row.get("failure_kind")
    status = next(
        (v for v in (selected.get("http_status"), row.get("_ng_failure_http_status")) if type(v) is int), None
    )
    if FAILURE_ID not in row and row.get("failure_kind") is None and row.get("_ng_failure_http_status") is None:
        return _legacy_agent_outcome(row)
    return (kind if isinstance(kind, str) else None), status


def _recovery_action(
    row: Mapping[str, Any], kind: str, policy: RecoveryOutcomePolicyConfig, *, http_status: int | None
) -> OutcomeAction:
    """Apply recovery rules to shared classification without changing the selected evidence or budgets."""
    classification = _applied_class(row)
    if classification is None:
        classification = classify_failure(
            failure_kind=kind, stage=_policy_stage(row), http_status=http_status, config=policy.classification
        )
    return choose_action(failure_kind=kind, http_status=http_status, failure_class=classification, config=policy)


def _sidecar_action(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> OutcomeAction:
    return _recovery_action(row, _retry_kind(row) or "unknown", policy, http_status=_policy_evidence(row)[1])


def _retry_kind(row: Mapping[str, Any]) -> str | None:
    """The kind a retry decision selects on: the policy evidence, else, for a row that selects no event, the one kind
    every saved event names, else the failure record's own kind. Final-zero and judge decisions do not use it."""
    from nemo_gym.rollouts.rollout_recovery import holds_response

    kind = _policy_evidence(row)[0]
    if kind is not None or FAILURE_ID in row:
        return kind
    events = row.get(FAILURE_EVENTS)
    kinds = (
        {e.get("failure_kind") for e in events.values() if isinstance(e, Mapping)}
        if isinstance(events, Mapping)
        else set()
    )
    # A row that keeps a producer response reports its final outcome there (a validated agent outcome): its
    # unselected events are attempts, not causes, and never stand in for an outcome that failed validation.
    if len(kinds) == 1 and isinstance(next(iter(kinds)), str) and not holds_response(row):
        return next(iter(kinds))
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    value = failure.get("failure_kind") if isinstance(failure, Mapping) else None
    if isinstance(value, str):
        return value
    value = row.get("_ng_failure_class")
    return value if isinstance(value, str) and is_registered(value) else None


def episode_recovery_action(
    row: Mapping[str, Any],
    *,
    policy: RecoveryOutcomePolicyConfig,
    retry_terminal_timeouts: bool = False,
) -> OutcomeAction:
    """Choose the action for a failed attempt, including failures with no classified cause.

    A persisted terminal decision stays terminal unless the existing timeout override
    applies. Unknown execution and incomplete diagnostic reports are evidence, not
    independent retry vetoes. Saved-answer ownership is checked by the executor.
    """
    if is_terminal_failure(row, retry_terminal_timeouts=retry_terminal_timeouts):
        return OutcomeAction.FAIL
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    stage = failure.get("stage") if isinstance(failure, Mapping) else None
    not_started = (
        isinstance(record, Mapping)
        and stage != "verification"
        and (record.get("delivery") == "not_sent" or stage == "admission")
    )
    if row.get("_ng_failure_legacy_class") == KILL_SHAPED and not not_started:
        return _recovery_action(row, KILL_SHAPED, policy, http_status=_policy_evidence(row)[1])
    if matches_failure_class(row, JUDGE_FAILURE_KINDS):
        return _recovery_action(row, JUDGE_FAILED, policy, http_status=_policy_evidence(row)[1])
    return _sidecar_action(row, policy)


# Compatibility with older metadata-only projections of a judge's saved answer.
SAVED_ANSWER_KEY = "_ng_collected_saved_answer"


def _saved_answer(row: Mapping[str, Any]) -> bool:
    if SAVED_ANSWER_KEY in row:
        return row[SAVED_ANSWER_KEY] is True
    from nemo_gym.rollouts.rollout_recovery import has_saved_generation

    # Match the saved input used by _rollout_response: diagnostic response bodies are not verifier inputs.
    # A present response takes precedence over the native judge-failure partial response, even if unusable.
    answer = row.get("response") if "response" in row else row.get("_ng_failure_partial_response")
    # The failure's stage stays with it, so this is the Store's predicate: an agent-stage response the outcome policy
    # rejected as failed is diagnostic output (an earlier tool turn, say), not an answer a rerun would replace.
    stage = {key: row[key] for key in ("_ng_failure_record", "_ng_failure_stage") if key in row}
    return has_saved_generation({"response": answer, **stage})


def _cause_chain(row: Mapping[str, Any]) -> list[tuple[str | None, ...]] | None:
    """The event ``failure_id`` selects and its explicit ``cause_id`` chain; other saved events are observations only.

    Legacy rows with no event report retain their explicit terminal classification. Otherwise a selected event
    and every explicit cause must exist; missing references and cycles remain unknown.
    """
    events = row.get(FAILURE_EVENTS)
    if row.get(FAILURE_ID) is None and not events:
        return []
    if (
        not isinstance(row.get(FAILURE_ID), str)
        or not isinstance(events, Mapping)
        or not isinstance(events.get(row.get(FAILURE_ID)), Mapping)
    ):
        return None
    chain: list[tuple[str | None, ...]] = []
    seen: set[str] = set()
    event_id = row.get(FAILURE_ID)
    while event_id is not None:
        event = events.get(event_id) if isinstance(event_id, str) else None
        if not isinstance(event, Mapping) or event_id in seen:
            return None
        seen.add(event_id)
        chain.append(tuple(event.get(k) for k in ("boundary", "operation", "operation_id", "attempt_id", "cause_id")))
        event_id = event.get("cause_id")
    return chain or None


def _replay_refusal(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """Why another episode would replace an answer or grade; None when it would not."""
    from nemo_gym.rollouts.rollout_recovery import (
        has_recorded_grade,
        has_saved_generation,
        is_infrastructure_agent_timeout,
        keeps_partial_response,
    )

    # An agent timeout whose window held a server-side fault kept no answer or grade of the model's: it runs again.
    infrastructure = is_infrastructure_agent_timeout(row, classification=classification)
    if not infrastructure and (has_saved_generation(row) or _saved_answer(row) or keeps_partial_response(row)):
        return "the row kept an answer that a rerun would replace"
    if not infrastructure and has_recorded_grade(row):
        return "the verifier recorded a grade that a rerun would replace"
    chain = _cause_chain(row)
    if chain and any(
        boundary == FailureBoundary.RESOURCES_SERVER and operation == "verify" for boundary, operation, *_ in chain
    ):
        return "verification failed after the agent answered"
    return None
