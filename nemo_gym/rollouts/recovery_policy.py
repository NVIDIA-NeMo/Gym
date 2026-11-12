# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode recovery decisions shared by collection and orchestration.

Failure evidence is interpreted once; the Store enforces result ownership and attempt
budgets without inventing a second retry policy.
"""

import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, ValidationError

from nemo_gym.agents.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.config.config_types import ConfigError
from nemo_gym.core.failure_kinds import (
    AGENT_ROUTE_FAILED,
    AGENT_TIMEOUT_INFRASTRUCTURE,
    COLLECTION_DISPATCH_FAILED,
    ENVIRONMENT_SERVER_FAILED,
    EPISODE_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    JUDGE_FAILED,
    JUDGE_FAILURE_KINDS,
    JUDGE_UNPARSEABLE,
    KILL_SHAPED,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    VERIFIER_ERROR,
    FailureClass,
    FailureClassificationConfig,
    classify_failure,
    is_registered,
)
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import FailureReportRow
from nemo_gym.core.outcome_policy import (
    OutcomeAction,
    OutcomeDecision,
    OutcomePolicyConfig,
    choose_action,
    is_policy_counted_zero,
)
from nemo_gym.rollouts.rollout_recovery import (
    applied_failure_class as _applied_class,
)
from nemo_gym.rollouts.rollout_recovery import (
    failure_stage as _policy_stage,
)
from nemo_gym.rollouts.rollout_recovery import (
    has_recorded_grade,
    has_saved_generation,
    holds_response,
    is_infrastructure_agent_timeout,
    is_terminal_failure,
    keeps_partial_response,
    matches_failure_class,
)


FAILURE_EVENTS = "_ng_failure_events"
FAILURE_ID = "_ng_failure_id"


class EpisodeOutcomePolicyConfig(OutcomePolicyConfig):
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
    saved_answer_default_action: Literal[OutcomeAction.VERIFY, OutcomeAction.FAIL] = OutcomeAction.VERIFY
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)

    supported_actions = frozenset({OutcomeAction.VERIFY, OutcomeAction.RETRY, OutcomeAction.FAIL})


# Keep the launcher's existing configuration/API name; this is the collection boundary's
# policy, not a second policy applied by resume.
RecoveryOutcomePolicyConfig = EpisodeOutcomePolicyConfig
EPISODE_OUTCOME_KEY = "_ng_episode_outcome"


def saved_episode_outcome(row: Mapping[str, Any]) -> OutcomeDecision | None:
    """Read a selected episode decision. Invalid saved decisions must not become new retries."""
    if EPISODE_OUTCOME_KEY not in row:
        return None
    try:
        return OutcomeDecision.model_validate(row[EPISODE_OUTCOME_KEY])
    except ValidationError as error:
        identity = (row.get("_ng_task_index"), row.get("_ng_rollout_index"))
        raise ConfigError(
            f"Invalid saved episode decision for rollout {identity}, attempt {row.get('_ng_attempt_index')}: {error}"
        ) from error


def episode_outcome(
    row: Mapping[str, Any], *, policy: EpisodeOutcomePolicyConfig, retry_terminal_timeouts: bool = False
) -> OutcomeDecision:
    """Consume a saved boundary decision, or select one for a failure lacking it.

    The collection boundary persists this existing core decision type beside the v1
    failure envelope. Producer stage decisions remain separate: retrying verification
    means re-judging its answer, not repeating the whole episode. Explicit producer
    fail/zero decisions remain settled. This boundary never creates a scored zero.
    """
    saved = saved_episode_outcome(row)
    if saved is not None:
        return saved
    kind = _retry_kind(row) or "unknown"
    if not is_registered(kind) or kind == ENVIRONMENT_SERVER_FAILED:
        kind = _cause_kind(row)[0] or kind
    status = _policy_evidence(row)[1]
    attribution = _episode_attribution(row, kind, policy, http_status=status)
    if is_policy_counted_zero(row) or (
        row.get("_ng_failure_counts_as_zero") is True and row.get("_ng_failure_terminal") is True
    ):
        action = OutcomeAction.COUNT_AS_ZERO
    elif row.get("failure_outcome_action") == OutcomeAction.FAIL:
        action = OutcomeAction.FAIL
    else:
        action = _choose_episode_action(row, policy=policy, retry_terminal_timeouts=retry_terminal_timeouts)
    return OutcomeDecision(action=action, attribution=attribution)


def episode_recovery_action(
    row: Mapping[str, Any], *, policy: EpisodeOutcomePolicyConfig, retry_terminal_timeouts: bool = False
) -> OutcomeAction:
    """The saved episode action, with the same policy as fallback for older rows."""
    return episode_outcome(row, policy=policy, retry_terminal_timeouts=retry_terminal_timeouts).action


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


def _episode_attribution(
    row: Mapping[str, Any], kind: str, policy: EpisodeOutcomePolicyConfig, *, http_status: int | None
) -> FailureClass:
    """Preserve applied attribution, including timeouts whose window was lost to server faults."""
    applied = _applied_class(row)
    if applied is not None:
        return applied
    if is_infrastructure_agent_timeout(row, classification=policy.classification):
        kind, http_status = AGENT_TIMEOUT_INFRASTRUCTURE, None
    return classify_failure(
        failure_kind=kind, stage=_policy_stage(row), http_status=http_status, config=policy.classification
    )


def _recovery_action(
    row: Mapping[str, Any], kind: str, policy: RecoveryOutcomePolicyConfig, *, http_status: int | None
) -> OutcomeAction:
    """Apply recovery rules to shared classification without changing the selected evidence or budgets."""
    saved = saved_episode_outcome(row)
    if saved is not None:
        return saved.action
    classification = _episode_attribution(row, kind, policy, http_status=http_status)
    return choose_action(failure_kind=kind, http_status=http_status, failure_class=classification, config=policy)


def _sidecar_action(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> OutcomeAction:
    return _recovery_action(row, _retry_kind(row) or "unknown", policy, http_status=_policy_evidence(row)[1])


def _retry_kind(row: Mapping[str, Any]) -> str | None:
    """The kind a retry decision selects on: the policy evidence, else, for a row that selects no event, the one kind
    every saved event names, else the failure record's own kind. Final-zero and judge decisions do not use it."""
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


def verification_recovery_action(row: Mapping[str, Any], *, policy: EpisodeOutcomePolicyConfig) -> OutcomeAction:
    """Consume the saved decision for an admitted saved-answer recovery, or apply its shared fallback."""
    saved = saved_episode_outcome(row)
    if saved is not None:
        return saved.action
    if row.get("failure_outcome_action") == OutcomeAction.FAIL or is_terminal_failure(row):
        return OutcomeAction.FAIL
    kind, status = _policy_evidence(row)
    if matches_failure_class(row, JUDGE_FAILURE_KINDS) and kind == COLLECTION_DISPATCH_FAILED:
        kind = JUDGE_FAILED
    saved_policy = policy.model_copy(update={"default_action": policy.saved_answer_default_action})
    return _recovery_action(row, kind or JUDGE_FAILED, saved_policy, http_status=status)


def _choose_episode_action(
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
    # A relay deadline ends its wait, not the upstream agent invocation. Without a
    # cancellation acknowledgement, another episode could overlap a still-running one.
    events, selected = row.get(FAILURE_EVENTS), row.get(FAILURE_ID)
    event = events.get(selected) if isinstance(events, Mapping) and isinstance(selected, str) else None
    chain = _cause_chain(row) or []
    if (
        isinstance(event, Mapping)
        and event.get("boundary") == FailureBoundary.ENVIRONMENT
        and event.get("operation") == "episode.deadline"
    ) or any(
        boundary == FailureBoundary.ENVIRONMENT and operation == "episode.deadline"
        for boundary, operation, *_ in chain
    ):
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
    if matches_failure_class(row, JUDGE_FAILURE_KINDS) or (
        _saved_answer(row)
        and (
            _policy_stage(row) == "verification"
            or (row.get("failure_kind") == VERIFIER_ERROR and _saved_model_patch(row))
            or _verify_transport_failure(row)
        )
    ):
        return verification_recovery_action(row, policy=policy)
    kind = _retry_kind(row)
    if not is_registered(kind) or kind == ENVIRONMENT_SERVER_FAILED:
        derived, model_final, _ = _cause_kind(row)
        if model_final:
            # An owned policy-model rejection is a final result, never another model attempt.
            return OutcomeAction.FAIL
        if derived is not None:
            return _recovery_action(row, derived, policy, http_status=_policy_evidence(row)[1])
    return _sidecar_action(row, policy)


# Compatibility with older metadata-only projections of a judge's saved answer.
SAVED_ANSWER_KEY = "_ng_collected_saved_answer"


def _saved_answer(row: Mapping[str, Any]) -> bool:
    if SAVED_ANSWER_KEY in row:
        return row[SAVED_ANSWER_KEY] is True
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


def _model_cause_facts(row: Mapping[str, Any]) -> dict:
    """The same bounded model-ownership facts carried by the remote evidence projection."""
    policy = row.get("ng_policy_model_server") or "policy_model"
    capture = row.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, Mapping) else None
    event_calls = {}
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, Mapping):
            continue
        report = call.get("failure_report")
        events = report.get("events") if isinstance(report, Mapping) else None
        if not isinstance(events, Mapping):
            continue
        ref = call.get("model_ref")
        owner = True if ref is None else ref.get("name") == policy if isinstance(ref, Mapping) else None
        context = "context_length_exceeded" in (call.get("local_response_reason"), call.get("error_category"))
        for event_id in events:
            if isinstance(event_id, str):
                event_calls[event_id] = {"policy": owner, "context_limit": context}
    record = row.get("_ng_failure_record")
    failure = record.get("failure") if isinstance(record, Mapping) else None
    return {
        "event_calls": event_calls,
        "episode_deadline": (
            isinstance(record, Mapping)
            and record.get("source") == "environment"
            and isinstance(failure, Mapping)
            and failure.get("failure_reason") == "Episode timed out"
            and not failure.get("failure_kind")
        ),
    }


def _cause_kind(row: Mapping[str, Any]) -> tuple[str | None, bool, str | None]:
    """The kind of a failure whose producer recorded none, or an unregistered one, from its recorded cause.

    Returns ``(kind, model_final, reason)``; the kind is None when the cause cannot be determined, with the reason.
    The cause is the event the row selects, else every event it saved, with the collector's facts of the model call
    that recorded each one (``COLLECTED_CAUSE``):

    - a model request that failed with HTTP 5xx or in transport is infrastructure: its own kind, retried under the cap;
    - an HTTP 4xx of the policy's own call is the model's final result: ``model_context_limit`` when the call recorded
      the context limit, else ``model_request_error``;
    - an HTTP 4xx of a harness call (a user simulator, a judge, any model that is not the policy) is evaluation-side:
      ``environment_episode_failed``, retried under the cap;
    - with no event, the environment's own episode deadline is evaluation-side too: ``environment_episode_failed``.

    As for a kind (``_retry_kind``), the unselected events of a row that keeps a response are attempts, not its cause,
    and a cause chain that is broken or omitted names no cause.
    """
    facts = row.get("_ng_collected_cause")
    facts = facts if isinstance(facts, Mapping) else _model_cause_facts(row)
    events, selected = row.get(FAILURE_EVENTS), row.get(FAILURE_ID)
    if isinstance(events, Mapping) and events:
        if row.get("_ng_failure_events_omitted") or isinstance(selected, str) and _cause_chain(row) is None:
            return None, False, "its recorded cause is incomplete"
        if not isinstance(selected, str) and holds_response(row):
            return None, False, "it kept a response, so its saved model failures are attempts, not its cause"
        ids = [selected] if isinstance(selected, str) else [key for key in events if isinstance(key, str)]
        chosen = [events.get(key) for key in ids]
        if not chosen or not all(
            isinstance(e, Mapping)
            and e.get("boundary") == FailureBoundary.MODEL
            and e.get("operation") == "model_request"
            for e in chosen
        ):
            return None, False, "its recorded cause is not a failed model request"
        statuses = [e.get("http_status") for e in chosen]
        if any(e.get("failure_kind") == MODEL_TRANSPORT_ERROR for e in chosen):
            return MODEL_TRANSPORT_ERROR, False, None
        if any(type(s) is int and 500 <= s <= 599 for s in statuses):
            return MODEL_REQUEST_ERROR, False, None
        if not all(type(s) is int and 400 <= s <= 499 for s in statuses):
            return None, False, "its failed model request has no HTTP status"
        calls = facts.get("event_calls") if isinstance(facts.get("event_calls"), Mapping) else {}
        owners = [calls.get(key) if isinstance(calls.get(key), Mapping) else {} for key in ids]
        if any(owner.get("policy") not in (True, False) for owner in owners):
            return None, False, "the model call that failed is not recorded, so whose call it was is unknown"
        if all(owner["policy"] is False for owner in owners):
            return EPISODE_FAILED, False, None
        if all(owner["policy"] is True for owner in owners):
            context = any(owner.get("context_limit") is True for owner in owners)
            return (MODEL_CONTEXT_LIMIT if context else MODEL_REQUEST_ERROR), True, None
        return None, False, "the failed model calls belong to both the policy and the harness"
    if facts.get("episode_deadline") is True:
        return EPISODE_FAILED, False, None
    return None, False, None


def _transport_kind(kind: Any) -> bool:
    return isinstance(kind, str) and kind.startswith("transport_")


def _verify_transport_chain(events: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    """Whether ``event`` and every explicit cause below it name only a lost verify call: an agent route failure or a
    transport kind, with no reply status. A verifier's own failure or an HTTP reply anywhere in the chain is not one."""
    seen: set[str] = set()
    current: Any = event
    while isinstance(current, Mapping):
        kind = current.get("failure_kind")
        foreign = current.get("foreign_error")
        if not (kind == AGENT_ROUTE_FAILED or _transport_kind(kind)):
            return False
        if current.get("http_status") is not None or (
            isinstance(foreign, Mapping) and foreign.get("http_status") is not None
        ):
            return False
        cause = current.get("cause_id")
        if cause is None:
            return True
        if not isinstance(cause, str) or cause in seen:
            return False
        seen.add(cause)
        current = events.get(cause)
    return False


def _verify_transport_failure(row: Mapping[str, Any]) -> bool:
    """Whether the row was masked because its verify call was lost in transport: its kind is an agent route failure
    or a transport kind, and a saved failure event of the ``verify`` operation has a transport-only cause chain."""
    kind = row.get("failure_kind")
    events = row.get(FAILURE_EVENTS)
    if not (kind == AGENT_ROUTE_FAILED or _transport_kind(kind)) or not isinstance(events, Mapping):
        return False
    return any(
        isinstance(event, Mapping) and event.get("operation") == "verify" and _verify_transport_chain(events, event)
        for event in events.values()
    )


def _saved_model_patch(row: Mapping[str, Any]) -> bool:
    """Whether a verifier collected and retained the intact patch needed for verification-only recovery."""
    facts = row.get("_ng_collected_grading_facts")
    if isinstance(facts, Mapping) and "saved_model_patch" in facts:
        return facts["saved_model_patch"] is True and facts.get("collected_patch") is True
    patch, digest = row.get("model_patch"), row.get("model_patch_sha256")
    return (
        isinstance(patch, str)
        and isinstance(digest, str)
        and hashlib.sha256(patch.encode("utf-8")).hexdigest() == digest
        and all(type(row.get(key)) in (int, float) for key in ("patch_collection_time_s", "verification_time_s"))
    )
