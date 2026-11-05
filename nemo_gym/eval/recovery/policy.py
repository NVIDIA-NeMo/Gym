# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The recovery outcome policy: which saved failure is retried, and the failure kind it is selected on."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import Field, ValidationError, model_validator

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    OutcomeAction,
    OutcomePolicyConfig,
    choose_action,
)
from nemo_gym.core.failure_kinds import (
    COLLECTION_DISPATCH_FAILED,
    EPISODE_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    JUDGE_FAILED,
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
)
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import FailureReportRow
from nemo_gym.eval.recovery.decision import (
    COLLECTED_CAUSE,
    FAILURE_EVENTS,
    FAILURE_ID,
    TERMINAL,
    IdentityState,
)


class RecoveryOutcomePolicyConfig(OutcomePolicyConfig):
    """Shared policy for eligible recovery failures. Retry permits another episode within the existing guards;
    verify re-judges a saved judge answer; fail leaves the failure ungraded. Settled outcomes are immutable.
    Unsupported context/action combinations stop, and explicit-zero scoring is not supported here.
    """

    # Infrastructure failures retry whatever the default action is: overrides merge into these entries, and each
    # retry still spends the identity's attempt cap and the run's infra-retry cap.
    by_failure_kind: dict[str, OutcomeAction] = Field(
        default_factory=lambda: {
            JUDGE_FAILED: OutcomeAction.VERIFY,
            COLLECTION_DISPATCH_FAILED: OutcomeAction.RETRY,
            HARNESS_TERMINATION_UNCONFIRMED: OutcomeAction.RETRY,
        }
    )
    default_action: OutcomeAction = OutcomeAction.RETRY

    @model_validator(mode="after")
    def require_recovery_actions(self) -> "RecoveryOutcomePolicyConfig":
        """Recovery cannot score an explicit zero, so ``count_as_zero`` is refused at configuration load."""
        if OutcomeAction.COUNT_AS_ZERO in {
            self.default_action,
            *self.by_failure_kind.values(),
            *self.by_status_code.values(),
        }:
            raise ValueError("recovery outcome policy does not support count_as_zero")
        return self


class OutcomePolicySource(StrEnum):
    """Where the recovery outcome policy in effect came from."""

    DEFAULT = "default"
    CONFIG = "config"
    PYTHON = "python"


def configured_outcome_policy(overrides: Mapping[str, Any] | None) -> RecoveryOutcomePolicyConfig | None:
    """A benchmark's ``resumable.recovery_outcome_policy`` section as a policy, or None when it is not set.

    It is a partial override: ``RecoveryOutcomePolicyConfig`` merges the entries it names into its defaults, and
    validates them exactly as the Python API does. A bad section raises, naming each problem.
    """
    if overrides is None:
        return None
    try:
        return RecoveryOutcomePolicyConfig.model_validate(overrides)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'policy'}: {e['msg'].removeprefix('Value error, ')}"
            for e in error.errors()
        )
        raise ValueError(f"invalid recovery_outcome_policy: {problems}") from None


def effective_outcome_policy(
    explicit: RecoveryOutcomePolicyConfig | None, overrides: Mapping[str, Any] | None
) -> tuple[RecoveryOutcomePolicyConfig, OutcomePolicySource]:
    """The policy recovery applies and its source. Precedence: an explicit Python ``outcome_policy`` argument
    replaces the run's saved config section (it is not merged with it); the section overrides the defaults. The
    budget options (``extend_retries``, ``max_walltime``) are separate and never change the policy."""
    if explicit is not None:
        return explicit, OutcomePolicySource.PYTHON
    configured = configured_outcome_policy(overrides)
    if configured is not None:
        return configured, OutcomePolicySource.CONFIG
    return RecoveryOutcomePolicyConfig(), OutcomePolicySource.DEFAULT


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


def _sidecar_action(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> OutcomeAction:
    return choose_action(
        failure_kind=_retry_kind(row) or "unknown", http_status=_policy_evidence(row)[1], config=policy
    )


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


def _judge_state(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> "IdentityState":
    """A judge failure's treatment: verify re-judges the saved answer only (none saved: stop), fail leaves it
    ungraded; any other action would run the policy again, so it stops."""
    if row.get(TERMINAL) is True:
        return IdentityState.TERMINAL
    kind, status = _policy_evidence(row)
    action = choose_action(failure_kind=kind or JUDGE_FAILED, http_status=status, config=policy)
    if action == OutcomeAction.VERIFY:
        return IdentityState.JUDGE_RECOVERABLE if _saved_answer(row) else IdentityState.UNKNOWN
    return IdentityState.TERMINAL if action == OutcomeAction.FAIL else IdentityState.UNKNOWN


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


def _replay_refusal(row: Mapping[str, Any]) -> str | None:
    """Why replay needs a saved-answer path or model-budget accounting unavailable to recovery; None when it does not."""
    from nemo_gym.rollouts.rollout_recovery import (
        has_recorded_grade,
        has_saved_generation,
        has_unclassified_execution_failure,
        is_infrastructure_agent_timeout,
        keeps_partial_response,
    )

    # An agent timeout whose window held a server-side fault kept no answer or grade of the model's: it runs again.
    infrastructure = is_infrastructure_agent_timeout(row)
    if not infrastructure and (has_saved_generation(row) or _saved_answer(row) or keeps_partial_response(row)):
        return "the row kept an answer that a rerun would replace"
    if not infrastructure and has_recorded_grade(row):
        return "the verifier recorded a grade that a rerun would replace"
    if has_unclassified_execution_failure(row):
        return "its recorded cause does not permit another agent run"
    chain = _cause_chain(row)
    if chain is None:
        # A legacy row may carry observations without selecting one; they are not its causal failure.
        return "its selected failure event is missing" if row.get(FAILURE_ID) is not None else None
    if any(
        boundary == FailureBoundary.RESOURCES_SERVER and operation == "verify" for boundary, operation, *_ in chain
    ):
        return "verification failed after the agent answered"
    if (
        bool(chain)
        and chain[0][0] == FailureBoundary.COLLECTION
        and any(boundary == FailureBoundary.MODEL for boundary, *_ in chain)
    ):
        return "a model failure inside collection has no saved model-attempt budget"
    return None


def _policy_replay_is_unsupported(row: Mapping[str, Any]) -> bool:
    """Whether replay needs a saved-answer path or model-budget accounting unavailable to recovery."""
    return _replay_refusal(row) is not None


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
    return value if isinstance(value, str) else None


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
    from nemo_gym.rollouts.rollout_recovery import holds_response

    facts = row.get(COLLECTED_CAUSE)
    facts = facts if isinstance(facts, Mapping) else {}
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
