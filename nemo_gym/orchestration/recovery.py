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
"""Whether `gym eval resubmit` may queue a run again, decided from the run's persisted evidence only.

Every expected identity (``_ng_task_index``, ``_ng_rollout_index``) of the materialized inventory gets one state:

- ``scored``: a result row that is not a judge failure. Kept as saved; never re-run.
- ``model_final``: a failures-sidecar row marked ``_ng_failure_counts_as_zero``: the evaluated model's terminal failure,
  a scored zero that no infra retry may replace.
- ``judge_recoverable``: a result row of class ``judge_failed``. Its saved answer is re-judged by
  ``gym eval reverify --judge-failed-only``; queueing the policy again is never needed for it.
- ``retryable``: the last sidecar row is non-terminal and under its recorded attempt cap: a Gym resume retries it.
- ``exhausted``: the last sidecar row is non-terminal and at its attempt cap: only a resubmit gives it a fresh budget.
- ``terminal``: a terminal sidecar row that is not a scored zero; excluded, and the run cannot become complete.
- ``missing``: no row at all (interrupted or never started); a resume restarts its input.
- ``unknown``: a failure whose kind is unregistered or absent, or a row whose identity is invalid; nobody owns it.

The run-level action follows: ``stop`` for any unknown identity or a spent or unknown resubmit budget, ``noop`` when
nothing is left to do, ``reverify`` when only judge failures remain, else ``resubmit``. A decision never reports a run
as complete while an identity is terminal, missing or unknown.
"""

import hashlib
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.agent_outcome import OutcomeAction, OutcomePolicyConfig, choose_action
from nemo_gym.config.config_types import ConfigError
from nemo_gym.core.failure_kinds import (
    JUDGE_FAILED,
    MODEL_CONTEXT_LIMIT,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    is_namespaced,
    is_registered,
)
from nemo_gym.orchestration.executors.resume_script import ACCOUNTING_UNKNOWN_PREFIX
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.task_materialization import TASK_ID_FIELDS


TASK_INDEX = "_ng_task_index"
ROLLOUT_INDEX = "_ng_rollout_index"
FAILURE_CLASS = "_ng_failure_class"
TERMINAL = "_ng_failure_terminal"
COUNTS_AS_ZERO = "_ng_failure_counts_as_zero"
MAX_ATTEMPTS = "_ng_failure_max_attempts"
FAILURE_EVENTS = "_ng_failure_events"
FAILURE_ID = "_ng_failure_id"

Identity = tuple[int, int]


class IdentityState(StrEnum):
    SCORED = "scored"
    MODEL_FINAL = "model_final"
    JUDGE_RECOVERABLE = "judge_recoverable"
    RETRYABLE = "retryable"
    EXHAUSTED = "exhausted"
    TERMINAL = "terminal"
    MISSING = "missing"
    UNKNOWN = "unknown"


class RecoveryAction(StrEnum):
    NOOP = "noop"
    REVERIFY = "reverify"
    RESUBMIT = "resubmit"
    STOP = "stop"


class ManagedRecoveryFact(BaseModel):
    """Read-only facts from the authoritative Store, not a second retry policy."""

    identity: Identity
    attempts: int = Field(ge=0, strict=True)
    max_attempts: int = Field(ge=1, strict=True)
    pending: bool
    reserved: bool


class RecoveryBudget(BaseModel):
    """The run's stored infra-retry and walltime budget and what its job chain used, as the chain files record them.

    ``max_infra_retries`` and ``max_walltime_s`` come from the benchmark's saved ``resume_config``;
    ``infra_retries_used`` from ``.gym_infra_retries`` and ``walltime_used_s`` from ``.gym_accumulated_walltime``
    (see ``resume_script``). The chain records admitted retries and allows another only while
    ``infra_retries_used < max_infra_retries``. ``None`` means the evidence is unavailable and stops the decision.
    """

    max_infra_retries: int | None = Field(default=None, ge=0)
    infra_retries_used: int | None = Field(default=None, ge=0)
    max_walltime_s: float | None = Field(default=None, gt=0, description="None: the run sets no walltime cap.")
    walltime_capped: bool = Field(default=False, description="Whether resume_config sets max_walltime.")
    walltime_used_s: float | None = Field(default=None, ge=0)
    live_job: bool | None = Field(default=None, description="A job runs or waits in this job dir.")
    job_chain: list[str] = Field(default_factory=list, description="Job ids already run in this job dir.")


class RecoveryOutcome(BaseModel):
    """What ``recover`` did: ``dispatched`` names the one scheduler or judge action taken, None when nothing ran."""

    decision: "RecoveryDecision"
    ok: bool
    dispatched: str | None
    job_id: str | None = None
    unsupported: str | None = None
    # The Day-0 gate task entry (status, expected/scored samples, score, errors, warnings); None without evidence.
    gate_task: dict[str, Any] | None = None
    # What was known about MLflow: the local files, and the server status only as the caller observed it.
    mlflow: dict[str, Any] | None = None


class IdentityEvidence(BaseModel):
    identity: Identity
    state: IdentityState
    failure_kind: str | None = None
    failure_id: str | None = None
    attempts: int = 0
    max_attempts: int | None = None
    # (boundary, operation, operation_id, attempt_id, cause_id) of each saved failure event, in saved order.
    events: list[tuple[str | None, str | None, str | None, str | None, str | None]] = Field(default_factory=list)


class RecoveryDecision(BaseModel):
    action: RecoveryAction
    reason: str
    expected: int
    counts: dict[str, int]
    identities: dict[str, list[Identity]]
    evidence: list[IdentityEvidence]
    duplicate_results: list[Identity] = Field(default_factory=list)
    scheduler_live: bool = False

    @property
    def complete(self) -> bool:
        return (
            self.counts.get(IdentityState.SCORED, 0) + self.counts.get(IdentityState.MODEL_FINAL, 0) == self.expected
        )


def _identity(row: Mapping[str, Any]) -> Identity | None:
    task, rollout = row.get(TASK_INDEX), row.get(ROLLOUT_INDEX)
    if all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (task, rollout)):
        return task, rollout
    return None


def _known_kind(kind: Any) -> bool:
    return isinstance(kind, str) and (is_registered(kind) or is_namespaced(kind))


def _events(row: Mapping[str, Any]) -> list[tuple[str | None, str | None, str | None, str | None, str | None]]:
    events = row.get(FAILURE_EVENTS) or {}
    if not isinstance(events, Mapping):
        return []
    return [
        tuple(
            e.get(key) if isinstance(e.get(key), str) else None
            for key in ("boundary", "operation", "operation_id", "attempt_id", "cause_id")
        )
        for e in events.values()
        if isinstance(e, Mapping)
    ]


class RecoveryOutcomePolicyConfig(OutcomePolicyConfig):
    """Shared policy for eligible recovery failures. Retry permits another episode within the existing guards;
    verify re-judges a saved judge answer; fail leaves the failure ungraded. Settled outcomes are immutable.
    Unsupported context/action combinations stop, and explicit-zero scoring is not supported here.
    """

    by_failure_kind: dict[str, OutcomeAction] = Field(default_factory=lambda: {JUDGE_FAILED: OutcomeAction.VERIFY})
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


def _policy_evidence(row: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """The failure kind and HTTP status the policy selects on: the event ``failure_id`` selects first, then the
    row's own producer fields ``failure_kind`` and ``_ng_failure_http_status``. No other saved event is searched."""
    events = row.get(FAILURE_EVENTS)
    selected_id = row.get(FAILURE_ID)
    selected = events.get(selected_id) if isinstance(events, Mapping) and isinstance(selected_id, str) else None
    selected = selected if isinstance(selected, Mapping) else {}
    kind = selected.get("failure_kind") if isinstance(selected.get("failure_kind"), str) else row.get("failure_kind")
    status = next(
        (v for v in (selected.get("http_status"), row.get("_ng_failure_http_status")) if type(v) is int), None
    )
    return (kind if isinstance(kind, str) else None), status


def _sidecar_action(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> OutcomeAction:
    kind, status = _policy_evidence(row)
    return choose_action(failure_kind=kind or "unknown", http_status=status, config=policy)


# Compatibility with older metadata-only projections of a judge's saved answer.
SAVED_ANSWER_KEY = "_ng_collected_saved_answer"


def _saved_answer(row: Mapping[str, Any]) -> bool:
    if SAVED_ANSWER_KEY in row:
        return row[SAVED_ANSWER_KEY] is True
    from nemo_gym.rollout_recovery import has_saved_generation

    # Match the saved input used by _rollout_response: diagnostic response bodies are not verifier inputs.
    # A present response takes precedence over the native judge-failure partial response, even if unusable.
    answer = row.get("response") if "response" in row else row.get("_ng_failure_partial_response")
    return has_saved_generation({"response": answer})


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


def _policy_replay_is_unsupported(row: Mapping[str, Any]) -> bool:
    """Whether replay needs a saved-answer path or model-budget accounting unavailable to recovery."""
    from nemo_gym.rollout_recovery import has_saved_generation, has_unclassified_execution_failure

    if has_unclassified_execution_failure(row) or has_saved_generation(row) or _saved_answer(row):
        return True
    chain = _cause_chain(row)
    if chain is None:
        # A legacy row may carry observations without selecting one; they are not its causal failure.
        return row.get(FAILURE_ID) is not None
    return (
        any(
            boundary == FailureBoundary.RESOURCES_SERVER and operation == "verify" for boundary, operation, *_ in chain
        )
        or (
            bool(chain)
            and chain[0][0] == FailureBoundary.COLLECTION
            and any(boundary == FailureBoundary.MODEL for boundary, *_ in chain)
        )
        or (
            # Unconfirmed harness termination after the agent saved an answer: replay would ask the model again.
            any(
                boundary == FailureBoundary.HARNESS and operation == "harness.terminate"
                for boundary, operation, *_ in chain
            )
            and _saved_answer(row)
        )
    )


def decide_recovery(
    *,
    materialized: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    budget: RecoveryBudget,
    default_max_attempts: int = 3,
    resubmit_offset: int = 0,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    managed_selection: list[ManagedRecoveryFact] | None = None,
    planned_missing_continuation: bool = False,
) -> RecoveryDecision:
    """Classify every materialized identity and choose the run-level action.

    ``failures`` are every failures-sidecar row in file order (``_line`` is the row's line number when known). The
    standing row of a rollout is chosen as rollout collection chooses it (``_latest_failure_rows``): rows that
    ``is_non_superseding_rollout_row`` names are skipped, and an interrupted omitted row means the rollout is missing.
    Only rows at or after ``resubmit_offset`` count towards the attempt cap. The rows are only read.
    """
    from nemo_gym.rollout_collection import (
        NG_UNROUTED_RUN_ERROR_KEY,
        is_non_superseding_rollout_row,
        is_omitted_rollout_row,
        is_terminal_failure,
    )

    policy = outcome_policy or RecoveryOutcomePolicyConfig()
    managed = {fact.identity: fact for fact in managed_selection or []}

    expected: list[Identity] = []
    invalid_inventory = 0
    for row in materialized:
        identity = _identity(row)
        if identity is None:
            invalid_inventory += 1
        else:
            expected.append(identity)
    expected_set = set(expected)

    result_rows: dict[Identity, Mapping[str, Any]] = {}
    seen: Counter[Identity] = Counter()
    unknown_rows = 0
    for row in results:
        identity = _identity(row)
        if identity is None or identity not in expected_set:
            unknown_rows += 1
            continue
        seen[identity] += 1
        result_rows[identity] = row
    sidecar: dict[Identity, list[Mapping[str, Any]]] = {}
    for row in failures:
        identity = _identity(row)
        if identity is None or identity not in expected_set:
            unknown_rows += 1
            continue
        sidecar.setdefault(identity, []).append(row)

    evidence: list[IdentityEvidence] = []
    policy_stopped = False
    for identity in sorted(expected_set):
        if identity in result_rows:
            row = result_rows[identity]
            if row.get("_ng_omitted_rollout") is True:
                # A collector-written stand-in for a rollout that never produced a row: its producer's accepted-partial
                # semantics are not modelled here, so it is unsupported rather than scored or retried.
                evidence.append(
                    IdentityEvidence(
                        identity=identity, state=IdentityState.UNKNOWN, failure_kind="omitted_rollout_unsupported"
                    )
                )
                continue
            judge = row.get(FAILURE_CLASS) == JUDGE_FAILED
            reward = row.get("reward")
            graded = isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
            if not judge and (
                not graded or row.get("mask_sample") is True or row.get("evaluation_completed") is False
            ):
                evidence.append(
                    IdentityEvidence(
                        identity=identity,
                        state=IdentityState.UNKNOWN,
                        failure_kind=row.get("failure_kind") or "result_without_usable_grade",
                    )
                )
                continue
            state = _judge_state(row, policy) if judge else IdentityState.SCORED
            policy_stopped |= judge and state == IdentityState.TERMINAL
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=state,
                    failure_kind=row.get("failure_kind") if judge else None,
                    failure_id=row.get(FAILURE_ID) if isinstance(row.get(FAILURE_ID), str) else None,
                    events=_events(row),
                )
            )
            continue
        standing = [row for row in sidecar.get(identity, []) if not is_non_superseding_rollout_row(row)]
        unrouted = [row for row in sidecar.get(identity, []) if row.get(NG_UNROUTED_RUN_ERROR_KEY) is True]
        if not standing and unrouted:
            # A diagnostic does not spend a failure attempt. Only explicit not-sent evidence admits retry;
            # a possibly delivered request must not regenerate an answer whose response was lost.
            last = unrouted[-1]
            record = last.get("_ng_failure_record")
            not_sent = isinstance(record, Mapping) and record.get("delivery") == "not_sent"
            fact = managed.get(identity)
            spent = fact.attempts if fact else 0
            cap = fact.max_attempts if fact else default_max_attempts
            state = IdentityState.UNKNOWN
            if not_sent and not _policy_replay_is_unsupported(last):
                action = _sidecar_action(last, policy)
                if action == OutcomeAction.FAIL or last.get(TERMINAL) is True:
                    policy_stopped = True
                    state = IdentityState.TERMINAL
                elif action == OutcomeAction.RETRY:
                    state = IdentityState.RETRYABLE if spent < cap else IdentityState.EXHAUSTED
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=state,
                    attempts=spent,
                    max_attempts=cap,
                    failure_kind=last.get("failure_kind"),
                    failure_id=last.get(FAILURE_ID) if isinstance(last.get(FAILURE_ID), str) else None,
                    events=_events(last),
                )
            )
            continue
        if not standing:
            # No attempt ran, or the latest one was interrupted: the input restarts.
            evidence.append(IdentityEvidence(identity=identity, state=IdentityState.MISSING))
            continue
        last = standing[-1]
        # Spent attempts are counted as Collection counts them: an interrupted or omitted row ran no attempt.
        rows = [
            row
            for index, row in enumerate(standing)
            if (row.get("_line") if isinstance(row.get("_line"), int) else index) >= resubmit_offset
            and not is_omitted_rollout_row(row)
        ]
        kind = last.get("failure_kind")
        if is_omitted_rollout_row(last):
            # Only a recorded interruption restarts its input. An adapter's omitted outcome is not a result.
            interrupted = last.get(FAILURE_CLASS) == "kill_shaped"
            state = IdentityState.MISSING if interrupted else IdentityState.UNKNOWN
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=state,
                    failure_kind=kind,
                    failure_id=last.get(FAILURE_ID) if isinstance(last.get(FAILURE_ID), str) else None,
                    events=_events(last),
                )
            )
            continue
        fact = managed.get(identity)
        cap = (
            fact.max_attempts
            if fact
            else (last.get(MAX_ATTEMPTS) if isinstance(last.get(MAX_ATTEMPTS), int) else default_max_attempts)
        )
        spent = fact.attempts if fact else len(rows)
        if last.get(FAILURE_CLASS) == JUDGE_FAILED:
            # A judge that failed is re-judged from the saved answer, never by running the policy again.
            state = _judge_state(last, policy)
            policy_stopped |= state == IdentityState.TERMINAL
        elif last.get(COUNTS_AS_ZERO) is True:
            # Explicit scoring is required, but cannot make an unowned or infrastructure failure a model zero.
            model_cause, _ = _policy_evidence(last)
            if model_cause is None and fact and not last.get(FAILURE_ID) and not last.get(FAILURE_EVENTS):
                # Managed native outcomes persist the canonical kind as their sidecar class. Keep explicit
                # selected causes authoritative; this does not infer a kind for legacy or ambiguous evidence.
                model_cause = last.get(FAILURE_CLASS)
            model_kind = model_cause in {
                MODEL_CONTEXT_LIMIT,
                MODEL_REQUEST_ERROR,
                MODEL_TRANSPORT_ERROR,
                MODEL_POLICY_RESPONSE_TIMEOUT,
            }
            chain = _cause_chain(last)
            contradictory = chain is None or any(
                boundary not in (None, "model", "agent") or boundary == "agent" and operation != "agent.policy_request"
                for boundary, operation, *_ in chain
            )
            state = (
                IdentityState.MODEL_FINAL
                if model_kind and is_terminal_failure(last) and not contradictory
                else IdentityState.UNKNOWN
            )
        elif not _known_kind(_policy_evidence(last)[0]):
            state = IdentityState.UNKNOWN
        elif last.get(TERMINAL) is True:
            state = IdentityState.TERMINAL
        elif (action := _sidecar_action(last, policy)) == OutcomeAction.FAIL:
            # Unrestricted resume would still retry this nonterminal saved row.
            policy_stopped = True
            state = IdentityState.TERMINAL
        elif action == OutcomeAction.VERIFY or _policy_replay_is_unsupported(last):
            # Only a judge failure has a supported saved-answer path. A failed verification can retain a finished
            # agent answer, so replay must not generate another. For collection-wrapped model failures, the sidecar projection
            # has no durable model-operation budget accounting: replay must not silently replenish that budget.
            state = IdentityState.UNKNOWN
        elif spent < cap:
            state = IdentityState.RETRYABLE
        else:
            state = IdentityState.EXHAUSTED
        evidence.append(
            IdentityEvidence(
                identity=identity,
                state=state,
                failure_kind=kind,
                failure_id=last.get(FAILURE_ID) if isinstance(last.get(FAILURE_ID), str) else None,
                attempts=spent,
                max_attempts=cap,
                events=_events(last),
            )
        )

    for item in evidence:
        fact = managed.get(item.identity)
        if fact is None:
            continue
        item.attempts, item.max_attempts = fact.attempts, fact.max_attempts
        if item.state == IdentityState.MISSING and (fact.reserved or not fact.pending):
            # A reservation without an outcome is not proof that work never reached the agent.
            item.state = IdentityState.UNKNOWN
        elif item.state == IdentityState.RETRYABLE and not fact.pending:
            item.state = IdentityState.EXHAUSTED if fact.attempts >= fact.max_attempts else IdentityState.UNKNOWN

    by_state: dict[str, list[Identity]] = {}
    for item in evidence:
        by_state.setdefault(item.state.value, []).append(item.identity)
    counts = {state: len(ids) for state, ids in by_state.items()}
    duplicates = sorted(i for i, n in seen.items() if n > 1)

    def decision(action: RecoveryAction, reason: str) -> RecoveryDecision:
        return RecoveryDecision(
            action=action,
            reason=reason,
            expected=len(expected_set),
            counts=counts,
            identities=by_state,
            evidence=evidence,
            duplicate_results=duplicates,
        )

    if invalid_inventory or len(expected) != len(expected_set) or not expected_set:
        return decision(
            RecoveryAction.STOP, "the materialized inventory is empty, has invalid or duplicate identities"
        )
    if unknown_rows or duplicates:
        return decision(
            RecoveryAction.STOP,
            f"{unknown_rows} saved row(s) outside the inventory or invalid, "
            f"{len(duplicates)} duplicated result identity(ies)",
        )
    if counts.get(IdentityState.UNKNOWN):
        return decision(
            RecoveryAction.STOP,
            f"{counts[IdentityState.UNKNOWN]} failure(s) have no registered kind or no supported action, so nobody owns them",
        )
    # A live job is still writing the files read above, whatever they show now.
    if budget.live_job:
        live = decision(RecoveryAction.STOP, "a job of this run is still pending or running; it is not queued twice")
        return live.model_copy(update={"scheduler_live": True})
    if policy_stopped:
        return decision(
            RecoveryAction.STOP,
            "the policy stops an ungraded failure; mixed dispatch cannot enforce separate task decisions",
        )
    # Only what rollout collection would run again: a resume keeps the per-rollout attempts, so it gates an
    # exhausted identity (`_load_from_cache` maxed_out) and queueing for it alone runs nothing.
    needs_policy = sum(counts.get(s, 0) for s in (IdentityState.MISSING, IdentityState.RETRYABLE))
    if not needs_policy:
        if counts.get(IdentityState.JUDGE_RECOVERABLE):
            return decision(
                RecoveryAction.REVERIFY,
                f"{counts[IdentityState.JUDGE_RECOVERABLE]} judge failure(s) only: "
                "re-judge the saved answers with `gym eval reverify --judge-failed-only`",
            )
        if counts.get(IdentityState.TERMINAL):
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.TERMINAL]} terminal unscored failure(s) need an "
                "owner decision; the run is not complete",
            )
        if counts.get(IdentityState.EXHAUSTED):
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.EXHAUSTED]} identity(ies) used every attempt since the last resubmit, so a "
                "resume would run nothing; the run is not complete",
            )
        return decision(RecoveryAction.NOOP, "every identity is scored or a final model failure")
    if counts.get(IdentityState.JUDGE_RECOVERABLE):
        return decision(
            RecoveryAction.STOP,
            "saved judge answers and policy work need separate recovery; mixed recovery is unsupported",
        )
    if budget.live_job is None or budget.max_infra_retries is None or budget.infra_retries_used is None:
        return decision(
            RecoveryAction.STOP, "the job chain or scheduler state is unavailable, so the budget is unknown"
        )
    segment = planned_missing_continuation and bool(managed_selection) and not counts.get(IdentityState.RETRYABLE)
    if not segment and budget.infra_retries_used >= budget.max_infra_retries:
        return decision(
            RecoveryAction.STOP,
            f"the stored infra-retry budget is used ({budget.infra_retries_used} of {budget.max_infra_retries})",
        )
    if budget.walltime_capped:
        if budget.walltime_used_s is None or budget.max_walltime_s is None:
            return decision(RecoveryAction.STOP, "the run caps its walltime but the used walltime is unavailable")
        if budget.walltime_used_s >= budget.max_walltime_s:
            return decision(
                RecoveryAction.STOP,
                f"the stored walltime budget is used ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s)",
            )
    return decision(
        RecoveryAction.RESUBMIT,
        f"{needs_policy} identity(ies) need the policy again: a Gym resume runs only missing and retryable ones",
    )


# Run on the cluster over the executor's SSH connection: prints ONE JSON object with identity, failure and cause
# fields only (never answers or prompts), and the job chain's own budget files. argv[1] is the job dir.
COLLECT_SCRIPT = (
    "ACCOUNTING_UNKNOWN_PREFIX = "
    + repr(ACCOUNTING_UNKNOWN_PREFIX)
    + "\n"
    + "TASK_ID_FIELDS = "
    + repr(TASK_ID_FIELDS)
    + "\n"
    + r"""
import hashlib, json, math, os, sys
d = sys.argv[1]
def job_identity():
    try:
        stat = os.stat(d)
        return [os.path.realpath(d), stat.st_dev, stat.st_ino]
    except OSError:
        return None
initial_job_identity = job_identity()
a = os.path.join(d, "artifacts")
KEEP = ("_ng_task_index", "_ng_rollout_index", "_ng_rollout_id", "_ng_attempt_index", "_ng_failure_class", "failure_kind",
        "_ng_failure_terminal", "_ng_failure_counts_as_zero", "_ng_failure_max_attempts", "_ng_failure_id",
        "_ng_failure_events", "_ng_failure_events_omitted", "_ng_omitted_rollout", "_ng_unrouted_run_error", "_ng_dispatch_drained", "_line", "reward",
        "mask_sample", "evaluation_completed", "_ng_failure_http_status")
# Only structural response facts leave the run host; response contents stay local.
# A rollout store's attempt manifest, its attempt journal, or a run-tagged row: only the store can select those
# attempts. Only their presence leaves the run host, never their content or the run id.
managed_store = [os.path.exists(os.path.join(a, n)) for n in ("rollouts_manifest.json", "rollouts_attempts.jsonl")]
def trial_refs(items):
    if not isinstance(items, list):
        return None
    return [{"model_call_id": item.get("model_call_id"), "response_id": item.get("response_id"),
             "model_ref": {k: item["model_ref"].get(k) for k in ("type", "name")} if isinstance(item.get("model_ref"), dict) else None}
            if isinstance(item, dict) else None for item in items]
def trial_metadata(value):
    if not isinstance(value, dict):
        return None
    result = {k: value.get(k) for k in ("response_id", "status_code", "response_status", "error_category")}
    result["model_ref"] = {k: value["model_ref"].get(k) for k in ("type", "name")} if isinstance(value.get("model_ref"), dict) else None
    operations = value.get("model_operations")
    result["model_operations"] = [{k: operation.get(k) for k in ("operation_id", "outcome", "final_attempt_id", "failure_id", "http_attempts", "transport_failures")}
                                  if isinstance(operation, dict) else None for operation in operations] if isinstance(operations, list) else None
    report = value.get("failure_report")
    result["failure_report"] = None
    if isinstance(report, dict) and isinstance(report.get("events"), dict):
        result["failure_report"] = {"failure_id": report.get("failure_id"), "omitted": report.get("omitted"),
            "events": {event_id: {k: event.get(k) for k in ("event_id", "boundary", "operation", "operation_id", "attempt_id")}
                       if isinstance(event, dict) else None for event_id, event in report["events"].items()}}
    return result
import hashlib
def grading_result_binding(result):
    # Bind saved grading input before metadata projection.
    if not isinstance(result, dict) or not isinstance(result.get("response"), dict):
        return None
    response = result["response"]
    if not isinstance(response.get("id"), str) or not response["id"] or not isinstance(response.get("output"), list):
        return None
    reward = result.get("reward")
    if type(reward) not in (int, float) or (isinstance(reward, float) and not math.isfinite(reward)):
        return None
    value = {"response": response, "reward": reward}
    if "reward_components" in result:
        components = result["reward_components"]
        if not isinstance(components, dict) or not all(
            isinstance(key, str) and type(number) in (int, float)
            and (not isinstance(number, float) or math.isfinite(number))
            for key, number in components.items()
        ):
            return None
        value["reward_components"] = components

    def canonical(item):
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite grading input")
            return int(item) if item.is_integer() else item
        if isinstance(item, list):
            return [canonical(child) for child in item]
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            return {key: canonical(child) for key, child in item.items()}
        raise ValueError("non-JSON grading input")

    try:
        payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None
def trial_grading(value):
    if not isinstance(value, list):
        return None
    return [({k: item.get(k) for k in ("operation_id", "kind", "outcome", "result_binding_sha256")}
             | {"dependencies": trial_refs(item.get("dependencies"))}) if isinstance(item, dict) else None
            for item in value]
def trial_items(value, key, fields):
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        return None
    return [{k: item[k] for k in fields if k in item} if isinstance(item, dict) else None
            for item in value[key]]
def rows(name, skip=0):
    p = os.path.join(a, name)
    if not os.path.exists(p):
        return None
    out = []
    with open(p, "rb") as fh:
        for i, line in enumerate(fh):
            if i < skip or not line.strip():
                continue
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Binary iteration gives an unterminated line only at EOF; inventory must always be intact.
                if name != "rollouts_materialized_inputs.jsonl" and not line.endswith(b"\n"):
                    break
                raise
            if name == "rollouts_failures.jsonl":
                r["_line"] = i
            if isinstance(r, dict) and r.get("_ng_run_id") is not None:
                managed_store.append(True)
            if name == "rollouts.jsonl" and any(managed_store):
                r["_line"] = i
            selected = {k: r[k] for k in KEEP if k in r}
            if name == "rollouts.jsonl":
                binding = grading_result_binding(r)
                if binding is not None:
                    selected["_ng_collected_grading_binding"] = binding
            record = r.get("_ng_failure_record")
            if isinstance(record, dict):
                failure = record.get("failure")
                selected["_ng_failure_record"] = {"source": record.get("source"), "delivery": record.get("delivery"),
                    "failure": {"stage": failure.get("stage")} if isinstance(failure, dict) else None}
            for key in ("response", "_ng_failure_response", "_ng_failure_partial_response"):
                answer = r.get(key)
                if key in r and not isinstance(answer, dict):
                    selected[key] = None  # Preserve unusable-slot presence and verifier-input precedence.
                if isinstance(answer, dict):
                    output = answer.get("output")
                    selected[key] = {k: answer[k] for k in ("object", "id", "status") if k in answer}
                    if answer.get("error") is not None:
                        selected[key]["error"] = {}  # Presence only; diagnostic messages stay on the run host.
                    if isinstance(output, list):
                        selected[key]["output"] = [
                            {k: item[k] for k in ("type", "role", "status") if k in item}
                            if isinstance(item, dict) else None for item in output]

            if name == "rollouts_materialized_inputs.jsonl":
                identity = r.get("trajectory_identity")
                if isinstance(identity, dict) and all(isinstance(identity.get(k), str) and identity[k] for k in ("task_id", "rollout_id")):
                    task_id = identity["task_id"]
                else:
                    task_id = next((r[k] for k in TASK_ID_FIELDS if r.get(k) is not None), r.get("_ng_task_index"))
                # Only the scalar task identity leaves the host; dataset content stays in the saved input.
                selected["_ng_collected_task_id"] = str(task_id) if type(task_id) in (str, int) else None
            # Operation evidence only: prompts, answers, conversations and model payloads stay on the run host.
            trajectory = r.get("ng_trajectory")
            if name == "rollouts.jsonl" and isinstance(trajectory, dict):
                selected["ng_trial_observations"] = {
                    "task_id": trajectory.get("task_id"), "rollout_id": trajectory.get("rollout_id"),
                    "invocations": [({k: item.get(k) for k in ("invocation_id", "status", "in_session", "verification_skipped", "error_type", "parent_invocation_id", "stop_reason")} | {"model_calls": trial_refs(item.get("model_calls"))})
                                    if isinstance(item, dict) else None for item in trajectory.get("invocations")] if isinstance(trajectory.get("invocations"), list) else None,
                    "verifier_invocations": [({k: item.get(k) for k in ("invocation_id", "status", "failure_kind")} | {"model_calls": trial_refs(item.get("model_calls")), "grading_operations": trial_grading(item.get("grading_operations"))})
                                            if isinstance(item, dict) else None for item in trajectory.get("verifier_invocations")] if isinstance(trajectory.get("verifier_invocations"), list) else None,
                    "model_calls": [{"model_call_id": item.get("model_call_id"), "response_metadata": trial_metadata(item.get("response_metadata")),
                                     "output_items": trial_items(item.get("response"), "output", ("type", "role", "status", "call_id", "name")),
                                     "input_items": trial_items(item.get("request"), "input", ("type", "call_id"))}
                                    if isinstance(item, dict) else None for item in trajectory.get("model_calls")] if isinstance(trajectory.get("model_calls"), list) else None,
                    "turns": [({k: item.get(k) for k in ("invocation_id", "task_id", "rollout_id", "turn_no", "step_count")} | {"model_calls": trial_refs(item.get("model_calls"))}) if isinstance(item, dict) else None for item in trajectory.get("turns")] if isinstance(trajectory.get("turns"), list) else None,
                    "tool_calls": [{k: item.get(k) for k in ("invocation_id", "tool_call_id", "tool_name", "status", "error_type")}
                                   if isinstance(item, dict) else None for item in trajectory.get("tool_calls")] if isinstance(trajectory.get("tool_calls"), list) else None,
                    "tool_call_count": len(trajectory["tool_calls"]) if isinstance(trajectory.get("tool_calls"), list) else None,
                    "gaps": [{"code": gap.get("code")} if isinstance(gap, dict) else None for gap in trajectory.get("gaps")] if isinstance(trajectory.get("gaps"), list) else None,
                }
                for key in ("agent_ref", "_ng_environment_server", "_ng_result_type"):
                    if key in r:
                        selected[key] = r[key]
            out.append(selected)
    return out
def num(name):
    # (value, state): "absent" when the chain never wrote the file, "malformed" when it cannot be read as a number.
    path = os.path.join(d, name)
    if not os.path.exists(path):
        return None, "absent"
    try:
        value = float(open(path).read().strip())
        if not math.isfinite(value) or value < 0 or (name == ".gym_infra_retries" and not value.is_integer()):
            return None, "malformed"
        return value, "ok"
    except (OSError, ValueError):
        return None, "malformed"
try:
    offset = int(open(os.path.join(a, "rollouts_resubmit_offset")).read().strip())
except (OSError, ValueError):
    offset = 0
def accounting_unknown():
    marker = os.path.join(d, ".gym_chain_incomplete")
    try:
        if os.path.exists(marker) and open(marker).read().startswith(ACCOUNTING_UNKNOWN_PREFIX):
            return True
        for name in os.listdir(d):
            if name.startswith(".gym_followup_queued."):
                value = open(os.path.join(d, name)).read().strip()
                if not value.isascii() or not value.isdecimal():
                    return True
    except (OSError, ValueError):
        return True
    return False
def sha256(path):
    try:
        source = open(path, "rb")
    except FileNotFoundError:
        return None
    h = hashlib.sha256()
    with source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
        after = os.fstat(source.fileno())
    current = os.stat(path)
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError("run artifact changed while hashing")
    return h.hexdigest()
# The live exporter's run id file, and the post-run export receipt checked against the same files as its version.
RECEIPT_FILES = {"rollouts_jsonl_fpath": "rollouts.jsonl", "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
                 "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
                 "failures_jsonl_fpath": "rollouts_failures.jsonl",
                 "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl"}
source_files = (*RECEIPT_FILES.values(), "quality_summary.json", "rollouts_manifest.json", "rollouts_attempts.jsonl", "rollouts_resubmit_offset",
            "rollouts_coverage.json", "../.gym_infra_retries", "../.gym_accumulated_walltime", "../.gym_job_chain")
def followup_snapshot():
    # Keep scheduler state outside the installed Store artifact protocol, but bind both the dynamic
    # claim set and its contents. Directory timestamps also detect transient creation/removal.
    try:
        before = os.stat(d)
        claims = {name: sha256(os.path.join(d, name)) for name in sorted(os.listdir(d))
                  if name.startswith(".gym_followup_queued.")}
        after = os.stat(d)
        identity = lambda st: (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns)
        return (identity(after), claims) if identity(before) == identity(after) else None
    except (OSError, RuntimeError):
        return None
initial_followups = followup_snapshot()
source_snapshot = {name: sha256(os.path.join(a, name)) for name in source_files}
chain = open(os.path.join(d, ".gym_job_chain")).read().split() if os.path.exists(os.path.join(d, ".gym_job_chain")) else []
unknown_accounting = accounting_unknown()
mlflow = {"live_run_id": None, "receipt": None}
try:
    mlflow["live_run_id"] = open(os.path.join(a, "rollouts_mlflow_run_id")).read().strip()
except OSError:
    pass
if os.path.exists(os.path.join(a, "rollouts_mlflow_export_done")):
    try:
        text = open(os.path.join(a, "rollouts_mlflow_export_done")).read()
    except OSError:
        text = ""
    try:
        receipt = json.loads(text)
    except ValueError:
        receipt = None
    if (isinstance(receipt, dict) and type(receipt.get("version")) is int and receipt["version"] == 1
            and isinstance(receipt.get("run_id"), str) and receipt["run_id"].strip()
            and isinstance(receipt.get("artifacts"), dict)
            and set(receipt["artifacts"]) == set(RECEIPT_FILES)):
        current = {k: sha256(os.path.join(a, n)) for k, n in RECEIPT_FILES.items()}
        mlflow["receipt"] = {"run_id": receipt["run_id"], "current": receipt["artifacts"] == current,
                             "uploaded_rollouts": receipt.get("uploaded_rollouts"),
                             "rollout_projection_version": receipt.get("rollout_projection_version")}
    elif not isinstance(receipt, (dict, list)) and len(text.split()) == 1:
        # A marker that names only the backend run: it carries no digests to check the artifacts against.
        mlflow["receipt"] = {"run_id": text.strip(), "current": None}
    else:
        mlflow["receipt"] = "malformed"
quality = None
try:
    run = json.load(open(os.path.join(a, "quality_summary.json")))["run"]
    acc = run.get("artifacts") or {}
    sources = run.get("rollouts")
    source_matches = (isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], str)
                      and os.path.abspath(os.path.join(a, sources[0])) == os.path.abspath(os.path.join(a, "rollouts.jsonl")))
    provenance = run.get("provenance")
    bindings = provenance.get("rollout_artifact_digests") if isinstance(provenance, dict) else None
    binding = bindings[0] if isinstance(bindings, list) and len(bindings) == 1 else None
    expected_files = {k: v for k, v in RECEIPT_FILES.items() if k != "aggregate_metrics_json_fpath"}
    binding_valid = (isinstance(binding, dict) and set(binding) == set(expected_files)
                     and all(value is None or isinstance(value, str) and len(value) == 64
                             and all(c in "0123456789abcdef" for c in value) for value in binding.values())
                     and isinstance(binding.get("rollouts_jsonl_fpath"), str))
    source_matches = source_matches and binding_valid and binding == {k: sha256(os.path.join(a, n)) for k, n in expected_files.items()}
    quality = {"source_matches": source_matches, "rollouts": sources, "artifacts": {k: acc.get(k) for k in ("expected", "missing", "healthy", "unhealthy_reasons", "partial")},
               "verdicts": run.get("verdicts"), "issues": run.get("issues"), "coverage": acc.get("coverage"),
               "ignored_checks": run.get("ignored_checks"), "responses": (run.get("stats") or {}).get("responses"),
               "provenance": run.get("provenance"),
               "observed_upstream_recovery": (run.get("stats") or {}).get("observed_upstream_recovery"),
               "observed_runtime": (run.get("stats") or {}).get("observed_runtime"),
               "observed_request_effort": (run.get("stats") or {}).get("observed_request_effort"),
               "ended_on_failed_model_call": ((run.get("stats") or {}).get("model_call_errors") or {}).get("ended_on_error"),
               "verification_skipped": ((run.get("stats") or {}).get("model_call_errors") or {}).get("verification_skipped"),
               "context_window_exceeded": ((run.get("stats") or {}).get("model_call_errors") or {}).get("context_window_exceeded")}
except (OSError, ValueError, KeyError, TypeError, AttributeError):
    # Optional diagnostics cannot make the authoritative saved inventory unreadable.
    pass
score = None
if len(sys.argv) > 2:
    try:
        value = json.load(open(os.path.join(a, "rollouts_aggregate_metrics.json")))
        selector = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        if isinstance(value, list):
            if not selector or any(k not in ("agent", "environment") or not isinstance(v, str) or not v
                                   for k, v in selector.items()):
                raise ValueError("aggregate lists require a stable agent or environment selector")
            matches = [r for r in value if isinstance(r, dict)
                       and ("agent" not in selector or isinstance(r.get("agent_ref"), dict)
                            and r["agent_ref"].get("name") == selector["agent"])
                       and ("environment" not in selector or r.get("_ng_environment_server") == selector["environment"])]
            if len(matches) != 1:
                raise ValueError("aggregate identity is absent or ambiguous")
            value = matches[0]
        elif selector:
            raise ValueError("identity selectors require producer aggregate entries")
        for part in json.loads(sys.argv[2]):  # the metric path as a JSON list: keys may contain "/"
            # A string key into an object, a non-negative integer index into an array; anything else reads nothing.
            if isinstance(value, dict) and isinstance(part, str):
                value = value[part]
            elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
                value = value[part]
            else:
                raise KeyError(part)
        score = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
collected = {"materialized": rows("rollouts_materialized_inputs.jsonl"), "results": rows("rollouts.jsonl"),
                  "failures": rows("rollouts_failures.jsonl") or [], "resubmit_offset": offset, "infra_retries_used": num(".gym_infra_retries")[0],
                  "infra_retries_state": "unknown" if unknown_accounting else num(".gym_infra_retries")[1],
                  "walltime_used_s": num(".gym_accumulated_walltime")[0],
                  "walltime_state": "unknown" if unknown_accounting else num(".gym_accumulated_walltime")[1], "job_chain": chain,
                  "chain_incomplete": os.path.exists(os.path.join(d, ".gym_chain_incomplete")),
                  "quality": quality, "score": score, "mlflow": mlflow, "managed_store": any(managed_store)}
if quality is not None and source_snapshot != {name: sha256(os.path.join(a, name)) for name in source_files}:
    quality["source_matches"] = False
stable_sources = (source_snapshot == {name: sha256(os.path.join(a, name)) for name in source_files}
                  and initial_followups is not None and initial_followups == followup_snapshot())
collected["source_digests"] = source_snapshot if stable_sources else None
# Do not upgrade a stale scheduler-state read into a fresh budget. The executor and generated job
# recheck the durable unknown marker under the existing submission/admission boundary as well.
if not stable_sources or accounting_unknown():
    collected["infra_retries_state"] = collected["walltime_state"] = "unknown"
collected["source_job_identity"] = initial_job_identity if initial_job_identity == job_identity() else None
print(json.dumps(collected))
"""
)


class CollectedEvidence(BaseModel):
    materialized: list[dict[str, Any]] | None
    results: list[dict[str, Any]] | None
    failures: list[dict[str, Any]]
    resubmit_offset: int = 0
    infra_retries_used: float | None = None
    infra_retries_state: str = "absent"
    walltime_used_s: float | None = None
    walltime_state: str = "absent"
    job_chain: list[str] = Field(default_factory=list)
    chain_incomplete: bool = False
    accounted_walltime_s: float | None = None
    quality: dict[str, Any] | None = None
    score: float | None = None
    mlflow: dict[str, Any] | None = None
    managed_store: bool = False
    source_digests: dict[str, str | None] | None = None
    source_job_identity: tuple[str, int, int] | None = None
    managed_selection: list[ManagedRecoveryFact] | None = None
    managed_reader_error: str | None = None
    # Effective setting derived by the installed runtime from the bound saved configuration, not an observation.
    checkpoint_enabled: bool | None = Field(default=None, strict=True)


def _managed_source_digest(path: Path) -> str | None:
    """Hash one stable file with bounded memory, using the exporter's existing identity checks."""
    try:
        source = path.open("rb")
    except FileNotFoundError:
        return None
    digest = hashlib.sha256()
    with source:
        before = os.fstat(source.fileno())
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
        after = os.fstat(source.fileno())
    current = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after) or identity(after) != identity(current):
        raise RuntimeError("managed source changed while hashing")
    return digest.hexdigest()


def _managed_trial_metadata(
    artifacts: Path, source_digests: dict, quality: dict | None, results: list[dict], store: object
) -> tuple[list[dict], bool | None]:
    """Session trial metadata from the bound saved config and the Store's own inputs, after Store selection.

    Returns the results with ``materialized_task_id`` on each session row and the effective checkpoint setting, or
    the results unchanged and ``None`` when any fact is missing. Self-contained: the remote Store reader runs this
    source inside the saved install, on that install's Store, materialization and checkpoint APIs.
    """
    import hashlib

    # A config that differs from the collection snapshot is a changed source: raise, never report it as unknown.
    expected = source_digests["rollouts_resolved_config.yaml"]
    try:
        config_bytes = (artifacts / "rollouts_resolved_config.yaml").read_bytes()
    except FileNotFoundError:
        config_bytes = None
    if (None if config_bytes is None else hashlib.sha256(config_bytes).hexdigest()) != expected:
        raise ValueError("resolved config changed after collection")
    if config_bytes is None:
        return results, None
    try:
        import yaml

        from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
        from nemo_gym.task_materialization import materialize_task

        try:
            config = yaml.safe_load(config_bytes)
        except yaml.YAMLError as error:
            raise ValueError("resolved config is not YAML") from error
        if not isinstance(config, dict):
            raise ValueError("resolved config is not a mapping")
        checkpoint_enabled = checkpoint_settings(config) is not None
        servers = ((quality or {}).get("provenance") or {}).get("servers", {})
        bound_results = []
        for result, original in zip(results, store.inputs_for(results), strict=True):
            projected = dict(result)
            if result.get("_ng_result_type") == "single_agent_turn_legacy":
                agent = servers[result["agent_ref"]["name"]]
                taskset = original.get("_ng_task_source", agent["resources_server"])
                task = materialize_task(original, taskset=taskset)
                projected["materialized_task_id"] = task["task_id"]["task_id"]
            bound_results.append(projected)
        return bound_results, checkpoint_enabled
    except (ImportError, AttributeError, OSError, ValueError, KeyError, TypeError):
        return results, None


def select_managed_evidence(collected: CollectedEvidence, job_dir: Path) -> CollectedEvidence:
    """Read locally accessible Store history in this installed Gym process; never copy its selection rules.

    The plain remote collector remains dependency-free. Call only on the same host as the installed caller,
    after collection. Both snapshots must include the manifest and journal and remain byte-identical.
    """
    if not collected.managed_store:
        return collected
    try:
        from contextlib import redirect_stdout
        from io import StringIO

        from nemo_gym.rollout_records import logical_rollout_id
        from nemo_gym.rollout_store import RolloutStore

        artifacts = job_dir / "artifacts"
        names = (
            "rollouts.jsonl",
            "rollouts_failures.jsonl",
            "rollouts_materialized_inputs.jsonl",
            "rollouts_resolved_config.yaml",
            "rollouts_aggregate_metrics.json",
            "quality_summary.json",
            "rollouts_manifest.json",
            "rollouts_attempts.jsonl",
            "rollouts_resubmit_offset",
            "rollouts_coverage.json",
            "../.gym_infra_retries",
            "../.gym_accumulated_walltime",
            "../.gym_job_chain",
        )

        def snapshot():
            return {name: _managed_source_digest(artifacts / name) for name in names}

        if collected.source_digests is None or collected.source_digests != snapshot():
            raise ValueError("managed source changed or has no stable collection snapshot")
        store = RolloutStore.read(artifacts / "rollouts.jsonl", import_legacy=False)
        if store is None or collected.materialized is None:
            raise ValueError("managed source has no readable manifest or inventory")
        if store.manifest.max_rollout_attempts is None:
            raise ValueError("managed manifest does not preserve its rollout attempt cap")
        coverage = store.coverage()
        if len(collected.materialized) != coverage["expected"] or not coverage["identity_verified"]:
            raise ValueError("managed inventory is ambiguous or its identity is not verified")
        with redirect_stdout(StringIO()):
            pending = {logical_rollout_id(row) for row in store.pending()}
        facts = []
        for row in collected.materialized:
            identity = _identity(row)
            if identity is None:
                raise ValueError("managed input has no supported task/rollout identity")
            logical = logical_rollout_id(row)
            facts.append(
                ManagedRecoveryFact(
                    identity=identity,
                    attempts=store.attempt_count(row),
                    max_attempts=coverage["max_rollout_attempts"],
                    pending=logical in pending,
                    reserved=logical in store.manifest.next_attempt,
                )
            )
        selected = {"rollouts.jsonl": set(), "rollouts_failures.jsonl": set()}
        for disposition in ("success", "failure", "omitted"):
            for record in store.selected_records(disposition).values():
                matches = [name for name in selected if (artifacts / name).resolve() == record.path.resolve()]
                if len(matches) != 1:
                    raise ValueError("selected outcome is outside the collected source files")
                selected[matches[0]].add(record.line_number - 1)
        results = [row for row in collected.results or [] if row.get("_line") in selected["rollouts.jsonl"]]
        failures = [
            row
            for row in collected.failures
            if row.get("_line") in selected["rollouts_failures.jsonl"] or row.get("_ng_unrouted_run_error") is True
        ]
        if len(results) != len(selected["rollouts.jsonl"]) or len(
            [row for row in failures if row.get("_line") in selected["rollouts_failures.jsonl"]]
        ) != len(selected["rollouts_failures.jsonl"]):
            raise ValueError("selected outcomes are absent from the collected projection")
        results, checkpoint_enabled = _managed_trial_metadata(
            artifacts, collected.source_digests, collected.quality, results, store
        )
        if collected.source_digests != snapshot():
            raise ValueError("managed source changed during authoritative selection")
        return collected.model_copy(
            update={
                "results": results,
                "failures": failures,
                "managed_selection": facts,
                "checkpoint_enabled": checkpoint_enabled,
            }
        )
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError, ConfigError) as error:
        return collected.model_copy(update={"managed_selection": None, "managed_reader_error": str(error)})


def decide_from_collected(
    collected: CollectedEvidence,
    *,
    max_infra_retries: int | None,
    max_walltime_s: float | None,
    live_job: bool | None,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    planned_missing_continuation: bool = False,
) -> RecoveryDecision:
    """Decide from what ``COLLECT_SCRIPT`` read; a missing inventory or result file stops the decision."""
    # resume_script creates .gym_infra_retries on the first infra retry, so its absence means none ran; it adds every
    # previous hop to .gym_accumulated_walltime, so after a follow-up hop that file must exist. Malformed is unknown.
    retries = {"absent": 0, "ok": collected.infra_retries_used}.get(collected.infra_retries_state)
    # The prologue's counter covers predecessors, not necessarily the last allocation. A nonempty chain
    # therefore also needs terminal scheduler accounting for every recorded allocation.
    accounted = collected.accounted_walltime_s
    if collected.job_chain:
        walltime = (
            max(accounted, collected.walltime_used_s or 0)
            if accounted is not None
            and math.isfinite(accounted)
            and accounted >= 0
            and collected.walltime_state in ("absent", "ok")
            else None
        )
    elif collected.walltime_state == "ok":
        walltime = collected.walltime_used_s
    elif collected.walltime_state == "absent":
        walltime = 0.0
    else:
        walltime = None
    budget = RecoveryBudget(
        max_infra_retries=max_infra_retries,
        infra_retries_used=int(retries) if retries is not None else None,
        max_walltime_s=max_walltime_s,
        walltime_capped=max_walltime_s is not None,
        walltime_used_s=walltime,
        live_job=live_job,
        job_chain=collected.job_chain,
    )
    if collected.managed_store and collected.managed_selection is None:
        # Selecting from these files here could pick a superseded attempt or lose a reservation, so every known
        # input stays unknown and no saved row is read as its outcome.
        known = sorted({i for i in map(_identity, collected.materialized or []) if i is not None})
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=(
                "the run was written by a rollout store (an attempt manifest, an attempt journal or run-tagged rows); "
                "an authoritative Store reader is unavailable or could not verify these artifacts; recovery stops"
                + (f": {collected.managed_reader_error}" if collected.managed_reader_error else "")
            ),
            expected=len(known),
            counts={IdentityState.UNKNOWN.value: len(known)} if known else {},
            identities={IdentityState.UNKNOWN.value: known} if known else {},
            evidence=[IdentityEvidence(identity=i, state=IdentityState.UNKNOWN) for i in known],
        )
    if collected.materialized is None:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason="the run has no materialized inventory to recover against",
            expected=0,
            counts={},
            identities={},
            evidence=[],
        )
    decision = decide_recovery(
        materialized=collected.materialized,
        results=collected.results or [],
        failures=collected.failures,
        budget=budget,
        resubmit_offset=collected.resubmit_offset,
        outcome_policy=outcome_policy,
        managed_selection=collected.managed_selection,
        planned_missing_continuation=planned_missing_continuation,
    )
    if collected.results is None and not decision.scheduler_live:
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": "the results file is absent; collection cannot safely resume its saved history",
            }
        )
    return decision


def continuation_admission(
    job_dir: str,
    max_retries: int,
    max_walltime_s: float | None,
    exit_code: str,
    previous_job_id: str,
    previous_state: str = "FAILED",
) -> str:
    """Read saved state in the driver's installed runtime before a terminal allocation continues.

    The shell has already validated and persisted terminal elapsed accounting. This reader never changes
    artifacts, starts servers, resets budgets, or treats an exit code as sufficient retry evidence.
    """
    import json
    import subprocess
    import sys

    job = Path(job_dir)
    if exit_code == "76:0":
        return "stop"
    try:
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)],
            capture_output=True,
            text=True,
            check=True,
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        if not collected.managed_store:
            return "stop"
        collected = select_managed_evidence(collected, job)
        if collected.walltime_state != "ok":
            return "stop"
        collected.accounted_walltime_s = collected.walltime_used_s
        # A completed outcome plus untouched pending inputs establishes planned segment work. Actual failed
        # attempts still consume infra retries, even if the driver happened to exit75. A missing output,
        # reservation, judge answer or unowned selected cause is rejected by the ordinary decision below.
        # A scheduler allocation timeout continues untouched work, but never a failed agent attempt.
        segment = previous_state == "TIMEOUT"
        if previous_state == "FAILED" and exit_code == "75:0" and collected.managed_selection is not None:
            from nemo_gym.rollout_store import RolloutStore

            report_path = job / "artifacts" / "rollouts_coverage.json"
            if collected.source_digests and collected.source_digests.get(
                "rollouts_coverage.json"
            ) == _managed_source_digest(report_path):
                report = json.loads(report_path.read_text())
                store = RolloutStore.read(job / "artifacts" / "rollouts.jsonl", import_legacy=False)
                progress = report.get("progress_outcomes_recorded")
                segment = (
                    store is not None
                    and report.get("allocation_id") == previous_job_id
                    and bool(previous_job_id)
                    and type(progress) is int
                    and progress > 0
                    and type(report.get("outcomes_recorded")) is int
                    and progress <= report["outcomes_recorded"]
                    and all(report.get(key) == value for key, value in store.coverage().items())
                )
        if select_managed_evidence(collected, job).managed_selection is None:
            return "stop"
        decision = decide_from_collected(
            collected,
            max_infra_retries=max_retries,
            max_walltime_s=max_walltime_s,
            live_job=False,
            planned_missing_continuation=segment,
        )
        print(decision.reason, file=sys.stderr)
        if decision.action != RecoveryAction.RESUBMIT:
            return "stop"
        return "segment" if segment and not decision.counts.get(IdentityState.RETRYABLE) else "retry"
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError):
        return "stop"


def accounted_chain_walltime(
    job_chain: list[str],
    accounting: str,
    *,
    stored_walltime_s: float = 0,
) -> float | None:
    """Elapsed seconds of all recorded allocations, only when each has one terminal accounting row."""
    terminal = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
        "PREEMPTED",
    }
    if not job_chain or len(set(job_chain)) != len(job_chain):
        return None
    elapsed: dict[str, int] = {}
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            return None
        job_id, state, seconds = parts
        if (
            job_id not in job_chain
            or job_id in elapsed
            or (state.split()[0].rstrip("+") if state.split() else "") not in terminal
        ):
            return None
        if not seconds.isdecimal():
            return None
        elapsed[job_id] = int(seconds)
    if set(elapsed) != set(job_chain):
        return None
    total = sum(elapsed.values())
    # The prologue records predecessors. A stopped guard may already have accounted for the last job too.
    # Other values cannot identify the budget epoch (e.g. an intentional manual reset in an old chain).
    if stored_walltime_s not in (total - elapsed[job_chain[-1]], total):
        return None
    return float(total)


def gate_task_summary(
    decision: RecoveryDecision,
    *,
    score: float | None,
    fe_valid: bool | None,
    fe_reasons: list[str],
    health_passed: bool | None,
    health_reasons: list[str],
) -> dict[str, Any]:
    """One task entry of the Day-0 gate's run summary, filled from the decision and the existing validators.

    ``expected_samples`` is always the full materialized inventory; ``scored_samples`` counts only scored and
    final-model identities, so a judge, infra, terminal or missing identity can never make them equal. FE validity
    and response/trial health stay separate; ``None`` (unknown or missing) is a blocker, never a pass.
    """
    errors: list[str] = []
    warnings: list[str] = []
    states = decision.counts
    for state, label in (
        (IdentityState.JUDGE_RECOVERABLE, "judge failures awaiting re-judgment"),
        (IdentityState.RETRYABLE, "retryable failures awaiting another attempt"),
        (IdentityState.EXHAUSTED, "failures with spent attempts"),
        (IdentityState.TERMINAL, "terminal unscored failures"),
        (IdentityState.MISSING, "identities with no row"),
        (IdentityState.UNKNOWN, "failures nobody owns"),
    ):
        if states.get(state):
            errors.append(f"{states[state]} {label}: {decision.identities[state.value][:5]}")
    if score is None or not math.isfinite(score):
        errors.append("score unavailable: no requested metric was read")
    if decision.duplicate_results:
        errors.append(f"duplicate result rows: {decision.duplicate_results[:5]}")
    if decision.action == RecoveryAction.STOP:
        errors.append(f"recovery stopped: {decision.reason}")
    if fe_valid is not True:
        errors.append(
            "FE validity " + ("unknown" if fe_valid is None else "failed") + (f": {fe_reasons}" if fe_reasons else "")
        )
    if health_passed is not True:
        errors.append(
            "response/trial health "
            + ("unknown" if health_passed is None else "failed")
            + (f": {health_reasons}" if health_reasons else "")
        )
    if states.get(IdentityState.MODEL_FINAL):
        warnings.append(
            f"{states[IdentityState.MODEL_FINAL]} final evaluated-model failures scored zero: "
            f"{decision.identities[IdentityState.MODEL_FINAL.value][:5]}"
        )
    if decision.scheduler_live:
        status = "RUNNING"
    elif decision.action in (RecoveryAction.RESUBMIT, RecoveryAction.REVERIFY):
        status = "RESUMING"
    elif decision.action == RecoveryAction.NOOP and decision.complete and not errors:
        status = "SUCCESS"
    else:
        status = "FAILED"
    return {
        "status": status,
        "expected_samples": decision.expected,
        "scored_samples": states.get(IdentityState.SCORED, 0) + states.get(IdentityState.MODEL_FINAL, 0),
        "score": score,
        "errors": errors,
        "warnings": warnings,
    }


RecoveryOutcome.model_rebuild()


def quality_inputs(
    decision: RecoveryDecision, collected: CollectedEvidence, *, max_unhealthy_fraction: float | None
) -> tuple[bool | None, list[str], bool | None, list[str]]:
    """FE validity and per-rollout health from the run's ``quality_summary.json``, kept separate.

    Validity is the health check's own accounting verdict (``run.artifacts.healthy``, null for a partial summary),
    and must count the same expected identities as the inventory. Per-rollout health is the rollout verdict count
    (``run.verdicts``), judged only against an explicit ``max_unhealthy_fraction``; it is not ModelOpt response or
    trial health, which this adapter has no separate evidence for. Any missing piece is ``None``.
    """
    quality = collected.quality
    if quality is None:
        return None, ["no quality_summary.json"], None, ["no quality_summary.json"]
    if quality.get("source_matches") is not True:
        reason = "quality summary is not bound to this run's sole rollouts.jsonl source"
        return None, [reason], None, [reason]
    artifacts = quality.get("artifacts") or {}
    fe_reasons = [str(r) for r in artifacts.get("unhealthy_reasons") or []]
    fe_valid = artifacts.get("healthy") if isinstance(artifacts.get("healthy"), bool) else None
    if artifacts.get("partial"):
        fe_valid, fe_reasons = None, fe_reasons + ["partial summary: per-record checks not evaluated"]
    if (
        type(artifacts.get("expected")) is not int
        or artifacts["expected"] < 0
        or artifacts["expected"] != decision.expected
    ):
        fe_valid = None if fe_valid is None else False
        fe_reasons.append(f"summary counts {artifacts.get('expected')} expected, inventory {decision.expected}")
    verdicts = quality.get("verdicts")
    if not isinstance(verdicts, dict) or not all(
        type(verdicts.get(k)) is int and verdicts[k] >= 0 for k in ("healthy", "unhealthy", "unobserved")
    ):
        return fe_valid, fe_reasons, None, ["no valid rollout verdict counts (non-negative integers)"]
    counts = f"{verdicts['healthy']} healthy, {verdicts['unhealthy']} unhealthy, {verdicts['unobserved']} unobserved"
    if verdicts["healthy"] + verdicts["unhealthy"] + verdicts["unobserved"] != decision.expected:
        return fe_valid, fe_reasons, None, [f"verdict counts ({counts}) do not cover the {decision.expected} expected"]
    if max_unhealthy_fraction is None:
        return fe_valid, fe_reasons, None, [f"no response-health criterion configured ({counts})"]
    if (
        isinstance(max_unhealthy_fraction, bool)
        or not isinstance(max_unhealthy_fraction, (int, float))
        or not math.isfinite(max_unhealthy_fraction)
        or not 0 < max_unhealthy_fraction <= 1
    ):
        return fe_valid, fe_reasons, None, [f"invalid response-health criterion {max_unhealthy_fraction!r}"]
    judged = verdicts["healthy"] + verdicts["unhealthy"]
    if verdicts["unobserved"] or not judged:
        return fe_valid, fe_reasons, None, [f"health not observed for every rollout ({counts})"]
    share = verdicts["unhealthy"] / judged
    # The criterion is a strict bound, as the Day-0 gate's "< limit": a share equal to it fails.
    if share >= max_unhealthy_fraction:
        return (
            fe_valid,
            fe_reasons,
            False,
            [f"unhealthy {share:.2%} not below {max_unhealthy_fraction:.2%} ({counts})"],
        )
    return fe_valid, fe_reasons, True, [counts]


def mlflow_errors(
    local: Mapping[str, Any] | None,
    observed_status: str | None,
    *,
    configured: bool | None = None,
    observed_run_id: str | None = None,
) -> list[str]:
    """MLflow delivery errors from the collected exporter files and the server status the caller observed.

    ``local`` holds the live exporter's run id (``rollouts_mlflow_run_id``) and the post-run export receipt
    (``rollouts_mlflow_export_done``, checked against the current artifact digests). Only a current post-run receipt
    with an observed FINISHED passes. A live run is not accepted here: live uploads log failures and continue, so
    neither its run id nor FINISHED shows the configured payloads arrived. A live run beside a post-run export leaves
    the observed status unbound. ``configured`` is the health report's `run.provenance.mlflow.configured`, a
    names-only fact about the saved config snapshot (None when not reported); it only qualifies the message. The
    observed status is bound to a run: with a post-run receipt, ``observed_run_id`` must name the receipt's run.
    `BaseExporter.export` writes the receipt only after every upload returned and the artifacts did not change, but
    its teardown still sets FINISHED on a partial export (``run_failed`` is the process's run, not the export's), so
    a FINISHED from any other run proves nothing. No
    evidence is always an error, never a run that needed none: exporter activation and whether tracking was required
    or disabled are not recorded.
    """
    local = local or {}
    live, receipt = local.get("live_run_id"), local.get("receipt")
    if live is not None and not isinstance(live, str):
        return [f"MLflow live run id evidence is malformed: {live!r}"]
    if receipt not in (None, "malformed") and not (
        isinstance(receipt, dict)
        and isinstance(receipt.get("run_id"), str)
        and receipt["run_id"].strip()
        and "current" in receipt
        and (receipt["current"] is None or type(receipt["current"]) is bool)
    ):
        return [f"MLflow post-run export evidence has an unknown shape: {receipt!r}"]
    if live is None and receipt is None:
        if observed_status:
            return [f"MLflow status {observed_status!r} given for a run with no MLflow run id"]
        if configured is True:
            return [
                "no MLflow run evidence: no live run id and no post-run export receipt, although the saved config "
                "snapshot names a tracking URI, experiment and run name; exporter activation and whether tracking was "
                "required are not recorded"
            ]
        if configured is False:
            return [
                "no MLflow run evidence: no live run id and no post-run export receipt, and the saved config snapshot "
                "does not name all of the tracking URI, experiment and run name; exporter activation and whether "
                "tracking was required or disabled are not recorded"
            ]
        return ["no MLflow run evidence: no live run id and no post-run export receipt"]
    errors = []
    if live is not None and not live.strip():
        errors.append("MLflow live run id file is empty")
    if receipt == "malformed":
        errors.append("MLflow post-run export receipt is malformed or not a version-1 receipt")
    elif isinstance(receipt, dict) and receipt.get("current") is None:
        errors.append(
            f"MLflow post-run export marker of run {receipt.get('run_id')} has no artifact digests, so the exported "
            "content is not verified"
        )
    elif isinstance(receipt, dict) and receipt.get("current") is not True:
        errors.append(
            f"MLflow post-run export receipt of run {receipt.get('run_id')} does not match the current artifacts"
        )
    # A post-run export beside an earlier live run is allowed (BaseExporter.export warns and keeps the live id), so a
    # current receipt whose own run is the observed one is bound; any other observation stays ambiguous.
    bound = (
        isinstance(receipt, dict)
        and receipt.get("current") is True
        and isinstance(observed_run_id, str)
        and bool(observed_run_id.strip())
        and observed_run_id == receipt.get("run_id")
    )
    if live and receipt is not None and not bound:
        errors.append(f"MLflow live run {live} and a post-run export both exist: the observed status names neither")
    elif live is not None and live.strip() and receipt is None:
        errors.append(
            f"MLflow live run {live}: configured payload delivery is not verified (live uploads log failures and "
            "continue), so live acceptance is unsupported"
        )
    if isinstance(receipt, dict) and receipt.get("current") is True and receipt.get("uploaded_rollouts") is not True:
        # The receipt hashes the local files whatever the export uploaded; only its own record of the rollouts upload
        # says the backend run holds them. An older receipt without it, or an export that skipped them, is not delivery.
        scope = receipt.get("uploaded_rollouts")
        errors.append(
            f"MLflow post-run export receipt of run {receipt.get('run_id')} "
            + (
                "records that the rollouts were not uploaded"
                if scope is False
                else "does not record whether the rollouts were uploaded"
            )
        )
    if (
        isinstance(receipt, dict)
        and receipt.get("current") is True
        and receipt.get("uploaded_rollouts") is True
        and (type(receipt.get("rollout_projection_version")) is not int or receipt["rollout_projection_version"] != 1)
    ):
        errors.append(
            "MLflow export receipt does not verify delivery of canonical trajectory metadata "
            "(rollout projection version 1 required)"
        )
    if isinstance(receipt, dict) and not live and observed_status is not None:
        bound = receipt.get("run_id")
        if not isinstance(observed_run_id, str) or not observed_run_id.strip():
            errors.append(f"MLflow observed status names no run, so it is not bound to the receipt's run {bound}")
        elif observed_run_id != bound:
            errors.append(f"MLflow observed status is for run {observed_run_id}, not the receipt's run {bound}")
    if observed_status is None:
        errors.append("MLflow terminal status not observed")
    elif observed_status != "FINISHED":
        errors.append(f"MLflow run status {observed_status}")
    return errors


# The Day-0 bounded failure policy: a gate passes only below this failure rate, measured before rounding.
GATE_FAILURE_LIMIT = 0.02


class GateCriteria(BaseModel):
    """The named response and trial acceptance criteria. Each gate passes only strictly below its limit, measured
    before rounding. They are separate from retry policy, budgets and benchmark rewards, and default to the bounded
    failure policy's limit; an invalid value is refused, never clamped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_response_failure_fraction: float = GATE_FAILURE_LIMIT
    max_trial_failure_fraction: float = GATE_FAILURE_LIMIT

    @field_validator("max_response_failure_fraction", "max_trial_failure_fraction", mode="before")
    @classmethod
    def _fraction(cls, value: Any) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 < value <= 1
        ):
            raise ValueError("expected a finite number in (0, 1]")
        return float(value)

    def report(self) -> dict[str, Any]:
        """The effective criteria as reported beside the gate results."""
        rule = "strictly below, before rounding"
        return {
            "response": {"max_failure_fraction": self.max_response_failure_fraction, "passes": rule},
            "trial": {"max_failure_fraction": self.max_trial_failure_fraction, "passes": rule},
        }


def _percent(limit: float) -> str:
    return f"{limit * 100:g}%"


def response_health(
    decision: RecoveryDecision, quality: Mapping[str, Any] | None, *, limit: float = GATE_FAILURE_LIMIT
) -> tuple[bool | None, str]:
    """The response gate from ``run.stats.responses`` of the health report: faulted over selected responses.

    Evaluated only when every scored rollout has a verified response population and every selected call id is
    unique across the run (a report that predates it, an unverified call scope, a response selected twice, or zero
    selected responses is unknown, neither pass nor fail). Failed trials have no response row and are left out, as
    the policy leaves failed attempts out of the response denominator.
    """
    r = (quality or {}).get("responses")
    fields = ("rollouts", "unknown", "selected", "faulted", "duplicated")
    if not isinstance(r, dict) or not all(type(r.get(k)) is int and r[k] >= 0 for k in fields):
        return None, "response: the health report has no verified response population (run.stats.responses)"
    scored = decision.counts.get(IdentityState.SCORED, 0)
    if not decision.complete or r["rollouts"] != scored or r["unknown"] or r["duplicated"] or not r["selected"]:
        return None, (
            f"response: population not verified for every scored rollout ({r['rollouts']} reported, {scored} scored, "
            f"{r['unknown']} unknown, {r['duplicated']} duplicated, {r['selected']} selected)"
        )
    if r["faulted"] > r["selected"]:
        return None, f"response: {r['faulted']} faulted exceeds {r['selected']} selected responses"
    rate = r["faulted"] / r["selected"]
    counts = f"{r['faulted']} faulted of {r['selected']} selected raw responses ({rate:.2%})"
    if rate >= limit:
        return False, f"response: {counts} is not below {_percent(limit)}"
    return True, f"response: {counts} is below {_percent(limit)}"


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _clip_text(quality: Mapping[str, Any]) -> str:
    counts = quality.get("context_window_exceeded")
    fields = ("true", "false", "absent")
    if isinstance(counts, dict) and all(_count(counts.get(k)) is not None for k in fields):
        return str({k: counts[k] for k in fields})
    return "not reported or malformed"


def trial_boundary(quality: Mapping[str, Any] | None) -> str:
    """Why trial failures outside the sidecar cannot be ruled out, from the run's own persisted facts.

    Every persisted value is type-checked: a malformed report is named as malformed, never read as a fact.
    """
    quality = quality if isinstance(quality, Mapping) else {}
    provenance = quality.get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or provenance.get("resolved_config") != "read":
        state = provenance.get("resolved_config") if isinstance(provenance, dict) else "not reported"
        facts = f"run provenance {state if state in ('absent', 'unreadable', 'not reported') else 'malformed'}"
    elif not isinstance(servers, dict) or not all(isinstance(f, dict) for f in servers.values()):
        facts = "run provenance malformed"
    else:
        agents = {
            str(name): (fact.get("implementation"), fact.get("resources_server"), fact.get("skip_verification"))
            for name, fact in servers.items()
            if fact.get("kind") == "responses_api_agents"
        }
        environments = sorted(str(n) for n, f in servers.items() if f.get("kind") == "environment_servers")
        route = provenance.get("route_failures_to_sidecar")
        facts = (
            f"agents (implementation, resources server, skip_verification) {agents}, environment servers "
            f"{environments}, route_failures_to_sidecar {route if isinstance(route, bool) else None}"
        )
    runtime = quality.get("observed_runtime")
    runtime_count = _count(runtime.get("invocation_observations")) if isinstance(runtime, dict) else None
    runtime_valid = runtime_count is not None and all(
        isinstance(runtime.get(key), dict)
        and all(_count(runtime[key].get(k)) is not None for k in ("true", "false", "absent"))
        and sum(runtime[key][k] for k in ("true", "false", "absent")) == runtime_count
        for key in ("in_session", "verification_skipped")
    )
    if runtime_valid:
        session_text = f"observed invocation in_session {runtime['in_session']}; runtime verification_skipped {runtime['verification_skipped']}; these observations do not certify the full trial population"
    else:
        session_text = "agent session (in_session) runtime observations are absent or malformed, so it is unknown"
    ended = _count(quality.get("ended_on_failed_model_call"))
    skipped = quality.get("verification_skipped")
    fields = ("true", "false", "absent")
    if isinstance(skipped, dict) and all(_count(skipped.get(k)) is not None for k in fields):
        skipped_text = str({k: skipped[k] for k in fields})
    else:
        skipped_text = "not reported or malformed"
    # The producer sets ended_on_error only where the rollout's model calls were captured, so the count is a floor.
    ended_text = f"at least {ended} (counted where calls were captured)" if ended is not None else "unknown"
    return (
        f"{facts}, rollouts ended on a failed model call {ended_text}, rows by verification_skipped {skipped_text}, "
        f"rows whose step_usage records context_window_exceeded {_clip_text(quality)} (a signal from a best-effort "
        "error-text match or model metadata, not proof that the input was clipped); "
        f"{session_text}; model-generated "
        "code that fails, raises or times out is scored by the benchmark protocol, and whether it counts as a trial "
        "failure is open for the benchmark PIC"
    )


def _observed_trial_operations(decision: RecoveryDecision, collected: CollectedEvidence | None) -> bool:
    """Join observed model/tool operations and the outer verifier invocation for every expected trial.

    A source-reviewed programmatic grading operation must bind the saved response and score. Empty captured
    model calls alone never establish grading coverage; old or uninstrumented verifiers remain unknown.
    """
    from nemo_gym.core.episode import maybe_rollout_id_from_run_body

    if (
        collected is None
        or not collected.quality
        or collected.quality.get("source_matches") is not True
        or decision.counts.get(IdentityState.SCORED, 0) != decision.expected
    ):
        return False
    rows = collected.results
    expected = set(decision.identities.get(IdentityState.SCORED.value, []))
    if rows is None or len(rows) != decision.expected or {_identity(row) for row in rows} != expected:
        return False
    provenance = collected.quality.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("resolved_config") != "read":
        return False
    servers = provenance.get("servers")
    if not isinstance(servers, dict):
        return False
    input_task_ids = {_identity(item): item.get("_ng_collected_task_id") for item in collected.materialized or []}
    populations = (set(), set(), set(), set(), set())
    grading_ids = set()

    def nonempty(value):
        return isinstance(value, str) and bool(value.strip())

    def matches(ref, call):
        meta = call["response_metadata"]
        return (
            isinstance(ref, dict)
            and (
                ref.get("model_call_id") == call["model_call_id"]
                if ref.get("model_call_id")
                else ref.get("model_ref") == meta["model_ref"] and ref.get("response_id") == meta["response_id"]
            )
            and (ref.get("model_ref") is None or ref["model_ref"] == meta["model_ref"])
            and (ref.get("response_id") is None or ref["response_id"] == meta["response_id"])
        )

    try:
        for row in rows:
            agent = servers[row["agent_ref"]["name"]]
            resource = servers[agent["resources_server"]]
            environment = servers[row["_ng_environment_server"]]
            if (
                agent.get("kind") != "responses_api_agents"
                or resource.get("kind") != "resources_servers"
                or environment.get("kind") != "environment_servers"
                or row.get("_ng_result_type") != environment.get("implementation")
                or row.get(FAILURE_CLASS) is not None
                or row.get("failure_kind") is not None
                or row.get(FAILURE_ID) is not None
                or row.get("_ng_failure_events_omitted", False) is not False
                or not isinstance(row.get(FAILURE_EVENTS, {}), dict)
                or row.get(FAILURE_EVENTS, {})
            ):
                return False
            observed = row["ng_trial_observations"]
            invocations, verifiers = observed["invocations"], observed["verifier_invocations"]
            calls, turns, tools = observed["model_calls"], observed["turns"], observed["tool_calls"]
            if (
                not all(isinstance(items, list) for items in (invocations, verifiers, calls, turns, tools))
                or len(invocations) != 1
                or len(verifiers) != 1
                or not calls
                or len(turns) != len(calls)
                or type(observed["tool_call_count"]) is not int
                or observed["tool_call_count"] != len(tools)
                or not isinstance(observed["gaps"], list)
                or any(gap.get("code") != "resolution_unavailable" for gap in observed["gaps"])
            ):
                return False
            invocation, verifier = invocations[0], verifiers[0]
            session = invocation["in_session"]
            task_id = row.get("materialized_task_id") if session else input_task_ids.get(_identity(row))
            if (
                type(session) is not bool
                or (session and (collected.checkpoint_enabled is not False or collected.managed_selection is None))
                or not nonempty(task_id)
                or observed["task_id"] != task_id
                or observed["rollout_id"] != maybe_rollout_id_from_run_body(row)
                or not nonempty(invocation["invocation_id"])
                or invocation["status"] != "completed"
                or invocation.get("parent_invocation_id") is not None
                or invocation.get("error_type") is not None
                or invocation.get("stop_reason") is not None
                or invocation["verification_skipped"] is not (None if session else False)
                or (session and invocation["invocation_id"] != "root")
                or verifier["status"] != "completed"
                or verifier["failure_kind"] is not None
                or verifier["model_calls"] != []
                or not nonempty(verifier["invocation_id"])
                or verifier["invocation_id"] in populations[3]
                or len(invocation["model_calls"]) != len(calls)
            ):
                return False
            grading = verifier.get("grading_operations")
            if not isinstance(grading, list) or len(grading) != 1 or not isinstance(grading[0], dict):
                return False
            grade = grading[0]
            binding = row.get("_ng_collected_grading_binding")
            if (
                grade.get("kind") != "programmatic_calculation"
                or grade.get("outcome") != "completed"
                or grade.get("dependencies") != []
                or not nonempty(grade.get("operation_id"))
                or grade["operation_id"] in grading_ids
                or not isinstance(binding, str)
                or len(binding) != 64
                or grade.get("result_binding_sha256") != binding
            ):
                return False
            grading_ids.add(grade["operation_id"])
            populations[3].add(verifier["invocation_id"])
            ordered = []
            for number, turn in enumerate(turns, 1):
                if (
                    turn["invocation_id"] != invocation["invocation_id"]
                    or turn["task_id"] != task_id
                    or turn["rollout_id"] != observed["rollout_id"]
                    or type(turn["turn_no"]) is not int
                    or turn["turn_no"] != number
                    or not isinstance(turn["model_calls"], list)
                    or len(turn["model_calls"]) != 1
                ):
                    return False
                joined = [call for call in calls if matches(turn["model_calls"][0], call)]
                if len(joined) != 1 or not matches(invocation["model_calls"][number - 1], joined[0]):
                    return False
                ordered.append(joined[0])
            requested = {}
            tool_by_id = {tool["tool_call_id"]: tool for tool in tools}
            if len(tool_by_id) != len(tools):
                return False
            for number, (call, turn) in enumerate(zip(ordered, turns), 1):
                meta = call["response_metadata"]
                operations, report = meta["model_operations"], meta["failure_report"]
                if (
                    type(meta["status_code"]) is not int
                    or not 200 <= meta["status_code"] < 300
                    or meta.get("error_category") is not None
                    or meta.get("response_status") != "completed"
                    or len(operations) != 1
                    or not isinstance(report["events"], dict)
                    or report["events"]
                    or report["omitted"] is not False
                    or report["failure_id"] is not None
                ):
                    return False
                operation = operations[0]
                if (
                    operation["outcome"] != "completed"
                    or operation["failure_id"] is not None
                    or type(operation["http_attempts"]) is not int
                    or operation["http_attempts"] != 1
                    or type(operation["transport_failures"]) is not int
                    or operation["transport_failures"] != 0
                    or not all(
                        nonempty(value)
                        for value in (
                            call["model_call_id"],
                            meta["model_ref"]["type"],
                            meta["model_ref"]["name"],
                            meta["response_id"],
                            operation["operation_id"],
                            operation["final_attempt_id"],
                        )
                    )
                ):
                    return False
                if operation["final_attempt_id"] in populations[4]:
                    return False
                populations[4].add(operation["final_attempt_id"])
                values = (
                    call["model_call_id"],
                    (meta["model_ref"]["type"], meta["model_ref"]["name"], meta["response_id"]),
                    operation["operation_id"],
                )
                if any(value in seen for value, seen in zip(values, populations)):
                    return False
                for value, seen in zip(values, populations):
                    seen.add(value)
                inputs, outputs = call["input_items"], call["output_items"]
                if not isinstance(inputs, list) or not isinstance(outputs, list) or not outputs:
                    return False
                # This proof joins client-executed function tools. Hosted/computer/custom operations
                # need their own runtime evidence; an assistant message beside them does not certify them.
                if any(item.get("type") not in {"message", "reasoning", "function_call"} for item in outputs) or any(
                    item.get("type") not in {None, "message", "reasoning", "function_call", "function_call_output"}
                    for item in inputs
                ):
                    return False
                consumed = [item["call_id"] for item in inputs if item.get("type") == "function_call_output"]
                if len(consumed) != len(set(consumed)) or set(consumed) != set(requested):
                    return False
                emitted = [item for item in outputs if item.get("type") == "function_call"]
                for item in emitted:
                    tool_id, name = item["call_id"], item["name"]
                    if not nonempty(tool_id) or not nonempty(name) or tool_id in requested:
                        return False
                    tool = tool_by_id.get(tool_id)
                    if (
                        tool is None
                        or tool["invocation_id"] != invocation["invocation_id"]
                        or tool["tool_name"] != name
                        or tool["status"] != "completed"
                        or tool["error_type"] is not None
                    ):
                        return False
                    requested[tool_id] = name
                # step_count is producer-specific (before/after a tool execution); the explicit tool
                # request/execution/result joins above establish coverage without reinterpreting that counter.
                if type(turn["step_count"]) is not int or turn["step_count"] < 0:
                    return False
                if number == len(ordered):
                    if emitted or not any(
                        item.get("type") == "message"
                        and item.get("role") == "assistant"
                        and item.get("status") == "completed"
                        for item in outputs
                    ):
                        return False
                elif not emitted:
                    return False
            saved_response = row.get("response")
            if (
                set(requested) != set(tool_by_id)
                or not isinstance(saved_response, dict)
                or saved_response.get("id") != ordered[-1]["response_metadata"]["response_id"]
                or saved_response.get("status") != "completed"
                or saved_response.get("error") is not None
            ):
                return False
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    return True


def trial_health(
    decision: RecoveryDecision, collected: CollectedEvidence | None = None, *, limit: float = GATE_FAILURE_LIMIT
) -> tuple[bool | None, str]:
    """The trial gate's known failures: settled zero-scored failures over the expected trials, including repeats.

    A failure counts when rollout collection settled it as a scored zero (MODEL_FINAL: marked counts-as-zero by its
    producer and terminal), so it is recorded as incorrect under the benchmark protocol. The denominator is the full
    materialized inventory. A run that is not complete has unaccounted trials, so the gate is not evaluated. Other
    terminal trial failures (a judge, verifier or harness error scored as a zero in the main output) carry no failure
    record and cannot be ruled out, so their count is a lower bound. At or above the limit it fails; below it the
    gate remains unknown unless actual per-identity operation outcomes cover the observed model/tool and programmatic grading path.
    """
    if not decision.complete or decision.expected <= 0:
        return None, f"trial: not evaluated, the run is not complete ({dict(decision.counts)})"
    failed = decision.counts.get(IdentityState.MODEL_FINAL, 0)
    kinds = Counter(e.failure_kind for e in decision.evidence if e.state == IdentityState.MODEL_FINAL)
    rate = failed / decision.expected
    counts = f"{failed} settled zero-scored failure(s) of {decision.expected} expected trials ({rate:.2%})"
    detail = f", kinds {dict(sorted(kinds.items(), key=lambda kv: str(kv[0])))}" if kinds else ""
    if rate >= limit:
        return False, f"trial: {counts} is not below {_percent(limit)}{detail}"
    recovery = (collected.quality or {}).get("observed_upstream_recovery") if collected else None
    if recovery is not None and (
        not isinstance(recovery, dict)
        or any(_count(recovery.get(key)) is None for key in ("recovered_attempt_errors", "recovered_operations"))
    ):
        return None, "trial: recovered attempt/operation counters are malformed or incomplete"
    if isinstance(recovery, dict) and any(
        recovery[key] > 0 for key in ("recovered_attempt_errors", "recovered_operations")
    ):
        return (
            None,
            "trial: recovered attempt/operation failures are observed; their trial-health treatment is not defined",
        )
    if _observed_trial_operations(decision, collected):
        return (
            True,
            f"trial: {failed} failed of {decision.expected} expected trials ({rate:.2%}); actual model/tool and bound programmatic grading operations cover every identity and repeat",
        )
    return None, (
        f"trial: known failures are a lower bound, {counts}{detail}; failures recorded only as zero rewards "
        "cannot be ruled out"
    )


_MAX_FACTS = 20


def _bounded(items: list[str]) -> str:
    shown = ", ".join(items[:_MAX_FACTS])
    return shown + (f", and {len(items) - _MAX_FACTS} more" if len(items) > _MAX_FACTS else "")


def health_check_facts(quality: Mapping[str, Any] | None) -> list[str]:
    """Named health-check facts from ``quality_summary.json`` for the gate warnings, kept even below any criterion.

    Nonzero ``run.issues`` counts, checks ``run.artifacts.coverage`` did not fully evaluate (any unobserved or
    ignored), and ``run.ignored_checks``. A field that is absent or malformed is reported as unknown, never as clean.
    """
    if quality is None:
        return []
    facts: list[str] = []
    issues = quality.get("issues")
    if not isinstance(issues, dict) or not all(
        isinstance(k, str) and type(v) is int and v >= 0 for k, v in issues.items()
    ):
        facts.append("health-check issue counts unknown: run.issues absent or not non-negative integers")
    elif nonzero := sorted(f"{k[:80]} {v}" for k, v in issues.items() if v):
        facts.append(f"health-check issues (nonzero): {_bounded(nonzero)}")
    coverage = quality.get("coverage")
    fields = ("evaluated", "unobserved", "ignored")
    if not isinstance(coverage, dict) or not all(
        isinstance(k, str) and isinstance(c, dict) and all(type(c.get(f)) is int and c[f] >= 0 for f in fields)
        for k, c in coverage.items()
    ):
        facts.append("health-check coverage unknown: run.artifacts.coverage absent or malformed")
    elif partial := sorted(
        f"{k[:80]} (evaluated {c['evaluated']}, unobserved {c['unobserved']}, ignored {c['ignored']})"
        for k, c in coverage.items()
        if c["unobserved"] or c["ignored"]
    ):
        facts.append(f"health checks not fully evaluated: {_bounded(partial)}")
    ignored = quality.get("ignored_checks")
    if not isinstance(ignored, list) or not all(isinstance(k, str) for k in ignored):
        facts.append("ignored health checks unknown: run.ignored_checks absent or malformed")
    elif ignored:
        facts.append(f"ignored health checks: {_bounded(sorted(k[:80] for k in ignored))}")
    return facts


def gate_summary_from_collected(
    decision: RecoveryDecision,
    collected: CollectedEvidence,
    *,
    max_unhealthy_fraction: float | None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    criteria: GateCriteria | None = None,
) -> dict[str, Any]:
    """The Day-0 gate task entry from the collected run: decision counts, the named score and the quality summary.

    ``mlflow_status`` and ``mlflow_run_id`` are the tracking run's status and id as the caller read them together.
    ``criteria`` are the response and trial limits (default: the bounded failure policy's); the entry reports the
    effective values under ``criteria``.
    """
    criteria = criteria or GateCriteria()
    fe_valid, fe_reasons, rollout_health, rollout_reasons = quality_inputs(
        decision, collected, max_unhealthy_fraction=max_unhealthy_fraction
    )
    # The Day-0 gate's response and trial health need their own counters and denominators; quality_summary's
    # verdicts are per rollout, so they are reported in warnings, never used as either. A positive trial gate
    # requires source-bound operation evidence for the supported path; recovered failures remain unknown.
    trial_passed, trial_reason = trial_health(decision, collected, limit=criteria.max_trial_failure_fraction)
    if trial_passed is None and decision.complete:
        trial_reason = f"{trial_reason} ({trial_boundary(collected.quality)})"
    bound_quality = (
        collected.quality if collected.quality and collected.quality.get("source_matches") is True else None
    )
    response_passed, response_reason = response_health(
        decision, bound_quality, limit=criteria.max_response_failure_fraction
    )
    gates = [(response_passed, response_reason), (trial_passed, trial_reason)]
    if any(passed is False for passed, _ in gates):
        health = False
    elif all(passed is True for passed, _ in gates):
        health = True
    else:
        health = None
    health_reasons = [reason for passed, reason in gates if passed is not True]
    task = gate_task_summary(
        decision,
        score=collected.score,
        fe_valid=fe_valid,
        fe_reasons=fe_reasons,
        health_passed=health,
        health_reasons=health_reasons,
    )
    task["criteria"] = criteria.report()
    judged = {True: "within", False: "beyond", None: "not judged against"}[rollout_health]
    task["warnings"] = (
        task["warnings"]
        + [f"per-rollout health {judged} the configured criterion: {rollout_reasons}"]
        + health_check_facts(collected.quality)
        + [f"health gate within the bounded failure policy: {reason}" for passed, reason in gates if passed]
    )
    extra: list[str] = []
    if collected.chain_incomplete and not decision.complete:
        extra.append("the job chain stopped with rollouts missing (.gym_chain_incomplete)")
    provenance = (collected.quality or {}).get("provenance")
    mlflow_facts = provenance.get("mlflow") if isinstance(provenance, dict) else None
    configured = mlflow_facts.get("configured") if isinstance(mlflow_facts, dict) else None
    extra.extend(
        mlflow_errors(
            collected.mlflow,
            mlflow_status,
            configured=configured if isinstance(configured, bool) else None,
            observed_run_id=mlflow_run_id,
        )
    )
    if extra:
        task["errors"] = task["errors"] + extra
        if task["status"] == "SUCCESS":
            task["status"] = "FAILED"
    return task
