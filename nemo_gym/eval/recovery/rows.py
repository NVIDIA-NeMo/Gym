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
  a scored zero that no infra retry may replace; or an ``agent_timeout`` row that kept the agent's answer
  (``is_answered_agent_deadline``), the model's result under its own deadline, which no resume runs again.
- ``judge_recoverable``: a result row of class ``judge_failed``. Its saved answer is re-judged by
  ``gym eval reverify --judge-failed-only``; queueing the policy again is never needed for it.
- ``retryable``: the last sidecar row is non-terminal and under its recorded attempt cap: a Gym resume retries it.
- ``exhausted``: the last sidecar row is non-terminal and at its attempt cap: only a resubmit gives it a fresh budget.
- ``terminal``: a terminal sidecar row that is not a scored zero; excluded, and the run cannot become complete.
- ``missing``: no row at all (interrupted or never started); a resume restarts its input.
- ``masked``: a result row the producer masked or did not evaluate. Collection keeps it as completed and never runs it
  again, so it is retained, but it is not measured and the run is not complete.
- ``omitted``: a planned row that a staged collection's accepted partial stage omitted. It is retained, never
  planned again and not measured, and the run is complete without it.
- ``unknown``: a failure whose kind is unregistered or absent, or a row whose identity is invalid; nobody owns it.

The run-level action follows: ``stop`` for a spent or unknown resubmit budget, or for an unknown identity that a resume
would run again; ``noop`` when nothing is left to do, ``reverify`` when only judge failures remain, else ``resubmit``.
A run that ended before its inventory was written has nothing to recover: ``fresh-restart`` queues it again as a fresh
run with a longer walltime.
An unknown identity that a resume never runs again (the Store or the saved result holds it) does not block the
missing and retryable ones: the decision reports it apart from what is retried and what is retained. A decision never
reports a run as complete while an identity is terminal, missing or unknown.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from nemo_gym.agents.agent_outcome import (
    OutcomeAction,
    choose_action,
)
from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION, accepted_stage_omission
from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    JUDGE_FAILED,
    JUDGE_UNPARSEABLE,
    MODEL_CONTEXT_LIMIT,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
)
from nemo_gym.eval.recovery.budgets import (
    RecoveryBudget,
)
from nemo_gym.eval.recovery.decision import (
    COUNTS_AS_ZERO,
    EXTEND_ATTEMPTS_HINT,
    FAILURE_CLASS,
    FAILURE_EVENTS,
    FAILURE_ID,
    MAX_ATTEMPTS,
    RERUN_MASKED_FLAGS,
    RERUN_MASKED_REASON,
    RESERVATION_WITHOUT_OUTCOME,
    SCORED_UNPARSEABLE_REASON,
    STORE_DOES_NOT_REPLAY,
    TERMINAL,
    UNKNOWN_BUDGET_REASON,
    UNKNOWN_WALLTIME_REASON,
    IdentityEvidence,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    RerunMaskedApproval,
    _events,
    _known_kind,
)
from nemo_gym.eval.recovery.evidence import (
    Identity,
    ManagedRecoveryFact,
    _identity,
)
from nemo_gym.eval.recovery.grading import (
    REVERIFY_JUDGE,
    masked_without_artifact_kind,
    reverification_cause,
)
from nemo_gym.eval.recovery.policy import (
    RecoveryOutcomePolicyConfig,
    _cause_chain,
    _cause_kind,
    _judge_state,
    _policy_evidence,
    _policy_replay_is_unsupported,
    _replay_refusal,
    _retry_kind,
    _sidecar_action,
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
    allocation_cut: bool = False,
    rerun_masked_without_artifact: bool = False,
) -> RecoveryDecision:
    """Classify every materialized identity and choose the run-level action.

    ``failures`` are every failures-sidecar row in file order (``_line`` is the row's line number when known). The
    standing row of a rollout is chosen as rollout collection chooses it (``_latest_failure_rows``): rows that
    ``is_non_superseding_rollout_row`` names are skipped, and an interrupted omitted row means the rollout is missing.
    Only rows at or after ``resubmit_offset`` count towards the attempt cap. The rows are only read.

    ``allocation_cut`` says that the allocation that last wrote the Store has ended, by any path (one of
    ``ALLOCATION_ENDED_STATES``), and no allocation of the run is live: a reservation with no saved row was in flight
    when it ended and saved nothing, so it is not started for recovery purposes and the Store runs it again.

    A masked result with a saved answer, no gradeable artifact and no recorded grade (``masked_without_artifact_kind``)
    stays MASKED, named in ``masked_without_artifact``, unless the Store reports an approval for it
    (``ManagedRecoveryFact.rerun_approved``) or ``rerun_masked_without_artifact`` asks for it now: it is then planned
    like an infrastructure-masked result, one failed attempt within its cap, and ``rerun_masked`` lists the retries
    this call would have journaled.
    """
    from nemo_gym.rollouts.rollout_collection import (
        NG_UNROUTED_RUN_ERROR_KEY,
        is_non_superseding_rollout_row,
        is_omitted_rollout_row,
        is_terminal_failure,
    )
    from nemo_gym.rollouts.rollout_recovery import (
        is_answered_agent_deadline,
        is_infrastructure_agent_timeout,
        masked_infrastructure_kind,
    )

    policy = outcome_policy or RecoveryOutcomePolicyConfig()
    managed = {fact.identity: fact for fact in managed_selection or []}

    expected: list[Identity] = []
    omitted: dict[Identity, str] = {}
    invalid_inventory = 0
    for row in materialized:
        identity = _identity(row)
        if identity is None:
            invalid_inventory += 1
        else:
            expected.append(identity)
            if label := accepted_stage_omission(row):
                omitted[identity] = label
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
    masked_without_artifact: list[Identity] = []
    rerun_masked: list[RerunMaskedApproval] = []
    # Reopened by this call and not yet counted by the Store: the Store facts do not describe them.
    asked: set[Identity] = set()
    # Results re-verified from their saved answers (``reverification_cause``): masked ones, and scored ones whose judge
    # gave no verdict or whose grading did not resolve (``judge_unparseable``). No failure attempt is spent. A scored
    # result whose grading was never recorded stays SCORED here (the trial gate reports it unobserved); the run's
    # completion pass grades it again.
    saved_results: set[Identity] = set()
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
            if not judge and identity in managed and is_infrastructure_agent_timeout(row):
                # A graded agent timeout whose window held a server-side fault (Harbor's graded zero): the Store
                # selects it as a failure and runs it again under the attempt cap. A run without a Store cannot
                # reopen a saved result, so there it stays scored and the gate counts its fault evaluation-side.
                fact = managed[identity]
                evidence.append(
                    IdentityEvidence(
                        identity=identity,
                        state=IdentityState.RETRYABLE
                        if fact.attempts < fact.max_attempts
                        else IdentityState.EXHAUSTED,
                        failure_kind=AGENT_TIMEOUT_INFRASTRUCTURE,
                        attempts=fact.attempts,
                        max_attempts=fact.max_attempts,
                        events=_events(row),
                    )
                )
                continue
            reward = row.get("reward")
            graded = isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
            if not judge and (row.get("mask_sample") is True or row.get("evaluation_completed") is False):
                if reverification_cause(row) is not None:
                    # Masked by a judge or verify-transport failure: its saved answer is graded again, never rerun.
                    saved_results.add(identity)
                    evidence.append(
                        IdentityEvidence(
                            identity=identity,
                            state=IdentityState.JUDGE_RECOVERABLE,
                            failure_kind=row.get("failure_kind"),
                            events=_events(row),
                        )
                    )
                    continue
                infrastructure = masked_infrastructure_kind(row)
                fact = managed.get(identity)
                spent = fact.attempts if fact is not None else 0
                kept = masked_without_artifact_kind(row) if infrastructure is None and fact is not None else None
                if kept is not None:
                    if fact.rerun_approved:
                        infrastructure = kept  # the Store counts the approved row as a failed attempt
                    elif rerun_masked_without_artifact:
                        infrastructure, spent = kept, fact.attempts + 1
                        asked.add(identity)
                    else:
                        masked_without_artifact.append(identity)
                        evidence.append(
                            IdentityEvidence(
                                identity=identity,
                                state=IdentityState.MASKED,
                                failure_kind=row.get("failure_kind"),
                                reason=RERUN_MASKED_REASON,
                            )
                        )
                        continue
                if infrastructure is not None and fact is not None:
                    # Infrastructure masked it: the Store counts it as a failed attempt and runs it again, and the new
                    # attempt supersedes it. The policy decides it as it decides a failure of that kind.
                    action = choose_action(failure_kind=infrastructure, http_status=None, config=policy)
                    if action == OutcomeAction.FAIL:
                        # The Store would still run it again, so a policy that fails it stops the run.
                        policy_stopped = True
                        state = IdentityState.TERMINAL
                    elif action == OutcomeAction.RETRY:
                        state = IdentityState.RETRYABLE if spent < fact.max_attempts else IdentityState.EXHAUSTED
                    else:
                        state = IdentityState.UNKNOWN
                    if identity in asked and state == IdentityState.RETRYABLE:
                        rerun_masked.append(
                            RerunMaskedApproval(
                                identity=identity,
                                attempts=fact.attempts,
                                attempt=int(row.get("_ng_attempt_index") or 0),
                            )
                        )
                    evidence.append(
                        IdentityEvidence(
                            identity=identity,
                            state=state,
                            failure_kind=infrastructure,
                            attempts=spent,
                            max_attempts=fact.max_attempts,
                            reason=(
                                "the policy verifies a masked result that has no saved judge answer"
                                if state == IdentityState.UNKNOWN
                                else None
                            ),
                        )
                    )
                    continue
                # Collection reuses a masked result as completed: it is retained, never run again, and not measured.
                evidence.append(
                    IdentityEvidence(
                        identity=identity, state=IdentityState.MASKED, failure_kind=row.get("failure_kind")
                    )
                )
                continue
            if not judge and not graded:
                evidence.append(
                    IdentityEvidence(
                        identity=identity,
                        state=IdentityState.UNKNOWN,
                        failure_kind=row.get("failure_kind") or "result_without_usable_grade",
                    )
                )
                continue
            if not judge and reverification_cause(row) == REVERIFY_JUDGE:
                # Scored, but its reward is no grade: the judge gave no verdict or the grading did not resolve. Its
                # saved answer is judged again, never rerun.
                saved_results.add(identity)
                evidence.append(
                    IdentityEvidence(
                        identity=identity,
                        state=IdentityState.JUDGE_RECOVERABLE,
                        failure_kind=JUDGE_UNPARSEABLE,
                        events=_events(row),
                        reason=SCORED_UNPARSEABLE_REASON,
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
        if identity in omitted:
            # Its stage was accepted without it, so collection never runs it again: neither scored nor a failure.
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=IdentityState.OMITTED,
                    failure_kind=ACCEPTED_STAGE_OMISSION,
                    reason=omitted[identity],
                )
            )
            continue
        standing = [row for row in sidecar.get(identity, []) if not is_non_superseding_rollout_row(row)]
        unrouted = [row for row in sidecar.get(identity, []) if row.get(NG_UNROUTED_RUN_ERROR_KEY) is True]
        if not standing and unrouted:
            # A diagnostic does not spend a failure attempt. Explicit not-sent evidence admits a retry, and so does
            # a possibly delivered request that kept no response; the replay gate refuses any other delivery.
            last = unrouted[-1]
            fact = managed.get(identity)
            spent = fact.attempts if fact else 0
            cap = fact.max_attempts if fact else default_max_attempts
            state = IdentityState.UNKNOWN
            if not _policy_replay_is_unsupported(last):
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
                    reason=_replay_refusal(last) if state == IdentityState.UNKNOWN else None,
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
        kind = _retry_kind(last)
        reason = None
        unscored = False
        # A kind the producer did not record, or did not register, is read from the recorded cause instead.
        derived = None if _known_kind(kind) else _cause_kind(last)
        if derived is not None and derived[0] is not None:
            kind = derived[0]
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
        elif is_answered_agent_deadline(last):
            # The agent's own deadline ended it with an answer kept: the model's result, which no resume runs again.
            # Collection's metrics score it only when its producer marked it a counted zero.
            state = IdentityState.MODEL_FINAL
            unscored = last.get(COUNTS_AS_ZERO) is not True
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
        elif not _known_kind(kind):
            state = IdentityState.UNKNOWN
            reason = f"unregistered kind {_retry_kind(last)}" if _retry_kind(last) else "no failure kind"
            if derived is not None and derived[2]:
                reason = f"{reason}: {derived[2]}"
        elif derived is not None and derived[1]:
            # The policy's own request failed: the model's final result, unless a resume would run it again.
            if last.get(TERMINAL) is True or _policy_replay_is_unsupported(last):
                state = IdentityState.MODEL_FINAL
            else:
                state = IdentityState.UNKNOWN
                reason = f"the policy's request failed ({kind}), a final model result, but a resume would run it again"
        elif last.get(TERMINAL) is True:
            state = IdentityState.TERMINAL
        elif (
            action := choose_action(
                failure_kind=kind or "unknown", http_status=_policy_evidence(last)[1], config=policy
            )
        ) == OutcomeAction.FAIL:
            # Unrestricted resume would still retry this nonterminal saved row.
            policy_stopped = True
            state = IdentityState.TERMINAL
        elif action == OutcomeAction.VERIFY or _policy_replay_is_unsupported(last):
            # Only a judge failure has a supported saved-answer path. A failed verification can retain a finished
            # agent answer, so replay must not generate another. For collection-wrapped model failures, the sidecar projection
            # has no durable model-operation budget accounting: replay must not silently replenish that budget.
            state = IdentityState.UNKNOWN
            reason = _replay_refusal(last) or "the policy verifies a failure that has no saved judge answer"
        elif spent < cap:
            state = IdentityState.RETRYABLE
        else:
            state = IdentityState.EXHAUSTED
        evidence.append(
            IdentityEvidence(
                identity=identity,
                state=state,
                # The kind its retry selects on stays agent_timeout; the counted kind names the server-side fault.
                failure_kind=AGENT_TIMEOUT_INFRASTRUCTURE if is_infrastructure_agent_timeout(last) else kind,
                http_status=_policy_evidence(last)[1],
                failure_id=last.get(FAILURE_ID) if isinstance(last.get(FAILURE_ID), str) else None,
                attempts=spent,
                max_attempts=cap,
                events=_events(last),
                reason=reason if state == IdentityState.UNKNOWN else None,
                unscored=unscored,
            )
        )

    for item in evidence:
        fact = managed.get(item.identity)
        if fact is None:
            continue
        if item.identity in asked:
            # Not run by the Store until the approval is journaled: its facts say nothing of the row yet.
            item.max_attempts = fact.max_attempts
            continue
        item.attempts, item.max_attempts = fact.attempts, fact.max_attempts
        if (
            item.state == IdentityState.JUDGE_RECOVERABLE
            and item.identity not in saved_results
            and fact.attempts >= fact.max_attempts
        ):
            item.state = IdentityState.EXHAUSTED
        elif item.state == IdentityState.MISSING and (fact.reserved or not fact.pending):
            # A reservation without an outcome stays unknown while the allocation that held it may be alive or its
            # end is unknown. Once it has ended, no row of that attempt exists and the Store runs it again.
            if not (allocation_cut and fact.reserved and fact.pending):
                item.state = IdentityState.UNKNOWN
                item.failure_kind = item.failure_kind or RESERVATION_WITHOUT_OUTCOME
        elif item.state == IdentityState.RETRYABLE and not fact.pending:
            if fact.attempts >= fact.max_attempts:
                item.state = IdentityState.EXHAUSTED
            else:
                # The Store reader runs at the run's pinned Gym revision and applies that revision's replay rules.
                item.state, item.reason = IdentityState.UNKNOWN, STORE_DOES_NOT_REPLAY

    by_state: dict[str, list[Identity]] = {}
    for item in evidence:
        by_state.setdefault(item.state.value, []).append(item.identity)
    counts = {state: len(ids) for state, ids in by_state.items()}
    duplicates = sorted(i for i, n in seen.items() if n > 1)

    def decision(action: RecoveryAction, reason: str, **fields: Any) -> RecoveryDecision:
        if masked_without_artifact:
            reason = (
                f"{reason}; {len(masked_without_artifact)} masked result(s) without a gradeable artifact rerun only "
                f"with {RERUN_MASKED_FLAGS} TEXT"
            )
        return RecoveryDecision(
            action=action,
            reason=reason,
            masked_without_artifact=sorted(masked_without_artifact),
            rerun_masked=sorted(rerun_masked, key=lambda approval: approval.identity),
            expected=len(expected_set),
            counts=counts,
            identities=by_state,
            evidence=evidence,
            duplicate_results=duplicates,
            **fields,
        )

    # A live job is still writing the files read above, whatever they show now: classify nothing from them.
    if budget.live_job:
        live = decision(RecoveryAction.NOOP, _live_reason(budget.live_job_name))
        return live.model_copy(update={"scheduler_live": True})
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
    # Only what rollout collection would run again: a resume keeps the per-rollout attempts, so it gates an
    # exhausted identity (`_load_from_cache` maxed_out) and queueing for it alone runs nothing.
    needs_policy = sum(counts.get(s, 0) for s in (IdentityState.MISSING, IdentityState.RETRYABLE))
    populations = _populations(counts, evidence)
    if counts.get(IdentityState.UNKNOWN):
        # A resume never runs an identity the Store does not select as pending, nor a saved result row (both
        # collection paths gate it); any other unknown identity would run again without an owner.
        rerun = [
            item.identity
            for item in evidence
            if item.state == IdentityState.UNKNOWN
            and (managed[item.identity].pending if item.identity in managed else item.identity not in result_rows)
        ]
        if not needs_policy or rerun:
            unknown = [item for item in evidence if item.state == IdentityState.UNKNOWN]
            reserved = sum(item.failure_kind == RESERVATION_WITHOUT_OUTCOME for item in unknown)
            parts = []
            if len(unknown) > reserved:
                kinds = Counter(
                    item.failure_kind or "no kind"
                    for item in unknown
                    if item.failure_kind != RESERVATION_WITHOUT_OUTCOME
                )
                named = ", ".join(kind for kind, _ in kinds.most_common(3)) + (", ..." if len(kinds) > 3 else "")
                parts.append(
                    f"{len(unknown) - reserved} failure(s) ({named}) have no registered kind, no supported action or "
                    "no rerun in the Store, so nobody owns them"
                )
            if reserved:
                parts.append(
                    f"{reserved} reservation(s) have no saved outcome and the allocation that made them is not known "
                    "to have ended, so whether they ran is unknown"
                )
            return decision(
                RecoveryAction.STOP,
                "; ".join(parts)
                + (f" and a resume would run {len(rerun)} of them again" if needs_policy and rerun else "")
                + f"; {populations}",
            )
    if policy_stopped:
        return decision(
            RecoveryAction.STOP,
            "the policy stops an ungraded failure; mixed dispatch cannot enforce separate task decisions",
        )
    if not needs_policy:
        if counts.get(IdentityState.JUDGE_RECOVERABLE):
            if saved_results:
                unparsed = sum(1 for item in evidence if item.reason == SCORED_UNPARSEABLE_REASON)
                why = [
                    *(
                        [f"{len(saved_results) - unparsed} masked by a judge or verify-transport failure"]
                        if len(saved_results) > unparsed
                        else []
                    ),
                    *([f"{unparsed} scored, but the {SCORED_UNPARSEABLE_REASON}"] if unparsed else []),
                ]
                return decision(
                    RecoveryAction.REVERIFY,
                    f"{counts[IdentityState.JUDGE_RECOVERABLE]} saved answer(s) only need grading again "
                    f"({'; '.join(why)}): re-verify "
                    "the saved answers with `gym eval reverify --judge-failed-only --settle-into-source`",
                )
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
                "resume would run nothing; the run is not complete; "
                + _EXTEND_RETRIES_SCOPE.format(
                    cap="/".join(
                        str(cap)
                        for cap in sorted(
                            {item.max_attempts for item in evidence if item.state == IdentityState.EXHAUSTED} - {None}
                        )
                    )
                )
                + f"; {EXTEND_ATTEMPTS_HINT}",
            )
        if counts.get(IdentityState.MASKED):
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.MASKED]} completed result(s) are masked and not measured: they are kept and a "
                f"resume would run nothing; the run is not complete; {populations}",
            )
        if counts.get(IdentityState.OMITTED):
            return decision(
                RecoveryAction.NOOP,
                f"every identity is scored, a final model failure or omitted by an accepted partial stage: "
                f"{omission_summary(evidence)}",
            )
        return decision(RecoveryAction.NOOP, "every identity is scored or a final model failure")
    # Saved answers to grade again and policy work: the resumed run re-verifies the saved answers first, while its
    # servers are up, and then runs the missing and retryable identities.
    reverify_first = sorted(by_state.get(IdentityState.JUDGE_RECOVERABLE.value, []))
    if budget.live_job is None or budget.max_infra_retries is None or budget.infra_retries_used is None:
        return decision(RecoveryAction.STOP, UNKNOWN_BUDGET_REASON, accounting_unknown="budget")
    segment = planned_missing_continuation and bool(managed_selection) and not counts.get(IdentityState.RETRYABLE)
    if not segment and budget.infra_retries_used >= budget.max_infra_retries:
        # A chain stops once less than a minute of its walltime remains, so name that budget too when it is spent.
        walltime_spent = (
            budget.walltime_capped
            and budget.walltime_used_s is not None
            and budget.max_walltime_s is not None
            and budget.max_walltime_s - budget.walltime_used_s < 60
        )
        return decision(
            RecoveryAction.STOP,
            f"{_INFRA_BUDGET_USED_PREFIX} ({budget.infra_retries_used} of {budget.max_infra_retries})"
            + (
                f", and the stored walltime budget is used ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s"
                "; less than a minute is left): raise both in one call"
                if walltime_spent
                else ""
            ),
        )
    if budget.walltime_capped:
        if budget.walltime_used_s is None or budget.max_walltime_s is None:
            return decision(RecoveryAction.STOP, UNKNOWN_WALLTIME_REASON, accounting_unknown="walltime")
        if budget.walltime_used_s >= budget.max_walltime_s:
            return decision(
                RecoveryAction.STOP,
                f"the stored walltime budget is used ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s)",
            )
    return decision(
        RecoveryAction.RESUBMIT,
        (
            f"{len(reverify_first)} saved answer(s) are re-verified first by the resumed run, then "
            if reverify_first
            else ""
        )
        + f"{needs_policy} identity(ies) need the policy again: a Gym resume runs only missing and retryable ones"
        + (f"; {populations}" if counts.get(IdentityState.UNKNOWN) or counts.get(IdentityState.MASKED) else ""),
        reverify_first=reverify_first,
    )


_INFRA_BUDGET_USED_PREFIX = "the stored infra-retry budget is used"
_EXTEND_RETRIES_SCOPE = (
    "--extend-retries raises infra retries only; it does not raise the per-identity attempt cap "
    "(max_rollout_attempts={cap})"
)


def omission_summary(evidence: Iterable[IdentityEvidence]) -> str:
    """The omitted identities by stage and policy, e.g. "1 omitted by an accepted partial stage (stage_index 1, tolerate_unresolved)"."""
    stages = Counter(item.reason for item in evidence if item.state == IdentityState.OMITTED)
    return ", ".join(f"{n} omitted by an accepted partial stage {stage}" for stage, n in sorted(stages.items()))


def _unknown_label(item: IdentityEvidence) -> str:
    """The kind of an unknown identity with the rule that refused it, without naming the kind twice."""
    if item.reason and item.failure_kind and item.failure_kind not in item.reason:
        return f"{item.failure_kind}: {item.reason}"
    return item.reason or item.failure_kind or "no kind"


def _populations(counts: Mapping[str, int], evidence: list[IdentityEvidence]) -> str:
    """What a resume runs, what nobody owns (by kind, largest first) and the saved rows kept as they are."""
    retryable = counts.get(IdentityState.MISSING, 0) + counts.get(IdentityState.RETRYABLE, 0)
    retained = counts.get(IdentityState.SCORED, 0) + counts.get(IdentityState.MODEL_FINAL, 0)
    kinds = Counter(_unknown_label(item) for item in evidence if item.state == IdentityState.UNKNOWN)
    detail = ", ".join(f"{n} {kind}" for kind, n in kinds.most_common(3))
    masked = counts.get(IdentityState.MASKED, 0)
    return (
        f"retryable {retryable} (resumed), unknown {counts.get(IdentityState.UNKNOWN, 0)}"
        + (f" ({detail}{', ...' if len(kinds) > 3 else ''})" if detail else "")
        + f" (reported, not retried), retained {retained} (saved rows kept)"
        + (f", masked {masked} (completed, not measured, kept)" if masked else "")
        + (
            f", omitted {omitted} (accepted partial stage, not run again)"
            if (omitted := counts.get(IdentityState.OMITTED, 0))
            else ""
        )
    )


def _live_reason(live_job_name: str | None) -> str:
    if live_job_name and live_job_name.startswith("judge-only job"):
        return f"{live_job_name} live: it grades the run's saved answers again; nothing to decide"
    if live_job_name:
        return f"continuation {live_job_name} live; nothing to decide"
    return "a job of this run is still pending or running; nothing to decide, it is not queued twice"
