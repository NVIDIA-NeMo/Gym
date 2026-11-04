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
"""Whether `gym eval resubmit` may queue a run again: ``decide_from_collected`` turns the evidence
``COLLECT_SCRIPT`` read (``CollectedEvidence``) into one ``RecoveryDecision``.

It classifies every expected identity (``rows``), applies the run's budgets and an operator's explicit budget,
and names in the decision's reason what the run's chain, its drift and any raised limit add. It reads no files
and starts nothing.
"""

import math
from typing import Any

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.eval.recovery.budgets import (
    OperatorBudget,
    RecoveryBudget,
    _extend_attempts_refusal,
    _rerun_masked_refusal,
    _saved_attempt_cap,
    _with_attempt_extension,
)
from nemo_gym.eval.recovery.decision import (
    FRESH_RESTART_REASON,
    LIVE_JOB_STATES,
    OPERATOR_CONTINUATION,
    UNKNOWN_BUDGET_HINT,
    IdentityEvidence,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    failed_judge_jobs_blocking,
    failed_judge_jobs_stop,
)
from nemo_gym.eval.recovery.evidence import (
    STORE_READER_UNAVAILABLE,
    UNOBSERVED_CLAIM_REASON,
    UNREADABLE_CLAIM_REASON,
    CollectedEvidence,
    _identity,
)
from nemo_gym.eval.recovery.markers import ACCUMULATED_WALLTIME_NAME, RETRY_EXTENSION_NAME, RETRY_FILE_NAME
from nemo_gym.eval.recovery.policy import (
    OutcomePolicySource,
    RecoveryOutcomePolicyConfig,
)
from nemo_gym.eval.recovery.rows import (
    _INFRA_BUDGET_USED_PREFIX,
    _live_reason,
    decide_recovery,
)
from nemo_gym.utils.path_utils import RERUN_MASKED_JOURNAL_NAME


def live_chain_job(rows: list[tuple[str, str, int, int]] | None, job_chain: list[str]) -> str | None:
    """The chain job that still waits or runs, as 'id (STATE)' from its latest incarnation; None when none does."""
    latest: dict[str, tuple[int, str]] = {}
    for job_id, state, _, restarts in rows or []:
        if job_id not in latest or restarts >= latest[job_id][0]:
            latest[job_id] = (restarts, state)
    for job_id in reversed(job_chain):
        if job_id in latest and latest[job_id][1] in LIVE_JOB_STATES:
            return f"{job_id} ({latest[job_id][1]})"
    return None


def _operator_continuation(
    stopped: RecoveryDecision,
    collected: CollectedEvidence,
    inputs: dict[str, Any],
    operator: OperatorBudget,
    *,
    configured_retries: int | None,
    max_walltime_s: float | None,
    extend_retries: int = 0,
) -> RecoveryDecision:
    """Decide again under the operator's explicit budget, or keep the stop when continuing could be unsafe.

    The caller has checked that the scheduler answered and shows no live or pending job of the run. It fills only
    unreadable counters: an unknown spent-retry count so that ``operator.retries`` retries remain under the cap, an
    unknown used walltime with zero, so the walltime cap is fresh for this continuation, and an unknown extension with
    zero. A readable counter keeps its value. When the spent-retry count is readable, an explicit ``extend_retries``
    (``--extend-retries``) is what it is on any recovery: that many more infra retries, recorded as the run's retry
    extension in the continuation's writes and journal. Saved rows are classified as for any decision and never
    changed.
    """

    def stop(why: str) -> RecoveryDecision:
        return stopped.model_copy(update={"reason": f"{stopped.reason}; {why}"})

    if collected.accounting_unknown_reason in (UNOBSERVED_CLAIM_REASON, UNREADABLE_CLAIM_REASON):
        return stop("a follow-up claim names no job, so an operator continuation cannot rule out a queued follow-up")
    states = collected.counter_states
    if states is None:
        return stop("the chain's accounting files changed while they were read, so an operator continuation waits")
    if configured_retries is None:
        return stop("the run configures no infra-retry cap, so an operator continuation has no budget to fill")
    unknown: list[str] = []
    writes: dict[str, dict[str, Any]] = {}
    if collected.infra_retry_extension_state in ("absent", "ok"):
        extension = int(collected.infra_retry_extension or 0)
    else:
        extension = 0
        unknown.append("infra_retry_extension")
        writes[RETRY_EXTENSION_NAME] = {"previous": "unknown", "value": 0}
    readable_retries = states.get("infra_retries") in ("absent", "ok")
    extended = extend_retries if readable_retries else 0
    if extended:
        # The spent count is known, so N cannot fill it: it raises the cap, as --extend-retries does elsewhere.
        writes[RETRY_EXTENSION_NAME] = {
            "previous": "unknown" if "infra_retry_extension" in unknown else extension,
            "value": extension + extended,
        }
        extension += extended
    cap = configured_retries + extension
    if readable_retries:
        used = int(collected.infra_retries_used or 0)
    else:
        used = max(cap - operator.retries, 0)
        unknown.append("infra_retries_used")
        writes[RETRY_FILE_NAME] = {"previous": "unknown", "value": used}
    if states.get("walltime") in ("absent", "ok"):
        walltime = float(collected.walltime_used_s or 0)
    else:
        walltime = 0.0
        unknown.append("walltime_used_s")
        writes[ACCUMULATED_WALLTIME_NAME] = {"previous": "unknown", "value": 0}
    budget = RecoveryBudget(
        max_infra_retries=cap,
        infra_retries_used=used,
        max_walltime_s=max_walltime_s,
        walltime_capped=max_walltime_s is not None,
        walltime_used_s=walltime,
        live_job=False,
        job_chain=collected.job_chain,
    )
    decision = decide_recovery(**inputs, budget=budget)
    remaining = max(cap - used, 0)
    summary = (
        f"{OPERATOR_CONTINUATION} by {operator.user} ({operator.reason}): {remaining} infra "
        f"retr{'y' if remaining == 1 else 'ies'} remain"
        + (f" ({extended} more recorded as the run's retry extension)" if extended else "")
        + f" and the walltime cap is {operator.max_walltime}; it fills "
        f"{', '.join(sorted(writes)) or 'no counter'} and keeps every other readable counter"
    )
    if decision.action != RecoveryAction.RESUBMIT:
        return decision.model_copy(update={"reason": f"{decision.reason}; {summary}, but nothing is resubmitted"})
    record = {
        "kind": OPERATOR_CONTINUATION,
        "user": operator.user,
        "reason": operator.reason,
        "budget": {
            "infra_retries_remaining": remaining,
            "infra_retries_requested": operator.retries,
            "max_walltime": operator.max_walltime,
            **({"infra_retry_extension_added": extended} if extended else {}),
        },
        "unknown": unknown,
        "accounting_unknown_reason": collected.accounting_unknown_reason,
        "writes": writes,
        "epoch": {"chain_length": len(collected.job_chain), "walltime_s": walltime},
    }
    return decision.model_copy(update={"reason": f"{decision.reason}; {summary}", "operator_continuation": record})


# At most this many drifted settings are named in a decision reason.
_DRIFT_KEYS_SHOWN = 20


def resume_drift_note(keys: list[str] | None) -> str | None:
    """The sentence a decision reason carries when the run's latest resume crossed operational drift."""
    if not keys:
        return None
    shown = sorted(set(keys))
    more = len(shown) - _DRIFT_KEYS_SHOWN
    listed = ", ".join(shown[:_DRIFT_KEYS_SHOWN]) + (f" and {more} more" if more > 0 else "")
    return f"resumed across operational drift: {listed}"


def decide_from_collected(collected: CollectedEvidence, **options: Any) -> RecoveryDecision:
    """``_decide_from_collected``, with the reason naming any operational drift the run's latest resume crossed (see
    ``resume_drift_note``) and a continuation that ended before it queued its follow-up, so an inspection, a recovery
    and a chain continuation all say it."""
    explicit = options.get("outcome_policy")
    source = options.pop("outcome_policy_source", None) or (
        OutcomePolicySource.PYTHON if explicit is not None else OutcomePolicySource.DEFAULT
    )
    policy = explicit or RecoveryOutcomePolicyConfig()
    decision = _decide_from_collected(collected, **options).model_copy(
        update={"outcome_policy": {"source": OutcomePolicySource(source).value, **policy.model_dump(mode="json")}}
    )
    notes = [resume_drift_note(collected.resume_drift_keys)]
    blocked = failed_judge_jobs_blocking(collected.judge_jobs, collected.judge_job_acknowledgements)
    if blocked is not None and decision.action == RecoveryAction.REVERIFY:
        notes.append(failed_judge_jobs_stop(blocked, "the run's reverification directory"))
    if collected.chain_continuation_note:
        notes.append(f"the chain recorded: {collected.chain_continuation_note}")
    notes = [note for note in notes if note]
    return decision if not notes else decision.model_copy(update={"reason": "; ".join([decision.reason, *notes])})


# The saved rows a decision reads; one that exists but cannot be read stops it.
SAVED_ROW_FILES = ("rollouts_materialized_inputs.jsonl", "rollouts.jsonl", "rollouts_failures.jsonl")


def _decide_from_collected(
    collected: CollectedEvidence,
    *,
    max_infra_retries: int | None,
    max_walltime_s: float | None,
    live_job: bool | None,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    planned_missing_continuation: bool = False,
    allocation_cut: bool = False,
    extend_retries: int = 0,
    live_job_name: str | None = None,
    operator_budget: OperatorBudget | None = None,
    extend_attempts: int = 0,
    rerun_masked_without_artifact: bool = False,
) -> RecoveryDecision:
    """Decide from what ``COLLECT_SCRIPT`` read; a missing result file stops the decision.

    ``extend_attempts`` is the attempts this recovery asks to add to every rollout's cap (``--extend-attempts``):
    the Store's facts are read as they would be once it is recorded (see ``_with_attempt_extension``).

    ``rerun_masked_without_artifact`` asks to run again the masked results that kept a saved answer but have no gradeable
    artifact and no grade (see ``decide_recovery``); the run's Store reader must apply the approval, or the ask is refused.

    When the chain accounting is unknown, ``operator_budget`` may continue the run instead (see
    ``_operator_continuation``); without it, the stop names the options that allow that.

    A run with no inventory and no saved row is a fresh restart once no job of it is live. ``allocation_cut`` (see
    ``decide_recovery``) applies only with no live job. The infra-retry cap is the configured one plus the run's
    recorded extension plus ``extend_retries`` that this recovery asks to record.
    """
    recorded = {"absent": 0, "ok": collected.infra_retry_extension}.get(collected.infra_retry_extension_state)
    configured = max_infra_retries
    if max_infra_retries is not None:
        max_infra_retries = None if recorded is None else max_infra_retries + int(recorded) + extend_retries
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
        live_job_name=live_job_name,
        job_chain=collected.job_chain,
    )
    unreadable_rows = [name for name in SAVED_ROW_FILES if f"artifacts/{name}" in collected.unreadable_files]
    if unreadable_rows and live_job is not True:
        # Saved rows that cannot be read are unknown, not absent: no fresh restart, no resume over them.
        known = sorted({i for i in map(_identity, collected.materialized or []) if i is not None})
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=f"saved rows are unreadable ({', '.join(unreadable_rows)}), so what they hold is unknown",
            expected=len(known),
            counts={IdentityState.UNKNOWN.value: len(known)} if known else {},
            identities={IdentityState.UNKNOWN.value: known} if known else {},
            evidence=[IdentityEvidence(identity=i, state=IdentityState.UNKNOWN) for i in known],
        )
    if collected.managed_store and collected.managed_selection is None:
        # Selecting from these files here could pick a superseded attempt or lose a reservation, so every known
        # input stays unknown and no saved row is read as its outcome.
        known = sorted({i for i in map(_identity, collected.materialized or []) if i is not None})
        unread = dict(
            expected=len(known),
            counts={IdentityState.UNKNOWN.value: len(known)} if known else {},
            identities={IdentityState.UNKNOWN.value: known} if known else {},
            evidence=[IdentityEvidence(identity=i, state=IdentityState.UNKNOWN) for i in known],
        )
        if collected.managed_reader_unavailable:
            # The reader read nothing, so this is no verdict on the artifacts: a live job keeps the run live, and
            # otherwise nothing is decided until the reader runs.
            if live_job:
                return RecoveryDecision(
                    action=RecoveryAction.NOOP, reason=_live_reason(live_job_name), scheduler_live=True, **unread
                )
            return RecoveryDecision(
                action=RecoveryAction.UNKNOWN,
                reason=f"{STORE_READER_UNAVAILABLE}: {collected.managed_reader_error or 'no cause reported'}; "
                "inspect again once the reader can run",
                **unread,
            )
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=(
                "the run was written by a rollout store (an attempt manifest, an attempt journal or run-tagged rows); "
                "an authoritative Store reader is unavailable or could not verify these artifacts; recovery stops"
                + (f": {collected.managed_reader_error}" if collected.managed_reader_error else "")
            ),
            **unread,
        )
    if collected.materialized is None:
        if live_job is False and not collected.results and not collected.failures:
            return RecoveryDecision(
                action=RecoveryAction.FRESH_RESTART,
                reason=FRESH_RESTART_REASON,
                expected=0,
                counts={},
                identities={},
                evidence=[],
            )
        live = live_job is True
        return RecoveryDecision(
            action=RecoveryAction.NOOP if live else RecoveryAction.STOP,
            reason=_live_reason(live_job_name) if live else "the run has no materialized inventory to recover against",
            expected=0,
            counts={},
            identities={},
            evidence=[],
            scheduler_live=live,
        )
    attempts_refused = _extend_attempts_refusal(collected) if extend_attempts else None
    rerun_refused = _rerun_masked_refusal(collected) if rerun_masked_without_artifact else None
    inputs = dict(
        materialized=collected.materialized,
        results=collected.results or [],
        failures=collected.failures,
        resubmit_offset=collected.resubmit_offset,
        outcome_policy=outcome_policy,
        managed_selection=collected.managed_selection
        if attempts_refused
        else _with_attempt_extension(collected.managed_selection, extend_attempts),
        planned_missing_continuation=planned_missing_continuation,
        allocation_cut=allocation_cut and live_job is False,
        rerun_masked_without_artifact=rerun_masked_without_artifact and rerun_refused is None,
    )
    decision = decide_recovery(**inputs, budget=budget)
    if rerun_refused:
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": f"--rerun-masked-without-artifact is refused: {rerun_refused}",
            }
        )
    if decision.rerun_masked and decision.action == RecoveryAction.RESUBMIT:
        decision = decision.model_copy(
            update={
                "reason": f"{decision.reason}; {len(decision.rerun_masked)} masked result(s) without a gradeable "
                f"artifact run again on the operator approval ({RERUN_MASKED_JOURNAL_NAME})"
            }
        )
    if attempts_refused:
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": f"--extend-attempts {extend_attempts} is refused: {attempts_refused}",
            }
        )
    saved = _saved_attempt_cap(collected)
    if extend_attempts and decision.action == RecoveryAction.RESUBMIT and saved is not None:
        applied = collected.attempt_extension_applied or 0
        decision = decision.model_copy(
            update={
                "reason": f"{decision.reason} (attempt cap {saved + applied + extend_attempts}: {saved} configured + "
                f"{applied} recorded extension + {extend_attempts} requested now, for rollouts the Store runs again)"
            }
        )
    if collected.unresolved_rejudgments and live_job is None:
        # With no job of the run live, the pass that admitted them died: its rows are still the selected judge
        # failures, re-verified like any other, and the recovery pass withdraws the admissions first. A live job may
        # still be in that pass (NOOP above); an unknown scheduler rules out neither.
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": (
                    f"admitted re-judgment outcome unknown for {collected.unresolved_rejudgments}: "
                    "the scheduler query failed, so the pass that admitted them may still run"
                ),
            }
        )
    if collected.results is None and not decision.scheduler_live:
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": "the results file is absent; collection cannot safely resume its saved history",
            }
        )
    unknown = decision.action == RecoveryAction.STOP and decision.accounting_unknown is not None
    if operator_budget is not None:
        # An operator continuation needs the scheduler's answer that no job of the run waits or runs.
        accounting_unknown = (
            budget.max_infra_retries is None
            or budget.infra_retries_used is None
            or (budget.walltime_capped and budget.walltime_used_s is None)
        )
        if live_job is None:
            why = "the scheduler query failed, so an operator continuation cannot rule out a live or pending job"
            return decision.model_copy(
                update={
                    "action": RecoveryAction.STOP,
                    "reason": f"{decision.reason}; {why}" if decision.action == RecoveryAction.STOP else why,
                }
            )
        if decision.scheduler_live and accounting_unknown:
            job = f"continuation {live_job_name}" if live_job_name else "a job of this run"
            why = f"the chain accounting is unknown and {job} is still pending or running"
            return decision.model_copy(
                update={"action": RecoveryAction.STOP, "reason": f"{why}, so an operator continuation is refused"}
            )
        if unknown:
            return _operator_continuation(
                decision,
                collected,
                inputs,
                operator_budget,
                configured_retries=configured,
                max_walltime_s=max_walltime_s,
                extend_retries=extend_retries,
            )
    # The cap note explains an infra-cap decision (a resubmit, or the stored budget is used); a stop for the
    # identities' own states (exhausted, terminal, masked) does not depend on it.
    infra_cap_decision = decision.action == RecoveryAction.RESUBMIT or decision.reason.startswith(
        _INFRA_BUDGET_USED_PREFIX
    )
    if infra_cap_decision and max_infra_retries is not None and max_infra_retries != configured:
        extension = (
            f"infra-retry cap {max_infra_retries}: {configured} configured + {int(recorded or 0)} recorded extension"
            + (f" + {extend_retries} requested now" if extend_retries else "")
        )
        decision = decision.model_copy(update={"reason": f"{decision.reason} ({extension})"})
    if (
        max_walltime_s is not None
        and walltime is not None
        and len(collected.job_chain) > 1
        and collected.walltime_state == "absent"
        and not decision.scheduler_live
    ):
        derived = (
            f"the used walltime {walltime:g} s is the sum of the chain's {len(collected.job_chain)} scheduler "
            "records, as no allocation stored it"
        )
        decision = decision.model_copy(update={"reason": f"{decision.reason}; {derived}"})
    if unknown:
        decision = decision.model_copy(update={"reason": f"{decision.reason}\n{UNKNOWN_BUDGET_HINT}"})
    return decision


RecoveryOutcome.model_rebuild()


__getattr__ = moved_names_getattr(__name__)
