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
"""The recovery decision: identity states, run-level actions, and the decision and outcome records."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nemo_gym.core.failure_kinds import (
    is_namespaced,
    is_registered,
)
from nemo_gym.eval.recovery.evidence import (
    OPERATOR_CONTINUATION_JOURNAL_NAME,
    Identity,
)
from nemo_gym.utils.path_utils import RERUN_MASKED_JOURNAL_NAME


FAILURE_CLASS = "_ng_failure_class"
TERMINAL = "_ng_failure_terminal"
COUNTS_AS_ZERO = "_ng_failure_counts_as_zero"
MAX_ATTEMPTS = "_ng_failure_max_attempts"
FAILURE_EVENTS = "_ng_failure_events"
FAILURE_ID = "_ng_failure_id"
# Facts of a failure's recorded cause that the collector reads on the run host (see ``_cause_kind``).
COLLECTED_CAUSE = "_ng_collected_cause"


# Scheduler states of an allocation that has ended, by any path: a scheduler cut, a failed or completed driver
# (its no-progress exit included), or a cancel. Its in-flight requests ended with it, and a reservation with no saved
# row saved nothing a rerun could duplicate. An operator's cancel counts only for an explicit
# `gym eval resubmit --recover`: the in-job guard never continues it.
ALLOCATION_ENDED_STATES = frozenset(
    {
        "TIMEOUT",
        "PREEMPTED",
        "NODE_FAIL",
        "BOOT_FAIL",
        "CANCELLED",
        "FAILED",
        "COMPLETED",
        "OUT_OF_MEMORY",
        "DEADLINE",
    }
)
# The kind a reservation without a saved outcome gets while the allocation that made it may be alive or its end is
# unknown.
RESERVATION_WITHOUT_OUTCOME = "reservation_without_outcome"
# Why a failure the policy retries is unknown: the Store does not select it as pending, so a resume never runs it.
STORE_DOES_NOT_REPLAY = "the policy retries it but the run's Store does not plan it again"
# Scheduler states of a job that still waits or runs: a chain job in one of them is a live continuation.
LIVE_JOB_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESIZING",
        "STOPPED",
        "SIGNALING",
        "STAGE_OUT",
    }
)


class IdentityState(StrEnum):
    SCORED = "scored"
    MODEL_FINAL = "model_final"
    JUDGE_RECOVERABLE = "judge_recoverable"
    RETRYABLE = "retryable"
    EXHAUSTED = "exhausted"
    TERMINAL = "terminal"
    MISSING = "missing"
    MASKED = "masked"
    OMITTED = "omitted"
    UNKNOWN = "unknown"


class RecoveryAction(StrEnum):
    NOOP = "noop"
    REVERIFY = "reverify"
    RESUBMIT = "resubmit"
    # The run has no task inventory, so nothing to recover: it is queued again as a fresh run with a longer walltime.
    FRESH_RESTART = "fresh-restart"
    STOP = "stop"
    # Nothing was decided: the evidence could not be read (its Store reader could not run), so ask again.
    UNKNOWN = "unknown"


# Gym writes the task inventory before any rollout runs: a run that ended before it saved nothing a resume keeps.
NO_INVENTORY = "the run ended before the task inventory was written"
FRESH_RESTART_REASON = f"nothing to recover: {NO_INVENTORY}; resubmit it as a fresh run with a longer walltime"


# What an exhausted-attempts stop says an operator can do; the extension is explicit and journaled.
EXTEND_ATTEMPTS_HINT = (
    "an exhausted identity whose attempts went to infrastructure failures or allocation cuts gets N more with "
    "gym eval resubmit --recover --extend-attempts N (explicit and journaled); a model failure or a judge failure "
    "that kept its saved answer does not"
)
# The most attempts one recovery may add to every rollout's cap.
MAX_EXTEND_ATTEMPTS = 100


# The chain's accounting counters in the job dir (see resume_script); an operator continuation fills unknown ones.
INFRA_RETRIES_FILE = ".gym_infra_retries"
ACCUMULATED_WALLTIME_FILE = ".gym_accumulated_walltime"
INFRA_RETRY_EXTENSION_FILE = ".gym_infra_retry_extension"
# The planner's stops when the chain accounting is unknown; an operator continuation may lift them (see below).
UNKNOWN_BUDGET_REASON = "the job chain or scheduler state is unavailable, so the budget is unknown"
UNKNOWN_WALLTIME_REASON = "the run caps its walltime but the used walltime is unavailable"
UNKNOWN_BUDGET_HINT = (
    "An operator may continue it with an explicit budget: gym eval resubmit --recover --max-walltime HH:MM:SS "
    "--walltime-reason TEXT [--extend-retries N]."
)
OPERATOR_CONTINUATION = "OPERATOR-CONTINUATION"
# Written over the unknown-accounting marker just before the continuation is queued; it has no unknown prefix.
OPERATOR_CONTINUATION_NOTE = f"Operator continuation: an explicit budget filled the unknown accounting; see {OPERATOR_CONTINUATION_JOURNAL_NAME}."
# A masked result that kept a saved answer has neither a gradeable artifact nor a recorded grade, so no re-verification
# can measure it and a resume keeps it: it runs again only with the explicit, journaled approval of the operator
# (``RERUN_MASKED_JOURNAL_NAME``, in the run artifacts directory), which the Store reads.
RERUN_MASKED_FLAGS = "--rerun-masked-without-artifact --rerun-reason"
RERUN_MASKED_REASON = (
    f"retained, never run again, not measured: it reruns only with {RERUN_MASKED_FLAGS} TEXT "
    f"(journaled in {RERUN_MASKED_JOURNAL_NAME})"
)
# How many judge-only jobs in a row may end without settling a saved answer before recovery stops and names them.
JUDGE_JOB_FAILURE_BOUND = 2
ACKNOWLEDGE_JUDGE_JOBS_FLAGS = "--acknowledge-failed-judge-jobs --acknowledge-reason"


def _judge_job_unsettled(entry: Mapping[str, Any]) -> bool:
    return entry.get("state") == "failed" or (entry.get("state") == "completed" and not entry.get("settled"))


def failed_judge_jobs_blocking(
    judge_jobs: Sequence[Mapping[str, Any]], acknowledgements: Sequence[Mapping[str, Any]] = ()
) -> list[str | None] | None:
    """The ids of the last ``JUDGE_JOB_FAILURE_BOUND`` judge-only jobs when all ended without settling a saved answer and no
    acknowledgement names exactly them; else None. An acknowledgement covers the failed jobs it names, so a later pair of
    failures (the one more job it let queue, and the one before it) needs an acknowledgement of its own."""
    recent = list(judge_jobs)[-JUDGE_JOB_FAILURE_BOUND:]
    if len(recent) < JUDGE_JOB_FAILURE_BOUND or not all(_judge_job_unsettled(entry) for entry in recent):
        return None
    ids = [
        entry.get("job_id") if isinstance(entry.get("job_id"), str) and entry.get("job_id") else None
        for entry in recent
    ]
    if None not in ids and any(
        isinstance(ack.get("job_ids"), list) and set(ack["job_ids"]) == set(ids) for ack in acknowledgements
    ):
        return None
    return ids


def failed_judge_jobs_stop(ids: Sequence[str | None], where: str) -> str:
    """The stop text of ``failed_judge_jobs_blocking``: the jobs, where to read them, and the explicit way on."""
    named = ", ".join(str(job_id or "(id unknown)") for job_id in ids)
    text = (
        f"the last {JUDGE_JOB_FAILURE_BOUND} judge-only job(s) ({named}) ended without settling a saved "
        f"answer; read their receipts and logs under {where} before another is queued"
    )
    if None in ids:
        return text + "; a job without an id cannot be acknowledged"
    return (
        f'{text}; once read, `gym eval resubmit ... --recover {ACKNOWLEDGE_JUDGE_JOBS_FLAGS} "<why>"` allows one more'
    )


class RecoveryOutcome(BaseModel):
    """What ``recover`` did: ``dispatched`` names the one scheduler or judge action taken, None when nothing ran."""

    decision: "RecoveryDecision"
    ok: bool
    dispatched: str | None
    job_id: str | None = None
    unsupported: str | None = None
    # The acceptance gate task entry (status, expected/scored samples, score, errors, warnings); None without evidence.
    # Its status is RUNNING, PLANNED, RESUMING, FAILED, UNVERIFIED or SUCCESS (``gate_task_summary``).
    gate_task: dict[str, Any] | None = None
    # What was known about MLflow: the local files, and the server status only as the caller observed it.
    mlflow: dict[str, Any] | None = None
    # The re-verification part of a RESUBMIT that names ``reverify_first``: what runs it and when.
    reverify: str | None = None
    # The judge-only job a REVERIFY of a model-served run queued: its id, profile (kind, partition, pools, services
    # and QOS, each with its source), walltime and its bounds, the rows it re-judges, the Gym commit it runs and its
    # receipt path (the job dir's judge-job journal entry).
    judge_job: dict[str, Any] | None = None
    # The GYM_READER_TOKEN a login-host Store reader of this recovery carries; None for a caller that sets none.
    reader_token: str | None = None
    # For a run pinned at another Gym commit than the launcher's: the installed and pinned commits, whether the
    # recovery moves the run to the installed one (``repin``), whether --no-auto-repin refused that, and why.
    pin_plan: dict[str, Any] | None = None
    # A --max-walltime raise checked against the partition before any evidence is read: the partition, its MaxTime
    # (as sinfo prints it, None when unknown), the requested chain cap, the walltime each segment requests, why it
    # was refused (None when it fits), and a note when MaxTime is unknown or segments are shorter than the raise.
    walltime_plan: dict[str, Any] | None = None
    # The scheduler refused to queue the job: its text verbatim, and what the job dir holds after the exit trap.
    scheduler_refusal: dict[str, Any] | None = None


# Why a scored row is planned for a re-judge: its judge gave no verdict or its grading did not resolve. Health still counts
# it as an evaluation-side fault of a scored trial (``_health_view``) while the re-judge is pending.
SCORED_UNPARSEABLE_REASON = "judge gave no verdict (unparseable); re-judge planned"


class IdentityEvidence(BaseModel):
    identity: Identity
    state: IdentityState
    failure_kind: str | None = None
    # The HTTP status the selected failure recorded, when it recorded one.
    http_status: int | None = None
    failure_id: str | None = None
    attempts: int = 0
    max_attempts: int | None = None
    # (boundary, operation, operation_id, attempt_id, cause_id) of each saved failure event, in saved order.
    events: list[tuple[str | None, str | None, str | None, str | None, str | None]] = Field(default_factory=list)
    # Why an unknown identity has no supported action, when a specific rule refused it.
    reason: str | None = None
    # A settled final outcome the metric does not score: an answered agent deadline that its producer did not mark
    # as a counted zero. It is never run again, but it is no scored sample.
    unscored: bool = False


class RerunMaskedApproval(BaseModel):
    """One masked result the operator approves to run again: its identity, the attempts counted for it before the
    approval, and the attempt id of the masked row the approval covers (a later attempt is never covered)."""

    identity: Identity
    attempts: int = Field(ge=0)
    attempt: int = Field(ge=0)


class RecoveryDecision(BaseModel):
    action: RecoveryAction
    reason: str
    expected: int
    counts: dict[str, int]
    identities: dict[str, list[Identity]]
    evidence: list[IdentityEvidence]
    duplicate_results: list[Identity] = Field(default_factory=list)
    scheduler_live: bool = False
    # The OPERATOR-CONTINUATION record a RESUBMIT under an operator budget carries: who, why, the budget, the unknown
    # fields, the counter values it fills with their previous states, and its epoch. The resubmit journals it.
    operator_continuation: dict[str, Any] | None = None
    # A RESUBMIT of a run that also has saved answers to grade again names them: the resumed run re-verifies them
    # from their saved answers first, while its servers are up, and then runs the policy work.
    reverify_first: list[Identity] = Field(default_factory=list)
    # The outcome policy this decision applied: its ``source`` (default, config or python) and its entries.
    outcome_policy: dict[str, Any] | None = None
    # Masked results that kept a saved answer, with no gradeable artifact and no recorded grade
    # (``masked_without_artifact_kind``), that a resume keeps: they rerun only with ``RERUN_MASKED_FLAGS``. Their
    # state stays MASKED.
    masked_without_artifact: list[Identity] = Field(default_factory=list)
    # The results of that kind this recovery asks to run again (``rerun_masked_without_artifact``) and plans as
    # retries: what the resubmit journals as the approval.
    rerun_masked: list[RerunMaskedApproval] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        settled = (IdentityState.SCORED, IdentityState.MODEL_FINAL, IdentityState.OMITTED)
        return sum(self.counts.get(state, 0) for state in settled) == self.expected


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
