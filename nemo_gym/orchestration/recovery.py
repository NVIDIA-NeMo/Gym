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

import hashlib
import math
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nemo_gym.agents.agent_outcome import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentOutcome,
    OutcomeAction,
    OutcomePolicyConfig,
    choose_action,
)
from nemo_gym.artifacts.run_artifacts import ACCEPTED_STAGE_OMISSION, accepted_stage_omission
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.secret_utils import mask_url_passwords
from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    COLLECTION_DISPATCH_FAILED,
    COLLECTION_RESUME_REFUSED,
    DRIVER_INSTALL_FAILED,
    EMPTY_ANSWER,
    ENVIRONMENT_PREPARE_FAILED,
    ENVIRONMENT_PROTOCOL_VIOLATION,
    EPISODE_FAILED,
    GRADING_WITHOUT_VERDICT,
    HARBOR_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    INFRASTRUCTURE_ERROR,
    INVALID_OUTPUT_BUDGET,
    INVALID_REQUEST_PARAMETER,
    JUDGE_FAILED,
    JUDGE_FAILURE_KINDS,
    JUDGE_SKIPPED,
    JUDGE_UNPARSEABLE,
    MALFORMED_ANSWER,
    MODEL_CONTEXT_LIMIT,
    MODEL_POLICY_RESPONSE_TIMEOUT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_QUOTA_EXHAUSTED,
    ROLLOUT_ALLOCATION_ENDED,
    SERVER_SPINUP_TIMEOUT,
    SERVER_START_FAILED,
    TRUNCATED_ANSWER,
    VERIFIER_ERROR,
    FailureClass,
    FailureClassificationConfig,
    FailureStage,
    classify_failure,
    is_namespaced,
    is_registered,
    is_server_failure,
    quota_reset_text,
    quota_reset_unix,
    reporting_failure_kind,
)
from nemo_gym.core.failure_records import FailureBoundary, FailureEvent
from nemo_gym.core.failure_transport import FailureReportRow
from nemo_gym.core.outcome_policy import OutcomeAction, is_policy_counted_zero
from nemo_gym.orchestration.collect_evidence import (
    COLLECT_SCRIPT,
    EPISODE_DEADLINE_REASON,  # noqa: F401
    JUDGE_JOB_ACK_JOURNAL_NAME,  # noqa: F401
    JUDGE_JOB_JOURNAL_NAME,  # noqa: F401
    OPERATOR_CONTINUATION_JOURNAL_NAME,
    UNOBSERVED_CLAIM_REASON,
    UNREADABLE_CLAIM_REASON,
)
from nemo_gym.orchestration.executors.resume_script import (
    ACCOUNTED_WALLTIME_PREFIX,
    CHAIN_ACCOUNTING_ENV,
    CONTINUATION_REASON_PREFIX,
    GUARDED_DRIVER_FAILURE_KINDS,
    PROVISIONAL_MARKER_PREFIX,
    REQUEUE_RECONCILE_STATE,
    REQUEUE_UNKNOWN_REASON,
)
from nemo_gym.rollouts.rollout_recovery import applied_failure_class as _applied_class
from nemo_gym.rollouts.rollout_recovery import failure_stage as _policy_stage
from nemo_gym.utils.path_utils import RERUN_MASKED_JOURNAL_NAME


TASK_INDEX = "_ng_task_index"
ROLLOUT_INDEX = "_ng_rollout_index"
FAILURE_CLASS = "_ng_failure_class"
TERMINAL = "_ng_failure_terminal"
COUNTS_AS_ZERO = "_ng_failure_counts_as_zero"
MAX_ATTEMPTS = "_ng_failure_max_attempts"
FAILURE_EVENTS = "_ng_failure_events"
FAILURE_ID = "_ng_failure_id"
# Facts of a failure's recorded cause that the collector reads on the run host (see ``_cause_kind``).
COLLECTED_CAUSE = "_ng_collected_cause"

Identity = tuple[int, int]

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
    # Verification skipped by configuration: complete, never re-judged or rerun, and unverified by construction.
    SKIPPED = "skipped"
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


class ManagedRecoveryFact(BaseModel):
    """Read-only facts from the authoritative Store, not a second retry policy."""

    identity: Identity
    attempts: int = Field(ge=0, strict=True)
    max_attempts: int = Field(ge=1, strict=True)
    pending: bool
    reserved: bool
    # Whether the Store would run it again under some attempt cap (its replay rule without the cap). None from a
    # Store reader that predates attempt extensions (`--extend-attempts`), which then cannot be applied to the run.
    replayable: bool | None = None
    # The selected result is a masked one the operator approved to run again (`--rerun-masked-without-artifact`):
    # the Store counts it as a failed attempt and runs it again within its cap.
    rerun_approved: bool = False


# What an exhausted-attempts stop says an operator can do; the extension is explicit and journaled.
EXTEND_ATTEMPTS_HINT = (
    "an exhausted identity whose attempts went to infrastructure failures or allocation cuts gets N more with "
    "gym eval resubmit --recover --extend-attempts N (explicit and journaled); a model failure or a judge failure "
    "that kept its saved answer does not"
)
# The most attempts one recovery may add to every rollout's cap.
MAX_EXTEND_ATTEMPTS = 100


def _with_attempt_extension(
    facts: list[ManagedRecoveryFact] | None, extend_attempts: int
) -> list[ManagedRecoveryFact] | None:
    """The Store's facts as they read once ``extend_attempts`` more attempts are recorded for every rollout.

    Only a rollout the Store would run again under some cap (``replayable``) gets the higher cap, and it is pending
    while it has attempts left. Any other rollout keeps its cap and state: a model failure, a judge failure that kept
    its saved answer (its re-judgment budget is the saved cap), a saved result.
    """
    if not extend_attempts or facts is None:
        return facts
    return [
        fact.model_copy(
            update={
                "max_attempts": fact.max_attempts + extend_attempts,
                "pending": fact.pending or fact.attempts < fact.max_attempts + extend_attempts,
            }
        )
        if fact.replayable
        else fact
        for fact in facts
    ]


def _saved_attempt_cap(collected: "CollectedEvidence") -> int | None:
    """The run's saved per-rollout attempt cap, from the Store's facts; None when they do not say."""
    applied = collected.attempt_extension_applied
    for fact in collected.managed_selection or []:
        if fact.replayable is None or applied is None:
            return None
        return fact.max_attempts - applied if fact.replayable else fact.max_attempts
    return None


def retry_budget(collected: "CollectedEvidence") -> dict[str, dict[str, int | None]]:
    """The run's two retry budgets as the gate reports them, each its configured cap and its recorded extension.

    ``infra`` is the chain's infra-retry cap (``--extend-retries``); ``attempts`` is every rollout's attempt cap
    (``--extend-attempts``). A value that could not be read is None.
    """
    recorded = {"absent": 0, "ok": collected.infra_retry_extension}.get(collected.infra_retry_extension_state)
    attempts = {"absent": 0, "ok": collected.attempt_extension}.get(collected.attempt_extension_state)
    return {
        "infra": {
            "configured": collected.infra_retries_configured,
            "extension": None if recorded is None else int(recorded),
        },
        "attempts": {"configured": _saved_attempt_cap(collected), "extension": attempts},
    }


def _extend_attempts_refusal(collected: "CollectedEvidence") -> str | None:
    """Why this run cannot take an attempt extension, or None when it can."""
    if collected.attempt_extension_state == "malformed":
        return "the recorded attempt extension (artifacts/rollouts_attempt_extension) is unreadable"
    if not collected.managed_store or not collected.managed_selection:
        return "only a run written by a rollout store, whose saved selection was read, keeps a per-rollout attempt cap"
    if collected.attempt_extension_applied is None or any(
        fact.replayable is None for fact in collected.managed_selection
    ):
        return (
            "the run's pinned Gym predates attempt extensions, so its Store would not run the extra attempts; repin the "
            "run to a Gym that has them (--repin COMMIT --repin-reason TEXT) in the same resubmit"
        )
    if collected.attempt_extension_applied != (collected.attempt_extension or 0):
        return "the recorded attempt extension changed while it was read; inspect again"
    return None


def _rerun_masked_refusal(collected: "CollectedEvidence") -> str | None:
    """Why this run cannot run masked results without an artifact again, or None when it can."""
    if not collected.managed_store or not collected.managed_selection:
        return "only a run written by a rollout store, whose saved selection was read, can run a masked result again"
    if collected.rerun_masked_applied is not True:
        return (
            "the run's pinned Gym predates the rerun of masked results, so its Store would not run them; repin the "
            "run to a Gym that has it (--repin COMMIT --repin-reason TEXT) in the same resubmit"
        )
    return None


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
    # A failed job that settled a saved answer before it failed made progress, as a completed one does.
    return entry.get("state") in ("failed", "completed") and not entry.get("settled")


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


# What ``failed_judge_jobs_stop`` says, read back by the operator verdict: the jobs, where to read them, and whether
# they can be acknowledged.
_FAILED_JUDGE_JOBS = re.compile(
    rf"the last {JUDGE_JOB_FAILURE_BOUND} judge-only job\(s\) \((?P<ids>.*?)\) ended without settling a saved answer; "
    r"read their receipts and logs under (?P<where>.*?) before another is queued(?P<unacknowledgeable>; a job without "
    r"an id cannot be acknowledged)?"
)


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


class OperatorBudget(BaseModel):
    """An operator's explicit budget for a run whose chain accounting is unknown.

    ``retries`` infra retries remain from now (``--extend-retries``, one by default) and ``max_walltime`` is the
    walltime cap (``--max-walltime``, raised as a recovery raises it). It only fills counters that are unknown: a
    readable counter is never lowered or reset.
    """

    model_config = ConfigDict(extra="forbid")

    max_walltime: str
    reason: str = Field(min_length=1)
    retries: int = Field(default=1, ge=1, le=1000)
    user: str


class OperatorEpoch(BaseModel):
    """Where the last operator continuation started the chain's accounting: its chain length and walltime baseline."""

    chain_length: int = Field(ge=0)
    walltime_s: float = Field(ge=0, allow_inf_nan=False)


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
    live_job_name: str | None = Field(default=None, description="The live chain job, as 'id (STATE)', when known.")
    job_chain: list[str] = Field(default_factory=list, description="Job ids already run in this job dir.")


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
    # The one verdict an operator acts on (``operator_verdict``): state, reason and action; set by the CLI.
    operator_verdict: dict[str, str] | None = None
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
    stage: FailureStage | None = None
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
    # Which chain accounting is unknown when this STOP is the unknown-budget or unknown-walltime stop: "budget" or
    # "walltime". The reason is display text that later notes extend; an operator continuation keys on this field.
    accounting_unknown: Literal["budget", "walltime"] | None = Field(default=None, exclude=True)
    # The guard that refuses (or, for an inspection, would refuse) an operator continuation of unknown accounting:
    # the scheduler did not answer, a job of the run is pending or running, a follow-up claim names no job, the
    # accounting files changed while read, or the run configures no infra-retry cap. The operator verdict keys on it.
    continuation_refused: (
        Literal["scheduler_unreachable", "job_live", "claim_unidentified", "counters_changed", "no_retry_cap"] | None
    ) = Field(default=None, exclude=True)
    # The options of the call this plan was made for, spelled as `gym eval resubmit --recover` takes them (reasons as
    # TEXT), so the operator verdict's command repeats them.
    requested: list[str] = Field(default_factory=list, exclude=True)
    # The run's `gym eval resubmit` positionals (job id and benchmark, shell-quoted) for the call this plan was made
    # for, so every command the operator verdict prints is complete; placeholders when no call named them.
    target: str = Field(default="GYM_JOB_ID BENCHMARK", exclude=True)
    # The typed driver failure the chain's incomplete marker opens with (one of GUARDED_DRIVER_FAILURE_KINDS) when the
    # chain stopped on one: the next recovery needs its remedy first, or the resumed segment stops the same way.
    chain_stop_kind: str | None = Field(default=None, exclude=True)
    # Why a judge-only job was not queued, as the dispatch classified it: ``cpu_partition_unknown`` (no CPU partition
    # is named), ``cpu_placement`` (the scheduler refused the CPU partition or QOS), ``served_placement`` (it refused the
    # saved node pool of a run that serves its judge, which no recovery option changes), ``unsupported_profile`` (the
    # saved config has no judge execution profile, decided locally before any scheduler call), ``refused`` (the
    # scheduler refused it for a definite reason that is not placement, such as an invalid time limit), ``concurrent``
    # (the dispatch found another recovery's lock or a job of the run on the scheduler: it waits) or ``other`` (the
    # scheduler did not answer, or whether the job was queued is not known). The operator verdict offers the judge
    # placement options only for the first two.
    judge_refusal: (
        Literal[
            "cpu_partition_unknown",
            "cpu_placement",
            "served_placement",
            "unsupported_profile",
            "refused",
            "concurrent",
            "other",
        ]
        | None
    ) = Field(default=None, exclude=True)
    # An option this call asked for that the run cannot take (``--extend-attempts`` or the masked rerun), and the
    # action the plan would take without it: a RESUBMIT or REVERIFY continues once the option is dropped.
    refused_option: str | None = Field(default=None, exclude=True)
    refused_over: RecoveryAction | None = Field(default=None, exclude=True)
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
        settled = (IdentityState.SCORED, IdentityState.MODEL_FINAL, IdentityState.OMITTED, IdentityState.SKIPPED)
        return sum(self.counts.get(state, 0) for state in settled) == self.expected

    @property
    def unverified(self) -> bool:
        """Whether results of the run were deliberately not verified (``judge_skipped``), so it is UNVERIFIED."""
        return bool(self.counts.get(IdentityState.SKIPPED))


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
    classification: FailureClassificationConfig = Field(default_factory=FailureClassificationConfig)

    supported_actions = frozenset({OutcomeAction.VERIFY, OutcomeAction.RETRY, OutcomeAction.FAIL})


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


def _quota_reset_text(row: Mapping[str, Any]) -> str | None:
    """The reset time a quota-stopped row names: the collected fact, else its reason (a live run's row)."""
    collected = row.get("_ng_quota_resets_at")
    return (
        collected
        if isinstance(collected, str)
        else (quota_reset_text(row.get("failure_reason")) or quota_reset_text(row.get("_ng_failure_message")))
    )


def _quota_wait_reason(row: Mapping[str, Any]) -> str | None:
    text = _quota_reset_text(row)
    return f"resets at {text}" if text else None


def _quota_reset_unix(row: Mapping[str, Any]) -> float | None:
    return quota_reset_unix("resets at " + text) if (text := _quota_reset_text(row)) else None


def _now() -> float:
    return time.time()


def _judge_state(row: Mapping[str, Any], policy: "RecoveryOutcomePolicyConfig") -> "IdentityState":
    """A judge failure's treatment: verify re-judges the saved answer only (none saved: stop), fail leaves it
    ungraded; any other action would run the policy again, so it stops."""
    if row.get(TERMINAL) is True:
        return IdentityState.TERMINAL
    kind, status = _policy_evidence(row)
    if kind in (PROVIDER_QUOTA_EXHAUSTED, PROVIDER_CREDENTIALS_REJECTED):
        # The grading provider refused until the operator acts (its quota resets, its key is fixed). The saved answer
        # is graded again, never regenerated; before a quota's named reset time nothing can be graded, so it stops.
        if not _saved_answer(row):
            return IdentityState.UNKNOWN
        reset = _quota_reset_unix(row) if kind == PROVIDER_QUOTA_EXHAUSTED else None
        return IdentityState.TERMINAL if reset is not None and _now() < reset else IdentityState.JUDGE_RECOVERABLE
    action = _recovery_action(row, kind or JUDGE_FAILED, policy, http_status=status)
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


def _replay_refusal(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """Why replay needs a saved-answer path or model-budget accounting unavailable to recovery; None when it does not."""
    from nemo_gym.rollouts.rollout_recovery import (
        has_recorded_grade,
        has_saved_generation,
        has_unclassified_execution_failure,
        is_infrastructure_agent_timeout,
        keeps_partial_response,
    )

    # An agent timeout whose window held a server-side fault kept no answer or grade of the model's: it runs again.
    infrastructure = is_infrastructure_agent_timeout(row, classification=classification)
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


def _policy_replay_is_unsupported(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> bool:
    """Whether replay needs a saved-answer path or model-budget accounting unavailable to recovery."""
    return _replay_refusal(row, classification=classification) is not None


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
            if (
                not judge
                and identity in managed
                and is_infrastructure_agent_timeout(row, classification=policy.classification)
            ):
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
            if not judge and row.get("mask_sample") is True and row.get("failure_kind") == JUDGE_SKIPPED:
                # Verification was skipped by configuration: complete and never re-judged or rerun.
                reason = row.get("_ng_skipped_reason")
                evidence.append(
                    IdentityEvidence(
                        identity=identity,
                        state=IdentityState.SKIPPED,
                        failure_kind=JUDGE_SKIPPED,
                        reason=reason if isinstance(reason, str) else None,
                    )
                )
                continue
            if (
                not judge
                and (row.get("mask_sample") is True or row.get("evaluation_completed") is False)
                and not is_policy_counted_zero(row)
            ):
                if row.get("failure_outcome_action") == OutcomeAction.FAIL:
                    # The owning boundary already ended this failure; recovery cannot grant a new retry budget.
                    policy_stopped = True
                    evidence.append(
                        IdentityEvidence(
                            identity=identity,
                            state=IdentityState.TERMINAL,
                            failure_kind=row.get("failure_kind"),
                            events=_events(row),
                        )
                    )
                    continue
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
                infrastructure = masked_infrastructure_kind(row, classification=policy.classification)
                fact = managed.get(identity)
                spent = fact.attempts if fact is not None else 0
                kept = (
                    masked_without_artifact_kind(row, classification=policy.classification)
                    if infrastructure is None and fact is not None
                    else None
                )
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
                    action = _recovery_action(row, infrastructure, policy, http_status=None)
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
            quota_wait = (
                judge and state == IdentityState.TERMINAL and row.get("failure_kind") == PROVIDER_QUOTA_EXHAUSTED
            )
            policy_stopped |= judge and state == IdentityState.TERMINAL and not quota_wait
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=state,
                    reason=_quota_wait_reason(row) if quota_wait else None,
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
            if not _policy_replay_is_unsupported(last, classification=policy.classification):
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
                    reason=_replay_refusal(last, classification=policy.classification)
                    if state == IdentityState.UNKNOWN
                    else None,
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
        quota_wait = False
        if last.get(FAILURE_CLASS) == JUDGE_FAILED:
            # A judge that failed is re-judged from the saved answer, never by running the policy again.
            state = _judge_state(last, policy)
            quota_wait = state == IdentityState.TERMINAL and last.get("failure_kind") == PROVIDER_QUOTA_EXHAUSTED
            policy_stopped |= state == IdentityState.TERMINAL and not quota_wait
        elif is_answered_agent_deadline(last, classification=policy.classification):
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
            if last.get(TERMINAL) is True or _policy_replay_is_unsupported(last, classification=policy.classification):
                state = IdentityState.MODEL_FINAL
            else:
                state = IdentityState.UNKNOWN
                reason = f"the policy's request failed ({kind}), a final model result, but a resume would run it again"
        elif last.get(TERMINAL) is True:
            state = IdentityState.TERMINAL
        elif (
            action := _recovery_action(last, kind or "unknown", policy, http_status=_policy_evidence(last)[1])
        ) == OutcomeAction.FAIL:
            # Unrestricted resume would still retry this nonterminal saved row.
            policy_stopped = True
            state = IdentityState.TERMINAL
        elif action == OutcomeAction.VERIFY or _policy_replay_is_unsupported(
            last, classification=policy.classification
        ):
            # Only a judge failure has a supported saved-answer path. A failed verification can retain a finished
            # agent answer, so replay must not generate another. For collection-wrapped model failures, the sidecar projection
            # has no durable model-operation budget accounting: replay must not silently replenish that budget.
            state = IdentityState.UNKNOWN
            reason = (
                _replay_refusal(last, classification=policy.classification)
                or "the policy verifies a failure that has no saved judge answer"
            )
        elif spent < cap:
            state = IdentityState.RETRYABLE
        else:
            state = IdentityState.EXHAUSTED
        evidence.append(
            IdentityEvidence(
                identity=identity,
                state=state,
                # The kind its retry selects on stays agent_timeout; the counted kind names the server-side fault.
                failure_kind=AGENT_TIMEOUT_INFRASTRUCTURE
                if is_infrastructure_agent_timeout(last, classification=policy.classification)
                else kind,
                http_status=_policy_evidence(last)[1],
                stage=_policy_stage(last),
                failure_id=last.get(FAILURE_ID) if isinstance(last.get(FAILURE_ID), str) else None,
                attempts=spent,
                max_attempts=cap,
                events=_events(last),
                reason=reason if state == IdentityState.UNKNOWN else _quota_wait_reason(last) if quota_wait else None,
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
                parts.append(f"{reserved} reservation(s) have no saved outcome and {_RESERVATIONS_UNKNOWN}")
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
            waiting = [
                item
                for item in evidence
                if item.state == IdentityState.TERMINAL and item.failure_kind == PROVIDER_QUOTA_EXHAUSTED
            ]
            if len(waiting) == counts[IdentityState.TERMINAL]:
                resets = sorted({text for item in waiting if (text := quota_reset_text(item.reason))})
                return decision(
                    RecoveryAction.STOP,
                    f"{len(waiting)} saved answer(s) wait on the grading provider's quota "
                    f"({PROVIDER_QUOTA_EXHAUSTED}), which {'resets at ' + resets[-1] if resets else 'has no known reset'}"
                    "; nothing can be graded before then. After the reset continue with gym eval resubmit --recover "
                    "(re-judges the saved answers, no model calls)",
                )
            violations = [
                item
                for item in evidence
                if item.state == IdentityState.TERMINAL and item.failure_kind == ENVIRONMENT_PROTOCOL_VIOLATION
            ]
            if len(violations) == counts[IdentityState.TERMINAL]:
                return decision(
                    RecoveryAction.STOP,
                    f"{len(violations)} rollout(s) got a reply the collector cannot use "
                    f"({ENVIRONMENT_PROTOCOL_VIOLATION}); this is a producer bug, retrying cannot help. "
                    "Fix the server or the benchmark pairing, then `gym eval submit` again",
                )
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.TERMINAL]} terminal unscored failure(s) need an "
                "owner decision; the run is not complete",
            )
        if counts.get(IdentityState.EXHAUSTED):
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.EXHAUSTED]} identity(ies) {_ATTEMPTS_USED_TOKEN}, so a "
                "resume would run nothing; the run is not complete; "
                + _EXTEND_RETRIES_SCOPE.format(
                    cap="/".join(
                        str(cap)
                        for cap in sorted(
                            {item.max_attempts for item in evidence if item.state == IdentityState.EXHAUSTED} - {None}
                        )
                    )
                )
                + f"; {EXTEND_ATTEMPTS_HINT}"
                # More attempts alone would stop again on the chain's budget, so name every blocker now.
                + _chain_blockers_note(budget),
            )
        if counts.get(IdentityState.MASKED):
            return decision(
                RecoveryAction.STOP,
                f"{counts[IdentityState.MASKED]} {_MASKED_STOP_TOKEN}: they are kept and a resume would run nothing; "
                f"the run is not complete; {populations}"
                + (_chain_blockers_note(budget) if masked_without_artifact else ""),
            )
        if counts.get(IdentityState.SKIPPED):
            reasons = sorted({item.reason for item in evidence if item.state == IdentityState.SKIPPED and item.reason})
            return decision(
                RecoveryAction.NOOP,
                f"{counts[IdentityState.SKIPPED]} result(s) not graded by configuration ({JUDGE_SKIPPED}); "
                f"nothing is re-judged or rerun{': ' + reasons[0] if reasons else ''}",
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
                f", and {_WALLTIME_USED_TOKEN} ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s"
                "; less than a minute is left): raise both in one call"
                if walltime_spent
                else ""
            )
            # A raised retry cap would stop next on the walltime nobody can read, so name it now.
            + (
                f", and {UNKNOWN_WALLTIME_REASON}: raise the retries and give an explicit budget in one call"
                if budget.walltime_capped and budget.walltime_used_s is None
                else ""
            ),
        )
    if budget.walltime_capped:
        if budget.walltime_used_s is None or budget.max_walltime_s is None:
            return decision(RecoveryAction.STOP, UNKNOWN_WALLTIME_REASON, accounting_unknown="walltime")
        if budget.walltime_used_s >= budget.max_walltime_s:
            return decision(
                RecoveryAction.STOP,
                f"{_WALLTIME_USED_TOKEN} ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s)",
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
_RESERVATIONS_UNKNOWN = "the allocation that made them is not known to have ended, so whether they ran is unknown"
_ATTEMPTS_USED_TOKEN = "used every attempt since the last resubmit"
_WALLTIME_USED_TOKEN = "the stored walltime budget is used"
# A refused or stopped recovery records none of the extensions it asked for, so the next one must ask again.
_NOT_RECORDED_TOKEN = "asked in this call is not recorded, as nothing was queued: ask it again in the same call"

_EXTEND_RETRIES_SCOPE = (
    "--extend-retries raises infra retries only; it does not raise the per-identity attempt cap "
    "(max_rollout_attempts={cap})"
)


_CHAIN_BLOCKERS_LEAD = "; a resume of them would also stop, as "
_MASKED_STOP_TOKEN = "completed result(s) are masked and not measured"


def _chain_blockers_note(budget: RecoveryBudget) -> str:
    """For a stop on the identities' own states (exhausted, masked): what in the chain's budget would stop the resume
    an extension or a rerun allows, as the stop texts name it, so one call can lift every blocker; else ""."""
    blockers = []
    if budget.max_infra_retries is None or budget.infra_retries_used is None:
        blockers.append(UNKNOWN_BUDGET_REASON)
    elif budget.infra_retries_used >= budget.max_infra_retries:
        blockers.append(f"{_INFRA_BUDGET_USED_PREFIX} ({budget.infra_retries_used} of {budget.max_infra_retries})")
    if budget.walltime_capped and budget.max_walltime_s is not None:
        if budget.walltime_used_s is None:
            if UNKNOWN_BUDGET_REASON not in blockers:
                blockers.append(UNKNOWN_WALLTIME_REASON)
        # A chain stops once less than a minute of its walltime remains.
        elif budget.max_walltime_s - budget.walltime_used_s < 60:
            blockers.append(f"{_WALLTIME_USED_TOKEN} ({budget.walltime_used_s:g} of {budget.max_walltime_s:g} s)")
    if not blockers:
        return ""
    return f"{_CHAIN_BLOCKERS_LEAD}{' and '.join(blockers)}: lift every blocker in one call"


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
    # Job-dir accounting file -> sha256 (None when absent) at collection, plus the saved resume_config.
    accounting_snapshot: dict[str, str | None] | None = None
    source_job_identity: tuple[str, int, int] | None = None
    managed_selection: list[ManagedRecoveryFact] | None = None
    managed_reader_error: str | None = None
    # Rollouts with an admitted re-judgment and no saved outcome: its row is still the selected judge failure.
    unresolved_rejudgments: list[str] = Field(default_factory=list)
    # True when the Store reader could not run (install failure, crash, timeout, lost transport): nothing was read,
    # unlike a reader that ran and could not verify the artifacts.
    managed_reader_unavailable: bool = False
    # Effective setting derived by the installed runtime from the bound saved configuration, not an observation.
    checkpoint_enabled: bool | None = Field(default=None, strict=True)
    # Artifacts that exist but could not be read, relative to the job dir: what they hold is unknown.
    unreadable_files: list[str] = Field(default_factory=list)
    # rollouts_coverage.json's allocation_id: the scheduler allocation that last opened the Store for writing.
    coverage_allocation_id: str | None = None
    # rollouts_coverage.json's (scored, expected): the rollouts collection scored into the metrics, of those expected.
    coverage_scored: tuple[int, int] | None = None
    # The aggregate's own (metric_coverage/trials, metric_coverage/expected, metric_coverage/name or None): the trials
    # the benchmark's headline metric covered, as the benchmark declares them.
    benchmark_metric_coverage: tuple[int, int, str | None] | None = None
    # Why the chain's accounting is unknown (the marker's reason or an unobserved claim), and the counter files' own
    # states (absent, ok, malformed) when the sources were stable.
    accounting_unknown_reason: str | None = None
    counter_states: dict[str, str] | None = None
    # Set by an executor that reconciled a requeued chain from complete terminal scheduler records.
    accounting_reconciled: bool = False
    # Infra retries an explicit recovery added to the configured cap (.gym_infra_retry_extension), and its state.
    infra_retry_extension: float | None = None
    infra_retry_extension_state: str = "absent"
    # The follow-up job id each allocation of the chain claimed (.gym_followup_queued.<allocation>).
    followup_claims: dict[str, str] = Field(default_factory=dict)
    # The sentinel job id each allocation queued behind its follow-up (.gym_followup_sentinel.<allocation>).
    followup_sentinels: dict[str, str] = Field(default_factory=dict)
    # The last operator continuation's epoch (OPERATOR_CONTINUATION_JOURNAL_NAME): the chain is accounted after it.
    operator_epoch: OperatorEpoch | None = None
    # Where the heavy readers ran: "login" or "cpu:<job id>" with a reader step (NEMO_GYM_REMOTE_READER_SRUN), and why
    # one stayed on the login host; all None without a reader step.
    collector_host: str | None = None
    collector_fallback_reason: str | None = None
    reader_host: str | None = None
    reader_fallback_reason: str | None = None
    # The operational settings the run's latest resume crossed (its resume-drift journal), None when it crossed none.
    resume_drift_keys: list[str] | None = None
    # Attempts an explicit recovery added to every rollout's cap (artifacts/rollouts_attempt_extension) and its state
    # (absent, ok, malformed), as the plain collector read them; and the extension the run's Store reader applied,
    # None from a reader that predates attempt extensions.
    attempt_extension: int | None = None
    attempt_extension_state: str = "absent"
    attempt_extension_applied: int | None = None
    # True when the run's Store reader applies the approvals of ``RERUN_MASKED_JOURNAL_NAME``; None from a reader
    # that predates them, which would not run an approved result again.
    rerun_masked_applied: bool | None = None
    # The run's configured infra-retry cap (its saved resume_config), set by the executor that read it.
    infra_retries_configured: int | None = None
    # The marker line of a continuation that ended before it queued its follow-up, None when there is none.
    chain_continuation_note: str | None = None
    # The judge-only jobs queued for the run's saved answers (JUDGE_JOB_JOURNAL_NAME), oldest first: id, partition,
    # profile kind and the services it starts, walltime, rows, receipt path and ``state`` ("queued" without a receipt,
    # else the receipt's "completed" or "failed" with its counts); the executor adds the scheduler's state of a job
    # without a receipt.
    judge_jobs: list[dict[str, Any]] = Field(default_factory=list)
    # The acknowledgements of failed judge-only jobs (JUDGE_JOB_ACK_JOURNAL_NAME): the job ids each covers.
    judge_job_acknowledgements: list[dict[str, Any]] = Field(default_factory=list)


# The host sets the reader placement; a reader neither receives it nor replaces it.
READER_PLACEMENT_FIELDS = ("collector_host", "collector_fallback_reason", "reader_host", "reader_fallback_reason")


def _managed_source_digest(path: Path) -> str | None:
    """Hash one stable file with bounded memory, using the exporter's existing identity checks.

    Two streaming passes must agree on stat identity and bytes: a same-size in-place rewrite within one timestamp
    tick changes no stat field. This doubles source reads; it does not make the reader atomic against ABA writes.
    """
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    first = None
    for _ in range(2):
        try:
            source = path.open("rb")
        except FileNotFoundError:
            if first is None:
                return None
            raise RuntimeError("managed source changed while hashing")
        digest = hashlib.sha256()
        with source:
            before = os.fstat(source.fileno())
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
            after = os.fstat(source.fileno())
        current = path.stat()
        if identity(before) != identity(after) or identity(after) != identity(current):
            raise RuntimeError("managed source changed while hashing")
        if first is not None and first != (identity(before), digest.hexdigest()):
            raise RuntimeError("managed source changed while hashing")
        first = (identity(before), digest.hexdigest())
    return first[1]


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

        from nemo_gym.data.task_materialization import materialize_task
        from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings

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


# Row values the Store selection never reads, and the largest ones in agent runs. The model-call capture is read: an
# agent timeout's captured calls decide whether its window held a server-side fault (is_infrastructure_agent_timeout).
STORE_SKIP_KEYS = frozenset({"ng_trajectory", "ng_agent_observations"})


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

        from nemo_gym.rollouts import RolloutStore, logical_rollout_id

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
        # No selection rule reads a trajectory or model-call capture: large rows are read without them.
        store = RolloutStore.read(artifacts / "rollouts.jsonl", import_legacy=False, skip_keys=STORE_SKIP_KEYS)
        if store is None or collected.materialized is None:
            raise ValueError("managed source has no readable manifest or inventory")
        if store.manifest.max_rollout_attempts is None:
            raise ValueError("managed manifest does not preserve its rollout attempt cap")
        unresolved_rejudgments = store.unresolved_rejudgments()
        coverage = store.coverage()
        if len(collected.materialized) != coverage["expected"] or not coverage["identity_verified"]:
            raise ValueError("managed inventory is ambiguous or its identity is not verified")
        with redirect_stdout(StringIO()):
            pending = {logical_rollout_id(row) for row in store.pending()}
        replayable = store.replayable_identities()
        approved = store.rerun_approved_identities()
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
                    # The recorded extension reaches only a rollout the Store would run again; any other keeps the
                    # saved cap that its own readers (the judge re-judgment budget) apply.
                    max_attempts=coverage["max_rollout_attempts"]
                    - (0 if logical in replayable else store.attempt_extension),
                    pending=logical in pending,
                    reserved=logical in store.manifest.next_attempt,
                    replayable=logical in replayable,
                    rerun_approved=logical in approved,
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
                "unresolved_rejudgments": unresolved_rejudgments,
                "attempt_extension_applied": store.attempt_extension,
                "rerun_masked_applied": True,
            }
        )
    except ImportError as error:
        update = {"managed_selection": None, "managed_reader_error": str(error), "managed_reader_unavailable": True}
        return collected.model_copy(update=update)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ConfigError) as error:
        return collected.model_copy(update={"managed_selection": None, "managed_reader_error": str(error)})


# A Store reader that could not run read nothing; the decision says so instead of stopping the run.
STORE_READER_UNAVAILABLE = (
    "the run was written by a rollout store and its Store reader could not run, so nothing was decided and the "
    "saved artifacts were not judged"
)


def _live_reason(live_job_name: str | None) -> str:
    if live_job_name and live_job_name.startswith("judge-only job"):
        return f"{live_job_name} live: it grades the run's saved answers again; nothing to decide"
    if live_job_name:
        return f"continuation {live_job_name} live; nothing to decide"
    return "a job of this run is still pending or running; nothing to decide, it is not queued twice"


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

    def stop(why: str, guard: str) -> RecoveryDecision:
        return stopped.model_copy(update={"reason": f"{stopped.reason}; {why}", "continuation_refused": guard})

    if collected.accounting_unknown_reason in (UNOBSERVED_CLAIM_REASON, UNREADABLE_CLAIM_REASON):
        return stop(
            "a follow-up claim names no job, so an operator continuation cannot rule out a queued follow-up",
            "claim_unidentified",
        )
    states = collected.counter_states
    if states is None:
        return stop(
            "the chain's accounting files changed while they were read, so an operator continuation waits",
            "counters_changed",
        )
    if configured_retries is None:
        return stop(
            "the run configures no infra-retry cap, so an operator continuation has no budget to fill", "no_retry_cap"
        )
    unknown: list[str] = []
    writes: dict[str, dict[str, Any]] = {}
    if collected.infra_retry_extension_state in ("absent", "ok"):
        extension = int(collected.infra_retry_extension or 0)
    else:
        extension = 0
        unknown.append("infra_retry_extension")
        writes[INFRA_RETRY_EXTENSION_FILE] = {"previous": "unknown", "value": 0}
    readable_retries = states.get("infra_retries") in ("absent", "ok")
    extended = extend_retries if readable_retries else 0
    if extended:
        # The spent count is known, so N cannot fill it: it raises the cap, as --extend-retries does elsewhere.
        writes[INFRA_RETRY_EXTENSION_FILE] = {
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
        writes[INFRA_RETRIES_FILE] = {"previous": "unknown", "value": used}
    if states.get("walltime") in ("absent", "ok"):
        walltime = float(collected.walltime_used_s or 0)
    else:
        walltime = 0.0
        unknown.append("walltime_used_s")
        writes[ACCUMULATED_WALLTIME_FILE] = {"previous": "unknown", "value": 0}
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


# The delivery reason of an export that did not upload the rollouts (``upload_rollouts: false``): unverified by
# configuration, which no export changes.
MLFLOW_ROLLOUTS_NOT_UPLOADED = "records that the rollouts were not uploaded"
_CHAIN_STOP_KIND = re.compile(r"^(?:Previous job \S+ ended [^:]*: )?(?P<kind>[a-z_]+): ")


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
    if unrecorded := _unrecorded_notes(decision, options):
        # Beside the stop they belong to, before any hint line that ends the reason.
        first, *rest = decision.reason.split("\n", 1)
        decision = decision.model_copy(update={"reason": "\n".join(["; ".join([first, *unrecorded]), *rest])})
    notes = [resume_drift_note(collected.resume_drift_keys)]
    blocked = failed_judge_jobs_blocking(collected.judge_jobs, collected.judge_job_acknowledgements)
    if blocked is not None and decision.action == RecoveryAction.REVERIFY:
        notes.append(failed_judge_jobs_stop(blocked, "the run's reverification directory"))
    if collected.chain_continuation_note:
        notes.append(f"the chain recorded: {collected.chain_continuation_note}")
        # The guard's line names the typed kind first, after the previous job's end ("Previous job <id> ended
        # <state>: <kind>: ..."; resume_script's *_REASON texts).
        typed = _CHAIN_STOP_KIND.match(collected.chain_continuation_note)
        if typed and typed["kind"] in GUARDED_DRIVER_FAILURE_KINDS:
            decision = decision.model_copy(update={"chain_stop_kind": typed["kind"]})
    notes = [note for note in notes if note]
    return decision if not notes else decision.model_copy(update={"reason": "; ".join([decision.reason, *notes])})


def _unrecorded_notes(decision: RecoveryDecision, options: Mapping[str, Any]) -> list[str]:
    """A stop that a budget, unknown accounting or a refused extension blocks records none of the options this call
    asked for (nothing is queued): one note per option the reason does not already name, so the next advice keeps
    it and one call can carry them all."""
    if decision.action != RecoveryAction.STOP or decision.operator_continuation is not None:
        return []
    if not _continuation_options(decision):
        return []
    reason, budget = decision.reason, options.get("operator_budget")
    asked = (
        (f"--extend-attempts {options.get('extend_attempts')}", _ATTEMPTS_USED_TOKEN, options.get("extend_attempts")),
        (
            f"--extend-retries {options.get('extend_retries')}",
            _INFRA_BUDGET_USED_PREFIX,
            options.get("extend_retries"),
        ),
        (f"--max-walltime {getattr(budget, 'max_walltime', '')}", _WALLTIME_USED_TOKEN, budget),
        # A rerun this call planned and did not queue.
        (RERUN_MASKED_FLAGS.split()[0], _NOT_RECORDED_TOKEN, bool(decision.rerun_masked)),
    )
    return [f"{flag} {_NOT_RECORDED_TOKEN}" for flag, named, value in asked if value and named not in reason]


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
        # The stop it would have lifted stays named, so the advice keeps every option the next call needs.
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": f"--rerun-masked-without-artifact is refused: {rerun_refused}; {decision.reason}",
                "refused_option": RERUN_MASKED_FLAGS.split()[0],
                "refused_over": decision.action,
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
                "reason": f"--extend-attempts {extend_attempts} is refused: {attempts_refused}; {decision.reason}",
                "refused_option": "--extend-attempts",
                "refused_over": decision.action,
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
                # The scheduler's answer comes first; the run's accounting state stays for the continuation after.
                "continuation_refused": "scheduler_unreachable",
            }
        )
    if collected.results is None and not decision.scheduler_live:
        return decision.model_copy(
            update={
                "action": RecoveryAction.STOP,
                "reason": "the results file is absent; collection cannot safely resume its saved history",
                # No budget continues a run without its results file, so unknown accounting is not this stop's cause.
                "accounting_unknown": None,
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
                    "continuation_refused": "scheduler_unreachable",
                }
            )
        if decision.scheduler_live and accounting_unknown:
            job = f"continuation {live_job_name}" if live_job_name else "a job of this run"
            why = f"the chain accounting is unknown and {job} is still pending or running"
            return decision.model_copy(
                update={
                    "action": RecoveryAction.STOP,
                    "reason": f"{why}, so an operator continuation is refused",
                    "continuation_refused": "job_live",
                }
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
    if unknown and operator_budget is None and live_job is False:
        # What an explicit budget alone would do: refused by a guard, or stopped on a budget it cannot fill. The
        # advice then names the guard to clear first, or every option the continuation needs in one call.
        alone = _operator_continuation(
            decision,
            collected,
            inputs,
            OperatorBudget(max_walltime="HH:MM:SS", reason="inspection", retries=extend_retries or 1, user="-"),
            configured_retries=configured,
            max_walltime_s=max_walltime_s,
            extend_retries=extend_retries,
        )
        if alone.continuation_refused is not None:
            decision = alone
            unknown = False
        elif alone.action == RecoveryAction.STOP:
            still = alone.reason.split(f"; {OPERATOR_CONTINUATION} by")[0]
            decision = decision.model_copy(
                update={"reason": f"{decision.reason}; an explicit budget alone would still stop: {still}"}
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


def continuation_admission(
    job_dir: str,
    max_retries: int,
    max_walltime_s: float | None,
    exit_code: str,
    previous_job_id: str,
    previous_state: str = "FAILED",
    outcome_policy: str | None = None,
) -> str:
    """Read saved state in the driver's installed runtime before a terminal allocation continues.

    The shell has already validated and persisted terminal elapsed accounting. This reader never changes
    artifacts, starts servers, resets budgets, or treats an exit code as sufficient retry evidence.
    ``outcome_policy`` is the run's ``recovery_outcome_policy`` section as JSON, rendered from its saved config, so
    an automatic continuation decides with the policy a recovery would apply.
    """
    import json
    import subprocess
    import sys

    job = Path(job_dir)
    if previous_state == REQUEUE_RECONCILE_STATE:
        return _requeue_continuation(job, max_retries, max_walltime_s)
    if exit_code == "76:0":
        print(
            f"{CONTINUATION_REASON_PREFIX}the previous driver exited 76: none of its saved failures is eligible for "
            "another attempt; `gym eval resubmit --recover --inspect` names each identity's state"
        )
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
            print(f"{CONTINUATION_REASON_PREFIX}the saved artifacts have no rollout store manifest to continue from")
            return "stop"
        collected = select_managed_evidence(collected, job)
        # The marker holds this allocation's own provisional line until it queues its follow-up. It says nothing about
        # the run: quoted into a stop, it would read as this allocation dying before its guard, so its sentinel would
        # run the same stop again.
        own = f"{PROVISIONAL_MARKER_PREFIX}{os.environ.get('SLURM_JOB_ID', '')} started"
        if os.environ.get("SLURM_JOB_ID") and (collected.chain_continuation_note or "").startswith(own):
            collected.chain_continuation_note = None
        if collected.walltime_state != "ok":
            print(f"{CONTINUATION_REASON_PREFIX}the chain's accumulated walltime is {collected.walltime_state}")
            return "stop"
        collected.accounted_walltime_s = collected.walltime_used_s
        # A completed outcome plus untouched pending inputs establishes planned segment work. Actual failed
        # attempts still consume infra retries, even if the driver happened to exit 75. A missing output,
        # reservation, judge answer or unowned selected cause is rejected by the ordinary decision below.
        # A scheduler allocation timeout continues untouched work, but never a failed agent attempt.
        segment = previous_state == "TIMEOUT"
        if previous_state == "FAILED" and exit_code == "75:0" and collected.managed_selection is not None:
            from nemo_gym.rollouts import RolloutStore

            report_path = job / "artifacts" / "rollouts_coverage.json"
            if collected.source_digests and collected.source_digests.get(
                "rollouts_coverage.json"
            ) == _managed_source_digest(report_path):
                report = json.loads(report_path.read_text())
                store = RolloutStore.read(
                    job / "artifacts" / "rollouts.jsonl", import_legacy=False, skip_keys=STORE_SKIP_KEYS
                )
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
        # The selection above verified the snapshot and read the Store once; a second read only hashes it again.
        if collected.managed_selection is None and collected.managed_reader_unavailable:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store reader could not run, so nothing was decided: "
                f"{collected.managed_reader_error or 'no cause reported'}; the saved artifacts were not judged: "
                "`gym eval resubmit --recover --inspect` reads them again"
            )
            return "unknown"
        if collected.managed_selection is None:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store could not verify the saved artifacts: "
                f"{collected.managed_reader_error or 'no authoritative selection'}"
            )
            return "stop"
        policy, source = effective_outcome_policy(None, json.loads(outcome_policy) if outcome_policy else None)
        decision = decide_from_collected(
            collected,
            max_infra_retries=max_retries,
            max_walltime_s=max_walltime_s,
            live_job=False,
            outcome_policy=policy,
            outcome_policy_source=source,
            planned_missing_continuation=segment,
            allocation_cut=_store_writer_ended(collected, previous_job_id),
        )
        print(CONTINUATION_REASON_PREFIX + decision.reason)
        if decision.action != RecoveryAction.RESUBMIT:
            return "stop"
        return "segment" if segment and not decision.counts.get(IdentityState.RETRYABLE) else "retry"
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{CONTINUATION_REASON_PREFIX}saved recovery evidence is unreadable: {type(error).__name__}: {error}")
        return "stop"


def _requeue_continuation(job: Path, max_retries: int, max_walltime_s: float | None) -> str:
    """In a sentinel: reconcile a requeued chain from the terminal records in ``CHAIN_ACCOUNTING_ENV`` and decide.

    The same reconcile and decision as ``gym eval resubmit --recover``: nothing else of the chain runs or waits (the
    sentinel checks the queue), and every allocation ended. Prints the reconciled walltime when the run continues.
    """
    import subprocess
    import sys

    try:
        rows = chain_accounting(os.environ.get(CHAIN_ACCOUNTING_ENV, ""))
        raw = subprocess.run(
            [sys.executable, "-I", "-S", "-c", COLLECT_SCRIPT, str(job)], capture_output=True, text=True, check=True
        )
        collected = CollectedEvidence.model_validate_json(raw.stdout)
        if collected.accounting_unknown_reason != REQUEUE_UNKNOWN_REASON:
            print(f"{CONTINUATION_REASON_PREFIX}the chain's accounting is not the requeue marker")
            return "stop"
        if not rows:
            print(f"{CONTINUATION_REASON_PREFIX}the requeued chain's scheduler records are malformed")
            return "stop"
        if not collected.managed_store:
            print(f"{CONTINUATION_REASON_PREFIX}the saved artifacts have no rollout store manifest to continue from")
            return "stop"
        collected = select_managed_evidence(collected, job)
        if collected.managed_selection is None:
            print(
                f"{CONTINUATION_REASON_PREFIX}the Store could not verify the saved artifacts: "
                f"{collected.managed_reader_error or 'no authoritative selection'}"
            )
            return "stop"
        collected = reconcile_requeued_chain(collected, rows)
        if not collected.accounting_reconciled or collected.accounted_walltime_s is None:
            print(f"{CONTINUATION_REASON_PREFIX}the terminal scheduler records do not account for the requeued chain")
            return "stop"
        decision = decide_from_collected(
            collected,
            max_infra_retries=max_retries,
            max_walltime_s=max_walltime_s,
            live_job=False,
            allocation_cut=allocation_cut_from_accounting(rows, collected.coverage_allocation_id),
        )
        requeued = requeue_note(collected.job_chain, rows)
        print(CONTINUATION_REASON_PREFIX + decision.reason + (f"; {requeued}" if requeued else ""))
        if decision.action != RecoveryAction.RESUBMIT:
            return "stop"
        print(f"{ACCOUNTED_WALLTIME_PREFIX}{int(collected.accounted_walltime_s)}")
        return "retry"
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{CONTINUATION_REASON_PREFIX}saved recovery evidence is unreadable: {type(error).__name__}: {error}")
        return "stop"


def chain_accounting(accounting: str) -> list[tuple[str, str, int, int]] | None:
    """Rows of ``sacct -D -X -n -P --format=CHAIN_ACCOUNTING_FORMAT`` as (job id, state, elapsed s, restarts).

    The state keeps its first word without a trailing ``+`` ("CANCELLED by 7" is CANCELLED). None when any row is
    malformed.
    """
    rows = []
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4 or not parts[0] or not parts[2].isdecimal() or not parts[3].isdecimal():
            return None
        state = parts[1].split()[0].rstrip("+") if parts[1].split() else ""
        rows.append((parts[0], state, int(parts[2]), int(parts[3])))
    return rows


FOLLOWUP_ACCOUNTING_FORMAT = "JobIDRaw,State,ElapsedRaw,ExitCode"
# Terminal states of a follow-up that started; a follow-up its dependency cancelled never starts (no elapsed time).
_FOLLOWUP_ENDED_STATES = frozenset(
    {"FAILED", "CANCELLED", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "OUT_OF_MEMORY", "TIMEOUT", "DEADLINE"}
)


def stopped_followup(job_id: str, accounting: str) -> str | None:
    """Why the chain stopped when the follow-up ``job_id`` that its last allocation claimed is not in the chain.

    ``accounting`` is ``sacct -D -X -n -P --format=FOLLOWUP_ACCOUNTING_FORMAT -j job_id``. The prologue records an
    allocation in the chain before anything else, so a follow-up that started and ended without that entry ran no
    work. None when the records do not show that: the follow-up never started, is live, or its record is ambiguous.
    """
    rows = [line.strip().split("|") for line in accounting.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != job_id or not rows[0][2].isdecimal():
        return None
    _, state, elapsed, exit_code = rows[0]
    state = state.split()[0].rstrip("+") if state.split() else ""
    if state not in _FOLLOWUP_ENDED_STATES or int(elapsed) == 0:
        return None
    code, _, signal = exit_code.partition(":")
    signaled = signal not in ("", "0")
    detail = f" with signal {signal}" if signaled else f" with exit code {code}" if code != "0" else ""
    # Only the scheduler's own state names the infrastructure; the record does not say who sent a signal or why.
    cause = " on infrastructure" if state in {"NODE_FAIL", "BOOT_FAIL", "PREEMPTED"} else ""
    unknown = " (cause not recorded)" if signaled and not cause else ""
    return (
        f"the chain stopped{cause}: its follow-up {job_id} ended {state}{detail} after {elapsed} s{unknown}, "
        "before it recorded itself in the chain, so no job continued the run"
    )


def allocation_cut_from_accounting(rows: list[tuple[str, str, int, int]], allocation_id: str | None) -> bool:
    """Whether the first incarnation of ``allocation_id`` has ended, by any path (``ALLOCATION_ENDED_STATES``).

    Only the first incarnation (no restarts) runs work: the prologue stops a requeued one before any. A state that is
    absent, ambiguous or not terminal is not an ending.
    """
    later = any(job_id == allocation_id and restarts > 0 for job_id, _, _, restarts in rows)
    ended = ALLOCATION_ENDED_STATES | {"REQUEUED"} if later else ALLOCATION_ENDED_STATES
    states = {state for job_id, state, _, restarts in rows if job_id == allocation_id and restarts == 0}
    return allocation_id is not None and len(states) == 1 and states <= ended


def _store_writer_ended(collected: CollectedEvidence, previous_job_id: str) -> bool:
    """In the in-job guard: whether the allocation that last wrote the Store has ended.

    The guard runs after its predecessor ended, and a chain runs one allocation at a time: every allocation the chain
    recorded before this one has ended. The writer is often an earlier allocation than the predecessor, for example
    when the predecessor timed out before it opened the Store. A writer the chain does not name stays unknown.
    """
    writer = collected.coverage_allocation_id
    if not writer or not previous_job_id:
        return False
    return writer == previous_job_id or (writer in collected.job_chain and writer != os.environ.get("SLURM_JOB_ID"))


def accounted_requeued_chain_walltime(
    job_chain: list[str], rows: list[tuple[str, str, int, int]], *, stored_walltime_s: float = 0
) -> float | None:
    """Elapsed seconds of every incarnation of a chain with a requeued allocation, when the records are complete.

    Every chain job needs terminal records numbered 0..n by restarts, and some job a requeue (n > 0). The stored
    counter must equal the elapsed time of every incarnation of the jobs before the last requeued one: an earlier
    requeue was reconciled and stored before the chain went on, and the last requeued incarnation stops before it
    adds anything, as does every later allocation. Otherwise the budget epoch is unknown.
    """
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
    if not job_chain or len(set(job_chain)) != len(job_chain) or {row[0] for row in rows} != set(job_chain):
        return None
    last = {job_id: max(row[3] for row in rows if row[0] == job_id) for job_id in job_chain}
    restarts: dict[str, list[int]] = {}
    for job_id, state, _, restart in rows:
        # A REQUEUED incarnation that a later one followed has ended (its Start and End are recorded, so its elapsed
        # time is known); the last incarnation still REQUEUED waits to run again and stays unknown.
        if state not in terminal and not (state == "REQUEUED" and restart < last[job_id]):
            return None
        restarts.setdefault(job_id, []).append(restart)
    if any(sorted(numbers) != list(range(len(numbers))) for numbers in restarts.values()):
        return None
    requeued = [index for index, job_id in enumerate(job_chain) if len(restarts[job_id]) > 1]
    if not requeued:
        return None
    before = set(job_chain[: requeued[-1]])
    if stored_walltime_s != sum(elapsed for job_id, _, elapsed, _ in rows if job_id in before):
        return None
    return float(sum(elapsed for _, _, elapsed, _ in rows))


def reconcile_requeued_chain(collected: CollectedEvidence, rows: list[tuple[str, str, int, int]]) -> CollectedEvidence:
    """Know a requeued chain's budget again once every incarnation has a terminal scheduler record.

    Only the requeue marker is reconciled: the counters themselves stayed readable and the missing fact is the elapsed
    time of the requeued incarnations, which complete terminal records give. Any other unknown reason stays unknown.
    """
    states = collected.counter_states or {}
    if (
        collected.accounting_unknown_reason != REQUEUE_UNKNOWN_REASON
        or states.get("infra_retries") not in ("absent", "ok")
        or states.get("walltime") not in ("absent", "ok")
    ):
        return collected
    total = accounted_requeued_chain_walltime(
        collected.job_chain, rows, stored_walltime_s=collected.walltime_used_s or 0
    )
    if total is None:
        return collected
    return collected.model_copy(
        update={
            "infra_retries_state": states["infra_retries"],
            "walltime_state": states["walltime"],
            "accounted_walltime_s": total,
            "accounting_reconciled": True,
        }
    )


def requeue_note(job_chain: list[str], rows: list[tuple[str, str, int, int]]) -> str | None:
    """The last requeued job of a chain and what its allocations account for, as a decision states it."""
    requeued = [job_id for job_id in job_chain if sum(row[0] == job_id for row in rows) > 1]
    if not requeued:
        return None
    incarnations = sorted((row for row in rows if row[0] == requeued[-1]), key=lambda row: row[3])
    cut = "was preempted" if incarnations[0][1] == "PREEMPTED" else f"ended {incarnations[0][1]}"
    return (
        f"job {requeued[-1]} {cut} and requeued; its {len(incarnations)} allocations account for "
        f"{sum(row[2] for row in incarnations)} s"
    )


def accounted_chain_walltime(
    job_chain: list[str],
    accounting: str,
    *,
    stored_walltime_s: float | None = 0,
) -> float | None:
    """Elapsed seconds of all recorded allocations, only when each has one terminal accounting row.

    ``stored_walltime_s`` None means no allocation stored a walltime: the prologue that accounts the previous
    allocation stopped first (after an operator's cancel, for one). The records' sum is the used walltime then.
    """
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
    states: dict[str, str] = {}
    for line in accounting.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            return None
        job_id, state, seconds = parts
        state = state.split()[0].rstrip("+") if state.split() else ""
        if job_id not in job_chain or job_id in elapsed or state not in terminal:
            return None
        if not seconds.isdecimal():
            return None
        elapsed[job_id] = int(seconds)
        states[job_id] = state
    if set(elapsed) != set(job_chain):
        return None
    total = sum(elapsed.values())
    # The prologue records predecessors. A stopped guard may already have accounted for the last job too.
    # After an operator cancels a segment, the next prologue stops at the cancel before recording it, so neither
    # the cancelled segment nor the one that stopped there is stored. A segment before the cancel must have stored
    # its time, so a chain that stored nothing still has to match exactly.
    # Other values cannot identify the budget epoch (e.g. an intentional manual reset in an old chain).
    accepted = {total - elapsed[job_chain[-1]], total}
    if len(job_chain) >= 3 and states[job_chain[-2]] == "CANCELLED":
        accepted.add(total - elapsed[job_chain[-1]] - elapsed[job_chain[-2]])
    if stored_walltime_s is not None and stored_walltime_s not in accepted:
        return None
    return float(total)


def unsettled_chain_note(decision: RecoveryDecision) -> str:
    """The chain stopped before the run settled: how many rollouts are unsettled, by state, so the stop reads
    as an incomplete run and never as a finished one. Unsettled is wider than the coverage report's ``open`` (the
    count in the chain's own stop line): it also holds final failures that were not counted as zero."""
    settled = (IdentityState.SCORED, IdentityState.MODEL_FINAL, IdentityState.OMITTED)
    unsettled = decision.expected - sum(decision.counts.get(state, 0) for state in settled)
    by_state = ", ".join(
        f"{count} {state}"
        for state, count in sorted(decision.counts.items())
        if count and state not in {s.value for s in settled}
    )
    return (
        f"the job chain stopped with {unsettled} of {decision.expected} rollouts unsettled"
        + (f" ({by_state})" if by_state else "")
        + "; final failures are included, the stop line's open count leaves them out (.gym_chain_incomplete)"
    )


def gate_task_summary(
    decision: RecoveryDecision,
    *,
    score: float | None,
    summary_valid: bool | None,
    summary_reasons: list[str],
    health_passed: bool | None,
    health_reasons: list[str],
) -> dict[str, Any]:
    """One task entry of the acceptance gate's run summary, filled from the decision and the existing validators.

    ``expected_samples`` is always the full materialized inventory; ``scored_samples`` counts only scored and
    final-model identities, so a judge, infra, terminal or missing identity can never make them equal. A settled
    final outcome the metric does not score (``IdentityEvidence.unscored``) is stated, not counted as scored. Summary validity
    and response/trial health stay separate. ``False`` is a failure and an error; ``None`` (unknown or missing) is no
    failure but never a pass: its reason is a warning and it withholds SUCCESS.

    ``status``, first match wins: RUNNING (a job is live); PLANNED (recovery work is eligible; only a caller that
    dispatches it reports RESUMING); FAILED (anything in ``errors``, or the run is not a complete NOOP); UNVERIFIED
    (nothing failed, but summary validity or response/trial health is unknown, so acceptance is not proven); SUCCESS.
    """
    errors: list[str] = []
    warnings: list[str] = []
    unverified = summary_valid is None or health_passed is None or decision.unverified
    states = decision.counts
    for state, label in (
        (IdentityState.JUDGE_RECOVERABLE, "judge failures awaiting re-judgment"),
        (IdentityState.RETRYABLE, "retryable failures awaiting another attempt"),
        (IdentityState.EXHAUSTED, "failures with spent attempts"),
        (IdentityState.TERMINAL, "terminal unscored failures"),
        (IdentityState.MISSING, "identities with no row"),
        (IdentityState.MASKED, "masked results, completed but not measured"),
        (IdentityState.UNKNOWN, "failures nobody owns"),
    ):
        if states.get(state):
            errors.append(f"{states[state]} {label}: {decision.identities[state.value][:5]}")
    if decision.unverified:
        # Deliberately unverified: there is no score to read, and that is no failure.
        score = None
        warnings.append(decision.reason)
    elif score is None or not math.isfinite(score):
        # An incomplete run has no score to read yet, whatever score options were passed: say that, not the options.
        settled = sum(
            states.get(state, 0) for state in (IdentityState.SCORED, IdentityState.MODEL_FINAL, IdentityState.OMITTED)
        )
        errors.append(
            "score unavailable: no requested metric was read (pass --score-key, and --score-agent or "
            "--score-environment for a Gym aggregate)"
            if decision.complete
            else f"score unavailable: the run is not complete ({settled}/{decision.expected} rollouts settled); "
            "the score is read once every rollout is settled"
        )
    if decision.duplicate_results:
        errors.append(f"duplicate result rows: {decision.duplicate_results[:5]}")
    if decision.action == RecoveryAction.STOP:
        errors.append(f"recovery stopped: {decision.reason}")
    if decision.action == RecoveryAction.UNKNOWN:
        errors.append(f"recovery undecided: {decision.reason}")
    if summary_valid is not True:
        (errors if summary_valid is False else warnings).append(
            "summary validity "
            + ("unknown" if summary_valid is None else "failed")
            + (f": {summary_reasons}" if summary_reasons else "")
        )
    if health_passed is not True:
        (errors if health_passed is False else warnings).append(
            "response/trial health "
            + ("unknown" if health_passed is None else "failed")
            + (f": {health_reasons}" if health_reasons else "")
        )
    unscored = [e.identity for e in decision.evidence if e.state == IdentityState.MODEL_FINAL and e.unscored]
    if states.get(IdentityState.MODEL_FINAL, 0) > len(unscored):
        final = [e.identity for e in decision.evidence if e.state == IdentityState.MODEL_FINAL and not e.unscored]
        warnings.append(f"{len(final)} final evaluated-model failures scored zero: {final[:5]}")
    if unscored:
        warnings.append(
            f"{len(unscored)} answered agent deadlines settled with their kept answer, not scored by the metric: "
            f"{unscored[:5]}"
        )
    if states.get(IdentityState.OMITTED):
        # Complete by the stage's own policy: stated, never folded into the scored or failed counts.
        warnings.append(omission_summary(decision.evidence))
    if decision.scheduler_live:
        status = "RUNNING"
    elif decision.action in (RecoveryAction.RESUBMIT, RecoveryAction.REVERIFY, RecoveryAction.FRESH_RESTART):
        # A plan, not a dispatch: only the caller that dispatches it reports RESUMING.
        status = "PLANNED"
    elif decision.action == RecoveryAction.NOOP and decision.complete and not errors:
        status = "UNVERIFIED" if unverified else "SUCCESS"
    else:
        status = "FAILED"
    return {
        "status": status,
        "expected_samples": decision.expected,
        "scored_samples": states.get(IdentityState.SCORED, 0)
        + states.get(IdentityState.MODEL_FINAL, 0)
        - len(unscored),
        "omitted_samples": states.get(IdentityState.OMITTED, 0),
        "score": score,
        "errors": errors,
        "warnings": warnings,
    }


RecoveryOutcome.model_rebuild()


def quality_inputs(
    decision: RecoveryDecision, collected: CollectedEvidence, *, max_unhealthy_fraction: float | None
) -> tuple[bool | None, list[str], bool | None, list[str]]:
    """Summary validity and per-rollout health from the run's ``quality_summary.json``, kept separate.

    Validity is the health check's own accounting verdict (``run.artifacts.healthy``, null for a partial summary),
    and must count the same expected identities as the inventory. Per-rollout health is the rollout verdict count
    (``run.verdicts``), judged only against an explicit ``max_unhealthy_fraction``; it is not separately validated response or
    trial health, which this adapter has no separate evidence for. Any missing piece is ``None``.
    """
    quality = collected.quality
    if quality is None:
        return None, ["no quality_summary.json"], None, ["no quality_summary.json"]
    if quality.get("source_matches") is not True:
        reason = "quality summary is not bound to this run's sole rollouts.jsonl source"
        return None, [reason], None, [reason]
    artifacts = quality.get("artifacts") or {}
    raw_reasons = artifacts.get("unhealthy_reasons")
    if raw_reasons is not None and not isinstance(raw_reasons, list):
        reason = "quality summary artifacts.unhealthy_reasons is malformed"
        return None, [reason], None, [reason]
    summary_reasons = [str(r) for r in raw_reasons or []]
    summary_valid = artifacts.get("healthy") if isinstance(artifacts.get("healthy"), bool) else None
    if artifacts.get("partial"):
        summary_valid, summary_reasons = None, summary_reasons + ["partial summary: per-record checks not evaluated"]
    if (
        type(artifacts.get("expected")) is not int
        or artifacts["expected"] < 0
        or artifacts["expected"] != decision.expected
    ):
        summary_valid = None if summary_valid is None else False
        summary_reasons.append(f"summary counts {artifacts.get('expected')} expected, inventory {decision.expected}")
    verdicts = quality.get("verdicts")
    if not isinstance(verdicts, dict) or not all(
        type(verdicts.get(k)) is int and verdicts[k] >= 0 for k in ("healthy", "unhealthy", "unobserved")
    ):
        return summary_valid, summary_reasons, None, ["no valid rollout verdict counts (non-negative integers)"]
    counts = f"{verdicts['healthy']} healthy, {verdicts['unhealthy']} unhealthy, {verdicts['unobserved']} unobserved"
    if verdicts["healthy"] + verdicts["unhealthy"] + verdicts["unobserved"] != decision.expected:
        return (
            summary_valid,
            summary_reasons,
            None,
            [f"verdict counts ({counts}) do not cover the {decision.expected} expected"],
        )
    if max_unhealthy_fraction is None:
        return summary_valid, summary_reasons, None, [f"no response-health criterion configured ({counts})"]
    if (
        isinstance(max_unhealthy_fraction, bool)
        or not isinstance(max_unhealthy_fraction, (int, float))
        or not math.isfinite(max_unhealthy_fraction)
        or not 0 < max_unhealthy_fraction <= 1
    ):
        return summary_valid, summary_reasons, None, [f"invalid response-health criterion {max_unhealthy_fraction!r}"]
    judged = verdicts["healthy"] + verdicts["unhealthy"]
    if verdicts["unobserved"] or not judged:
        return summary_valid, summary_reasons, None, [f"health not observed for every rollout ({counts})"]
    share = verdicts["unhealthy"] / judged
    # The criterion is a strict bound, as the acceptance gate's "< limit": a share equal to it fails.
    if share >= max_unhealthy_fraction:
        return (
            summary_valid,
            summary_reasons,
            False,
            [f"unhealthy {share:.2%} not below {max_unhealthy_fraction:.2%} ({counts})"],
        )
    return summary_valid, summary_reasons, True, [counts]


def mlflow_notes(local: Mapping[str, Any] | None) -> list[str]:
    """What the gate states about a passing MLflow part: a current post-run export beside a live run wins."""
    local = local or {}
    live, receipt = local.get("live_run_id"), local.get("receipt")
    if isinstance(live, str) and live.strip() and isinstance(receipt, dict) and receipt.get("current") is True:
        return [
            f"MLflow live run {live} and post-run export run {receipt.get('run_id')} both exist: the post-run export "
            "receipt is current for the saved artifacts, so it is the run of record"
        ]
    return []


MLFLOW_STATUS_NOT_OBSERVED = "MLflow terminal status not observed"
# The longest a status query of the tracking server may take per HTTP request, and its retries.
MLFLOW_STATUS_QUERY_TIMEOUT_S = "10"
MLFLOW_STATUS_QUERY_RETRIES = "1"


def query_mlflow_status(
    tracking_uri: str, run_id: str, client_factory: Callable[[str], Any] | None = None
) -> tuple[str | None, str | None]:
    """The status of ``run_id`` on the tracking server, as ``(status, None)``, or ``(None, error)`` when it could not
    be read: a missing client, an unreachable or refusing server, a missing run. Short timeout and one retry;
    ``client_factory`` builds the client from the URI (default ``mlflow.MlflowClient``)."""
    knobs = {
        "MLFLOW_HTTP_REQUEST_TIMEOUT": MLFLOW_STATUS_QUERY_TIMEOUT_S,
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": MLFLOW_STATUS_QUERY_RETRIES,
    }
    saved = {key: os.environ.get(key) for key in knobs}
    os.environ.update(knobs)
    try:
        if client_factory is None:
            from mlflow import MlflowClient

            def client_factory(uri: str) -> Any:
                return MlflowClient(tracking_uri=uri)

        status = client_factory(tracking_uri).get_run(run_id).info.status
    except Exception as error:  # a status that cannot be read is reported, whatever the reason
        # Request errors carry the URL, which may hold a password: mask it before cutting the text.
        return None, f"{type(error).__name__}: {mask_url_passwords(str(error))[:200]}"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if not isinstance(status, str) or not status.strip():
        return None, f"the server reported no status for run {run_id}"
    return status, None


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
        # The post-run export receipt, current for the saved artifacts, is the run of record; a live run is not.
        if isinstance(receipt, dict) and receipt.get("current") is True:
            if observed_run_id not in (None, live, receipt.get("run_id")):
                errors.append(
                    f"MLflow observed status is for run {observed_run_id}, neither the live run {live} nor the "
                    f"post-run export run {receipt.get('run_id')}"
                )
        else:
            stale = receipt.get("run_id") if isinstance(receipt, dict) else "with a malformed receipt"
            errors.append(
                f"MLflow live run {live} and a post-run export both exist, and the post-run export run {stale} is "
                "stale: its receipt is not current for the saved artifacts, so neither is the run of record"
            )
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
                MLFLOW_ROLLOUTS_NOT_UPLOADED
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
        errors.append(MLFLOW_STATUS_NOT_OBSERVED)
    elif observed_status != "FINISHED":
        errors.append(f"MLflow run status {observed_status}")
    return errors


# The acceptance bounded failure policy: a gate passes only below this failure rate, measured before rounding.
GATE_FAILURE_LIMIT = 0.02


class GateCriteria(BaseModel):
    """The response and trial failure fractions a caller may still pass. They are validated (an invalid value is
    refused, never clamped) and reported, but no longer applied: the gate's only health verdict is evaluation health
    on the bounded failure policy's limit (``trial_verdict``), and response and trial failures are reported as facts
    (``failure_facts``) for consumers to judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_response_failure_fraction: float = GATE_FAILURE_LIMIT
    max_trial_failure_fraction: float = GATE_FAILURE_LIMIT
    # Whether a policy-model response timeout (``model_policy_response_timeout``: every attempt of the policy's
    # request timed out after its upload) is the model's fault. True by default: the deployment serving the evaluated
    # model is part of what is evaluated, so its slowness is on the model's side, never the evaluation's. A
    # self-hosted evaluation that owns the serving deployment may set it to false, and the timeout then counts as a
    # server fault (evaluation-side). Applied by the trial gate (``classify_failure``).
    policy_timeout_counts_as_model: bool = Field(default=True, strict=True)

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

    def report(self, *, classification: FailureClassificationConfig | None = None) -> dict[str, Any]:
        """The effective criterion and the accepted, unapplied fractions, as reported beside the gate results."""
        rule = "strictly below, before rounding"
        return {
            "evaluation": {
                "max_failure_fraction": GATE_FAILURE_LIMIT,
                "passes": rule,
                "policy_timeout_counts_as_model": self.policy_timeout_counts_as_model,
                "classification": (classification or FailureClassificationConfig()).model_dump(mode="json"),
            },
            "response": {"max_failure_fraction": self.max_response_failure_fraction, "applied": False},
            "trial": {"max_failure_fraction": self.max_trial_failure_fraction, "applied": False},
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
    decision = _health_view(decision)
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


def _cut_judge_reply(call: Mapping[str, Any]) -> bool:
    """Whether a captured judge reply was cut short: an incomplete Responses reply, or a chat completion that stopped
    on its output cap or a content filter."""
    meta = call.get("response_metadata") if isinstance(call, Mapping) else None
    return isinstance(meta, Mapping) and (
        (meta.get("dialect") == "responses" and meta.get("response_status") == "incomplete")
        or (
            meta.get("dialect") == "chat"
            and meta.get("response_status") is None
            and meta.get("finish_reason") in ("length", "content_filter")
        )
    )


def _observed_trial_operations(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    *,
    classification: FailureClassificationConfig | None = None,
    judge_recovery: dict[str, int] | None = None,
    grading_failures: list[str] | None = None,
    policy_recovery: dict[str, int] | None = None,
    verifier_recorded: list[Identity] | None = None,
    artifact_recorded: list[Identity] | None = None,
    trajectory_incomplete: dict[Identity, list[str]] | None = None,
    agent_failed_graded: list[Identity] | None = None,
) -> bool:
    """Whether every expected trial that is not a counted failure is observed: its saved answer is bound to its grading.

    Observed means a recorded grading bound to the saved answer and score: one completed verifier's grading operation
    bound to the row, its own grading record, a saved grading artifact (``GRADING_ARTIFACTS``) with the collector's
    binding, or the model's own failed answer. Only that decides coverage. Each trial's model/tool trajectory is
    joined too (its one agent invocation, turns, calls, tools and failure events), but that is observability
    evidence: a trial whose trajectory does not join is reported in ``trajectory_incomplete`` with the kinds of its
    missing evidence (``_trajectory_kinds``), and it is still observed.

    Join observed model/tool operations and the outer verifier invocation for every expected trial.

    A trial whose verifier recorded no grading operation and no grading record of its own is observed through a
    saved grading artifact (``GRADING_ARTIFACTS``) or through the model's own failed answer the benchmark's rule
    scored (``_model_failure``); those trials go to ``artifact_recorded``, and the verifier's captured calls are its
    own. An environment reward record stands in for a missing verifier invocation.

    A source-reviewed programmatic grading operation must bind the saved response and score. A verifier that
    recorded no grading operation and owns no model call is observed through its own grading record instead
    (``_verifier_recorded``); those trials go to ``verifier_recorded``. Empty captured model calls or a reward
    alone never establish grading coverage; old or uninstrumented verifiers remain unknown.
    Recovered grading attempts are reported separately and do not count toward the trial-health threshold;
    every failed attempt must join the same completed, result-bound grading dependency. A policy request retried
    inside one completed operation is recovered the same way (``policy_recovery`` receives its counts): each failed
    attempt is an event of that operation and of the row, and the answer is the final attempt's.
    """
    classification = decision_classification(decision, classification)
    decision = _health_view(decision)
    from nemo_gym.core.episode_context import maybe_rollout_id_from_run_body

    # A settled model failure (MODEL_FINAL) or a rollout that ended on a failed policy request is a counted trial
    # failure with its recorded kind; only the other trials need their operations and grading observed. A row an
    # accepted partial stage omitted is no trial (``trial_population``), so it needs no observation.
    if (
        collected is None
        or not collected.quality
        or collected.quality.get("source_matches") is not True
        or decision.counts.get(IdentityState.SCORED, 0) + decision.counts.get(IdentityState.MODEL_FINAL, 0)
        != trial_population(decision)
    ):
        return False
    rows = collected.results
    expected = set(decision.identities.get(IdentityState.SCORED.value, []))
    if rows is None or len(rows) != len(expected) or {_identity(row) for row in rows} != expected:
        return False
    final_zero = final_zero_trials(decision, collected)
    provenance = collected.quality.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("resolved_config") != "read":
        return False
    servers = provenance.get("servers")
    if not isinstance(servers, dict):
        return False
    input_task_ids = {_identity(item): item.get("_ng_collected_task_id") for item in collected.materialized or []}
    populations = (set(), set(), set(), set(), set())
    grading_ids = set()
    recovered = {
        "judge_recovered_attempts": 0,
        "judge_recovered_operations": 0,
        "observed_grading_attempt_errors": 0,
        "observed_grading_failed_operations": 0,
    }
    recovered_policy = {"policy_recovered_attempts": 0, "policy_recovered_operations": 0}
    recovered_events = {}
    invalid_grades = []
    recorded_trials = []
    artifact_trials = []
    incomplete: dict[Identity, list[str]] = {}
    agent_failed: list[Identity] = []

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

    def completed_once(call, *, judge=False, valid_grade=True, truncated=False):
        """One successful HTTP response with one completed physical attempt, ids unique in the run.

        ``truncated``: the policy's final answer that ran into its output cap (a length finish on an incomplete or
        chat response); the model answered, so it is a model fault the trial gate counts, not missing evidence.
        """
        meta = call["response_metadata"]
        operations, report = meta["model_operations"], meta["failure_report"]
        if truncated and not (
            meta.get("finish_reason") == "length"
            and meta.get("error_category") in (None, "incomplete")
            and (
                meta.get("response_status") == "incomplete"
                or (meta.get("dialect") == "chat" and meta.get("response_status") is None)
            )
        ):
            return False
        if (
            type(meta["status_code"]) is not int
            or not 200 <= meta["status_code"] < 300
            or (meta.get("error_category") is not None and not truncated)
            or not (
                truncated
                or meta.get("response_status") == "completed"
                or (
                    meta.get("dialect") == "chat"
                    and meta.get("response_status") is None
                    and meta.get("finish_reason") == "stop"
                )
                or (
                    judge
                    and not valid_grade
                    and (
                        (meta.get("dialect") == "responses" and meta.get("response_status") == "incomplete")
                        or (
                            meta.get("dialect") == "chat"
                            and meta.get("response_status") is None
                            and meta.get("finish_reason") in ("length", "content_filter")
                        )
                    )
                )
            )
            or not isinstance(operations, list)
            or not operations
            or not isinstance(report["events"], dict)
            or report["omitted"] is not False
            or report["failure_id"] is not None
        ):
            return False
        if judge and (len(operations) > 1 or report["events"]):
            events = report["events"]
            used = set()
            failed_operations = 0
            for number, operation in enumerate(operations):
                final = number == len(operations) - 1
                op_id, attempt_id = operation.get("operation_id"), operation.get("final_attempt_id")
                attempts = operation.get("http_attempts")
                transport = operation.get("transport_failures")
                if (
                    not nonempty(op_id)
                    or not nonempty(attempt_id)
                    or op_id in populations[2]
                    or attempt_id in populations[4]
                    or operation.get("outcome") != ("completed" if final else "failed")
                    or type(attempts) is not int
                    or attempts < 1
                    or type(transport) is not int
                    or not 0 <= transport <= attempts
                    or any(
                        type(operation.get(key)) not in (int, float)
                        or not math.isfinite(operation[key])
                        or operation[key] < 0
                        for key in ("retry_wait_s", "request_elapsed_ms")
                    )
                ):
                    return False
                owned = {
                    key: event
                    for key, event in events.items()
                    if isinstance(event, dict) and event.get("operation_id") == op_id
                }
                # A failed HTTP attempt or a transport failure (no HTTP status) before the final attempt is its own
                # recorded event of this operation, as for a policy request.
                if len(owned) != attempts - int(final) + transport or transport != sum(
                    event.get("http_status") is None for event in owned.values()
                ):
                    return False
                ids = set()
                for key, event in owned.items():
                    eid = event.get("attempt_id")
                    if (
                        key != event.get("event_id")
                        or not nonempty(key)
                        or not nonempty(eid)
                        or eid in ids
                        or eid in populations[4]
                        or key in used
                        or key in recovered_events
                        or event.get("boundary") != "model"
                        or event.get("operation") != "model_request"
                        or event.get("failure_kind") != "model_request_error"
                        or event.get("cause_id") is not None
                        or not (
                            event.get("http_status") is None
                            or (type(event["http_status"]) is int and 400 <= event["http_status"] <= 599)
                        )
                    ):
                        return False
                    ids.add(eid)
                    used.add(key)
                if final:
                    if operation.get("failure_id") is not None or attempt_id in ids:
                        return False
                elif (
                    operation.get("failure_id") not in owned
                    or owned[operation["failure_id"]].get("attempt_id") != attempt_id
                ):
                    return False
                if owned:
                    failed_operations += 1
                populations[2].add(op_id)
                populations[4].update(ids | {attempt_id})
            if used != set(events):
                return False
            if not all(
                nonempty(value)
                for value in (
                    call["model_call_id"],
                    meta["model_ref"]["type"],
                    meta["model_ref"]["name"],
                    meta["response_id"],
                )
            ):
                return False
            identity = (meta["model_ref"]["type"], meta["model_ref"]["name"], meta["response_id"])
            if call["model_call_id"] in populations[0] or identity in populations[1]:
                return False
            populations[0].add(call["model_call_id"])
            populations[1].add(identity)
            recovered_events.update(events)
            recovered["observed_grading_attempt_errors"] += len(used)
            recovered["observed_grading_failed_operations"] += failed_operations
            if valid_grade:
                recovered["judge_recovered_attempts"] += len(used)
                recovered["judge_recovered_operations"] += failed_operations
            return True
        if len(operations) != 1:
            return False
        operation = operations[0]
        events = report["events"]
        attempts, transport = operation["http_attempts"], operation["transport_failures"]
        if (
            operation["outcome"] != "completed"
            or operation["failure_id"] is not None
            or type(attempts) is not int
            or attempts < 1
            or type(transport) is not int
            or transport < 0
            # Every failed attempt before the final one is its own recorded event of this operation.
            or len(events) != attempts - 1 + transport
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
        # A policy request retried inside one completed operation (a timeout, a reset or a 5xx, then an answer) is a
        # recovered attempt, not a trial failure: each failed attempt must join this operation exactly once.
        failed_ids = set()
        for key, event in events.items():
            status = event.get("http_status") if isinstance(event, dict) else None
            if (
                not isinstance(event, dict)
                or key != event.get("event_id")
                or not nonempty(event.get("attempt_id"))
                or event["attempt_id"] in failed_ids
                or event["attempt_id"] in populations[4]
                or event["attempt_id"] == operation["final_attempt_id"]
                or key in recovered_events
                or event.get("boundary") != "model"
                or event.get("operation") != "model_request"
                or event.get("operation_id") != operation["operation_id"]
                or event.get("failure_kind") != "model_request_error"
                or event.get("cause_id") is not None
                or not (status is None or (type(status) is int and 400 <= status <= 599))
            ):
                return False
            failed_ids.add(event["attempt_id"])
        if sum(event.get("http_status") is None for event in events.values()) != transport:
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
        populations[4].update(failed_ids)
        recovered_events.update(events)
        recovered_policy["policy_recovered_attempts"] += len(events)
        recovered_policy["policy_recovered_operations"] += bool(events)
        return True

    def trajectory_complete(row, observed, calls, artifact, truncated):
        """The trial's model/tool trajectory joins its one agent invocation, turn by turn, call by call."""
        try:
            invocations, turns, tools = observed["invocations"], observed["turns"], observed["tool_calls"]
            if (
                not all(isinstance(items, list) for items in (invocations, calls, turns, tools))
                or len(invocations) != 1
                or not calls
                or len(turns) != len(calls)
                or type(observed["tool_call_count"]) is not int
                or observed["tool_call_count"] != len(tools)
                or not isinstance(observed["gaps"], list)
                or any(gap.get("code") != "resolution_unavailable" for gap in observed["gaps"])
            ):
                return False
            invocation = invocations[0]
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
                or len(invocation["model_calls"]) != len(calls)
            ):
                return False
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
                if not completed_once(call, truncated=truncated and number == len(ordered)):
                    return False
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
                    # A truncated or empty final answer may hold only reasoning or a cut-off message.
                    if (
                        emitted
                        or not (truncated or artifact == "model_failure")
                        and not any(
                            item.get("type") == "message"
                            and item.get("role") == "assistant"
                            and item.get("status") == "completed"
                            for item in outputs
                        )
                    ):
                        return False
                elif not emitted:
                    return False
            saved_response = row["response"]
            return (
                set(requested) == set(tool_by_id)
                and saved_response.get("id") == ordered[-1]["response_metadata"]["response_id"]
                and saved_response.get("status") == ("incomplete" if truncated else "completed")
            )
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            return False

    try:
        for row in rows:
            if _identity(row) in final_zero:
                continue
            if _scored_kind(row) is not None:
                continue  # A counted failure with its own recorded kind (``_scored_fault``), like a final zero.
            if _answer_fault(row, collected):
                # A counted model failure (``_answer_fault``), like a final zero: only its answer must be observed.
                if not _observed_incomplete_answer(row):
                    return False
                continue
            if _unparsed_judge_verdict(row):
                continue  # A counted grading fault (``_scored_fault``): its reward is no grade to observe.
            truncated = _incomplete_answer(row)
            prior_recovered_events = set(recovered_events)
            owner = _verification_owner(row, servers)
            if owner is None:
                return False
            # An agent without a resources server grades its own rollouts (verifiers_agent); it has no resources
            # server to look up, and only its own grading record can observe such a trial (below).
            resource = servers[owner] if servers[owner].get("kind") == "resources_servers" else None
            if (
                row.get(FAILURE_CLASS) is not None
                or row.get("failure_kind") is not None
                or row.get(FAILURE_ID) is not None
                or row.get("_ng_failure_events_omitted", False) is not False
                or not isinstance(row.get(FAILURE_EVENTS, {}), dict)
            ):
                return False
            observed = row["ng_trial_observations"]
            verifiers, calls = observed["verifier_invocations"], observed.get("model_calls")
            judge_calls = None
            single = verifiers[0] if isinstance(verifiers, list) and len(verifiers) == 1 else None
            graded = single.get("grading_operations") if isinstance(single, dict) else None
            # Without a grading operation, a saved grading artifact or the model's own failed answer records the
            # grading; the verifier's captured calls (a judge's) are then its own, never the policy's.
            artifact = None
            recorded = single is not None and _verifier_recorded(single, row, owner)
            if not graded and not recorded:
                artifact = grading_artifact(row) or ("model_failure" if _model_failure(row) else None)
            # An agent without a resources server is observed through its own grading record, or, when no verifier
            # ran, through a saved grading artifact (the environment's reward record, or the agent rubric's own saved
            # score); its own failed answer alone does not observe it.
            if resource is None and not recorded and not (artifact not in (None, "model_failure") and verifiers == []):
                return False
            if artifact is not None and single is not None and isinstance(calls, list):
                owned = single.get("model_calls")
                if not isinstance(owned, list):
                    return False
                judge_calls = []
                for ref in owned:
                    hit = [call for call in calls if matches(ref, call)]
                    if len(hit) != 1 or any(hit[0] is seen for seen in judge_calls):
                        return False
                    judge_calls.append(hit[0])
                calls = [call for call in calls if not any(call is judge for judge in judge_calls)]
            if isinstance(graded, list) and len(graded) == 1 and (graded[0] or {}).get("kind") == "llm_judge":
                # Every dependency is exactly one captured call owned by this verifier invocation, every owned call is
                # a dependency, and only the remaining calls are the policy's.
                dependencies, owned = graded[0].get("dependencies"), single.get("model_calls")
                if not dependencies or not isinstance(dependencies, list) or not isinstance(owned, list):
                    return False
                if not isinstance(calls, list):
                    return False
                judge_calls = []
                for ref in dependencies:
                    hit = [call for call in calls if matches(ref, call)]
                    if len(hit) != 1 or any(hit[0] is seen for seen in judge_calls):
                        return False
                    if sum(1 for item in owned if matches(item, hit[0])) != 1:
                        return False
                    judge_calls.append(hit[0])
                if len(owned) != len(judge_calls):
                    return False
                calls = [call for call in calls if not any(call is judge for judge in judge_calls)]
            # Grading coverage: the trial's one verifier (none only beside an environment reward) and how it graded.
            if not isinstance(verifiers, list) or len(verifiers) != (
                0 if not verifiers and (artifact == "environment_reward" or resource is None) else 1
            ):
                return False
            verifier = verifiers[0] if verifiers else None
            if verifier is not None and (
                verifier["status"] != "completed"
                or verifier["failure_kind"] is not None
                or _grading_sandbox_failed(verifier.get("sandbox_observations"))
                or (judge_calls is None and verifier["model_calls"] != [])
                or not nonempty(verifier["invocation_id"])
                or verifier["invocation_id"] in populations[3]
            ):
                return False
            grading = verifier.get("grading_operations") if verifier is not None else None
            if artifact is not None:
                # A saved grading artifact must be bound to the saved answer and score; the model's own failed
                # answer is a counted model failure instead.
                if artifact != "model_failure" and not _bound_answer(row):
                    return False
                # The verifier's own judge replies are part of that grading: each completed once.
                if judge_calls and not all(completed_once(call, judge=True) for call in judge_calls):
                    return False
                grade = None
                artifact_trials.append(_identity(row))
            elif judge_calls is None and _verifier_recorded(verifier, row, owner):
                grade = None
                recorded_trials.append(_identity(row))
            elif not isinstance(grading, list) or len(grading) != 1 or not isinstance(grading[0], dict):
                return False
            else:
                grade = grading[0]
            binding = row.get("_ng_collected_grading_binding")
            if grade is not None and (
                grade.get("kind") != ("programmatic_calculation" if judge_calls is None else "llm_judge")
                or grade.get("outcome")
                not in (
                    ("completed", "unknown")
                    if judge_calls is not None and grading_failures is not None
                    else ("completed",)
                )
                or (judge_calls is None and grade.get("dependencies") != [])
                or not nonempty(grade.get("operation_id"))
                or grade["operation_id"] in grading_ids
                or not isinstance(binding, str)
                or len(binding) != 64
                or grade.get("result_binding_sha256") != binding
            ):
                return False
            if grade is not None:
                grading_ids.add(grade["operation_id"])
            if verifier is not None:
                populations[3].add(verifier["invocation_id"])
            # A completed grade that rests on a judge reply cut by its output cap or a content filter has no valid
            # verdict behind it: where the caller counts those, it is an observed grading without a valid verdict.
            cut_reply = (
                grade is not None
                and judge_calls is not None
                and grade["outcome"] == "completed"
                and grading_failures is not None
                and any(_cut_judge_reply(call) for call in judge_calls)
            )
            if (
                grade is not None
                and judge_calls is not None
                and not all(
                    completed_once(call, judge=True, valid_grade=grade["outcome"] == "completed" and not cut_reply)
                    for call in judge_calls
                )
            ):
                return False
            if cut_reply:
                invalid_grades.append(_identity(row))
            if (
                grade is not None
                and judge_calls is not None
                and grade["outcome"] == "unknown"
                and not _unknown_grading_outcome(row)  # counted once, as ``judge_unparseable`` (``_scored_fault``)
            ):
                invalid_grades.append(_identity(row))
            # An agent invocation that says verification was skipped contradicts any recorded grading.
            if any(
                isinstance(invocation, dict) and invocation.get("verification_skipped") is True
                for invocation in (observed.get("invocations") or [])
            ):
                return False
            # A judge's call is the verifier's own: the agent never asked for it.
            agent_refs = [
                ref
                for invocation in (observed.get("invocations") or [])
                if isinstance(invocation, dict)
                for ref in (invocation.get("model_calls") or [])
            ]
            if judge_calls and any(matches(ref, call) for ref in agent_refs for call in judge_calls):
                return False
            saved_response = row.get("response")
            if not isinstance(saved_response, dict) or saved_response.get("error") is not None:
                return False
            # The agent's own run failed, yet its saved answer was graded: observed, and reported as a fact.
            if any(
                isinstance(invocation, dict) and invocation.get("status") == "failed"
                for invocation in (observed.get("invocations") or [])
            ):
                agent_failed.append(_identity(row))
            # Trajectory evidence: reported, never a coverage condition.
            if not trajectory_complete(row, observed, calls, artifact, truncated):
                incomplete[_identity(row)] = _trajectory_kinds(observed) or ["operation_join_incomplete"]
            # Every recorded model failure is accounted for: a failure is never hidden behind a graded answer.
            fresh = {key: value for key, value in recovered_events.items() if key not in prior_recovered_events}
            # A graded row whose rollout ended on the model's own failed request (its context limit, say) is a model
            # result at any reward: the verifier graded the state the model reached. That request's failure is
            # accounted for; one of an evaluation-side kind is not.
            ended = _ended_on_failed_request(row) if _reward(row) not in (None, 0) else None
            if ended is not None and _row_fault_class(row, ended[0], classification) == FailureClass.MODEL:
                report = (ended[1].get("response_metadata") or {}).get("failure_report")
                events = report.get("events") if isinstance(report, dict) else None
                if isinstance(events, dict):
                    fresh |= {key: event for key, event in events.items() if isinstance(event, dict)}
            if _unaccounted_failures(row, fresh):
                return False
    except (KeyError, TypeError, ValueError, AttributeError):
        return False
    if judge_recovery is not None:
        judge_recovery.update(recovered)
    if grading_failures is not None:
        grading_failures.extend(invalid_grades)
    if policy_recovery is not None:
        policy_recovery.update(recovered_policy)
    if verifier_recorded is not None:
        verifier_recorded.extend(recorded_trials)
    if artifact_recorded is not None:
        artifact_recorded.extend(artifact_trials)
    if trajectory_incomplete is not None:
        trajectory_incomplete.update(incomplete)
    if agent_failed_graded is not None:
        agent_failed_graded.extend(agent_failed)
    return True


# A sandbox outcome that reports the sandbox did not finish its work; "unknown" means the provider did not report one.
_SANDBOX_FAILURES = frozenset({"failed", "timeout", "oom", "sandbox_error", "cancelled"})


def _teardown_failure_only(item: Mapping) -> bool:
    """Whether a sandbox observation's only failure is its teardown: the producer records a failed close or release
    after the work as outcome ``sandbox_error`` or ``unknown`` with that cleanup error as its ``error_type``."""
    errors = item.get("cleanup_errors")
    return (
        isinstance(errors, list)
        and bool(errors)
        and item.get("outcome") in ("unknown", "sandbox_error")
        and item.get("error_type") == errors[0]
    )


def _grading_sandbox_failed(observations: Any) -> bool:
    """Whether a verifier's sandbox observations report that a grading sandbox failed, or are malformed.

    They are lifecycle evidence of the sandboxes the verification used, not a grading record: a completed or
    unreported outcome, an unconfirmed or failed teardown, a failing test command's exit code and the agent's own
    sandbox outcome do not leave a bound grading unobserved. A verifier sandbox whose outcome is a failure, or that
    records an error other than its teardown's, does.
    """
    if observations is None:
        return False
    if not isinstance(observations, list):
        return True
    return any(
        not isinstance(item, Mapping)
        or (
            item.get("role") == "verifier"
            and not _teardown_failure_only(item)
            and (item.get("outcome") in _SANDBOX_FAILURES or item.get("error_type") is not None)
        )
        for item in observations
    )


def _failed_request(call: Any) -> tuple[str, Any] | None:
    """The failure kind and HTTP status of a captured call whose final model operation failed with its own recorded
    request event, or None."""
    meta = call.get("response_metadata") if isinstance(call, dict) else None
    operations = meta.get("model_operations") if isinstance(meta, dict) else None
    report = meta.get("failure_report") if isinstance(meta, dict) else None
    events = report.get("events") if isinstance(report, dict) else None
    if not isinstance(operations, list) or not operations or not isinstance(operations[-1], dict):
        return None
    final = operations[-1]
    event = events.get(final.get("failure_id")) if isinstance(events, dict) else None
    if (
        final.get("outcome") == "failed"
        and isinstance(event, dict)
        and event.get("boundary") == "model"
        and event.get("operation") == "model_request"
        and event.get("operation_id") == final.get("operation_id")
        and isinstance(event.get("failure_kind"), str)
    ):
        return event["failure_kind"], event.get("http_status")
    return None


# The kind of a request the model server rejected, by the reason it recorded: the conversation outgrew the context
# window (the model's own limit), the request's output budget alone exceeds it, or a request parameter carries a value
# the deployment does not support (both setups the deployment cannot serve, evaluation-side).
LENGTH_REFUSAL_KINDS = {
    "context_length_exceeded": MODEL_CONTEXT_LIMIT,
    "invalid_output_token_parameter": INVALID_OUTPUT_BUDGET,
    "invalid_request_parameter": INVALID_REQUEST_PARAMETER,
}


def _length_refusal_kind(meta: Any) -> str | None:
    """The kind of a captured call rejected with HTTP 4xx on length (``LENGTH_REFUSAL_KINDS``), or None.

    The reason is the call's ``error_category`` or its ``local_response_reason``; the status is the call's own, its
    upstream's, or that of a failure event it reported. A 5xx among them is the server's fault, and an event of another
    kind than a rejected request (a transport failure) is its own cause, never the length's.
    """
    if not isinstance(meta, dict):
        return None
    kinds = {
        LENGTH_REFUSAL_KINDS[reason]
        for reason in (meta.get("error_category"), meta.get("local_response_reason"))
        if isinstance(reason, str) and reason in LENGTH_REFUSAL_KINDS
    }
    if len(kinds) != 1:
        return None
    report = meta.get("failure_report")
    events = report.get("events") if isinstance(report, dict) else None
    events = [event for event in events.values() if isinstance(event, dict)] if isinstance(events, dict) else []
    if any(event.get("failure_kind") not in (None, MODEL_REQUEST_ERROR) for event in events):
        return None
    statuses = [
        status
        for status in [
            meta.get("status_code"),
            meta.get("upstream_status_code"),
            *(event.get("http_status") for event in events),
        ]
        if type(status) is int and status >= 400
    ]
    return kinds.pop() if statuses and all(status <= 499 for status in statuses) else None


def _policy_call(row: Mapping[str, Any], call: Any) -> bool:
    """Whether a captured call is the evaluated policy's: its model is the row's policy server, or none is named."""
    policy = row.get("ng_policy_model_server")
    meta = call.get("response_metadata") if isinstance(call, dict) else None
    model = meta.get("model_ref") if isinstance(meta, dict) else None
    return not isinstance(policy, str) or (isinstance(model, dict) and model.get("name") == policy)


def _last_captured_call(calls: list) -> Any:
    """The call a rollout ended on, for an agent whose invocations reference none of its calls.

    The highest capture position (``call_index``, the durable append order). When every call recorded its start
    time, that call must also be the latest started; otherwise the end is ambiguous and there is none.
    """
    if not calls:
        return None
    last = calls[-1]
    starts = [call.get("started_at") if isinstance(call, dict) else None for call in calls]
    if all(type(start) in (int, float) for start in starts) and starts[-1] < max(starts):
        return None
    return last


def infrastructure_timeouts(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    *,
    classification: FailureClassificationConfig | None = None,
) -> set[Identity]:
    """Scored identities whose graded result is an agent timeout with a server-side fault in its window.

    A Harbor graded zero a run without a Store cannot reopen (``is_infrastructure_agent_timeout``): it stays in the
    results, and the trial gate counts it once, as ``AGENT_TIMEOUT_INFRASTRUCTURE``, an evaluation-side fault.
    """
    from nemo_gym.rollouts.rollout_recovery import is_infrastructure_agent_timeout

    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    return {
        _identity(row)
        for row in ((collected.results or []) if collected is not None else [])
        if _identity(row) in scored
        and is_infrastructure_agent_timeout(row, classification=decision_classification(decision, classification))
    }


def final_zero_trials(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    *,
    classification: FailureClassificationConfig | None = None,
) -> dict[Identity, str]:
    """Scored zeros with a saved policy decision or a failed final policy request, with their failure kinds.

    A consistent saved counted-zero decision retains its own registered failure kind. Otherwise, the last
    policy call of the row's one invocation ends in a failed model operation whose failure event the call
    recorded, and the row's reward is zero. Such a trial is settled as a failure under the benchmark protocol: it
    has no selectable response, and it is counted once, in the trial family. An agent that records no call of its own
    (it has no invocation, or its invocations reference none) ended on a failed request when its last captured call (``_last_captured_call``)
    failed that way; an earlier failed call is no end and stays unaccounted. The kind is keyed on the failure's HTTP
    status (``status_fault_kind``): a 5xx reply is the server's fault, and another model's 4xx rejection is
    evaluation-side. A policy whose own last call was rejected on length (``_length_refusal_kind``) ended there,
    whatever its event's kind: ``model_context_limit`` for the context window, ``invalid_output_budget`` for an output
    budget the window cannot hold. The policy's last call is its last captured one when its invocations reference
    none, since another model's capture may follow.
    """
    classification = decision_classification(decision, classification)
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    found: dict[Identity, str] = {}
    infrastructure = infrastructure_timeouts(decision, collected, classification=classification)
    for row in (collected.results or []) if collected is not None else []:
        reward = row.get("reward")
        if _identity(row) not in scored or type(reward) not in (int, float) or reward != 0:
            continue
        if _identity(row) in infrastructure:
            continue  # Counted once, as its agent timeout's server-side fault.
        if (kind := _policy_zero_kind(row)) is not None:
            found[_identity(row)] = kind
            continue
        ended = _ended_on_failed_request(row, classification=classification)
        if ended is not None:
            found[_identity(row)] = ended[0]
    return found


def _ended_on_failed_request(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> tuple[Any, Mapping[str, Any]] | None:
    """The failure kind of the request a row's rollout ended on, with that call, or None (``final_zero_trials``).

    The last policy call of the row's one invocation ended in a failed model operation whose failure event the call
    recorded, or, for an agent that recorded no call of its own, its last captured call failed that way; a policy whose
    own last call was rejected on length ended there, whatever its event's kind.
    """
    observed = row.get("ng_trial_observations")
    invocations = observed.get("invocations") if isinstance(observed, dict) else None
    # No invocation names a call: an agent that recorded no tool exchange has none at all.
    unreferenced = isinstance(invocations, list) and all(
        isinstance(item, dict) and item.get("model_calls") == [] for item in invocations
    )
    if unreferenced and isinstance(observed.get("model_calls"), list):
        own = _last_captured_call([call for call in observed["model_calls"] if _policy_call(row, call)])
        if own is not None and (refused := _length_refusal_kind(own.get("response_metadata"))):
            return refused, own
        last = _last_captured_call(observed["model_calls"])
        failed = _failed_request(last)
        if failed is not None:
            return status_fault_kind(
                failed[0], failed[1], policy=_policy_call(row, last), classification=classification
            ), last
        return None
    if not isinstance(invocations, list) or len(invocations) != 1 or not isinstance(invocations[0], dict):
        return None
    refs, calls = invocations[0].get("model_calls"), observed.get("model_calls")
    if not isinstance(refs, list) or not refs or not isinstance(refs[-1], dict) or not isinstance(calls, list):
        return None
    ref = refs[-1]
    last = [
        call
        for call in calls
        if isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and (
            call.get("model_call_id") == ref["model_call_id"]
            if ref.get("model_call_id")
            else ref.get("response_id") is not None
            and call["response_metadata"].get("response_id") == ref["response_id"]
            and call["response_metadata"].get("model_ref") == ref.get("model_ref")
        )
    ]
    meta = last[0]["response_metadata"] if len(last) == 1 else None
    if meta is not None and _policy_call(row, last[0]) and (refused := _length_refusal_kind(meta)):
        return refused, last[0]
    operations = meta.get("model_operations") if isinstance(meta, dict) else None
    report = meta.get("failure_report") if isinstance(meta, dict) else None
    if not isinstance(operations, list) or not operations or not isinstance(operations[-1], dict):
        return None
    final = operations[-1]
    events = report.get("events") if isinstance(report, dict) else None
    event = events.get(final.get("failure_id")) if isinstance(events, dict) else None
    if (
        final.get("outcome") == "failed"
        and isinstance(event, dict)
        and event.get("boundary") == "model"
        and event.get("operation") == "model_request"
        and event.get("operation_id") == final.get("operation_id")
        and isinstance(event.get("failure_kind"), str)
    ):
        return status_fault_kind(event["failure_kind"], event.get("http_status"), classification=classification), last[
            0
        ]
    return None


def _grading_facts(row: Mapping[str, Any]) -> Mapping[str, Any]:
    facts = row.get("_ng_collected_grading_facts")
    return facts if isinstance(facts, dict) else {}


def _reward(row: Mapping[str, Any]) -> float | None:
    reward = row.get("reward")
    return reward if type(reward) in (int, float) and math.isfinite(reward) else None


def _unparsed_verdict(facts: Mapping[str, Any]) -> bool:
    """Saved judge evaluations of which one recorded no verdict: a null label, or a judge or parse error label."""
    return facts.get("judge_evaluations") is True and (
        facts.get("judge_unparsed") is True or facts.get("judge_error") is True
    )


def _scored_unmasked(row: Mapping[str, Any]) -> bool:
    """A scored row: unmasked, with a reward and no failure kind."""
    return (
        row.get(FAILURE_CLASS) is None
        and row.get("failure_kind") is None
        and row.get("mask_sample") is not True
        and _reward(row) is not None
    )


def _unparsed_judge_verdict(row: Mapping[str, Any]) -> bool:
    """Whether a scored row's grading is ``judge_unparseable`` for a verdict the judge did not give: a saved judge
    evaluation recorded no verdict (``_unparsed_verdict``), or a pairwise game has none (``games_unjudged``). Its
    reward is no grade."""
    facts = _grading_facts(row)
    return _scored_unmasked(row) and (_unparsed_verdict(facts) or facts.get("games_unjudged") is True)


def _unknown_grading_outcome(row: Mapping[str, Any]) -> bool:
    """Whether a scored row's verifier record reports a judge grading operation with outcome ``unknown``: the grading
    did not resolve, so its reward is no grade. Read from the verifier invocations the row already saves."""
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, Mapping) else None
    return (
        isinstance(verifiers, list)
        and _scored_unmasked(row)
        and any(
            isinstance(verifier, Mapping)
            and isinstance(verifier.get("grading_operations"), list)
            and any(
                isinstance(op, Mapping) and op.get("kind") == "llm_judge" and op.get("outcome") == "unknown"
                for op in verifier["grading_operations"]
            )
            for verifier in verifiers
        )
    )


def _judge_unparseable(row: Mapping[str, Any]) -> bool:
    """Whether a scored row is ``judge_unparseable``: its judge gave no verdict (``_unparsed_judge_verdict``) or its
    verifier's grading did not resolve (``_unknown_grading_outcome``). Judges are stateless: it is re-judged."""
    return _unparsed_judge_verdict(row) or _unknown_grading_outcome(row)


def _judge_evaluations(facts: Mapping[str, Any], reward: float | None) -> bool:
    """A non-empty ``judge_evaluations`` list without a judge error; when every evaluation's verdict label parses
    (``[[A=B]]``/``[[A!=B]]`` or ``Judgement: yes/no``), the reward is 1.0 exactly when every verdict is equal/yes."""
    verdicts = facts.get("judge_verdicts")
    if facts.get("judge_evaluations") is not True or facts.get("judge_error") is True or _unparsed_verdict(facts):
        return False
    if isinstance(verdicts, list) and verdicts and all(type(verdict) is bool for verdict in verdicts):
        return reward == (1.0 if all(verdicts) else 0.0)
    return True


def _judge_record(row: Mapping[str, Any]) -> bool:
    """A saved judge verdict: a ``judge_response``, a non-empty ``judgments`` list with a ``judge_score``, or
    non-empty ``judge_evaluations`` whose parsed verdicts agree with the reward (``_judge_evaluations``)."""
    facts = _grading_facts(row)
    return bool(facts.get("judge_response") or (facts.get("judgments") and facts.get("judge_score"))) or (
        _judge_evaluations(facts, _reward(row))
    )


def _rubric_verdict(row: Mapping[str, Any]) -> bool:
    """A saved rubric judge's verdict: ``verdict`` YES or NO, a finite ``rubric_score`` and ``scoring_threshold``, no
    ``invalid_judge_response``, and the reward 1.0 exactly when the verdict is YES and the score reaches the threshold
    (0.0 otherwise). The saved answer and score are bound."""
    record = _grading_facts(row).get("rubric")
    if not isinstance(record, dict):
        return False
    score, threshold, verdict = record.get("rubric_score"), record.get("scoring_threshold"), record.get("verdict")
    return (
        type(score) in (int, float)
        and type(threshold) in (int, float)
        and type(verdict) is bool
        and record.get("invalid_judge_response") is False
        and _reward(row) == (1.0 if verdict and score >= threshold else 0.0)
        and _bound_answer(row)
    )


def _symbolic_match(row: Mapping[str, Any]) -> bool:
    """A saved symbolic answer check: a ``library_reward`` the row's reward equals, or one beside a judge verdict
    (the judge fallback)."""
    library = _grading_facts(row).get("library_reward")
    return library is not None and (_reward(row) == library or _judge_record(row))


def _metric_record(row: Mapping[str, Any]) -> bool:
    """Saved in-process metric scores: numeric ``sentence_*`` fields (``sentence_chrf``, ``sentence_spbleu``)."""
    return bool(_grading_facts(row).get("sentence_metrics"))


def _test_execution(row: Mapping[str, Any]) -> bool:
    """A saved test run: ``evaluation_completed`` true with an integer ``verifier_exit_code`` and no
    ``verifier_error``."""
    facts = _grading_facts(row)
    return (
        row.get("evaluation_completed") is True
        and facts.get("verifier_exit_code") is not None
        and facts.get("verifier_error") is False
    )


def _bound_answer(row: Mapping[str, Any]) -> bool:
    """Whether the collector bound the saved answer and score together (``_ng_collected_grading_binding``)."""
    binding = row.get("_ng_collected_grading_binding")
    return isinstance(binding, str) and len(binding) == 64


def _repository_test_run(row: Mapping[str, Any]) -> bool:
    """A saved repository test run of the answer's patch: ``evaluation_completed`` true, the run's ``resolved``
    verdict with the row's reward equal to it (1.0 resolved, 0.0 not), saved ``test_results``, no ``error``, no
    evaluation timeout, and a resolved verdict only for a patch that applied. The saved answer and score are bound."""
    run = _grading_facts(row).get("test_run")
    return (
        isinstance(run, dict)
        and row.get("evaluation_completed") is True
        and type(run.get("resolved")) is bool
        and _reward(row) == float(run["resolved"])
        and run.get("test_results") is True
        and run.get("error") is False
        and run.get("timed_out") is False
        and (not run["resolved"] or run.get("patch_applied") is True)
        and _bound_answer(row)
    )


def _substep_tests(row: Mapping[str, Any]) -> bool:
    """Saved per-sub-step test results: a non-empty ``step_results`` list of booleans, one environment result per
    step, ``num_steps_total`` and ``num_steps_passed`` equal to its length and its passes, ``problem_accuracy`` true
    exactly when every step passed, and the row's reward equal to it (1.0 or 0.0). The saved answer and score are
    bound."""
    steps = _grading_facts(row).get("substep_tests")
    return (
        isinstance(steps, dict)
        and type(steps.get("steps")) is int
        and steps["steps"] >= 1
        and steps.get("num_steps_total") == steps["steps"]
        and steps.get("num_steps_passed") == steps.get("passed")
        and steps.get("environment_results") == steps["steps"]
        and type(steps.get("problem_accuracy")) is bool
        and steps["problem_accuracy"] is (steps.get("passed") == steps["steps"])
        and _reward(row) == float(steps["problem_accuracy"])
        and _bound_answer(row)
    )


def _unit_test_results(row: Mapping[str, Any]) -> bool:
    """Saved unit test results of the answer's extracted code: at least one test, and the row's reward 1.0 exactly
    when every test passed (0.0 otherwise), or 0.0, the default penalty, for an answer whose reasoning format
    violated the rule (``reasoning_format_violation_rate`` 1.0). A configured non-zero penalty is not saved on the row,
    so such a row stays unobserved. The saved answer and score are bound."""
    tests = _grading_facts(row).get("unit_tests")
    if not (
        isinstance(tests, dict)
        and type(tests.get("tests")) is int
        and tests["tests"] >= 1
        and type(tests.get("all_passed")) is bool
        and tests.get("code") is True
    ):
        return False
    violation = tests.get("reasoning_format_violation_rate")
    if violation not in (None, 0.0, 1.0):
        return False
    expected = 0.0 if violation == 1.0 else float(tests["all_passed"])
    return _reward(row) == expected and _bound_answer(row)


def _node_set_f1(row: Mapping[str, Any]) -> bool:
    """A saved node-set F1 re-graded from the answer: the collector re-parses the saved answer's final
    ``Final Answer: [...]`` line, reproduces the saved predicted nodes and parse outcome, and scores them against the
    saved expected node list as the verifier does; the row's reward equals that re-graded F1 and the saved ``f1``.
    The saved answer and score are bound."""
    f1 = _grading_facts(row).get("node_f1")
    return (
        isinstance(f1, dict)
        and f1.get("parse_matches") is True
        and type(f1.get("regraded")) is float
        and _reward(row) == f1["regraded"] == f1.get("f1")
        and _bound_answer(row)
    )


def _session_reward_file(row: Mapping[str, Any]) -> bool:
    """The verifier's own reward file of the row's session, read by the collector beside the artifacts
    (``resources/<session_id>/verifier/reward.json``): the evaluation completed with no infrastructure error and the
    row's reward equals the file's ``reward``. Test counts and partial credit are metadata, never the grade. The saved
    answer and score are bound."""
    recorded = _grading_facts(row).get("session_reward")
    return (
        recorded is not None
        and row.get("evaluation_completed") is True
        and _reward(row) == recorded
        and _bound_answer(row)
    )


def _guarded_objective_score(row: Mapping[str, Any]) -> bool:
    """A guardrail-gated objective score the agent's own rubric saved (``response.metrics``): the row's reward equals
    the saved ``aa_headline``, and that headline follows from the saved counts: 0.0 when a guardrail was violated,
    else the objectives passed over the objectives (0.0 without objectives). A row without any counted assertion (no
    world or no assertions also scores 0.0) is no evidence and stays unobserved. The counts are the rubric's own, not
    a re-run of its assertions. The saved answer and score are bound."""
    score = _grading_facts(row).get("guarded_objectives")
    if not isinstance(score, dict) or any(
        type(score.get(key)) not in (int, float) or score[key] < 0
        for key in ("aa_headline", "objectives_passed", "objectives_total", "guardrails_violated", "guardrails_total")
    ):
        return False
    passed, total = score["objectives_passed"], score["objectives_total"]
    if (
        passed > total
        or score["guardrails_violated"] > score["guardrails_total"]
        or total + score["guardrails_total"] == 0
    ):
        return False
    expected = 0.0 if score["guardrails_violated"] > 0 else (passed / total if total else 0.0)
    return _reward(row) == score["aa_headline"] == expected and _bound_answer(row)


def _native_scorer_score(row: Mapping[str, Any]) -> bool:
    """A native scorer's saved score (``native_metrics``): the row's reward equals the saved ``weighted_score`` in
    [0, 1], the scorer saved its binary ``execution`` flag and its diagnostics. The score is the scorer's own, not a
    re-run of it. The saved answer and score are bound."""
    score = _grading_facts(row).get("native_score")
    if not isinstance(score, dict):
        return False
    weighted = score.get("weighted_score")
    return (
        type(weighted) in (int, float)
        and 0.0 <= weighted <= 1.0
        and score.get("execution") in (0.0, 1.0)
        and score.get("diagnostics") is True
        and _reward(row) == weighted
        and _bound_answer(row)
    )


def _environment_reward(row: Mapping[str, Any]) -> bool:
    """The environment's own reward record: ``result.reward_info.reward`` equal to the row's reward."""
    recorded = _grading_facts(row).get("environment_reward")
    return recorded is not None and _reward(row) == recorded


# The saved grading artifacts that record how a trial was graded when its verifier recorded no grading operation and
# no grading record of its own. Each reads the scalar facts the collector projects (``_ng_collected_grading_facts``);
# a bare reward is never one.
GRADING_ARTIFACTS = (
    ("judge_record", _judge_record),
    ("rubric_verdict", _rubric_verdict),
    ("symbolic_match", _symbolic_match),
    ("metric_record", _metric_record),
    ("test_execution", _test_execution),
    ("environment_reward", _environment_reward),
    ("repository_test_run", _repository_test_run),
    ("substep_tests", _substep_tests),
    ("unit_test_results", _unit_test_results),
    ("node_set_f1", _node_set_f1),
    ("session_reward_file", _session_reward_file),
    ("guarded_objective_score", _guarded_objective_score),
    ("native_scorer_score", _native_scorer_score),
)


def grading_artifact(row: Mapping[str, Any]) -> str | None:
    """The name of the first saved grading artifact (``GRADING_ARTIFACTS``) that records the row's grading."""
    return next((name for name, holds in GRADING_ARTIFACTS if holds(row)), None)


def _empty_answer(row: Mapping[str, Any]) -> bool:
    """Whether the saved answer holds no assistant message with text: none at all, or only ones the collector saw
    without non-whitespace text (``has_text`` false). A message whose content was not saved counts as an answer."""
    answer = row.get("response")
    output = answer.get("output") if isinstance(answer, dict) else None
    return isinstance(output, list) and not any(
        isinstance(item, dict) and item.get("role") == "assistant" and item.get("has_text") is not False
        for item in output
    )


def _model_failure(row: Mapping[str, Any]) -> bool:
    """The benchmark's rule scored the model's own failed answer without grading it: the verifier marked the answer
    invalid (``invalid_model_response``), or the model left it incomplete, or it is empty and scored zero."""
    return (
        _grading_facts(row).get("invalid_model_response") is True
        or _incomplete_answer(row)
        or (_empty_answer(row) and _reward(row) == 0)
    )


def _graded(row: Mapping[str, Any]) -> bool:
    """Whether the row's one verifier recorded grading operations for it that all completed with a valid verdict.

    An operation without a valid verdict does not resolve the trial; the gate's join decides whether it is a fault.
    """
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, dict) else None
    operations = (
        verifiers[0].get("grading_operations") if isinstance(verifiers, list) and len(verifiers) == 1 else None
    )
    return (
        isinstance(operations, list)
        and bool(operations)
        and all(isinstance(item, dict) and item.get("outcome") == "completed" for item in operations)
    )


def _bound_grading(row: Mapping[str, Any], collected: CollectedEvidence) -> bool:
    """Whether the row is ``_graded`` and every grading operation binds its saved answer and score, as the gate
    requires (``_ng_collected_grading_binding``)."""
    observed = row.get("ng_trial_observations")
    binding = row.get("_ng_collected_grading_binding")
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    return (
        _graded(row)
        and _bound_answer(row)
        and isinstance(servers, dict)
        and _verification_owner(row, servers) is not None
        and all(
            item.get("result_binding_sha256") == binding
            for item in observed["verifier_invocations"][0]["grading_operations"]
        )
    )


def _bound_artifact(row: Mapping[str, Any], collected: CollectedEvidence) -> bool:
    """Whether a saved grading artifact or the model's own failed answer resolves the row, as the gate requires: an
    artifact other than the model's failed answer must bind the saved answer and score to its verification owner."""
    if grading_artifact(row) is None:
        return _model_failure(row)
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    return _bound_answer(row) and isinstance(servers, dict) and _verification_owner(row, servers) is not None


def _verification_owner(row: Mapping[str, Any], servers: Mapping[str, Any]) -> str | None:
    """Resolve verification from saved routing bindings, including taskset results without ``agent_ref``.

    The environment owns a session's verification. An agent reference on a compatibility result must agree with
    a declared primary agent; without one, its explicit reference retains its existing attribution.
    An agent that grades its own rollouts names itself when neither configuration binds a resources server.
    """
    environment_name = row.get("_ng_environment_server")
    environment = servers.get(environment_name) if isinstance(environment_name, str) else None
    if (
        not isinstance(environment, dict)
        or environment.get("kind") != "environment_servers"
        or environment.get("bindings_malformed", False) is not False
        or row.get("_ng_result_type") != environment.get("implementation")
    ):
        return None
    reference = row.get("agent_ref")
    if reference is None:
        agent_name = environment.get("agent_server")
    elif isinstance(reference, dict):
        agent_name = reference.get("name")
        if environment.get("agent_server") is not None and agent_name != environment["agent_server"]:
            return None
    else:
        return None
    agent = servers.get(agent_name) if isinstance(agent_name, str) and agent_name else None
    if not isinstance(agent, dict) or agent.get("kind") != "responses_api_agents":
        return None
    owner = environment.get("resources_server")
    if owner is None:
        if reference is None:
            return None
        owner = agent.get("resources_server")
    if owner is None:
        return agent_name
    resource = servers.get(owner) if isinstance(owner, str) and owner else None
    return owner if isinstance(resource, dict) and resource.get("kind") == "resources_servers" else None


def _verifier_recorded(verifier: Any, row: Mapping[str, Any], resources_server: Any) -> bool:
    """Whether the row's verifier saved its own grading record for it, in place of a recorded grading operation.

    The shared verify path records it only for a verify execution that completed without asking any model. It must
    name the row's resources server (or the agent, for an agent that verifies its own rollouts), carry the saved score with a ``scored`` verdict and bind the saved response and
    score. A verifier that recorded a grading operation or owns a model call is judged by those, never by the record.
    """
    record = verifier.get("grading_record") if isinstance(verifier, dict) else None
    binding, reward = row.get("_ng_collected_grading_binding"), row.get("reward")
    return (
        isinstance(record, dict)
        and verifier.get("status") == "completed"
        and verifier.get("failure_kind") is None
        and verifier.get("grading_operations") is None
        and verifier.get("model_calls") == []
        and isinstance(resources_server, str)
        and bool(resources_server)
        and record.get("verifier") == resources_server
        and record.get("verdict") == "scored"
        and type(record.get("score")) in (int, float)
        and type(reward) in (int, float)
        and record["score"] == reward
        and isinstance(binding, str)
        and len(binding) == 64
        and record.get("result_binding_sha256") == binding
    )


def _recorded_by_verifier(row: Mapping[str, Any], collected: CollectedEvidence) -> bool:
    """Whether the row's one verifier invocation saved its own grading record for it (``_verifier_recorded``)."""
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    owner = _verification_owner(row, servers) if isinstance(servers, dict) else None
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, dict) else None
    return (
        owner is not None
        and isinstance(verifiers, list)
        and len(verifiers) == 1
        and _verifier_recorded(verifiers[0], row, owner)
    )


def _verifying_servers(collected: CollectedEvidence) -> str:
    """The resources servers the run binds, from its provenance, for an unobserved-trial reason."""
    provenance = (collected.quality or {}).get("provenance")
    servers = provenance.get("servers") if isinstance(provenance, dict) else None
    names = sorted(
        {
            str(fact["resources_server"])
            for fact in (servers.values() if isinstance(servers, dict) else ())
            if isinstance(fact, dict)
            and fact.get("kind") in {"responses_api_agents", "environment_servers"}
            and fact.get("resources_server")
        }
    )
    return ", ".join(names) if names else "a resources server the run provenance does not name"


def _health_view(decision: RecoveryDecision) -> RecoveryDecision:
    """``decision`` as the trial gate counts it: a scored row planned for a re-judge because its judge gave no verdict
    (``SCORED_UNPARSEABLE_REASON``) is a scored trial whose fault, ``judge_unparseable``, is evaluation-side. The
    recovery action is unchanged; only health reads the row as it stands until the re-judge settles it."""
    planned = {
        item.identity
        for item in decision.evidence
        if item.state == IdentityState.JUDGE_RECOVERABLE and item.reason == SCORED_UNPARSEABLE_REASON
    }
    if not planned:
        return decision
    evidence = [
        item.model_copy(update={"state": IdentityState.SCORED, "failure_kind": None, "reason": None})
        if item.identity in planned and item.state == IdentityState.JUDGE_RECOVERABLE
        else item
        for item in decision.evidence
    ]
    identities: dict[str, list[Identity]] = {}
    for item in evidence:
        identities.setdefault(item.state.value, []).append(item.identity)
    counts = {state: len(ids) for state, ids in identities.items()}
    return decision.model_copy(update={"evidence": evidence, "identities": identities, "counts": counts})


def trial_population(decision: RecoveryDecision) -> int:
    """The trial denominator: every expected trial, including repeats, failed and missing ones, less the rows an
    accepted partial stage omitted (no trial: reported separately, never in a numerator or denominator)."""
    return decision.expected - decision.counts.get(IdentityState.OMITTED, 0)


def trial_counts(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    unobserved_identities: set[Identity] | None = None,
) -> dict[str, Any] | None:
    """Acceptance's trial counts: observed healthy, unhealthy and unobserved, over the expected trials.

    A saved grading artifact or the model's own failed answer (``grading_artifact``, ``_model_failure``) resolves a
    trial too. ``unhealthy`` are the failures the trial gate counts (``trial_faults``, model and evaluation-side). When the
    gate's observation join covers every other trial, its observed gradings without a valid verdict are unhealthy
    too and nothing is unobserved. Otherwise ``unobserved`` are the other scored trials whose verifier recorded
    neither completed grading operations bound to the saved row nor its own grading record: nothing resolved their
    grading, so they are unverified, never healthy. ``observed_healthy`` are the rest; ``verifier_recorded`` of them
    are observed only through the verifier's grading record. None until the run is complete. A caller that passes
    ``unobserved_identities`` gets the identities counted as unobserved added to it.
    """
    decision = _health_view(decision)
    trials = trial_population(decision)
    if not decision.complete or trials <= 0 or collected is None:
        return None
    unhealthy = sum(trial_faults(decision, collected).values())
    invalid_grades: list[Identity] = []
    covered: list[Identity] = []
    if _observed_trial_operations(decision, collected, grading_failures=invalid_grades, verifier_recorded=covered):
        unhealthy += len(invalid_grades)
        return {
            "observed_healthy": trials - unhealthy,
            "unhealthy": unhealthy,
            "unobserved": 0,
            "verifier_recorded": len(covered),
        }
    final_zero = final_zero_trials(decision, collected)
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    rows = [
        row
        for row in collected.results or []
        if _identity(row) in scored and _identity(row) not in final_zero and not _scored_fault(row, collected)
    ]
    # The same per-trial evidence the gate validates: a grading bound to the saved row, the verifier's own record, or
    # a bound grading artifact. A mismatched or missing binding stays unobserved, never healthy.
    recorded = sum(1 for row in rows if not _bound_grading(row, collected) and _recorded_by_verifier(row, collected))
    artifacts = sum(
        1
        for row in rows
        if not _bound_grading(row, collected)
        and not _recorded_by_verifier(row, collected)
        and _bound_artifact(row, collected)
    )
    unbound = [row for row in rows if not _bound_grading(row, collected)]
    unobserved = len(unbound) - recorded - artifacts
    if unobserved_identities is not None:
        unobserved_identities.update(
            _identity(row)
            for row in unbound
            if not _recorded_by_verifier(row, collected) and not _bound_artifact(row, collected)
        )
    counts: dict[str, Any] = {
        "observed_healthy": trials - unhealthy - unobserved,
        "unhealthy": unhealthy,
        "unobserved": unobserved,
        "verifier_recorded": recorded,
    }
    if unobserved:
        counts["unobserved_reason"] = (
            f"no completed grading operation or grading record was recorded by the verifier "
            f"({_verifying_servers(collected)})"
        )
    return counts


def _predates_operation_observations(decision: RecoveryDecision, collected: CollectedEvidence | None) -> bool:
    """Whether every scored row was written by a Gym pin from before operation observations were recorded.

    Current pins write ``response_metadata.model_operations`` on every captured model call (a list, or null), and
    the collector reports whether the key was there (``model_operations_recorded``). Only an explicit False on every
    call of every scored row counts. A row without captured calls, or read by a collector that does not report the
    key, proves nothing, so the strict rule applies to it.
    """
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    rows = [row for row in (collected.results or []) if _identity(row) in scored] if collected is not None else []
    if not scored or len(rows) != len(scored):
        return False
    for row in rows:
        observed = row.get("ng_trial_observations")
        calls = observed.get("model_calls") if isinstance(observed, dict) else None
        if not isinstance(calls, list) or not calls:
            return False
        for call in calls:
            meta = call.get("response_metadata") if isinstance(call, dict) else None
            if not isinstance(meta, dict) or meta.get("model_operations_recorded") is not False:
                return False
    return True


# Classification is shared with boundary policies; the gate only counts final trial outcomes.
def status_fault_kind(
    kind: Any, http_status: Any, *, policy: bool = True, classification: FailureClassificationConfig | None = None
) -> Any:
    """The shared reporting label for observed HTTP evidence; saved events remain unchanged."""
    return reporting_failure_kind(kind, http_status, policy=policy, config=classification)


UNVERIFIED_NOTE = "unverified: this withholds a pass and is no health failure"


def _health_classification(
    classification: FailureClassificationConfig | None, *, policy_timeout_counts_as_model: bool
) -> FailureClassificationConfig:
    """Express the existing timeout switch as a shared classification override."""
    config = classification or FailureClassificationConfig()
    if policy_timeout_counts_as_model:
        return config
    return config.model_copy(
        update={"by_failure_kind": config.by_failure_kind | {MODEL_POLICY_RESPONSE_TIMEOUT: FailureClass.EVALUATION}}
    )


def _incomplete_answer(row: Mapping[str, Any]) -> bool:
    """Whether the row's saved answer is a response left incomplete, without an error. An incomplete status alone
    says nothing about why: an agent also marks an answer incomplete that holds no message."""
    response = row.get("response")
    return isinstance(response, dict) and response.get("status") == "incomplete" and response.get("error") is None


# The reasons that say an answer ran into its output cap: a chat ``length`` finish, or a responses-API incomplete
# reason (also recorded as the call's finish reason).
CUTOFF_REASONS = frozenset({"length", "max_output_tokens", "max_tokens"})


def _answer_cut_off(response: Mapping[str, Any]) -> bool:
    """Whether the saved answer itself says it was cut off: its ``incomplete_details.reason`` is ``max_output_tokens``
    or ``max_tokens``, or its output tokens reached its ``max_output_tokens``."""
    details = response.get("incomplete_details")
    if isinstance(details, dict) and details.get("reason") in CUTOFF_REASONS - {"length"}:
        return True
    usage, cap = response.get("usage"), response.get("max_output_tokens")
    tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    return type(cap) is int and cap > 0 and type(tokens) is int and tokens >= cap


def _cut_off(row: Mapping[str, Any]) -> bool:
    """Whether the saved answer carries an actual cutoff signal, not just an incomplete status.

    The answer's ``incomplete_details.reason`` is ``max_output_tokens`` or ``max_tokens``; or the captured call that
    returned it finished for ``length`` or one of those reasons; or its output tokens reached its ``max_output_tokens``.
    """
    response = row.get("response")
    if not isinstance(response, dict):
        return False
    if _answer_cut_off(response):
        return True
    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    return any(
        isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and call["response_metadata"].get("response_id") == response.get("id")
        and call["response_metadata"].get("finish_reason") in CUTOFF_REASONS
        for call in (calls if isinstance(calls, list) else [])
    )


def _recorded_grade(row: Mapping[str, Any], collected: CollectedEvidence | None) -> bool:
    """Whether the row's reward came from a recorded grade: completed grading operations, the verifier's own grading
    record, or a saved grading artifact (``GRADING_ARTIFACTS``). The model's own failed answer is not one."""
    return (
        _graded(row)
        # A shared record also covers an early zero returned for an empty answer; unlike an explicit grading
        # operation, its completion alone does not show that an incomplete, text-free answer was judged.
        or (
            collected is not None
            and not (_incomplete_answer(row) and _empty_answer(row))
            and _recorded_by_verifier(row, collected)
        )
        or grading_artifact(row) is not None
    )


def _answer_fault(row: Mapping[str, Any], collected: CollectedEvidence | None) -> str | None:
    """The model fault of a scored-zero incomplete answer.

    ``TRUNCATED_ANSWER`` when the answer carries a cutoff signal (``_cut_off``), whether the benchmark's rule or a
    recorded grade scored it zero: the cutoff is why it scored zero. Without a cutoff signal, a row whose reward came
    from a recorded grade (``_recorded_grade``) has none, as its score is its grade; otherwise ``EMPTY_ANSWER`` when it
    holds no assistant message with text (``_empty_answer``) and ``MALFORMED_ANSWER`` when it does.
    """
    if type(row.get("reward")) not in (int, float) or row["reward"] != 0 or not _incomplete_answer(row):
        return None
    if _cut_off(row):
        return TRUNCATED_ANSWER
    if _recorded_grade(row, collected):
        return None
    return EMPTY_ANSWER if _empty_answer(row) else MALFORMED_ANSWER


def _policy_zero_kind(row: Mapping[str, Any]) -> str | None:
    """A consistent applied zero keeps its own kind, without reinterpreting diagnostic requests."""
    kind = row.get("failure_kind")
    return kind if is_policy_counted_zero(row) and _known_kind(kind) else None


def _scored_kind(row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None) -> str | None:
    """The registered failure kind a scored result row records as its own outcome, keyed on its recorded HTTP status
    (``status_fault_kind``), or None. For example Harbor's graded ``agent_timeout`` is a scored main result.

    A persisted outcome decision governs metric admission, independently of fault attribution and the existing
    evaluation-health threshold. Counting a provider failure as zero does not make it a model fault or healthy.
    """
    if (kind := _policy_zero_kind(row)) is not None:
        return kind
    kind, status = _policy_evidence(row)
    return status_fault_kind(kind, status, classification=classification) if _known_kind(kind) else None


def _scored_fault(
    row: Mapping[str, Any],
    collected: CollectedEvidence | None,
    *,
    classification: FailureClassificationConfig | None = None,
) -> str | None:
    """A scored row's one counted fault: its own recorded kind (``_scored_kind``), else its answer's
    (``_answer_fault``), else ``judge_unparseable`` when its saved judge evaluation recorded no verdict
    (``_unparsed_judge_verdict``)."""
    fault = _scored_kind(row, classification=classification) or _answer_fault(row, collected)
    if fault is None and _judge_unparseable(row):
        return JUDGE_UNPARSEABLE
    return fault


def _trajectory_kinds(observed: Any) -> list[str]:
    """The kinds of trajectory evidence a trial lacks: its recorded gaps, and its agent invocations' own shape.

    Recorded gap codes other than ``resolution_unavailable`` (for example ``sandbox_lifecycle_timing_unavailable``,
    ``conversation_unavailable``, ``model_call_ownership_unavailable``), then ``invocation_unavailable`` or
    ``multiple_invocations``, ``in_session_unrecorded`` and ``invocation_<status>`` for an invocation that did not
    record a completed status. Empty when none of these is named.
    """
    if not isinstance(observed, dict):
        return ["trajectory_unavailable"]
    gaps = observed.get("gaps")
    kinds = sorted(
        {
            gap["code"]
            for gap in (gaps if isinstance(gaps, list) else [])
            if isinstance(gap, dict) and isinstance(gap.get("code"), str) and gap["code"] != "resolution_unavailable"
        }
    )
    invocations = observed.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        return kinds + ["invocation_unavailable"]
    if len(invocations) > 1:
        kinds.append("multiple_invocations")
    statuses = sorted(
        {
            str(invocation.get("status"))
            for invocation in invocations
            if isinstance(invocation, dict) and invocation.get("status") != "completed"
        }
    )
    if any(
        isinstance(invocation, dict) and type(invocation.get("in_session")) is not bool for invocation in invocations
    ):
        kinds.append("in_session_unrecorded")
    return kinds + [f"invocation_{status}" for status in statuses]


def trajectory_evidence(incomplete: Mapping[Identity, list[str]], trials: int) -> dict[str, Any]:
    """Trials whose trajectory evidence is incomplete, over the trials, with the count of each missing kind."""
    kinds = Counter(kind for found in incomplete.values() for kind in found)
    return {"incomplete": len(incomplete), "trials": trials, "kinds": dict(sorted(kinds.items()))}


def _trajectory_text(evidence: Mapping[str, Any]) -> str:
    """``trajectory evidence incomplete: N of M (kinds: kind n, ...)``, reported and never a condition."""
    kinds = ", ".join(f"{kind} {n}" for kind, n in evidence["kinds"].items())
    return (
        f"trajectory evidence incomplete: {evidence['incomplete']} of {evidence['trials']} (kinds: {kinds}); "
        "reported, not a coverage or health condition"
    )


def _observed_incomplete_answer(row: Mapping[str, Any]) -> bool:
    """Whether a counted incomplete answer (``_answer_fault``) is the model's own observed answer.

    The trial gate counts such a row as a model failure and reports it as a response failure, so its grading is not
    joined. Its answer must still be observed: the row recorded no failure, its one agent
    invocation ended completed or incomplete (an agent marks a cut-off answer incomplete) without an error, its last
    model call is exactly one captured call that answered in one completed operation, and that call's response is
    the saved answer, or, for an agent that saves its own answer envelope (its own response id), the envelope says it
    was cut off (``_answer_cut_off``) and that call itself finished on a cutoff (``CUTOFF_REASONS``): two independent
    signals that the model's own last reply was cut. A verifier that ran must have completed: a verifier error is never
    read as the model's fault.
    """
    observed = row.get("ng_trial_observations")
    if (
        not isinstance(observed, dict)
        or row.get(FAILURE_CLASS) is not None
        or row.get("failure_kind") is not None
        or row.get(FAILURE_ID) is not None
    ):
        return False
    invocations, verifiers, calls = (
        observed.get(key) for key in ("invocations", "verifier_invocations", "model_calls")
    )
    if (
        not isinstance(invocations, list)
        or len(invocations) != 1
        or not isinstance(invocations[0], dict)
        or not isinstance(calls, list)
        or not isinstance(verifiers, list)
        or len(verifiers) > 1
    ):
        return False
    invocation = invocations[0]
    refs = invocation.get("model_calls")
    if (
        invocation.get("status") not in ("completed", "incomplete")
        or invocation.get("error_type") is not None
        or invocation.get("stop_reason") is not None
        or invocation.get("parent_invocation_id") is not None
        or not isinstance(refs, list)
        or not refs
        or not isinstance(refs[-1], dict)
    ):
        return False
    if verifiers and (
        not isinstance(verifiers[0], dict)
        or verifiers[0].get("status") != "completed"
        or verifiers[0].get("failure_kind") is not None
    ):
        return False
    ref = refs[-1]
    last = [
        call
        for call in calls
        if isinstance(call, dict)
        and isinstance(call.get("response_metadata"), dict)
        and (
            call.get("model_call_id") == ref["model_call_id"]
            if ref.get("model_call_id")
            else ref.get("response_id") is not None
            and call["response_metadata"].get("response_id") == ref["response_id"]
            and call["response_metadata"].get("model_ref") == ref.get("model_ref")
        )
    ]
    if len(last) != 1:
        return False
    meta = last[0]["response_metadata"]
    operations = meta.get("model_operations")
    status = meta.get("status_code")
    return (
        type(status) is int
        and 200 <= status < 300
        and isinstance(operations, list)
        and len(operations) == 1
        and isinstance(operations[0], dict)
        and operations[0].get("outcome") == "completed"
        and operations[0].get("failure_id") is None
        and isinstance(meta.get("response_id"), str)
        and bool(meta["response_id"])
        and (
            meta["response_id"] == (row.get("response") or {}).get("id")
            or (meta.get("finish_reason") in CUTOFF_REASONS and _answer_cut_off(row.get("response") or {}))
        )
    )


def _row_recovered_attempts(row: Mapping[str, Any]) -> list[tuple[str, dict[str, dict]]]:
    """The row's recovered model attempts, by the completed operation that recovered them (``recovered_attempt_joins``).

    Each entry is an operation id and its failure events, keyed by event id, from one answered call of the row.
    """

    def nonempty(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    saved = row.get(FAILURE_EVENTS)
    saved = saved if isinstance(saved, dict) else {}
    found: list[tuple[str, dict[str, dict]]] = []
    seen: set[str] = set()
    for call in calls if isinstance(calls, list) else []:
        meta = call.get("response_metadata") if isinstance(call, dict) else None
        report = meta.get("failure_report") if isinstance(meta, dict) else None
        events = report.get("events") if isinstance(report, dict) else None
        operations = meta.get("model_operations") if isinstance(meta, dict) else None
        status = meta.get("status_code") if isinstance(meta, dict) else None
        # The health counter's own test of an answered call: a success status and no failure recorded on it.
        if (
            not isinstance(events, dict)
            or not events
            or not isinstance(operations, list)
            or type(status) is not int
            or not 200 <= status < 400
            or meta.get("error_category")
            or meta.get("response_status") in ("failed", "error", "cancelled")
        ):
            continue
        for operation in operations:
            op_id = operation.get("operation_id") if isinstance(operation, dict) else None
            if not nonempty(op_id) or operation.get("outcome") != "completed" or op_id in seen:
                continue
            owned = {
                key: event
                for key, event in events.items()
                if isinstance(event, dict) and event.get("operation_id") == op_id
            }
            tries, transport = operation.get("http_attempts"), operation.get("transport_failures")
            answer = operation.get("final_attempt_id")
            ids = {event.get("attempt_id") for event in owned.values()}
            if (
                not owned
                or type(tries) is not int
                or type(transport) is not int
                or tries < 1
                or transport < 0
                or len(owned) != tries - 1 + transport
                or sum(event.get("http_status") is None for event in owned.values()) != transport
                or not nonempty(answer)
                or answer in ids
                or len(ids) != len(owned)
                or operation.get("failure_id") is not None
                or any(
                    key != event.get("event_id")
                    or not nonempty(event.get("attempt_id"))
                    or event.get("boundary") != "model"
                    or event.get("operation") != "model_request"
                    or event.get("failure_kind") != MODEL_REQUEST_ERROR
                    or event.get("cause_id") is not None
                    or not (
                        event.get("http_status") is None
                        or (type(event["http_status"]) is int and 400 <= event["http_status"] <= 599)
                    )
                    or (
                        key in saved
                        and (
                            not isinstance(saved[key], dict)
                            or any(saved[key].get(name) != value for name, value in event.items())
                        )
                    )
                    for key, event in owned.items()
                )
            ):
                continue
            seen.add(op_id)
            found.append((op_id, owned))
    return found


def _unaccounted_failures(row: Mapping[str, Any], recovered: Mapping[str, Any]) -> bool:
    """Whether the row recorded a model failure that no recovery accounts for.

    Every model failure event the row saved or its captured calls reported must be a recovered attempt (``recovered``: the
    events the coverage join joined, with the row's own ``_row_recovered_attempts``), a saved copy must agree with its
    call's, and every failed model operation must be one whose failure such a recovery accounts for. A trial that
    ended on its failed request is counted separately (``final_zero_trials``) and is never asked this.
    """
    joined = dict(recovered)
    for _, owned in _row_recovered_attempts(row):
        joined.update(owned)
    saved = row.get(FAILURE_EVENTS, {})
    observed = row.get("ng_trial_observations")
    calls = observed.get("model_calls") if isinstance(observed, dict) else None
    if not isinstance(saved, dict) or (calls is not None and not isinstance(calls, list)):
        return True
    for event_id, event in saved.items():
        if (
            isinstance(event, dict)
            and event.get("boundary") in FailureBoundary
            and event["boundary"] != FailureBoundary.MODEL
        ):
            # The enclosing graded result completed. Non-model observations cannot join model attempts and
            # do not contradict that outcome; a selected final failure is counted before this coverage check.
            try:
                observed = FailureEvent.model_validate(event)
            except ValidationError:
                return True
            if observed.event_id != event_id:
                return True
            continue
        if (
            event_id not in joined
            or not isinstance(event, dict)
            or any(event.get(key) != value for key, value in joined[event_id].items())
        ):
            return True
    for call in calls or []:
        meta = call.get("response_metadata") if isinstance(call, dict) else None
        if not isinstance(meta, dict):
            continue
        report = meta.get("failure_report")
        events = report.get("events") if isinstance(report, dict) else None
        if isinstance(events, dict) and any(event_id not in joined for event_id in events):
            return True
        for operation in meta.get("model_operations") or []:
            if isinstance(operation, dict) and operation.get("outcome") == "failed":
                if operation.get("failure_id") not in joined:
                    return True
    return False


def recovered_attempt_joins(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, int]:
    """Failed model attempts each joined to the completed operation that recovered them, in a scored trial's own call.

    Counted like the run's observed upstream recovery (``observed_upstream_recovery``): a model-request failure event
    whose operation completed in an answered call. Here the event must be in the failure report of the same call
    whose one completed operation it names, as a failed attempt other than the answer, and the operation's attempts
    must account for every failed attempt of it (``http_attempts - 1 + transport_failures``, the HTTP-less ones being
    the transport failures). A copy the row saved must agree, and no event, attempt or operation joins twice. The
    join holds for any model the trial called (its policy, a judge or a simulated user). It says nothing about how
    the trial was graded: whether the final answer is bound to its grading is the coverage join's question
    (``_observed_trial_operations``).
    """
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    events_seen: set[str] = set()
    attempts_seen: set[str] = set()
    operations_seen: set[str] = set()
    joined = {"attempts": 0, "operations": 0}
    for row in (collected.results or []) if collected is not None else []:
        if _identity(row) not in scored:
            continue
        for op_id, owned in _row_recovered_attempts(row):
            ids = {event["attempt_id"] for event in owned.values()}
            if op_id in operations_seen or events_seen & set(owned) or attempts_seen & ids:
                continue
            events_seen.update(owned)
            attempts_seen.update(ids)
            operations_seen.add(op_id)
            joined["attempts"] += len(owned)
            joined["operations"] += 1
    return joined


def _recovered_note(joined: Mapping[str, int]) -> str:
    """``N failed model attempt(s) were retried and completed in the same operation (...)``, or empty."""
    if not joined["attempts"]:
        return ""
    return (
        f"{joined['attempts']} failed model attempt(s) in {joined['operations']} operation(s) were retried and "
        "completed in the same operation (recovered, no trial failure); "
    )


def _row_fault_class(
    row: Mapping[str, Any], fault: str, classification: FailureClassificationConfig | None
) -> FailureClass:
    if (applied := _applied_class(row)) is not None:
        return applied
    kind, status = _policy_evidence(row)
    zero = _policy_zero_kind(row)
    kind = zero or kind
    # Only selected final evidence may provide stage/status; unrelated recovered events cannot classify a verdict.
    if zero == fault or (
        _known_kind(kind) and status_fault_kind(kind, status, classification=classification) == fault
    ):
        return classify_failure(failure_kind=kind, stage=_policy_stage(row), http_status=status, config=classification)
    ended = _ended_on_failed_request(row, classification=classification)
    if ended is not None and ended[0] == fault:
        meta = ended[1].get("response_metadata") or {}
        report, operations = meta.get("failure_report"), meta.get("model_operations")
        events = report.get("events") if isinstance(report, Mapping) else None
        last = operations[-1] if isinstance(operations, list) and operations else None
        event = (
            events.get(last.get("failure_id")) if isinstance(events, Mapping) and isinstance(last, Mapping) else None
        )
        if isinstance(event, Mapping):
            stage = event.get("stage")
            # A length refusal refines a generic or missing event kind. Otherwise keep the
            # observed kind so stage rules still precede HTTP-derived reporting labels.
            return classify_failure(
                failure_kind=fault if _length_refusal_kind(meta) == fault else event.get("failure_kind"),
                stage=cast(FailureStage, stage) if stage in get_args(FailureStage) else None,
                http_status=event.get("http_status"),
                policy=_policy_call(row, ended[1]),
                config=classification,
            )
    return classify_failure(failure_kind=fault, config=classification)


def decision_classification(
    decision: RecoveryDecision, explicit: FailureClassificationConfig | None = None
) -> FailureClassificationConfig:
    """The resolved recovery policy snapshot supplies the same mapping to all health readers."""
    if explicit is not None:
        return explicit
    saved = decision.outcome_policy or {}
    return FailureClassificationConfig.model_validate(saved.get("classification", {}))


def trial_faults(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None,
    *,
    attributions: Counter | None = None,
    classification: FailureClassificationConfig | None = None,
) -> Counter:
    """Count final trial failures once; classify selected outcomes, never recovered observations."""
    classification = decision_classification(decision, classification)
    decision = _health_view(decision)
    final_zero = final_zero_trials(decision, collected, classification=classification)
    infrastructure = infrastructure_timeouts(decision, collected, classification=classification)
    kinds = Counter()
    failures = {_identity(row): row for row in (collected.failures if collected else [])}
    for evidence in decision.evidence:
        if evidence.state != IdentityState.MODEL_FINAL:
            continue
        kind = status_fault_kind(evidence.failure_kind, evidence.http_status, classification=classification)
        kinds[kind] += 1
        if attributions is not None:
            applied = _applied_class(failures.get(evidence.identity, {}))
            selected = (
                applied
                if applied is not None
                else classify_failure(
                    failure_kind=evidence.failure_kind,
                    stage=evidence.stage,
                    http_status=evidence.http_status,
                    config=classification,
                )
            )
            attributions[kind, selected] += 1
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    for row in collected.results or [] if collected is not None else []:
        identity = _identity(row)
        if identity not in scored:
            continue
        fault = (
            AGENT_TIMEOUT_INFRASTRUCTURE
            if identity in infrastructure
            else final_zero.get(identity) or _scored_fault(row, collected, classification=classification)
        )
        if fault is None:
            continue
        kinds[fault] += 1
        if attributions is not None:
            attributions[fault, _row_fault_class(row, fault, classification)] += 1
    return +kinds


def fault_report(
    kinds: Counter,
    trials: int,
    *,
    policy_timeout_counts_as_model: bool = True,
    attributions: Counter | None = None,
    classification: FailureClassificationConfig | None = None,
) -> dict[str, Any]:
    """Report shared classes separately; server is only a subcategory of evaluation."""
    config = _health_classification(classification, policy_timeout_counts_as_model=policy_timeout_counts_as_model)

    def side(found: Counter) -> dict[str, Any]:
        count = sum(found.values())
        return {"count": count, "fraction": count / trials, "kinds": dict(sorted(found.items(), key=str))}

    remaining = kinds.copy()
    classes = {failure_class: Counter() for failure_class in FailureClass}
    for (kind, attribution), count in (attributions or {}).items():
        remaining[kind] -= count
        classes[FailureClass(attribution)][kind] += count
    for kind, count in remaining.items():
        if count:
            classes[classify_failure(failure_kind=kind, config=config)][kind] += count
    evaluation = classes[FailureClass.EVALUATION]
    server_kinds = {kind: n for kind, n in evaluation.items() if is_server_failure(kind, config=config)}
    report = {
        "trials": trials,
        "evaluation": side(evaluation) | {"server": sum(server_kinds.values()), "server_kinds": server_kinds},
        "model": side(classes[FailureClass.MODEL]),
        "unknown": side(classes[FailureClass.UNKNOWN]),
    }
    return report if policy_timeout_counts_as_model else report | {"policy_timeout_counts_as_model": False}


def _fault_text(report: dict[str, Any], limit: float) -> str:
    """``evaluation-side faults: N of M (x%) is [not] below L (server S: kind n; kind n); model faults: ...``."""
    trials, evaluation, model = report["trials"], report["evaluation"], report["model"]
    server = [f"{kind} {n}" for kind, n in evaluation["kinds"].items() if kind in evaluation["server_kinds"]]
    other = [f"{kind} {n}" for kind, n in evaluation["kinds"].items() if kind not in evaluation["server_kinds"]]
    named = ([f"server {evaluation['server']}: {', '.join(server)}"] if server else []) + other
    below = "is below" if evaluation["fraction"] < limit else "is not below"
    return (
        f"evaluation-side faults: {evaluation['count']} of {trials} ({evaluation['fraction']:.2%}) {below} "
        f"{_percent(limit)}" + (f" ({'; '.join(named)})" if named else "") + f"; model faults: {model['count']} of "
        f"{trials} ({model['fraction']:.2%})"
        + (f" ({', '.join(f'{kind} {n}' for kind, n in model['kinds'].items())})" if model["kinds"] else "")
    )


def trial_health(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None = None,
    *,
    limit: float = GATE_FAILURE_LIMIT,
    policy_timeout_counts_as_model: bool = True,
    classification: FailureClassificationConfig | None = None,
) -> tuple[bool | None, str]:
    """Evaluation health from the trial gate (``trial_verdict``): passed, failed or None, with its reason."""
    passed, reason, _ = trial_verdict(
        decision,
        collected,
        limit=limit,
        policy_timeout_counts_as_model=policy_timeout_counts_as_model,
        classification=classification,
    )
    return passed, reason


def trial_verdict(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None = None,
    *,
    limit: float = GATE_FAILURE_LIMIT,
    policy_timeout_counts_as_model: bool = True,
    classification: FailureClassificationConfig | None = None,
) -> tuple[bool | None, str, dict[str, Any] | None]:
    """Evaluation health: evaluation-side trial faults over the expected trials, including repeats.

    The counted failures are ``trial_faults``; a model fault (``classify_failure``) is reported with its kind and never
    fails. Only the evaluation-side share is judged against the limit: at or above it the gate fails. Below it, it
    passes only with an exact count, when actual per-identity operation outcomes cover the observed model/tool and
    grading path of every trial that is not a recorded failure (``_observed_trial_operations``); an observed grading
    without a valid verdict is an evaluation-side fault. Otherwise the run is UNVERIFIED: a judge, verifier or harness
    error scored as a zero in the main output carries no failure record and cannot be ruled out, so the known count is
    a lower bound. That withholds a pass, never fails health. The denominator is the full materialized inventory less
    accepted omissions. A run that is not complete has unaccounted trials, so the gate is not evaluated.

    The report is ``status`` (PASS, FAIL or UNVERIFIED), the evaluation-side rate, ``exact`` and the fault counts.
    """
    decision = _health_view(decision)
    # A row an accepted partial stage omitted is no trial: it is neither in the numerator nor the denominator.
    trials = trial_population(decision)
    if not decision.complete or trials <= 0:
        return None, f"trial: not evaluated, the run is not complete ({dict(decision.counts)})", None
    classification = _health_classification(
        decision_classification(decision, classification),
        policy_timeout_counts_as_model=policy_timeout_counts_as_model,
    )
    attributions = Counter()
    kinds = trial_faults(decision, collected, attributions=attributions, classification=classification)
    report = fault_report(
        kinds,
        trials,
        policy_timeout_counts_as_model=policy_timeout_counts_as_model,
        attributions=attributions,
        classification=classification,
    )

    def verdict(passed: bool | None, reason: str, *, exact: bool = False):
        state = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[passed]
        rate = report["evaluation"]["fraction"]
        return passed, reason, {"status": state, "evaluation_failure_rate": rate, "exact": exact} | report

    if report["evaluation"]["fraction"] >= limit:
        return verdict(False, f"trial: {_fault_text(report, limit)}")
    if report["unknown"]["count"]:
        return verdict(
            None,
            f"trial: UNVERIFIED, {report['unknown']['count']} trial failure(s) have unknown classification; {UNVERIFIED_NOTE}",
        )
    recovery = (collected.quality or {}).get("observed_upstream_recovery") if collected else None
    if recovery is not None and (
        not isinstance(recovery, dict)
        or any(_count(recovery.get(key)) is None for key in ("recovered_attempt_errors", "recovered_operations"))
    ):
        return verdict(None, "trial: recovered attempt/operation counters are malformed or incomplete")
    if _predates_operation_observations(decision, collected):
        # Those pins never recorded the operations the strict rule joins, so neither a trial nor a recovered attempt
        # can be joined to one; the known-failure limit above still holds.
        unjoined = recovery["recovered_attempt_errors"] if isinstance(recovery, dict) else 0
        return verdict(
            True,
            f"trial: {_fault_text(report, limit)}; exact failure count unavailable (earlier pin): the run's rows "
            "predate operation observations, so the complete run is certified on coverage"
            + (f"; {unjoined} recovered attempt error(s) cannot be joined to an operation" if unjoined else ""),
        )
    judge_recovery: dict[str, int] = {}
    policy_recovery: dict[str, int] = {}
    invalid_grades: list[str] = []
    recorded: list[Identity] = []
    incomplete: dict[Identity, list[str]] = {}
    agent_failed: list[Identity] = []
    covered = _observed_trial_operations(
        decision,
        collected,
        classification=classification,
        judge_recovery=judge_recovery,
        grading_failures=invalid_grades,
        policy_recovery=policy_recovery,
        verifier_recorded=recorded,
        trajectory_incomplete=incomplete,
        agent_failed_graded=agent_failed,
    )
    # A recovered attempt is no trial failure only when it joins the completed operation that recovered it, in a
    # scored trial's own call. That join is per attempt and does not wait on the coverage join above: a run whose
    # coverage is incomplete is judged on its coverage below, never on its recovered attempts. One the run counted
    # but no trial's call joins cannot be ruled out as a trial failure: the run is unverified. When the coverage join
    # holds, its own count stands too: it also joins a failed grading operation to the completed one after it.
    recovered_policy = policy_recovery.get("policy_recovered_attempts", 0)
    joined = recovered_attempt_joins(decision, collected)
    joined = {
        "attempts": max(
            joined["attempts"], judge_recovery.get("observed_grading_attempt_errors", 0) + recovered_policy
        ),
        "operations": max(
            joined["operations"],
            judge_recovery.get("observed_grading_failed_operations", 0)
            + policy_recovery.get("policy_recovered_operations", 0),
        ),
    }
    if isinstance(recovery, dict) and (
        recovery["recovered_attempt_errors"] > joined["attempts"]
        or recovery["recovered_operations"] > joined["operations"]
    ):
        return verdict(
            None,
            f"trial: UNVERIFIED, recovered attempts not joined: the run counted {recovery['recovered_attempt_errors']} "
            f"recovered model attempt error(s) in {recovery['recovered_operations']} operation(s), and "
            f"{joined['attempts']} in {joined['operations']} join the completed operation of a scored trial's own "
            "call; the others cannot be ruled out as trial failures; known so far: "
            f"{_fault_text(report, limit)}; {UNVERIFIED_NOTE}",
        )
    recovered = _recovered_note(joined) if isinstance(recovery, dict) else ""
    if covered:
        if invalid_grades:
            report = fault_report(
                kinds + Counter({GRADING_WITHOUT_VERDICT: len(invalid_grades)}),
                trials,
                policy_timeout_counts_as_model=policy_timeout_counts_as_model,
                attributions=attributions,
                classification=classification,
            )
        retried = (
            f"{recovered_policy} failed policy attempt(s) were retried and completed in the same operation "
            "(recovered, no trial failure); "
            if recovered_policy
            else (recovered if incomplete else "")
        )
        scored = decision.counts.get(IdentityState.SCORED, 0) - len(final_zero_trials(decision, collected))
        grading = (
            f"grading ({len(recorded)} verifier-recorded, {scored - len(recorded)} observed through a recorded grading "
            "operation)"
            if recorded
            else "bound grading operations"
        )
        # A reported fact, never a condition: trials whose agent run failed and whose answer was still graded.
        facts = {"agent_invocation_failed_grading_recorded": len(agent_failed)} if agent_failed else {}
        fact_text = f"; agent invocation failed, grading recorded: {len(agent_failed)}" if agent_failed else ""
        # A reported fact too: trials whose verifier recorded no grading operation and no grading record, graded
        # through a saved grading artifact (``GRADING_ARTIFACTS``), by artifact: their grading is derived from the
        # saved verifier output, not observed as it ran.
        verifier_recorded = set(recorded)
        derived = Counter(
            name
            for row in collected.results or []
            if not _graded(row) and _identity(row) not in verifier_recorded and (name := grading_artifact(row))
        )
        if derived:
            facts["grading_derived_from_saved_output"] = dict(sorted(derived.items()))
            fact_text += "; grading derived from saved verifier output: " + ", ".join(
                f"{name} {count}" for name, count in sorted(derived.items())
            )
        if incomplete:
            # Trajectory evidence is observability, never coverage: the grading bound to every answer decides.
            evidence = trajectory_evidence(incomplete, trials)
            passed, reason, result = verdict(
                report["evaluation"]["fraction"] < limit,
                f"trial: {_fault_text(report, limit)}; exact count: {len(invalid_grades)} observed returned grading "
                f"result(s) lack a valid verdict; {retried}recorded grading bound to the saved answer ({grading}) covers "
                f"every identity and repeat that is not a recorded failure; {_trajectory_text(evidence)}{fact_text}",
                exact=True,
            )
            return passed, reason, result | {"trajectory_evidence": evidence} | ({"facts": facts} if facts else {})
        passed, reason, result = verdict(
            report["evaluation"]["fraction"] < limit,
            f"trial: {_fault_text(report, limit)}; exact count: {len(invalid_grades)} observed returned grading "
            f"result(s) lack a valid verdict; {retried}actual model/tool and {grading} cover every identity and "
            f"repeat that is not a recorded failure{fact_text}",
            exact=True,
        )
        return passed, reason, result | ({"facts": facts} if facts else {})
    observed = trial_counts(decision, collected)
    if observed is not None and observed["unobserved"]:
        return verdict(
            None,
            f"trial: UNOBSERVED {observed['unobserved']} of {trials} expected trials: "
            f"{observed['unobserved_reason']}, so the run is unverified, not healthy; {recovered}observed healthy "
            f"{observed['observed_healthy']}"
            + (f" ({observed['verifier_recorded']} verifier-recorded)" if observed["verifier_recorded"] else "")
            + f", unhealthy {observed['unhealthy']}; known so far: {_fault_text(report, limit)}; {UNVERIFIED_NOTE}",
        )
    return verdict(
        None,
        f"trial: UNVERIFIED, known failures are a lower bound: {_fault_text(report, limit)}; {recovered}failures "
        f"recorded only as zero rewards cannot be ruled out; {UNVERIFIED_NOTE}",
    )


# Rollout states counted as trial failures: every failed, masked or unowned outcome, settled or not.
_TRIAL_FAILURE_STATES = (
    IdentityState.MODEL_FINAL,
    IdentityState.JUDGE_RECOVERABLE,
    IdentityState.RETRYABLE,
    IdentityState.EXHAUSTED,
    IdentityState.TERMINAL,
    IdentityState.MASKED,
    IdentityState.UNKNOWN,
)
RESPONSE_FAILURE_CAUSES = ("truncated", "empty", "malformed")
TRIAL_FAILURE_CAUSES = ("request", "judge", "sandbox", "harness", "timeout", "allocation", "other")


def trial_failure_cause(kind: Any) -> str:
    """A trial failure's cause from its failure kind: request, judge, sandbox, harness, timeout, allocation, or other.

    ``allocation``: the rollout was in flight at repeated ends of its allocation (``ROLLOUT_ALLOCATION_ENDED``).
    """
    if not isinstance(kind, str):
        return "other"
    if kind == ROLLOUT_ALLOCATION_ENDED:
        return "allocation"
    if "timeout" in kind:
        return "timeout"
    if kind in (JUDGE_FAILED, JUDGE_UNPARSEABLE, VERIFIER_ERROR, GRADING_WITHOUT_VERDICT):
        return "judge"
    if kind.startswith(("provider_", "session_")):
        return "sandbox"
    if kind.startswith(("model_", "transport_")):
        return "request"
    if kind.startswith(("agent_", "environment_", "harness_")) or kind in (
        HARBOR_FAILED,
        GRADING_WITHOUT_VERDICT,
        INFRASTRUCTURE_ERROR,
        SERVER_START_FAILED,
    ):
        return "harness"
    return "other"


def failure_facts(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, Any] | None:
    """Response and trial failures by cause, each with a count and a rate over the expected rollouts.

    Reported facts, never a verdict: no limit applies to them here. The denominator is the trial gate's
    (``trial_population``): every expected rollout, failed and missing ones included; rows an accepted partial stage
    omitted are no rollout of the run and are reported separately as ``omitted``.

    A response failure is a scored rollout without a recorded grade (``_recorded_grade``) whose saved answer is
    truncated (incomplete with a cutoff signal, ``_cut_off``), empty (no assistant text, ``_empty_answer``) or malformed (no response
    with an output list, an answer the verifier marked invalid, or one left incomplete without a cutoff signal). A trial failure is a rollout in a failed, masked or
    unowned state (``_TRIAL_FAILURE_STATES``), a scored rollout that ended on a failed policy request, a scored
    rollout that records its own failure kind or whose judge recorded no verdict, or an observed grading without a
    valid verdict, by the cause its failure kind names (``trial_failure_cause``).
    """
    decision = _health_view(decision)
    population = trial_population(decision)
    if population <= 0:
        return None
    classification = decision_classification(decision)
    final_zero = final_zero_trials(decision, collected, classification=classification)
    infrastructure = infrastructure_timeouts(decision, collected, classification=classification)
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    responses: Counter = Counter()
    for row in (collected.results or []) if collected is not None else []:
        if _identity(row) not in scored or _identity(row) in final_zero or _identity(row) in infrastructure:
            continue
        # A cut-off answer scored zero is truncated, also when a recorded grade scored it.
        if _answer_fault(row, collected) == TRUNCATED_ANSWER:
            responses["truncated"] += 1
            continue
        # Otherwise a row whose reward came from a recorded grade is no response failure: its score is its grade.
        if _recorded_grade(row, collected):
            continue
        answer = row.get("response")
        output = answer.get("output") if isinstance(answer, dict) else None
        if not isinstance(output, list):
            responses["malformed"] += 1
        elif _incomplete_answer(row) and _cut_off(row):
            responses["truncated"] += 1
        elif _empty_answer(row):
            responses["empty"] += 1
        elif _grading_facts(row).get("invalid_model_response") is True or _incomplete_answer(row):
            responses["malformed"] += 1
    invalid_grades: list[Identity] = []
    _observed_trial_operations(decision, collected, grading_failures=invalid_grades)
    causes = Counter(
        "judge" if e.state == IdentityState.JUDGE_RECOVERABLE else trial_failure_cause(e.failure_kind)
        for e in decision.evidence
        if e.state in _TRIAL_FAILURE_STATES
    )
    causes.update(trial_failure_cause(kind) for kind in final_zero.values())
    # A scored row's own failure kind or an unparsed judge verdict (``_scored_fault``); answer faults are response
    # failures above.
    causes.update(
        trial_failure_cause(fault)
        for row in ((collected.results or []) if collected is not None else [])
        if _identity(row) in scored
        and _identity(row) not in final_zero
        and _identity(row) not in infrastructure
        and (fault := _scored_fault(row, collected)) is not None
        and fault != _answer_fault(row, collected)
    )
    causes[trial_failure_cause(AGENT_TIMEOUT_INFRASTRUCTURE)] += len(infrastructure)
    causes["judge"] += len(invalid_grades)

    def family(found: Counter, names: tuple[str, ...]) -> dict[str, Any]:
        count = sum(found.values())
        return {
            "count": count,
            "rate": count / population,
            "by_cause": {name: {"count": found[name], "rate": found[name] / population} for name in names},
        }

    return {
        "expected": population,
        "omitted": decision.counts.get(IdentityState.OMITTED, 0),
        "response_failures": family(responses, RESPONSE_FAILURE_CAUSES),
        "trial_failures": family(causes, TRIAL_FAILURE_CAUSES),
        "metric_coverage": metric_coverage(decision, collected),
        "output_budget_refits": output_budget_refits(collected),
        "agent_adaptations": agent_adaptations(decision, collected),
    }


def output_budget_refits(collected: CollectedEvidence | None) -> dict[str, int]:
    """How many model calls in the saved results were sent again with an output budget lowered to fit the context
    window (``output_budget_refit`` in a call's response metadata), and in how many result rows.

    A count, never a failure: the call generated within the window it was served at. A reader flags a cell whose
    requests were budget-capped from it.
    """
    count = rows = 0
    for row in (collected.results or []) if collected is not None else []:
        observed = row.get("ng_trial_observations")
        calls = observed.get("model_calls") if isinstance(observed, dict) else None
        refits = sum(
            isinstance(call, dict)
            and isinstance(call.get("response_metadata"), dict)
            and isinstance(call["response_metadata"].get("output_budget_refit"), dict)
            for call in calls or []
        )
        count += refits
        rows += refits > 0
    return {"count": count, "rows": rows}


def agent_adaptations(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, dict[str, int]]:
    """The scored rollouts' own ``agent_adaptations`` counts, summed by name: ``{name: {"count": total, "rows": N}}``,
    ``rows`` being the rollouts with a nonzero count. An agent records what it adapted on its own before a result,
    such as a request the endpoint rejected for its input size; a reported fact, never a failure."""
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    latest: dict[Identity, Mapping[str, Any]] = {}
    for row in (collected.results or []) if collected is not None else []:
        if (identity := _identity(row)) in scored and isinstance(row.get("agent_adaptations"), Mapping):
            latest[identity] = row["agent_adaptations"]
    totals: dict[str, dict[str, int]] = {}
    for adaptations in latest.values():
        for name, value in adaptations.items():
            if isinstance(name, str) and type(value) is int and value >= 0:
                total = totals.setdefault(name, {"count": 0, "rows": 0})
                total["count"] += value
                total["rows"] += value > 0
    return totals


def metric_coverage(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, Any]:
    """How many expected trials the benchmark's metric covered: ``{"trials": N, "expected": M, "source": ...}``.

    A benchmark that declares the coverage of its headline metric in its aggregate (``metric_coverage/trials``,
    ``metric_coverage/expected`` and an optional ``metric_coverage/name``) is the source (``benchmark``), with its
    own denominator and the metric's name. Otherwise the source is collection's own count when its coverage report
    has one for this inventory (``coverage_report``):
    the rollouts whose score entered the metrics, a failure counted as zero by configuration included. Otherwise it is
    read from the saved rows (``saved_rows``): the scored results and the failures their producer marked as counted
    zeros, which is what collection hands the metrics unless it counts more by configuration. A failure the benchmark
    leaves out of its metric, such as an agent deadline that kept its answer, is a trial the metric did not cover.
    The denominator is the trial gate's: every expected trial, less the rows an accepted partial stage omitted.
    """
    decision = _health_view(decision)
    from nemo_gym.rollouts.rollout_collection import is_non_superseding_rollout_row

    declared = collected.benchmark_metric_coverage if collected is not None else None
    if declared is not None:
        return {"trials": declared[0], "expected": declared[1], "source": "benchmark", "name": declared[2] or "metric"}
    expected = trial_population(decision)
    report = collected.coverage_scored if collected is not None else None
    if report is not None and report[1] == decision.expected:
        return {"trials": report[0], "expected": expected, "source": "coverage_report"}
    standing: dict[Identity, Mapping[str, Any]] = {}
    for row in (collected.failures or []) if collected is not None else []:
        if (identity := _identity(row)) is not None and not is_non_superseding_rollout_row(row):
            standing[identity] = row
    counted = sum(
        standing.get(identity, {}).get(COUNTS_AS_ZERO) is True
        for identity in decision.identities.get(IdentityState.MODEL_FINAL.value, [])
    )
    trials = decision.counts.get(IdentityState.SCORED, 0) + counted
    return {"trials": trials, "expected": expected, "source": "saved_rows"}


def health_verdict(evaluation: dict[str, Any] | None, facts: dict[str, Any] | None) -> str:
    """``evaluation health: PASS; response failures: 4.70% (truncated 41, empty 6 of 1000); trial failures: ...;
    metric over 1000 of 1000 trials``."""
    text = f"evaluation health: {evaluation['status'] if evaluation else 'NOT EVALUATED'}"
    if facts is None:
        return text
    for name in ("response_failures", "trial_failures"):
        found = facts[name]
        named = ", ".join(f"{cause} {fact['count']}" for cause, fact in found["by_cause"].items() if fact["count"])
        text += f"; {name.replace('_', ' ')}: {found['rate']:.2%} ({named or 'none'} of {facts['expected']})"
    if facts["omitted"]:
        text += f"; {facts['omitted']} accepted stage omission(s) are not counted"
    coverage = facts.get("metric_coverage")
    if coverage is not None:
        text += f"; {coverage.get('name') or 'metric'} over {coverage['trials']} of {coverage['expected']} trials"
    return text


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


MLFLOW_NOT_REQUIRED_NOTE = "no MLflow export by design (mlflow_tracking_required is false in the saved config)"


def tracking_export_due(
    decision: RecoveryDecision, collected: CollectedEvidence | None, tracking_required: bool
) -> bool:
    """Whether a complete run lacks the MLflow export receipt its saved config requires, so that
    ``gym eval resubmit --recover`` exports it.

    Only a run whose saved config configures MLflow (``provenance.mlflow.configured`` is true: tracking URI,
    experiment and run name) and does not set ``mlflow_tracking_required`` to false, with no post-run receipt at all
    and no live job, is exported. A run that configured no exporter, or whose provenance does not say, is never
    exported, and a receipt, current or not, is never replaced.
    """
    if collected is None or not decision.complete or decision.scheduler_live or not tracking_required:
        return False
    provenance = (collected.quality or {}).get("provenance")
    facts = provenance.get("mlflow") if isinstance(provenance, dict) else None
    local = collected.mlflow
    return (
        isinstance(facts, dict)
        and facts.get("configured") is True
        and isinstance(local, dict)
        and "receipt" in local
        and local["receipt"] is None
    )


def mlflow_tracking_required(collected: CollectedEvidence | None) -> bool:
    """False only when the run's saved config snapshot sets ``mlflow_tracking_required`` to false; absent, unread or
    any other value means tracking is required. The health report's provenance states it when its pin did; for a run
    whose report does not, the collector's read of the saved ``rollouts_resolved_config.yaml`` does."""
    provenance = ((collected.quality or {}) if collected is not None else {}).get("provenance")
    facts = provenance.get("mlflow") if isinstance(provenance, dict) else None
    stated = facts.get("tracking_required") if isinstance(facts, dict) else None
    if not isinstance(stated, bool):
        local = collected.mlflow if collected is not None else None
        stated = local.get("config_tracking_required") if isinstance(local, dict) else None
    return stated is not False


def mlflow_delivery(
    problems: Sequence[str], *, tracking_required: bool, status_error: str | None = None
) -> dict[str, Any]:
    """The MLflow delivery fact of a run: ``not_required`` by design, ``verified`` when ``mlflow_errors`` found nothing,
    ``pending`` while there is no receipt and no live run to judge or only the run's status was not queried, else
    ``unverified`` with its reasons. A fact for consumers to read, never a verdict: it does not touch evaluation
    health or the gate's status. A status that was neither given nor queried reads "status not queried"; a query that
    failed reads "status query failed: ...": never a silent "not observed"."""
    if not tracking_required:
        return {"state": "not_required", "reasons": []}
    others = [problem for problem in problems if problem != MLFLOW_STATUS_NOT_OBSERVED]
    status_unknown = len(others) != len(problems)
    reasons = list(others)
    if status_unknown:
        reasons.append(f"status query failed: {status_error}" if status_error else "status not queried")
    if not reasons:
        return {"state": "verified", "reasons": []}
    # Pending: nothing to judge yet (no run evidence) or only the run's status unread. Anything else is unverified.
    pending = not status_error and all(problem.startswith("no MLflow run evidence") for problem in others)
    return {"state": "pending" if pending else "unverified", "reasons": reasons}


def gate_summary_from_collected(
    decision: RecoveryDecision,
    collected: CollectedEvidence,
    *,
    max_unhealthy_fraction: float | None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    criteria: GateCriteria | None = None,
    status_client_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """The acceptance gate task entry from the collected run: decision counts, the named score and the quality summary.

    ``mlflow_status`` and ``mlflow_run_id`` are the tracking run's status and id as the caller read them together.
    Without a status, the receipt's run is asked of the tracking server the saved config names (``query_mlflow_status``;
    ``status_client_factory`` replaces its client).
    ``criteria`` are accepted and reported under ``criteria`` but not applied (``GateCriteria``). The only health
    verdict is ``evaluation_health``; ``failures`` reports response and trial failures by cause, never a verdict.
    ``status`` follows ``gate_task_summary``: an UNVERIFIED evaluation health with nothing failed is UNVERIFIED, and
    a chain error turns SUCCESS or UNVERIFIED into FAILED. MLflow delivery is reported as
    ``mlflow_delivery``, a fact that never changes the status.
    """
    criteria = criteria or GateCriteria()
    summary_valid, summary_reasons, rollout_health, rollout_reasons = quality_inputs(
        decision, collected, max_unhealthy_fraction=max_unhealthy_fraction
    )
    # Only evaluation health (evaluation-side trial faults, ``trial_verdict``) decides the gate's health; quality_summary's
    # verdicts are per rollout, so they are reported in warnings. A positive trial gate requires source-bound
    # operation evidence. Only recovered grading failures with complete joins are accepted. Model health and the raw
    # response faults are reported beside it and never fail the gate.
    trial_passed, trial_reason, evaluation = trial_verdict(
        decision,
        collected,
        policy_timeout_counts_as_model=criteria.policy_timeout_counts_as_model,
        classification=decision_classification(decision),
    )
    cause = None
    if trial_passed is None and decision.complete:
        # The warning names the cause in one sentence; the full reason and the trial boundary stay in the JSON.
        cause = unverified_cause(decision, collected, evaluation, trial_reason)
        trial_reason = f"{trial_reason} ({trial_boundary(collected.quality)})"
        if evaluation is not None:
            evaluation = {**evaluation, "detail": trial_reason}
    task = gate_task_summary(
        decision,
        score=collected.score,
        summary_valid=summary_valid,
        summary_reasons=summary_reasons,
        health_passed=trial_passed,
        health_reasons=[] if trial_passed else [cause["reason"] if cause else trial_reason],
    )
    task["unverified_cause"] = cause
    task["rollout_quality"] = rollout_quality_facts(collected.quality)
    task["criteria"] = criteria.report(classification=decision_classification(decision))
    if collected.unreadable_files:
        # What an unreadable metric, summary or record holds is unknown: never a pass.
        task["errors"].append(f"{UNREADABLE_ARTIFACTS_ERROR}: {collected.unreadable_files[:20]}")
        if task["status"] in ("SUCCESS", "UNVERIFIED"):
            task["status"] = "FAILED"
    task["trial_counts"] = trial_counts(decision, collected)
    task["evaluation_health"] = evaluation
    task["failures"] = failure_facts(decision, collected)
    task["health_verdict"] = health_verdict(evaluation, task["failures"])
    judge_recovery: dict[str, int] = {}
    _observed_trial_operations(decision, collected, judge_recovery=judge_recovery, grading_failures=[])
    task["observed_grading_recovery"] = judge_recovery or None
    for name in READER_PLACEMENT_FIELDS:
        if getattr(collected, name) is not None:
            task[name] = getattr(collected, name)
    judged = {True: "within", False: "beyond", None: "not judged against"}[rollout_health]
    task["warnings"] = (
        task["warnings"]
        + [f"rollout quality {judged} the configured criterion: {rollout_reasons}"]
        + health_check_facts(collected.quality)
        + ([f"health gate within the bounded failure policy: {trial_reason}"] if trial_passed else [])
        + [task["health_verdict"]]
    )
    # The judge-only jobs queued for the run's saved answers, each with its outcome: facts, never a verdict.
    task["retry_budget"] = retry_budget(collected)
    task["judge_jobs"] = [dict(job) for job in collected.judge_jobs] or None
    task["warnings"] = task["warnings"] + [judge_job_note(job) for job in collected.judge_jobs]
    extra: list[str] = []
    if collected.chain_incomplete and not decision.complete:
        extra.append(unsettled_chain_note(decision))
    provenance = (collected.quality or {}).get("provenance")
    mlflow_facts = provenance.get("mlflow") if isinstance(provenance, dict) else None
    configured = mlflow_facts.get("configured") if isinstance(mlflow_facts, dict) else None
    task["mlflow_tracking_required"] = mlflow_tracking_required(collected)
    mlflow_problems: list[str] = []
    status_error = None
    receipt = (collected.mlflow or {}).get("receipt")
    uri = (collected.mlflow or {}).get("config_tracking_uri")
    if (
        task["mlflow_tracking_required"]
        and mlflow_status is None
        and isinstance(receipt, dict)
        and isinstance(receipt.get("run_id"), str)
        and isinstance(uri, str)
    ):
        # No caller-read status: ask the tracking server for the receipt's own run.
        mlflow_status, status_error = query_mlflow_status(uri, receipt["run_id"], status_client_factory)
        mlflow_run_id = receipt["run_id"] if mlflow_status is not None else mlflow_run_id
    if task["mlflow_tracking_required"] is False:
        task["warnings"] = task["warnings"] + [MLFLOW_NOT_REQUIRED_NOTE]
    else:
        mlflow_problems = mlflow_errors(
            collected.mlflow,
            mlflow_status,
            configured=configured if isinstance(configured, bool) else None,
            observed_run_id=mlflow_run_id,
        )
        if not mlflow_problems:
            task["warnings"] = task["warnings"] + mlflow_notes(collected.mlflow)
    # MLflow delivery is a fact, never a verdict: it neither fails evaluation health nor turns the status to FAILED.
    task["mlflow_delivery"] = mlflow_delivery(
        mlflow_problems,
        tracking_required=task["mlflow_tracking_required"],
        status_error=status_error,
    )
    # Whether a recovery would export the results, by the executor's own predicate: the operator verdict offers the
    # export command only then.
    task["mlflow_export_due"] = tracking_export_due(decision, collected, task["mlflow_tracking_required"])
    if task["mlflow_delivery"]["state"] in ("pending", "unverified"):
        task["warnings"] = task["warnings"] + [
            f"MLflow delivery {task['mlflow_delivery']['state']}: {'; '.join(task['mlflow_delivery']['reasons'])}"
        ]
    if extra:
        task["errors"] = task["errors"] + extra
        if task["status"] in ("SUCCESS", "UNVERIFIED"):
            task["status"] = "FAILED"
    return task


def _rejudged_after_judge_failure(
    decision: RecoveryDecision, collected: CollectedEvidence, only: set[Identity]
) -> tuple[set[Identity], Counter]:
    """The scored identities among ``only`` with a saved judge failure (their answer was judged again), and the HTTP
    statuses of it."""
    scored = set(decision.identities.get(IdentityState.SCORED.value, [])) & only
    found: set[Identity] = set()
    statuses: Counter = Counter()
    for row in collected.failures or []:
        kind, status = _policy_evidence(row)
        identity = _identity(row)
        if identity in scored and kind in JUDGE_FAILURE_KINDS:
            if identity not in found and status is not None:
                statuses[status] += 1
            found.add(identity)
    return found, statuses


def unverified_cause(
    decision: RecoveryDecision,
    collected: CollectedEvidence,
    evaluation: Mapping[str, Any] | None,
    trial_reason: str,
) -> dict[str, str]:
    """What made a complete run's health UNVERIFIED, as ``{"kind", "reason"}``: one cause-first sentence with counts.

    ``kind`` is ``judge_rejudged``, ``grading_unobserved``, ``unclassified_failures``, ``recovered_not_joined`` or
    ``zero_rewards_unprovable``. The last has no cause beyond the missing proof, so the operator verdict treats it as
    unknown. The full reason of the trial gate stays in ``evaluation_health.detail``.
    """
    trials = trial_population(_health_view(decision))
    unobserved_ids: set[Identity] = set()
    counts = trial_counts(decision, collected, unobserved_ids) or {}
    # Judge history only explains rows whose grading is still unresolved (the unobserved ones); a recovered one is not
    # the current cause, so on the lower-bound path, where nothing is unobserved, it never names the judge.
    rejudged, statuses = _rejudged_after_judge_failure(decision, collected, unobserved_ids)
    if rejudged and trial_reason.startswith("trial: UNOBSERVED"):
        why = ", ".join(f"{status} from the judge model server" for status in sorted(statuses)) or "judge failed"
        reason = f"{len(rejudged)} of {trials} rows were re-judged after a judge failure ({why})"
        if unrecorded := counts.get("unobserved", 0):
            reason += f"; {unrecorded} rows have no recorded grading"
        return {"kind": "judge_rejudged", "reason": reason}
    if trial_reason.startswith("trial: UNOBSERVED"):
        return {
            "kind": "grading_unobserved",
            "reason": f"{counts.get('unobserved', 0)} of {trials} rows have no recorded grading (the verifier saved "
            "no grading operation or record)",
        }
    if "have unknown classification" in trial_reason:
        unknown = ((evaluation or {}).get("unknown") or {}).get("count")
        return {
            "kind": "unclassified_failures",
            "reason": f"{unknown} of {trials} trial failures have an unknown cause",
        }
    if (
        trial_reason.startswith("trial: UNVERIFIED, recovered attempts not joined")
        or "counters are malformed" in trial_reason
    ):
        return {
            "kind": "recovered_not_joined",
            "reason": "recovered attempt errors cannot be tied to a completed call of a scored trial",
        }
    return {
        "kind": "zero_rewards_unprovable",
        "reason": f"failures recorded only as zero rewards cannot be ruled out over {trials} rows",
    }


def rollout_quality_facts(quality: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The rollout-quality signals of ``quality_summary.json`` the operator verdict reads: verdict counts and the
    nonzero issue counts. A signal, never a gate; None when the summary has neither."""
    if not isinstance(quality, Mapping):
        return None
    verdicts, issues = quality.get("verdicts"), quality.get("issues")
    verdicts = {k: v for k, v in verdicts.items() if type(v) is int} if isinstance(verdicts, dict) else None
    issues = {k: v for k, v in issues.items() if type(v) is int and v} if isinstance(issues, dict) else None
    # The producer flags one finding per failed model chain, so the issue total can exceed the rollouts it affects;
    # the run's distinct count of rollouts that ended on a failed model call is the unit the verdict compares.
    ended = quality.get("ended_on_failed_model_call")
    facts: dict[str, Any] = {"verdicts": verdicts, "issues": issues}
    if type(ended) is int:
        facts["rollouts_ended_on_failed_model_call"] = ended
    return facts if verdicts is not None or issues is not None else None


UNREADABLE_ARTIFACTS_ERROR = "unreadable artifacts, read as unknown"
_NO_KNOWN_COMMAND = "read the stop reason; no recovery command is known for it"
# The fix each typed stop needs before the recovery, as Gym's own stop texts give it, and the options it adds to it.
_TYPED_REMEDIES: dict[str, tuple[str, tuple[str, ...]]] = {
    PROVIDER_CREDENTIALS_REJECTED: ("fix the credentials, then run", ()),
    DRIVER_INSTALL_FAILED: (
        "fix the driver install (the ref, the network or the image; logs/driver-<job id>.log has the installer's "
        "output), then run",
        (),
    ),
    ENVIRONMENT_PREPARE_FAILED: ("fix the benchmark's preparation, then run", ()),
    SERVER_SPINUP_TIMEOUT: (
        "raise the server start-up limit above the recorded one with",
        ("--server-spinup-timeout-seconds N --spinup-timeout-reason TEXT",),
    ),
    PROVIDER_QUOTA_EXHAUSTED: ("wait until the grading provider's quota resets, then run", ()),
}
# The typed stops a fresh restart repeats until their fix is made.
_RESTART_FIXES = (PROVIDER_CREDENTIALS_REJECTED, DRIVER_INSTALL_FAILED, ENVIRONMENT_PREPARE_FAILED)
# The typed stops whose own text says a recovery cannot continue the saved run as it is.
_UNSAFE_RESUME = ("--allow-unsafe-resume", "--unsafe-resume-reason TEXT")
_PROTOCOL_VIOLATION_REMEDY = (
    "fix the server or the benchmark pairing (a producer bug), then `gym eval submit` again, as the stop says; "
    "a recovery would get the same replies"
)
# What a judge-only job the dispatch did not queue needs, by the cause it classified (``RecoveryDecision.judge_refusal``).
_JUDGE_PLACEMENT_REMEDIES = {
    "cpu_partition_unknown": (("--judge-partition PARTITION",), "name a CPU partition of the cluster"),
    "cpu_placement": (
        ("--judge-partition PARTITION", "--judge-qos QOS"),
        "correct the partition or QOS the scheduler refused (a GPU QOS can require GPUs: QOSMinGRES)",
    ),
}
# An UNVERIFIED cause no resubmit and no flag changes, and where it is fixed (benchmark-notes: "UNVERIFIED by
# construction").
_UNVERIFIED_PRODUCERS = {
    "grading_unobserved": "the verifier's grading record",
    "unclassified_failures": "the classification of those failures by their producer",
    "recovered_not_joined": "the record of the recovered attempts by their producer",
}
# The options `gym eval resubmit --recover --inspect` refuses; an inspect command never repeats them.
_RECOVER_ONLY_FLAGS = frozenset(
    {
        "--max-walltime",
        "--allow-unsafe-resume",
        "--server-spinup-timeout-seconds",
        "--judge-partition",
        "--judge-qos",
    }
)
_OPEN_STATES = (
    IdentityState.JUDGE_RECOVERABLE,
    IdentityState.RETRYABLE,
    IdentityState.MISSING,
    IdentityState.EXHAUSTED,
    IdentityState.TERMINAL,
    IdentityState.MASKED,
    IdentityState.UNKNOWN,
)


def _verdict(state: str, reason: str, action: str) -> dict[str, str]:
    return {"state": state, "reason": reason, "action": action}


_SHELL_QUOTED = re.compile(r"""'(?:[^']|'\\'')*'|"(?:[^"\\]|\\.)*\"""")


def _flags(option: str) -> set[str]:
    """The flags one option spells (``--max-walltime T --walltime-reason TEXT`` is two)."""
    # What a shell quotes is a value (a JSON metric key can read `--x`), never an option.
    unquoted = _SHELL_QUOTED.sub(" ", option)
    return {word.lstrip("[") for word in unquoted.split() if word.lstrip("[").startswith("--")}


def _command(
    decision: RecoveryDecision,
    *options: str,
    remedy: Sequence[str] = (),
    drop: Sequence[str] = (),
    inspect: bool = False,
) -> str:
    """The one recovery command the operator verdict prints: `gym eval resubmit <job id> <benchmark> --recover`,
    then every option the call asked for (``decision.requested``), then the typed remedy's options, then ``options``
    (what the stop itself needs, such as a budget extension). An option is named once: ``options`` replaces the
    call's own value, the call's own value replaces a remedy's placeholder, and ``drop`` names options the run
    refused. ``inspect`` makes it the inspection, without the options an inspection refuses."""
    named = set(drop).union(*map(_flags, options))
    if "--repin" in named.union(*map(_flags, remedy)):
        # The CLI refuses --no-auto-repin beside an explicit repin, which is the operator's own pin choice.
        named.add("--no-auto-repin")
    asked = [
        option
        for option in decision.requested
        if not _flags(option) & named and not (inspect and _flags(option) & _RECOVER_ONLY_FLAGS)
    ]
    held = named.union(*map(_flags, asked))
    fixes = [option for option in remedy if not _flags(option) & held]
    words = ["gym eval resubmit", decision.target, "--recover", *(["--inspect"] if inspect else []), *asked, *fixes]
    return "`" + " ".join([*words, *options]) + "`"


def _choices(options: Sequence[str]) -> str:
    """What in a printed command is the operator's to choose, and that its options go in one call."""
    choices = [
        name for name, flag in (("N", " N"), ("the walltime", "HH:MM:SS")) if any(flag in option for option in options)
    ]
    chosen = f"; {' and '.join(choices)} {'are' if len(choices) > 1 else 'is'} yours to choose" if choices else ""
    commit = "; COMMIT is this launcher's full commit" if any("COMMIT" in option for option in options) else ""
    together = "; every option named goes in the same call" if sum(not o.startswith("[") for o in options) > 1 else ""
    return chosen + commit + together


def _stop_kind(decision: RecoveryDecision, *, from_rows: bool = True) -> str | None:
    """The typed failure with a known remedy the run stopped on: the chain guard's typed stop, else the one failure
    kind every terminal identity of the decision records (Gym's own typed stop, such as the grading quota). Read from
    typed fields only, never from the reason text. ``from_rows=False`` reads the chain guard's stop alone: the row
    kind explains a STOP, and a decision that continues the saved run (RESUBMIT, REVERIFY) is not one."""
    if decision.chain_stop_kind:
        return decision.chain_stop_kind
    if not from_rows:
        return None
    terminal = [item for item in decision.evidence if item.state == IdentityState.TERMINAL]
    kinds = {item.failure_kind for item in terminal}
    if terminal and len(terminal) == decision.counts.get(IdentityState.TERMINAL, 0) and len(kinds) == 1:
        kind = kinds.pop()
        return (
            kind
            if kind in _TYPED_REMEDIES or kind in (COLLECTION_RESUME_REFUSED, ENVIRONMENT_PROTOCOL_VIOLATION)
            else None
        )
    return None


def _continue_with(
    decision: RecoveryDecision, *options: str, drop: Sequence[str] = (), kind_from_rows: bool = True
) -> str:
    """The recovery command that continues the run, after the typed stop's fix when it stopped on one, with what in
    it is the operator's to choose. A typed stop whose own text says a recovery cannot continue (a refused resume,
    a protocol violation) names that instead."""
    kind = _stop_kind(decision, from_rows=kind_from_rows)
    if kind == COLLECTION_RESUME_REFUSED:
        unsafe = _command(decision, *options, *_UNSAFE_RESUME, drop=drop)
        return (
            "the saved rollouts do not verify against the run's inputs, so start a fresh run as the stop says; only "
            f"to accept that exact difference, {unsafe}" + _choices([*options, *_UNSAFE_RESUME])
        )
    if kind == ENVIRONMENT_PROTOCOL_VIOLATION:
        return _PROTOCOL_VIOLATION_REMEDY
    fix, remedy = _TYPED_REMEDIES.get(kind or "", ("", ()))
    if kind == PROVIDER_QUOTA_EXHAUSTED and (resets := quota_reset_text(decision.reason)):
        fix = f"wait until {resets}, when the grading provider's quota resets, then run"
    if remedy and all(_flags(option) & set().union(*map(_flags, decision.requested)) for option in remedy):
        # The call already carries the fix (a raised start-up limit, say); the command repeats it.
        fix = ""
    command = _command(decision, *options, remedy=remedy, drop=drop)
    return (f"{fix} {command}" if fix else command) + _choices([*(remedy if fix else ()), *options])


def _flagged_everything(
    gate_task: Mapping[str, Any], decision: RecoveryDecision, expected: int
) -> tuple[str, str] | None:
    """``(what, flagged)`` when rollout quality flags every rollout or every task, else None; ``flagged`` counts in
    the unit of the check."""
    quality = gate_task.get("rollout_quality") or {}
    issues = quality.get("issues") or {}
    tasks = len({identity[0] for identities in decision.identities.values() for identity in identities})
    # Distinct rollouts, not findings: a rollout whose several model chains failed counts once. The distinct count is
    # a statistic the producer keeps even for an ignored check, so the check must also hold an active finding.
    check = "rollout_ended_on_failed_model_call"
    if expected and quality.get("rollouts_ended_on_failed_model_call", 0) >= expected and issues.get(check, 0) > 0:
        return "every rollout ended on a failed model call", f"{expected} of {expected} rollouts flagged {check}"
    if tasks and issues.get("task_no_successful_model_calls", 0) >= tasks:
        check = "task_no_successful_model_calls"
        return "no task had a successful model call", f"{tasks} of {tasks} tasks flagged {check}"
    return None


_EXPLICIT_BUDGET = "--max-walltime HH:MM:SS --walltime-reason TEXT"
_REPIN_IN_THE_SAME_RESUBMIT = "(--repin COMMIT --repin-reason TEXT) in the same resubmit"


def _continuation_options(decision: RecoveryDecision) -> list[str]:
    """The options of the one ``gym eval resubmit --recover`` call that continues a run this STOP blocks, else [].

    The stop's own cause decides: a spent budget (attempts, infra retries, walltime), unknown chain accounting (an
    explicit budget fills it), masked results without an artifact (an approved rerun), or an extension the run's
    pinned Gym cannot apply (a repin in the same call). What in the chain would stop next (``_chain_blockers_note``)
    counts only once that cause has a continuation, and the note on masked results without an artifact, which any
    stop carries, is no cause of its own. Every option a blocker needs is named, with each one the stopped call asked
    for and did not record, so following the advice never drops one."""
    reason = decision.reason
    notes = re.split(r"; |\n", reason)

    def asked(flag: str) -> bool:
        return any(note.startswith(f"{flag} ") and note.endswith(_NOT_RECORDED_TOKEN) for note in notes)

    rerun_flag = RERUN_MASKED_FLAGS.split()[0]
    # A refused extension or rerun continues only with a repin in the same call; refused for any other reason, it
    # cannot continue the run at all, and the refusal itself says what to do.
    refused = re.match(rf"(--extend-attempts \d+|{rerun_flag}) is refused: ", reason)
    repin = _REPIN_IN_THE_SAME_RESUBMIT in reason
    if refused and not repin:
        return []
    cause, _, blockers = reason.split("\n")[0].partition(_CHAIN_BLOCKERS_LEAD)
    rerun = (
        (_MASKED_STOP_TOKEN in cause and bool(decision.masked_without_artifact))
        or bool(refused and refused[1] == rerun_flag)
        or asked(rerun_flag)
    )
    attempts = _ATTEMPTS_USED_TOKEN in cause or bool(refused and refused[1].startswith("--extend-attempts"))
    stops = cause + (blockers if attempts or rerun else "")
    budget_unknown = decision.accounting_unknown == "budget" or UNKNOWN_BUDGET_REASON in stops
    unknown = budget_unknown or decision.accounting_unknown is not None or UNKNOWN_WALLTIME_REASON in stops
    retries = _INFRA_BUDGET_USED_PREFIX in stops
    walltime = _WALLTIME_USED_TOKEN in stops or unknown
    if not (attempts or retries or walltime or rerun or repin):
        return []
    options = {
        "--extend-attempts N": attempts or asked("--extend-attempts"),
        "--extend-retries N": retries or asked("--extend-retries"),
        f"{RERUN_MASKED_FLAGS} TEXT": rerun,
        _EXPLICIT_BUDGET: walltime or asked("--max-walltime"),
        "[--extend-retries N]": budget_unknown and not (retries or asked("--extend-retries")),
        "--repin COMMIT --repin-reason TEXT": repin,
    }
    return [option for option, needed in options.items() if needed]


def _partial_rerun_note(decision: RecoveryDecision, options: list[str]) -> str:
    """What a rerun in ``options`` leaves masked: it runs only the results without a gradeable artifact."""
    others = decision.counts.get(IdentityState.MASKED, 0) - len(decision.masked_without_artifact)
    if f"{RERUN_MASKED_FLAGS} TEXT" not in options or others <= 0:
        return ""
    return (
        f"; the rerun covers only the masked results without a gradeable artifact: the other {others} masked "
        "result(s) stay masked (kept, not measured), so the run stays incomplete"
    )


# Stops on evidence that cannot be read yet, and how to read it, instead of a fresh submit over the saved run.
_UNKNOWN_EVIDENCE_STOPS = (
    ("the scheduler query failed", "wait until the scheduler answers and inspect again"),
    ("saved rows are unreadable", "make the saved rows readable again and inspect again"),
    ("is unreadable", "make the file the reason names readable again and inspect again"),
    ("changed while it was read", "inspect again once nothing writes the run's files"),
    (
        "authoritative Store reader is unavailable",
        "inspect again once the run's Store reader can verify its artifacts",
    ),
    (
        "the results file is absent",
        "restore the results file if it was moved and inspect again; if it is lost, report that collection cannot "
        "resume the run's saved history, as the stop says",
    ),
    # A judge-only job the evidence changed under, or whose snapshot was missing, was not queued.
    ("collect the current evidence", "inspect again: the evidence changed after the recovery was planned"),
    ("collect the evidence again", "inspect again to collect the evidence the judge-only job needs"),
)
# A recovery whose judge-only job or plan needs the launcher's revision, and the option that moves the run to it.
_REPIN_STOPS = (
    re.compile(r"--repin <this launcher's full commit> --repin-reason TEXT runs the judge-only job at it"),
    re.compile(
        r"the pin was kept \(--no-auto-repin\): recover with --repin (?P<commit>[0-9a-f]{40}) --repin-reason TEXT"
    ),
)
# What clears each guard that refuses an operator continuation, before the continuation itself, and the state of
# the run while it holds.
_CONTINUATION_GUARDS = {
    "scheduler_unreachable": (
        "unknown",
        "the scheduler did not answer, so a live or pending job of the run cannot be ruled out",
        "wait until the scheduler answers and inspect again",
    ),
    "job_live": (
        "unknown",
        "a job of the run is pending or running while its chain accounting is unknown",
        "wait for that job to end and inspect again",
    ),
    "claim_unidentified": (
        "needs_action",
        "a follow-up claim names no job, so a queued follow-up cannot be ruled out",
        "reconcile the scheduler queue with the run's follow-up claims first (one names no job id)",
    ),
    "counters_changed": (
        "unknown",
        "the chain's accounting files changed while they were read",
        "inspect again once no job of the run writes them",
    ),
}


def _refusal_verdict(refusal: Mapping[str, Any]) -> dict[str, str]:
    """The verdict of a resubmit the scheduler refused: no job is known to be queued, or it is unknown whether one is."""
    text = refusal.get("scheduler_text") or "(no text)"
    if refusal.get("job_dir_restored"):
        return _verdict(
            "needs_action",
            f"the scheduler refused the resubmit: {text}; nothing was queued",
            "correct what the scheduler text names before redispatching with every option of the refused call; the "
            "same resubmit would be refused again",
        )
    return _verdict(
        "needs_action",
        f"the scheduler did not confirm the resubmit ({text}); whether a job was queued is unknown",
        "reconcile the scheduler queue and the run's chain files first, then decide whether to redispatch with every "
        "option of that call",
    )


def operator_verdict(
    gate_task: Mapping[str, Any] | None,
    decision: RecoveryDecision | None,
    scheduler_refusal: Mapping[str, Any] | None = None,
    walltime_plan: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """The one verdict an operator acts on, derived from the gate and the decision only.

    ``state`` is ``accept`` (gate SUCCESS, coverage complete, run health PASS, decision noop), ``needs_action`` (a
    STOP, FAILED, incomplete coverage or failed run health) or ``unknown`` (UNVERIFIED, which no recovery changes: by
    configuration, by a producer that records no grading, after a re-judgment whose grading was not recorded or
    without a known cause; a live job; evidence missing or unreadable). ``reason`` is one sentence naming the
    dominant cause with counts; ``action`` is the one thing to do. Every command it prints comes from ``_command``:
    `gym eval resubmit <job id> <benchmark> --recover` with every option the call asked for, the typed stop's fix
    and the options the stop needs. It never adds a fresh submit that Gym's own decision or stop text does not name,
    and where Gym decided nothing is left to run it says what to report and check. An accepted run whose MLflow
    delivery is not closed names the export, or that it is unverified by configuration. Rollout quality never changes
    the state: an accepted run whose every rollout is flagged says so in its reason and points at the quality read. A
    scheduler refusal of the resubmit and a walltime raise the partition cannot take (refused before any evidence is
    read) are known dispatch failures: they need the operator's correction, never a plain recover-now or a reader
    retry.
    """
    refused = (walltime_plan or {}).get("refused")
    if gate_task is None and decision is not None and refused:
        return _verdict(
            "needs_action",
            f"the walltime raise was refused before any evidence was read: {refused}",
            "correct the walltime or the partition so the segment fits, then run the recovery again with its other "
            "options",
        )
    if gate_task is None or decision is None:
        return _verdict("unknown", "no evidence was collected", "inspect again once the reader runs")
    status, errors = gate_task.get("status"), gate_task.get("errors") or []
    counts = decision.counts
    expected = gate_task.get("expected_samples") or decision.expected
    if status in ("RUNNING", "RESUMING") or decision.scheduler_live:
        return _verdict(
            "unknown", "a job of the run is live, so the result is not final", "wait for it to end, then inspect again"
        )
    if scheduler_refusal:
        return _refusal_verdict(scheduler_refusal)
    if decision.action == RecoveryAction.UNKNOWN:
        return _verdict("unknown", f"nothing was decided: {decision.reason}", "inspect again once the reader runs")
    if decision.action in (RecoveryAction.REVERIFY, RecoveryAction.STOP) and (
        blocked := _FAILED_JUDGE_JOBS.search(decision.reason)
    ):
        # The judge-only job is refused until the failed ones are read and, where they can be, acknowledged.
        read = f"read their receipts and logs under {blocked['where']}"
        return _verdict(
            "needs_action",
            f"the last {JUDGE_JOB_FAILURE_BOUND} judge-only jobs ({blocked['ids']}) ended without settling a saved "
            "answer, so recovery queues no other until they are read and acknowledged",
            f"{read}; a job without an id cannot be acknowledged, so no further judge-only job is allowed and no "
            "recovery command is known"
            if blocked["unacknowledgeable"]
            else f"{read}, then {_command(decision, f'{ACKNOWLEDGE_JUDGE_JOBS_FLAGS} TEXT')} allows one more "
            "judge-only job",
        )
    if unreadable := next((str(e) for e in errors if str(e).startswith(UNREADABLE_ARTIFACTS_ERROR)), None):
        # What an unreadable artifact holds is unknown, whatever the decision read around it.
        return _verdict("unknown", unreadable, "make those files readable again and inspect again")
    open_text = ", ".join(f"{counts[s]} {s.value}" for s in _OPEN_STATES if counts.get(s))
    if decision.action == RecoveryAction.FRESH_RESTART:
        # Gym's own decision: no task inventory was saved, and its reason names the command.
        named = re.search(r"--max-walltime is required: (gym eval resubmit \S+ \S+ --recover .*)$", decision.reason)
        command = f"`{named[1]}`" if named else _command(decision, _EXPLICIT_BUDGET)
        # A typed failure the restart would repeat (a failed install or preparation, rejected credentials) is fixed first.
        kind = _stop_kind(decision)
        fix = _TYPED_REMEDIES[kind][0].removesuffix(", then run") + ", then " if kind in _RESTART_FIXES else ""
        return _verdict(
            "needs_action",
            f"the run saved no task inventory ({decision.reason.split('; --max-walltime is required')[0]})",
            f"{fix}queue it again as a fresh run with a longer walltime (the walltime is yours to choose): {command}",
        )
    if decision.action in (RecoveryAction.RESUBMIT, RecoveryAction.REVERIFY):
        settled = expected - sum(counts.get(s, 0) for s in _OPEN_STATES)
        # A chain that stopped on a typed failure needs its remedy before the resume, or the resume stops the same way.
        # The decision continues the saved run, so a terminal row's kind (it explains a STOP only) never turns the
        # advice into a fresh submit; only the chain guard's own typed stop does.
        action = _continue_with(decision, kind_from_rows=False)
        return _verdict(
            "needs_action",
            f"{settled} of {expected} rollouts are settled; {open_text or 'recovery work'} remain",
            action if _stop_kind(decision, from_rows=False) else f"{action} now",
        )
    if decision.action == RecoveryAction.STOP:
        return _stop_verdict(decision, expected, open_text)
    if status == "FAILED" or errors:
        health = gate_task.get("evaluation_health") or {}
        if health.get("status") == "FAIL":
            faults = health.get("evaluation") or {}
            kinds = ", ".join(f"{k} {v}" for k, v in (faults.get("kinds") or {}).items())
            # Over the trials, not the expected samples: an accepted omission is no trial. Gym decided nothing is left
            # to run: the faults are settled outcomes that no recovery reopens.
            return _verdict(
                "needs_action",
                f"{faults.get('count')} of {health.get('trials', expected)} trials failed on the evaluation side ({kinds})",
                "none: report the run as failing run health; no recovery reopens settled trials, so check the "
                "evaluation-side faults evaluation_health names (their kinds and the services' logs)",
            )
        unread = [e for e in errors if str(e).startswith("score unavailable")]
        if decision.complete and errors and len(unread) == len(errors):
            return _verdict(
                "unknown",
                "no score was read",
                f"inspect again naming the score: {_command(decision, '--score-agent NAME', '--score-key JSON', inspect=True)}",
            )
        # A known failure dominates an unread score listed before it.
        known = next((str(e) for e in errors if not str(e).startswith("score unavailable")), None)
        return _verdict(
            "needs_action",
            (known[:200] + ("; the score was not read either" if unread else ""))
            if known
            else (str(errors[0])[:200] if errors else "the gate failed"),
            "read the error; no recovery command is known for it",
        )
    if not decision.complete or decision.action != RecoveryAction.NOOP:
        return _verdict(
            "needs_action",
            f"coverage is incomplete: {open_text or 'rollouts are unsettled'}",
            f"{_command(decision)} now",
        )
    delivery = _delivery_action(
        decision, gate_task.get("mlflow_delivery") or {}, gate_task.get("mlflow_export_due") is True
    )
    if status == "UNVERIFIED":
        cause = gate_task.get("unverified_cause")
        export = f"; {delivery[1]}" if delivery else ""
        if decision.unverified:
            # Verification was configured off: the run is complete and unverified by design, nothing to do.
            skipped = counts.get(IdentityState.SKIPPED, 0)
            return _verdict(
                "unknown",
                f"unverified by configuration: {skipped} of {expected} rows were not verified (verification skipped "
                "by configuration)",
                "none: report the run as unverified by configuration" + export,
            )
        if cause and cause.get("kind") == "judge_rejudged":
            # The re-judgment recorded no grading, so whether the re-judged rows were graded is not evidenced.
            return _verdict(
                "unknown",
                f"unverified: {cause['reason']}",
                "none: report the run as unverified: the rows were re-judged and no grading was recorded, so whether "
                "they were graded is unknown; no recovery command records it" + export,
            )
        if cause and cause.get("kind") in _UNVERIFIED_PRODUCERS:
            # UNVERIFIED by construction: a resubmit cannot change it, and no flag does.
            return _verdict(
                "unknown",
                f"unverified: {cause['reason']}",
                "none: report the run as unverified with this reason and do not present it as verified; a resubmit "
                f"cannot change it, and no flag does: the fix is {_UNVERIFIED_PRODUCERS[cause['kind']]}, in Gym"
                + export,
            )
        known = (cause or {}).get("reason") or "summary validity or run health is unknown"
        return _verdict(
            "unknown",
            f"unverified without a known cause: {known}",
            "none: report the run as unverified; no operator action is known" + export,
        )
    if status != "SUCCESS" or (gate_task.get("evaluation_health") or {}).get("status") != "PASS":
        return _verdict("unknown", "the gate states no SUCCESS with run health PASS", "inspect again")
    quality_facts = gate_task.get("rollout_quality") or {}
    findings = (quality_facts.get("issues") or {}).get("rollout_ended_on_failed_model_call", 0)
    if expected and findings >= expected and "rollouts_ended_on_failed_model_call" not in quality_facts:
        # One finding per failed model chain can exceed the rollouts it affects, and this collector reports no distinct
        # count: whether every rollout ended on a failed model call is unknown, and neither accepting nor rejecting
        # the score would rest on evidence.
        return _verdict(
            "unknown",
            f"unverified: the collector has no distinct affected-rollout count, and {findings} rollout_ended_on_failed_"
            f"model_call findings could cover all {expected} rollouts or fewer",
            "none: report the run as unverified and do not present it as verified; re-collect the evidence with a "
            "current runtime, then inspect again" + (f"; {delivery[1]}" if delivery else ""),
        )
    if flagged := _flagged_everything(gate_task, decision, expected):
        what, count = flagged
        # The gate's status stays SUCCESS (rollout quality never changes it), but a score from rollouts that all ended
        # on a failed model call or never had a successful model call most likely comes from a broken evaluation
        # (serving or credentials), so it is not an acceptance. The delivery state is a fact beside it.
        return _verdict(
            "needs_action",
            f"{what}, so the score most likely does not measure the model (a broken serving or credentials setup is "
            "the usual cause)" + (f"; {delivery[0]}" if delivery else ""),
            f"do not accept or report the score yet: check the served configuration and the model's credentials, "
            f"and read rollout quality ({count})" + (f"; {delivery[1]}" if delivery else ""),
        )
    omitted = gate_task.get("omitted_samples") or 0
    reason = (
        f"{gate_task.get('scored_samples', 0)} of {expected} rollouts scored"
        + (f", {omitted} omitted by an accepted stage" if omitted else "")
        + ", run health PASS, nothing to recover"
    )
    if delivery:
        return _verdict("accept", f"{reason}; {delivery[0]}", delivery[1])
    return _verdict("accept", reason, "none")


def _delivery_action(
    decision: RecoveryDecision, delivery: Mapping[str, Any], export_due: bool = False
) -> tuple[str, str] | None:
    """``(what, action)`` for a complete run whose MLflow delivery is not closed: the results stand, and the delivery
    is a fact beside them. Unverified by configuration (the export did not upload the rollouts) is reported, as no
    export changes it."""
    if delivery.get("state") not in ("pending", "unverified"):
        return None
    reasons = [str(r) for r in delivery.get("reasons") or []]
    what = f"MLflow delivery {delivery['state']}: {'; '.join(reasons) or 'no reason recorded'}"
    if any(MLFLOW_ROLLOUTS_NOT_UPLOADED in r for r in reasons):
        return (
            what,
            "report the MLflow delivery as unverified by configuration (upload_rollouts: false); an export does not change it",
        )
    if any(r.startswith("no MLflow run evidence") for r in reasons):
        if export_due:
            return (
                what,
                f"export the results to MLflow: {_command(decision)} runs the export of a complete run without a receipt (no model calls)",
            )
        # A recovery exports only a run whose saved provenance says MLflow is configured; it would repeat as a no-op.
        return (
            what,
            "report the MLflow delivery as not evidenced: the run's saved provenance does not say MLflow was configured "
            "(provenance.mlflow.configured), so a recovery does not export it and its results stand without an export; "
            "if MLflow was meant to be used, export them with the tracking tooling and inspect again with "
            "--mlflow-run-id and --mlflow-status",
        )
    confirm = _command(decision, "--mlflow-run-id ID", "--mlflow-status STATUS", inspect=True)
    return what, (
        f"confirm the MLflow export: {confirm} with the run id and status you observed on the tracking server, or "
        "repair the export the reason names"
    )


def _stop_verdict(decision: RecoveryDecision, expected: int, open_text: str) -> dict[str, str]:
    """The verdict of a STOP: the continuation its own cause allows, after any guard and typed fix, else the most
    specific known step, else no command."""
    counts = decision.counts
    stopped = f"recovery stopped with {open_text or 'no way to continue'}: {decision.reason.splitlines()[0][:160]}"
    options = _continuation_options(decision)
    kind = _stop_kind(decision)
    # A spent budget is the cause of this stop when it names a continuation: a protocol row among the saved rows then
    # explains nothing, and its remedy (a fresh submit) would discard the saved run the extension keeps.
    from_rows = not (options and decision.chain_stop_kind is None and kind == ENVIRONMENT_PROTOCOL_VIOLATION)
    if not from_rows:
        kind = None
    if kind == PROVIDER_QUOTA_EXHAUSTED and decision.chain_stop_kind is None:
        # Gym's own quota stop: every terminal identity waits on the grading provider's quota.
        waiting = counts.get(IdentityState.TERMINAL, 0)
        return _verdict(
            "needs_action",
            f"{waiting} of {expected} saved answers wait on the grading provider quota",
            _continue_with(decision, *options),
        )
    if kind in (COLLECTION_RESUME_REFUSED, ENVIRONMENT_PROTOCOL_VIOLATION):
        # The typed stop's own text says a recovery cannot continue the saved run as it is.
        return _verdict("needs_action", stopped, _continue_with(decision, *options))
    if decision.continuation_refused == "no_retry_cap":
        # No continuation exists: the explicit budget has no cap to fill, and Gym names no other way on.
        return _verdict(
            "needs_action",
            f"{stopped}; the saved run cannot continue, as it configures no infra-retry cap",
            "none: report it; an operator budget has no infra-retry cap to fill, so no recovery command is known",
        )
    if decision.continuation_refused is not None:
        # A guard refuses the continuation: clear it first, and keep every option the continuation needs.
        state, why, clear = _CONTINUATION_GUARDS[decision.continuation_refused]
        return _verdict(
            state,
            f"an operator continuation is refused: {why}",
            f"{clear}; the saved run then continues (completed work is kept): {_continue_with(decision, *options, kind_from_rows=from_rows)}"
            + _partial_rerun_note(decision, options),
        )
    if options:
        return _verdict(
            "needs_action",
            stopped,
            "your decision: continue the saved run (completed work is kept): "
            f"{_continue_with(decision, *options, kind_from_rows=from_rows)}"
            + ("; it is refused while a job of the run is pending or running" if _EXPLICIT_BUDGET in options else "")
            + _partial_rerun_note(decision, options),
        )
    if decision.refused_option and decision.refused_over in (RecoveryAction.RESUBMIT, RecoveryAction.REVERIFY):
        # Only the refused option stops it: without it the plan continues the run.
        return _verdict(
            "needs_action",
            stopped,
            f"drop {decision.refused_option}, which this run cannot take; the saved run continues without it "
            f"(completed work is kept): {_continue_with(decision, drop=(decision.refused_option,))}",
        )
    if repinned := next((match for stop in _REPIN_STOPS if (match := stop.search(decision.reason))), None):
        # Only the launcher's revision runs this recovery: the repin moves the run there and keeps it.
        commit = repinned.groupdict().get("commit") or "COMMIT"
        return _verdict(
            "needs_action",
            stopped,
            "your decision: continue the saved run (completed work is kept) at this launcher's revision: "
            + _continue_with(decision, f"--repin {commit} --repin-reason TEXT"),
        )
    if unread := next((action for token, action in _UNKNOWN_EVIDENCE_STOPS if token in decision.reason), None):
        # What the saved run holds is unknown for now: learn it before anything else.
        return _verdict("unknown", stopped, f"{unread} (completed work is kept)")
    if _RESERVATIONS_UNKNOWN in decision.reason and "nobody owns them" not in decision.reason:
        # Whether they ran is unknown until the allocation is known to have ended; then a recovery resumes them.
        return _verdict(
            "needs_action",
            stopped,
            "confirm with the scheduler that the allocation that made the reservations has ended, then inspect "
            "again: a recovery resumes them once that is known (completed work is kept)",
        )
    if decision.judge_refusal in _JUDGE_PLACEMENT_REMEDIES:
        placement, fix = _JUDGE_PLACEMENT_REMEDIES[decision.judge_refusal]
        return _verdict(
            "needs_action",
            stopped,
            f"{fix}: {_command(decision, *placement)} queues the judge-only job (completed work is kept)",
        )
    if decision.judge_refusal == "served_placement":
        # The judge runs on the run's own saved node pool; --judge-partition and --judge-qos name a CPU placement.
        return _verdict(
            "needs_action",
            stopped,
            "the scheduler refused the saved node pool this run serves its judge on, and no recovery option changes "
            "it (--judge-partition and --judge-qos apply to a CPU judge only): read the scheduler's refusal in the "
            "reason and have the pool's partition, QOS or account fixed on the cluster, then inspect again "
            "(completed work is kept)",
        )
    if decision.judge_refusal == "unsupported_profile":
        # Decided locally from the saved config: no scheduler call was made, so waiting for the scheduler changes nothing.
        return _verdict(
            "needs_action",
            stopped,
            "the saved run has no judge execution profile that a recovery can start, and no recovery option adds "
            "one: read what the reason says would allow it (completed work is kept)",
        )
    if decision.judge_refusal == "refused":
        # A definite refusal, not an outage: repeating the call without a change is refused the same way.
        return _verdict(
            "needs_action",
            stopped,
            "the judge-only job was refused for a definite reason that is not its placement (the reason has the "
            "refusal): remove what it names, then inspect again; waiting for the scheduler does not change it "
            "(completed work is kept)",
        )
    if decision.judge_refusal == "concurrent":
        # The dispatch's own guard: another recovery or a job of the run is active. Its lock and job are protection and
        # work in progress, not an obstacle to remove.
        return _verdict(
            "needs_action",
            stopped,
            "another recovery of this run or a job of it is active, so the judge-only job was not queued: wait until "
            "it ends and inspect again; do not remove its lock or cancel its job (completed work is kept)",
        )
    if decision.judge_refusal == "other":
        # The scheduler did not answer, or the job's acknowledgement is missing: its own diagnostic is in the reason.
        return _verdict(
            "needs_action",
            stopped,
            f"the judge-only job could not be confirmed queued, so it may exist (the reason has the diagnostic): "
            f"wait until the scheduler answers and inspect again, then {_command(decision)} (completed work is kept)",
        )
    if kind in _TYPED_REMEDIES and decision.chain_stop_kind is None:
        # Every terminal row is the typed failure: its fix, then the recovery its own text names. A chain's typed
        # stop alone never names a recovery for a stop of another cause.
        return _verdict("needs_action", stopped, _continue_with(decision))
    # A stop no rule here knows: its reason is the only guide, and a fresh submit would discard the saved run.
    return _verdict("needs_action", stopped, _NO_KNOWN_COMMAND)


def with_operator_verdict(outcome: RecoveryOutcome) -> RecoveryOutcome:
    """The outcome with ``operator_verdict`` set on it and, when there is a gate, inside ``gate_task``."""
    verdict = operator_verdict(outcome.gate_task, outcome.decision, outcome.scheduler_refusal, outcome.walltime_plan)
    gate_task = {**outcome.gate_task, "operator_verdict": verdict} if outcome.gate_task is not None else None
    return outcome.model_copy(update={"gate_task": gate_task, "operator_verdict": verdict})


def operator_verdict_line(verdict: Mapping[str, str]) -> str:
    return f"operator verdict: {verdict['state']}: {verdict['reason']}; action: {verdict['action']}"


def judge_job_note(job: Mapping[str, Any]) -> str:
    """One line on a judge-only job of the run (``CollectedEvidence.judge_jobs``): queued, live, ended without a
    receipt, or what its receipt says it settled."""
    name = f"judge-only job {job.get('job_id') or '(id unknown)'}"
    where = f" on {job['partition']}" if job.get("partition") else ""
    if job.get("services"):
        # A served-judge job starts the run's own judge services on their saved node pools.
        where += f" serving {', '.join(job['services'])}"
    state = job.get("state")
    if state == "completed":
        note = (
            f"{name}{where} completed: settled {job.get('settled') or 0} of {job.get('selected') or 0} selected saved "
            "answer(s)"
        )
        if job.get("withdrawn"):
            note += f", {job['withdrawn']} re-verified without a grade and left as they were"
        if job.get("unsupported"):
            note += f", {job['unsupported']} not re-verified (verifier not stateless)"
        if job.get("due_after"):
            note += f"; {job['due_after']} still due"
        return note
    if state == "failed":
        if job.get("ended_at"):
            note = f"{name}{where} failed: {job.get('error') or 'no error recorded'}; "
            settled, withdrawn = job.get("settled") or 0, job.get("withdrawn") or 0
            if not settled and not withdrawn:
                return note + "the saved answers it was to grade again stay as they were"
            # A grade it settled before it failed stays, and is progress (``_judge_job_unsettled``).
            note += f"before it failed it settled {settled} of {job.get('selected') or 0} selected saved answer(s)"
            if settled:
                note += ", which counts as progress"
            if withdrawn:
                note += f"{',' if settled else ''} and withdrew {withdrawn}, left as they were"
            return note + "; the others stay as they were"
        return (
            f"{name}{where} ended {job.get('scheduler_state') or 'in an unknown state'} with no receipt; the saved "
            "answers it was to grade again stay as they were"
        )
    if state in ("running", "pending"):
        return (
            f"{name}{where} is {state} ({job.get('scheduler_state')}): it grades {job.get('rows')} saved answer(s) "
            "again"
        )
    return f"{name}{where} queued at {job.get('at')} for {job.get('rows')} saved answer(s): no receipt yet"


# --- Saved-answer re-verification --------------------------------------------------------------------------------- #
# A saved result whose grading failed for a judge or transport cause, or whose grading was never recorded, is graded
# again from its saved answer: never by running the policy again. Collection settles it into the run at completion
# (``rollout_reverification.bind_judge_failures``) and ``--recover`` decides REVERIFY for it. No other masked
# result is re-verified: one that infrastructure left unmeasured runs again (``masked_infrastructure_kind``).
REVERIFY_JUDGE = "judge"
REVERIFY_VERIFY_TRANSPORT = "verify_transport"
REVERIFY_UNGRADED = "ungraded"
REVERIFY_VERIFIER = "verifier"
_REVERIFY_JUDGE_KINDS = frozenset({JUDGE_FAILED, JUDGE_UNPARSEABLE})


def _transport_kind(kind: Any) -> bool:
    return isinstance(kind, str) and kind.startswith("transport_")


def _verify_transport_chain(events: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    """Whether ``event`` and every explicit cause below it name only a lost verify call: an agent route failure or a
    transport kind, with no reply status. A verifier's own failure or an HTTP reply anywhere in the chain is not one."""
    from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED

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
    from nemo_gym.core.failure_kinds import AGENT_ROUTE_FAILED

    kind = row.get("failure_kind")
    events = row.get(FAILURE_EVENTS)
    if not (kind == AGENT_ROUTE_FAILED or _transport_kind(kind)) or not isinstance(events, Mapping):
        return False
    return any(
        isinstance(event, Mapping) and event.get("operation") == "verify" and _verify_transport_chain(events, event)
        for event in events.values()
    )


_FAILED_RESPONSE_STATUSES = frozenset({"failed", "incomplete", "cancelled"})
_CUT_FINISH = frozenset({"length", "content_filter", "max_tokens", "max_output_tokens"})


def _verifier_model_answered(row: Mapping[str, Any], verifiers: list[Any]) -> bool:
    """Whether one model call of the row's verifiers completed with a 2xx and a usable response, read from the call
    the row saved for each reference (by call id, else response id). A call the row does not hold has not answered."""
    refs = [ref for verifier in verifiers for ref in verifier.get("model_calls") or [] if isinstance(ref, Mapping)]
    trajectory = row.get("ng_trajectory")
    calls = trajectory.get("model_calls") if isinstance(trajectory, Mapping) else None
    if not refs or not isinstance(calls, list):
        return False
    by_id, by_response = {}, {}
    for call in calls:
        if isinstance(call, Mapping):
            metadata = call.get("response_metadata")
            by_id[call.get("model_call_id")] = call
            if isinstance(metadata, Mapping):
                by_response[metadata.get("response_id")] = call
    for ref in refs:
        call = by_id.get(ref.get("model_call_id")) or by_response.get(ref.get("response_id"))
        metadata = call.get("response_metadata") if call is not None else None
        if not isinstance(metadata, Mapping):
            continue
        code, status = metadata.get("status_code"), metadata.get("response_status")
        answered = 200 <= code < 300 if type(code) is int else status == "completed"
        # A chat reply carries no response status; a cut one says so in its finish reason.
        if answered and status not in _FAILED_RESPONSE_STATUSES and metadata.get("finish_reason") not in _CUT_FINISH:
            return True
    return False


def _judge_error_marker(row: Mapping[str, Any]) -> bool:
    """Whether the row names a judge error the server wrote: a judge failure kind, a saved judge error (a label or a
    top-level message) or an invalid judge response (top level or inside the rubric facts)."""
    facts = _grading_facts(row)
    rubric = facts.get("rubric")
    return (
        row.get("failure_kind") in JUDGE_FAILURE_KINDS
        or facts.get("judge_error") is True
        or facts.get("judge_error_message") is True
        or facts.get("invalid_judge_response") is True
        or (isinstance(rubric, Mapping) and rubric.get("invalid_judge_response") is True)
    )


def _grading_unrecorded(row: Mapping[str, Any]) -> bool:
    """Whether the row's verifier invocations were observed but recorded no grading: no grading operation, no
    verifier grading record, no saved grading artifact (``grading_artifact``) and no model-failure rule
    (``_model_failure``). A row from a pin without verifier observations proves nothing and is not selected. Nor is
    one whose verifier got an answer from a model (``_verifier_model_answered``) and wrote no judge-error marker
    (``_judge_error_marker``): a judge the verifier does not observe (``observe_llm_judge_grading``) records no
    grading however often it is asked again, and a grade it gave is never replaced. A row whose model calls all
    failed, or that names a judge error, was not graded."""
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, Mapping) else None
    if not isinstance(verifiers, list) or not verifiers:
        return False
    if any(
        not isinstance(verifier, Mapping) or verifier.get("grading_operations") or verifier.get("grading_record")
        for verifier in verifiers
    ):
        return False
    if _verifier_model_answered(row, verifiers) and not _judge_error_marker(row):
        return False
    return grading_artifact(row) is None and not _model_failure(row)


def _saved_model_patch(row: Mapping[str, Any]) -> bool:
    """Whether the row saved, intact, the workspace patch its verifier collected and graded: the collector's projected
    facts, else the same checks on a full saved row (``saved_model_patch``, ``collected_patch``). Only a verifier that
    collects the patch itself can be given it back; one that reads the agent session at verify time cannot."""
    facts = row.get("_ng_collected_grading_facts")
    if isinstance(facts, Mapping) and "saved_model_patch" in facts:
        return facts["saved_model_patch"] is True and facts.get("collected_patch") is True
    return (
        _collector_function("saved_model_patch")(row) is True and _collector_function("collected_patch")(row) is True
    )


def reverification_cause(row: Mapping[str, Any]) -> str | None:
    """Why a saved result row is re-verified from its saved answer, or None when it is not.

    ``REVERIFY_JUDGE``: masked because its judge failed (``judge_failed``, ``judge_unparseable``), or scored while
    its judge gave no verdict or its grading did not resolve (``_judge_unparseable``).
    ``REVERIFY_VERIFY_TRANSPORT``: masked because its verify call was lost in transport (``_verify_transport_failure``).
    ``REVERIFY_VERIFIER``: masked because its verifier produced no verdict (``verifier_error``), with the workspace
    patch it graded saved intact (``saved_model_patch``): the verifier runs again on that patch, never the agent.
    ``REVERIFY_UNGRADED``: scored, but no grading was recorded for it (``_grading_unrecorded``).
    Only a result with a saved answer qualifies; a failures-sidecar row, a result a re-verification already settled
    (its settlement is its grading) and every other masked result do not. ``row`` is a collector projection, or a
    saved row passed through ``saved_row_evidence``.
    """
    if (
        row.get(FAILURE_CLASS) is not None
        or row.get("_ng_omitted_rollout") is True
        or row.get("_ng_rejudge_operation_id") is not None
        or not _saved_answer(row)
    ):
        return None
    if row.get("mask_sample") is True:
        if row.get("failure_kind") in _REVERIFY_JUDGE_KINDS:
            return REVERIFY_JUDGE
        if row.get("failure_kind") == VERIFIER_ERROR and _saved_model_patch(row):
            return REVERIFY_VERIFIER
        return REVERIFY_VERIFY_TRANSPORT if _verify_transport_failure(row) else None
    if row.get("evaluation_completed") is False or _reward(row) is None:
        return None
    if _judge_unparseable(row):
        return REVERIFY_JUDGE
    return REVERIFY_UNGRADED if _grading_unrecorded(row) else None


def masked_without_artifact_kind(
    row: Mapping[str, Any], *, classification: FailureClassificationConfig | None = None
) -> str | None:
    """The kind of a masked result that is never measured nor run again, though infrastructure left it unmeasured; else None.

    A masked result ``masked_infrastructure_kind`` rejects only because it kept a saved answer (so every other rule of
    that predicate applies: the kind must be registered, evaluation-side and not graded again), that has no recorded
    grade (``has_recorded_grade``) and no re-verification cause (``reverification_cause``): its saved answer has no
    gradeable artifact, a resume keeps it and re-verification cannot grade it. It runs again only with the explicit
    ``--rerun-masked-without-artifact``. ``row`` is a collector projection, or a saved row passed through
    ``saved_row_evidence``.
    """
    from nemo_gym.rollouts.rollout_recovery import masked_kept_answer_kind

    kind = masked_kept_answer_kind(row, classification=classification)
    return kind if kind is not None and reverification_cause(row) is None else None


_COLLECTOR_FUNCTIONS: dict[str, Any] = {}


def _collector_function(name: str):
    """``COLLECT_SCRIPT``'s own projection ``name``, so an in-process reader projects a saved row as the collector does."""
    if not _COLLECTOR_FUNCTIONS:
        start, end = COLLECT_SCRIPT.index("\ndef finite("), COLLECT_SCRIPT.index("\ndef grading_result_binding(")
        namespace: dict[str, Any] = {"math": math}
        exec(COLLECT_SCRIPT[start:end], namespace)  # noqa: S102 - this module's own constant source
        _COLLECTOR_FUNCTIONS.update(namespace)
    return _COLLECTOR_FUNCTIONS[name]


def saved_row_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    """A full saved row with the evidence fields ``COLLECT_SCRIPT`` projects for the grading predicates: its scalar
    grading facts, whether each assistant message of its answer has text, and its verifier invocations."""
    evidence = dict(row)
    facts = _collector_function("grading_facts")(row)
    if any(value is not None and value is not False and value != [] for value in facts.values()):
        evidence["_ng_collected_grading_facts"] = facts
    answer = row.get("response")
    if isinstance(answer, Mapping) and isinstance(answer.get("output"), list):
        evidence["response"] = dict(answer) | {
            "output": [
                item | {"has_text": has_text}
                if isinstance(item, dict)
                and (has_text := _collector_function("answer_item")(item).get("has_text")) is not None
                else item
                for item in answer["output"]
            ]
        }
    trajectory = row.get("ng_trajectory")
    if isinstance(trajectory, Mapping):
        evidence["ng_trial_observations"] = {"verifier_invocations": trajectory.get("verifier_invocations")}
    return evidence
