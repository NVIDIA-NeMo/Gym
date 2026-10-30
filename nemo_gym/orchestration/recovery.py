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

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from nemo_gym.failure_kinds import JUDGE_FAILED, is_namespaced, is_registered


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


class RecoveryBudget(BaseModel):
    """The run's stored infra-retry and walltime budget and what its job chain used, as the chain files record them.

    ``max_infra_retries`` and ``max_walltime_s`` come from the benchmark's saved ``resume_config``;
    ``infra_retries_used`` from ``.gym_infra_retries`` and ``walltime_used_s`` from ``.gym_accumulated_walltime``
    (see ``resume_script``). The chain stops once the used retries exceed the cap, so one more is allowed while
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
        (e.get("boundary"), e.get("operation"), e.get("operation_id"), e.get("attempt_id"), e.get("cause_id"))
        for e in events.values()
        if isinstance(e, Mapping)
    ]


def decide_recovery(
    *,
    materialized: Iterable[Mapping[str, Any]],
    results: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    budget: RecoveryBudget,
    default_max_attempts: int = 3,
    resubmit_offset: int = 0,
) -> RecoveryDecision:
    """Classify every materialized identity and choose the run-level action.

    ``failures`` are every failures-sidecar row in file order (``_line`` is the row's line number when known). The
    standing row of a rollout is chosen as rollout collection chooses it (``_latest_failure_rows``): rows that
    ``is_non_superseding_rollout_row`` names are skipped, and an interrupted omitted row means the rollout is missing.
    Only rows at or after ``resubmit_offset`` count towards the attempt cap. The rows are only read.
    """
    from nemo_gym.rollout_collection import is_non_superseding_rollout_row, is_omitted_rollout_row, is_terminal_failure

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
            state = IdentityState.JUDGE_RECOVERABLE if judge else IdentityState.SCORED
            evidence.append(
                IdentityEvidence(
                    identity=identity,
                    state=state,
                    failure_kind=row.get("failure_kind") if judge else None,
                    failure_id=row.get(FAILURE_ID),
                    events=_events(row),
                )
            )
            continue
        standing = [row for row in sidecar.get(identity, []) if not is_non_superseding_rollout_row(row)]
        if not standing or is_omitted_rollout_row(standing[-1]):
            # No attempt ran, or the latest one was interrupted: the input restarts.
            evidence.append(IdentityEvidence(identity=identity, state=IdentityState.MISSING))
            continue
        last = standing[-1]
        rows = [
            row
            for index, row in enumerate(standing)
            if (row.get("_line") if isinstance(row.get("_line"), int) else index) >= resubmit_offset
        ]
        kind = last.get("failure_kind")
        cap = last.get(MAX_ATTEMPTS) if isinstance(last.get(MAX_ATTEMPTS), int) else default_max_attempts
        if last.get(FAILURE_CLASS) == JUDGE_FAILED:
            # A judge that failed is re-judged from the saved answer, never by running the policy again.
            state = IdentityState.JUDGE_RECOVERABLE
        elif last.get(COUNTS_AS_ZERO) is True:
            # Settled exactly as rollout collection settles it (_final_zero_failure_rows): marked by its producer and
            # final. A counts-as-zero row that is not final contradicts itself, so nobody owns it.
            state = IdentityState.MODEL_FINAL if is_terminal_failure(last) else IdentityState.UNKNOWN
        elif not _known_kind(kind):
            state = IdentityState.UNKNOWN
        elif last.get(TERMINAL) is True:
            state = IdentityState.TERMINAL
        elif len(rows) < cap:
            state = IdentityState.RETRYABLE
        else:
            state = IdentityState.EXHAUSTED
        evidence.append(
            IdentityEvidence(
                identity=identity,
                state=state,
                failure_kind=kind,
                failure_id=last.get(FAILURE_ID),
                attempts=len(rows),
                max_attempts=cap,
                events=_events(last),
            )
        )

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
            f"{counts[IdentityState.UNKNOWN]} failure(s) have no registered kind, so nobody owns them",
        )
    # A live job is still writing the files read above, whatever they show now.
    if budget.live_job:
        live = decision(RecoveryAction.STOP, "a job of this run is still pending or running; it is not queued twice")
        return live.model_copy(update={"scheduler_live": True})
    needs_policy = sum(
        counts.get(s, 0) for s in (IdentityState.MISSING, IdentityState.RETRYABLE, IdentityState.EXHAUSTED)
    )
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
        return decision(RecoveryAction.NOOP, "every identity is scored or a final model failure")
    if budget.live_job is None or budget.max_infra_retries is None or budget.infra_retries_used is None:
        return decision(
            RecoveryAction.STOP, "the job chain or scheduler state is unavailable, so the budget is unknown"
        )
    if budget.infra_retries_used >= budget.max_infra_retries:
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
COLLECT_SCRIPT = r"""
import json, math, os, sys
d = sys.argv[1]
a = os.path.join(d, "artifacts")
KEEP = ("_ng_task_index", "_ng_rollout_index", "_ng_attempt_index", "_ng_failure_class", "failure_kind",
        "_ng_failure_terminal", "_ng_failure_counts_as_zero", "_ng_failure_max_attempts", "_ng_failure_id",
        "_ng_failure_events", "_ng_omitted_rollout", "_ng_unrouted_run_error", "_ng_dispatch_drained", "_line", "reward",
        "mask_sample", "evaluation_completed")
def rows(name, skip=0):
    p = os.path.join(a, name)
    if not os.path.exists(p):
        return None
    out = []
    with open(p, "rb") as fh:
        for i, line in enumerate(fh):
            if i < skip or not line.strip():
                continue
            r = json.loads(line)
            if name == "rollouts_failures.jsonl":
                r["_line"] = i
            out.append({k: r[k] for k in KEEP if k in r})
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
chain = open(os.path.join(d, ".gym_job_chain")).read().split() if os.path.exists(os.path.join(d, ".gym_job_chain")) else []
mlflow = {"run_id_present": os.path.exists(os.path.join(a, "rollouts_mlflow_run_id")),
          "export_done": os.path.exists(os.path.join(a, "rollouts_mlflow_export_done"))}
quality = None
try:
    run = json.load(open(os.path.join(a, "quality_summary.json")))["run"]
    acc = run.get("artifacts") or {}
    quality = {"artifacts": {k: acc.get(k) for k in ("expected", "missing", "healthy", "unhealthy_reasons", "partial")},
               "verdicts": run.get("verdicts"), "issues": run.get("issues"), "coverage": acc.get("coverage"),
               "ignored_checks": run.get("ignored_checks")}
except (OSError, ValueError, KeyError, TypeError):
    pass
score = None
if len(sys.argv) > 2:
    try:
        value = json.load(open(os.path.join(a, "rollouts_aggregate_metrics.json")))
        for part in json.loads(sys.argv[2]):  # the metric path as a JSON list: keys may contain "/"
            value = value[part]
        score = value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
print(json.dumps({"materialized": rows("rollouts_materialized_inputs.jsonl"), "results": rows("rollouts.jsonl"),
                  "failures": rows("rollouts_failures.jsonl") or [], "resubmit_offset": offset, "infra_retries_used": num(".gym_infra_retries")[0],
                  "infra_retries_state": num(".gym_infra_retries")[1],
                  "walltime_used_s": num(".gym_accumulated_walltime")[0],
                  "walltime_state": num(".gym_accumulated_walltime")[1], "job_chain": chain,
                  "chain_incomplete": os.path.exists(os.path.join(d, ".gym_chain_incomplete")),
                  "quality": quality, "score": score, "mlflow": mlflow}))
"""


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
    quality: dict[str, Any] | None = None
    score: float | None = None
    mlflow: dict[str, bool] | None = None


def decide_from_collected(
    collected: CollectedEvidence,
    *,
    max_infra_retries: int | None,
    max_walltime_s: float | None,
    live_job: bool | None,
) -> RecoveryDecision:
    """Decide from what ``COLLECT_SCRIPT`` read; a missing inventory or result file stops the decision."""
    # resume_script creates .gym_infra_retries on the first infra retry, so its absence means none ran; it adds every
    # previous hop to .gym_accumulated_walltime, so after a follow-up hop that file must exist. Malformed is unknown.
    retries = {"absent": 0, "ok": collected.infra_retries_used}.get(collected.infra_retries_state)
    if collected.walltime_state == "ok":
        walltime = collected.walltime_used_s
    elif collected.walltime_state == "absent" and len(collected.job_chain) <= 1:
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
    if collected.materialized is None:
        return RecoveryDecision(
            action=RecoveryAction.STOP,
            reason="the run has no materialized inventory to recover against",
            expected=0,
            counts={},
            identities={},
            evidence=[],
        )
    return decide_recovery(
        materialized=collected.materialized,
        results=collected.results or [],
        failures=collected.failures,
        budget=budget,
        resubmit_offset=collected.resubmit_offset,
    )


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
        (IdentityState.RETRYABLE, "infra failures awaiting retry"),
        (IdentityState.EXHAUSTED, "infra failures with spent attempts"),
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
    artifacts = quality.get("artifacts") or {}
    fe_reasons = [str(r) for r in artifacts.get("unhealthy_reasons") or []]
    fe_valid = artifacts.get("healthy") if isinstance(artifacts.get("healthy"), bool) else None
    if artifacts.get("partial"):
        fe_valid, fe_reasons = None, fe_reasons + ["partial summary: per-record checks not evaluated"]
    if artifacts.get("expected") != decision.expected:
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


def mlflow_errors(local: dict[str, bool] | None, observed_status: str | None) -> list[str]:
    """MLflow evidence: the local run-id and export-done files, and the tracking server's status as the caller observed it.

    The files say a run exists and finished exporting; only the server knows its terminal status, so it is an input,
    never inferred. With a run, acceptance needs the export marker and an observed FINISHED; FINISHED alone accepts
    nothing, and FAILED, any other status or no observation is an error. A status for a run with no id contradicts.
    """
    local = local or {}
    if not local.get("run_id_present"):
        # No MLflow evidence is unknown, never a pass; a status without a run contradicts.
        if observed_status:
            return [f"MLflow status {observed_status!r} given for a run with no MLflow run id"]
        return ["no MLflow run evidence"]
    errors = [] if local.get("export_done") else ["MLflow run exists but its export did not finish"]
    if observed_status is None:
        errors.append("MLflow terminal status not observed")
    elif observed_status != "FINISHED":
        errors.append(f"MLflow run status {observed_status}")
    return errors


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
) -> dict[str, Any]:
    """The Day-0 gate task entry from the collected run: decision counts, the named score and the quality summary."""
    fe_valid, fe_reasons, rollout_health, rollout_reasons = quality_inputs(
        decision, collected, max_unhealthy_fraction=max_unhealthy_fraction
    )
    # The Day-0 gate's response and trial health need their own counters and denominators; quality_summary's
    # verdicts are per rollout, so they are reported in warnings, never used as either. Health stays unknown.
    health = None
    health_reasons = ["this adapter lacks separately validated ModelOpt response and trial health evidence"]
    task = gate_task_summary(
        decision,
        score=collected.score,
        fe_valid=fe_valid,
        fe_reasons=fe_reasons,
        health_passed=health,
        health_reasons=health_reasons,
    )
    judged = {True: "within", False: "beyond", None: "not judged against"}[rollout_health]
    task["warnings"] = (
        task["warnings"]
        + [f"per-rollout health {judged} the configured criterion: {rollout_reasons}"]
        + health_check_facts(collected.quality)
    )
    extra: list[str] = []
    if collected.chain_incomplete and not decision.complete:
        extra.append("the job chain stopped with rollouts missing (.gym_chain_incomplete)")
    extra.extend(mlflow_errors(collected.mlflow, mlflow_status))
    if extra:
        task["errors"] = task["errors"] + extra
        if task["status"] == "SUCCESS":
            task["status"] = "FAILED"
    return task
