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
"""The acceptance gate: response and trial health, metric coverage, MLflow delivery and the gate summary."""

import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.config.secret_utils import mask_url_passwords
from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    HARBOR_FAILED,
    INFRASTRUCTURE_ERROR,
    JUDGE_FAILED,
    JUDGE_UNPARSEABLE,
    ROLLOUT_ALLOCATION_ENDED,
    SERVER_START_FAILED,
    TRUNCATED_ANSWER,
    VERIFIER_ERROR,
)
from nemo_gym.eval.recovery.budgets import (
    retry_budget,
)
from nemo_gym.eval.recovery.decision import (
    COUNTS_AS_ZERO,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
)
from nemo_gym.eval.recovery.evidence import (
    READER_PLACEMENT_FIELDS,
    CollectedEvidence,
    Identity,
    _identity,
)
from nemo_gym.eval.recovery.grading import (
    _empty_answer,
    _grading_facts,
    _incomplete_answer,
    grading_artifact,
)
from nemo_gym.eval.recovery.rows import (
    omission_summary,
)
from nemo_gym.eval.recovery.trials import (
    GRADING_WITHOUT_VERDICT,
    UNVERIFIED_NOTE,
    _answer_fault,
    _count,
    _cut_off,
    _fault_text,
    _graded,
    _health_view,
    _observed_trial_operations,
    _percent,
    _predates_operation_observations,
    _recorded_grade,
    _recovered_note,
    _scored_fault,
    _trajectory_text,
    fault_report,
    final_zero_trials,
    infrastructure_timeouts,
    recovered_attempt_joins,
    trajectory_evidence,
    trial_boundary,
    trial_counts,
    trial_faults,
    trial_population,
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
    unverified = summary_valid is None or health_passed is None
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
    if score is None or not math.isfinite(score):
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
    # server fault (evaluation-side). Applied by the trial gate (``fault_class``).
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

    def report(self) -> dict[str, Any]:
        """The effective criterion and the accepted, unapplied fractions, as reported beside the gate results."""
        rule = "strictly below, before rounding"
        return {
            "evaluation": {
                "max_failure_fraction": GATE_FAILURE_LIMIT,
                "passes": rule,
                "policy_timeout_counts_as_model": self.policy_timeout_counts_as_model,
            },
            "response": {"max_failure_fraction": self.max_response_failure_fraction, "applied": False},
            "trial": {"max_failure_fraction": self.max_trial_failure_fraction, "applied": False},
        }


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


def trial_health(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None = None,
    *,
    limit: float = GATE_FAILURE_LIMIT,
    policy_timeout_counts_as_model: bool = True,
) -> tuple[bool | None, str]:
    """Evaluation health from the trial gate (``trial_verdict``): passed, failed or None, with its reason."""
    passed, reason, _ = trial_verdict(
        decision, collected, limit=limit, policy_timeout_counts_as_model=policy_timeout_counts_as_model
    )
    return passed, reason


def trial_verdict(
    decision: RecoveryDecision,
    collected: CollectedEvidence | None = None,
    *,
    limit: float = GATE_FAILURE_LIMIT,
    policy_timeout_counts_as_model: bool = True,
) -> tuple[bool | None, str, dict[str, Any] | None]:
    """Evaluation health: evaluation-side trial faults over the expected trials, including repeats.

    The counted failures are ``trial_faults``; a model fault (``fault_class``) is reported with its kind and never
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
    kinds = trial_faults(decision, collected)
    report = fault_report(kinds, trials, policy_timeout_counts_as_model=policy_timeout_counts_as_model)

    def verdict(passed: bool | None, reason: str, *, exact: bool = False):
        state = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}[passed]
        rate = report["evaluation"]["fraction"]
        return passed, reason, {"status": state, "evaluation_failure_rate": rate, "exact": exact} | report

    if report["evaluation"]["fraction"] >= limit:
        return verdict(False, f"trial: {_fault_text(report, limit)}")
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
    final_zero = final_zero_trials(decision, collected)
    infrastructure = infrastructure_timeouts(decision, collected)
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
        decision, collected, policy_timeout_counts_as_model=criteria.policy_timeout_counts_as_model
    )
    if trial_passed is None and decision.complete:
        trial_reason = f"{trial_reason} ({trial_boundary(collected.quality)})"
    task = gate_task_summary(
        decision,
        score=collected.score,
        summary_valid=summary_valid,
        summary_reasons=summary_reasons,
        health_passed=trial_passed,
        health_reasons=[] if trial_passed else [trial_reason],
    )
    task["criteria"] = criteria.report()
    if collected.unreadable_files:
        # What an unreadable metric, summary or record holds is unknown: never a pass.
        task["errors"].append(f"unreadable artifacts, read as unknown: {collected.unreadable_files[:20]}")
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
        + [f"per-rollout health {judged} the configured criterion: {rollout_reasons}"]
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
        extra.append("the job chain stopped with rollouts missing (.gym_chain_incomplete)")
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
    if task["mlflow_delivery"]["state"] in ("pending", "unverified"):
        task["warnings"] = task["warnings"] + [
            f"MLflow delivery {task['mlflow_delivery']['state']}: {'; '.join(task['mlflow_delivery']['reasons'])}"
        ]
    if extra:
        task["errors"] = task["errors"] + extra
        if task["status"] in ("SUCCESS", "UNVERIFIED"):
            task["status"] = "FAILED"
    return task


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
