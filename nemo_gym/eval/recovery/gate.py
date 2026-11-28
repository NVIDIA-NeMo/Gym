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
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.config.secret_utils import mask_url_passwords
from nemo_gym.core.failure_kinds import (
    AGENT_TIMEOUT_INFRASTRUCTURE,
    COLLECTION_RESUME_REFUSED,
    DRIVER_INSTALL_FAILED,
    ENVIRONMENT_PREPARE_FAILED,
    ENVIRONMENT_PROTOCOL_VIOLATION,
    GRADING_WITHOUT_VERDICT,
    HARBOR_FAILED,
    INFRASTRUCTURE_ERROR,
    JUDGE_FAILED,
    JUDGE_FAILURE_KINDS,
    JUDGE_UNPARSEABLE,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_QUOTA_EXHAUSTED,
    ROLLOUT_ALLOCATION_ENDED,
    SERVER_SPINUP_TIMEOUT,
    SERVER_START_FAILED,
    TRUNCATED_ANSWER,
    VERIFIER_ERROR,
    FailureClassificationConfig,
    quota_reset_text,
)
from nemo_gym.eval.recovery.budgets import (
    retry_budget,
)
from nemo_gym.eval.recovery.continuation import unsettled_chain_note
from nemo_gym.eval.recovery.decision import (
    _FAILED_JUDGE_JOBS,
    ACKNOWLEDGE_JUDGE_JOBS_FLAGS,
    COUNTS_AS_ZERO,
    JUDGE_JOB_FAILURE_BOUND,
    RERUN_MASKED_FLAGS,
    IdentityState,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
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
from nemo_gym.eval.recovery.planner import _EXPLICIT_BUDGET, MLFLOW_ROLLOUTS_NOT_UPLOADED, _continuation_options
from nemo_gym.eval.recovery.policy import _policy_evidence
from nemo_gym.eval.recovery.rows import (
    _RESERVATIONS_UNKNOWN,
    omission_summary,
)
from nemo_gym.eval.recovery.trials import (
    CUTOFF_REASONS,
    UNVERIFIED_NOTE,
    _answer_fault,
    _count,
    _cut_off,
    _fault_text,
    _graded,
    _health_classification,
    _health_view,
    _observed_trial_operations,
    _percent,
    _policy_call,
    _predates_operation_observations,
    _recorded_grade,
    _recovered_note,
    _scored_fault,
    _trajectory_text,
    decision_classification,
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
        "output_budget_refits": output_budget_refits(decision, collected),
        "repetition_stops": repetition_stops(decision, collected),
        "length_cutoffs": length_cutoffs(decision, collected),
        "agent_adaptations": agent_adaptations(decision, collected),
    }


def _deciding_rows(decision: RecoveryDecision, collected: CollectedEvidence | None) -> list[Mapping[str, Any]]:
    """The saved result row that decides each scored trial: the last one saved for its identity, so an attempt a later
    one superseded is not counted again."""
    scored = set(decision.identities.get(IdentityState.SCORED.value, []))
    latest: dict[Identity, Mapping[str, Any]] = {}
    for row in (collected.results or []) if collected is not None else []:
        if (identity := _identity(row)) in scored:
            latest[identity] = row
    return list(latest.values())


def _verifier_owned(row: Mapping[str, Any], call: Mapping[str, Any]) -> bool:
    """Whether one of the row's verifier invocations owns the captured call (a judge's call), matched by its call id,
    else by its model and response id: a judge served by the policy's own server is still the judge."""
    observed = row.get("ng_trial_observations")
    verifiers = observed.get("verifier_invocations") if isinstance(observed, Mapping) else None
    meta = call.get("response_metadata") if isinstance(call.get("response_metadata"), Mapping) else {}
    for verifier in verifiers if isinstance(verifiers, list) else []:
        owned = verifier.get("model_calls") if isinstance(verifier, Mapping) else None
        for ref in owned if isinstance(owned, list) else []:
            if not isinstance(ref, Mapping):
                continue
            if ref.get("model_call_id"):
                if ref["model_call_id"] == call.get("model_call_id"):
                    return True
            elif ref.get("response_id") and (ref.get("response_id"), ref.get("model_ref")) == (
                meta.get("response_id"),
                meta.get("model_ref"),
            ):
                return True
    return False


def output_budget_refits(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, int]:
    """How many model calls in the rows that decide the scored trials were sent again with an output budget lowered
    to fit the context window (``output_budget_refit`` in a call's response metadata), and in how many of those rows.

    A count, never a failure: the call generated within the window it was served at. A reader flags a cell whose
    requests were budget-capped from it.
    """
    count = rows = 0
    for row in _deciding_rows(decision, collected):
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


def repetition_stops(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, int]:
    """How many of the policy's model calls in the rows that decide the scored trials the provider stopped for
    repetition, and in how many of those rows: a ``repetition`` finish reason or a ``repetition_detected`` stop reason
    (vLLM's repetition detection), as the evidence collector flags each call.

    The model's own ending, never a failure: the answer it produced is scored as it is and nothing is retried, so the
    count only makes loops visible. A call a verifier owns (a judge's, also when it shares the policy's server) or
    another model's call is not counted, nor is an attempt a later one superseded.
    """
    count = rows = 0
    for row in _deciding_rows(decision, collected):
        observed = row.get("ng_trial_observations")
        calls = observed.get("model_calls") if isinstance(observed, dict) else None
        stops = sum(
            isinstance(call, dict)
            and call.get("repetition_stop") is True
            and _policy_call(row, call)
            and not _verifier_owned(row, call)
            for call in calls or []
        )
        count += stops
        rows += stops > 0
    return {"count": count, "rows": rows}


def length_cutoffs(decision: RecoveryDecision, collected: CollectedEvidence | None) -> dict[str, int]:
    """How many of the policy's model calls in the rows that decide the scored trials ended on a length cutoff, and in
    how many of those rows: a ``length`` finish reason, or its ``max_output_tokens`` and ``max_tokens`` spellings, as
    the evidence collector records each call.

    A fact, never a failure: the model server returned the shortened answer and the rollout carried on from it, so
    a run that lost output to the context window or the output budget does not read as clean. The finish reason does
    not tell the two limits apart. A call a verifier owns (a judge's, also when it shares the policy's server) or
    another model's call is not counted, nor is an attempt a later one superseded.
    """
    count = rows = 0
    for row in _deciding_rows(decision, collected):
        observed = row.get("ng_trial_observations")
        calls = observed.get("model_calls") if isinstance(observed, dict) else None
        cuts = sum(
            isinstance(call, dict)
            and isinstance(call.get("response_metadata"), dict)
            and call["response_metadata"].get("finish_reason") in CUTOFF_REASONS
            and _policy_call(row, call)
            and not _verifier_owned(row, call)
            for call in calls or []
        )
        count += cuts
        rows += cuts > 0
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


def subset_note(selection: Mapping[str, Any]) -> str:
    """The label of a run that selected its tasks by id: its score is the subset's, never the benchmark's."""
    if selection.get("task_ids_sha256") is None:
        return "subset run: the task selection record is unreadable; its score is a subset score, not the benchmark's"
    return (
        f"subset run: {selection['tasks']} tasks ({selection['rows']} rows) selected by task id from "
        f"{selection['source']} (task_ids sha256 {selection['task_ids_sha256'][:12]}); its score is a subset score, "
        "report it under the subset's name, never as the benchmark's"
    )


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
    stops = facts.get("repetition_stops")
    if stops and stops["count"]:
        text += (
            f"; repetition stops: {stops['count']} policy calls in {stops['rows']} trials (model, scored as answered)"
        )
    cuts = facts.get("length_cutoffs")
    if cuts and cuts["count"]:
        text += f"; length cutoffs: {cuts['count']} policy calls in {cuts['rows']} trials"
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
    task["task_selection"] = collected.task_selection
    if collected.task_selection is not None:
        task["warnings"] = [subset_note(collected.task_selection)] + task["warnings"]
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
