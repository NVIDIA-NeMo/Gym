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

import getpass
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import SchedulerRefusedError
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.resume_script import fresh_restart_command
from nemo_gym.orchestration.executors.slurm import SlurmExecutor, reader_token
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, load_submission_record
from nemo_gym.orchestration.recovery import (
    MLFLOW_NOT_REQUIRED_NOTE,
    GateCriteria,
    OperatorBudget,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryOutcomePolicyConfig,
    gate_summary_from_collected,
    gate_task_summary,
    mlflow_tracking_required,
)
from nemo_gym.utils.decorators import experimental


_EXECUTORS = {
    SlurmComputeConfig: SlurmExecutor,
    KubernetesComputeConfig: KubernetesExecutor,
}
# A record names its executor as a string, so a resubmit dispatches on that.
_EXECUTORS_BY_NAME = {executor.name: executor for executor in _EXECUTORS.values()}


@experimental
def submit(config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
    compute = next(iter(config.compute.values()))
    executor_cls = _EXECUTORS[type(compute)]

    if not executor_cls.supports_resumable:
        unsupported = [name for name, b in config.driver.benchmarks.items() if b.resume_config is not None]
        if unsupported:
            raise ValueError(
                f"Benchmark(s) {', '.join(unsupported)} set `resumable`, but {executor_cls.__name__} "
                "does not support resumable runs."
            )

    return executor_cls().run(config, dry_run=dry_run)


@experimental
def resubmit(gym_job_id: str, benchmark: str) -> BenchmarkJob:
    """Queue `benchmark` of submission `gym_job_id` again, in place, and return its updated record entry."""
    record = load_submission_record(gym_job_id)
    executor_cls = _EXECUTORS_BY_NAME.get(record.executor)
    if executor_cls is None:
        raise ValueError(f"Submission {gym_job_id} was made by unknown executor {record.executor!r}.")
    return executor_cls().resubmit(record, benchmark)


@experimental
def recover(
    gym_job_id: str,
    benchmark: str,
    *,
    score_key: list[str | int] | None = None,
    score_agent: str | None = None,
    score_environment: str | None = None,
    max_unhealthy_fraction: float | None = None,
    max_response_failure_fraction: float | None = None,
    max_trial_failure_fraction: float | None = None,
    policy_timeout_counts_as_model: bool | None = None,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    repin: str | None = None,
    repin_reason: str | None = None,
    extend_retries: int = 0,
    extend_attempts: int = 0,
    max_walltime: str | None = None,
    walltime_reason: str | None = None,
    unsafe_resume_reason: str | None = None,
    server_spinup_timeout_seconds: int | None = None,
    spinup_timeout_reason: str | None = None,
    judge_partition: str | None = None,
    judge_qos: str | None = None,
    auto_repin: bool = True,
    rerun_masked_without_artifact: bool = False,
    rerun_reason: str | None = None,
    acknowledge_failed_judge_jobs: bool = False,
    acknowledge_reason: str | None = None,
) -> RecoveryOutcome:
    """Recover a saved submission, preserving its recorded retry and walltime budgets.

    A run pinned at another Gym commit than this launcher's is planned with the installed Gym's rules. When that plan
    needs the installed revision to run (a re-judge, whose job runs the installed recovery source, or a resubmit the
    pinned revision plans differently), the run moves to the installed commit, journaled as an automatic repin with
    the reason; otherwise its pin stays. ``auto_repin=False`` refuses that case instead, naming ``repin``.

    ``repin`` (with ``repin_reason``) continues the run on another full Gym commit, which must be this launcher's; a
    judge-only job of the run runs at it too.
    ``judge_partition`` names the CPU partition of the judge-only job that grades a model-served run's saved answers
    again, when its saved config declares no CPU node pool and the launcher's reader placement names none, and
    ``judge_qos`` that job's QOS (else the saved CPU pool's, the reader placement's or the partition's default);
    neither is used for a run that serves its judge on a node pool of its own, whose job runs on that saved pool.
    ``extend_retries`` records that many more infra retries for the run: explicit and journaled, never a reset.
    ``extend_attempts`` records that many more attempts for every rollout the Store would run again (not a model
    failure); explicit, journaled with the infra extension, and named in the decision.
    ``max_walltime`` (with ``walltime_reason``) raises the run's Slurm walltime, never below the recorded one.
    ``unsafe_resume_reason`` resumes the job with ``allow_unsafe_resume``, past a refused run identity check; the
    reason and the user are journaled.
    ``server_spinup_timeout_seconds`` (with ``spinup_timeout_reason``) raises how long the run's Gym servers may
    take to become ready, never below the recorded limit; 0 removes it. The raise is journaled in the job dir.
    For a run whose chain accounting is unknown, the two are also an operator's explicit budget: the run continues
    as an OPERATOR-CONTINUATION with ``extend_retries`` (one when not given) infra retries remaining, recorded in the
    decision and the job dir's operator journal (see ``decide_from_collected``).
    ``rerun_masked_without_artifact`` (with a nonempty ``rerun_reason``) runs again the masked results that kept a saved
    answer but have neither a gradeable artifact nor a recorded grade, which a resume otherwise keeps and
    re-verification cannot grade: the identities the plan finds qualifying are journaled with the reason in the run's
    artifacts when the job is queued, and each counts one failed attempt within its cap.
    ``acknowledge_failed_judge_jobs`` (with a nonempty ``acknowledge_reason``) lets one more judge-only job queue when the
    last two ended without settling a saved answer; the acknowledgement, naming those jobs, is journaled with the reason
    when the new job is queued, and covers only them.
    The outcome names the ``reader_token`` its login-host Store readers carry.
    """
    outcome = _recover(
        gym_job_id,
        benchmark,
        score_key=score_key,
        score_agent=score_agent,
        score_environment=score_environment,
        max_unhealthy_fraction=max_unhealthy_fraction,
        max_response_failure_fraction=max_response_failure_fraction,
        max_trial_failure_fraction=max_trial_failure_fraction,
        policy_timeout_counts_as_model=policy_timeout_counts_as_model,
        outcome_policy=outcome_policy,
        mlflow_status=mlflow_status,
        mlflow_run_id=mlflow_run_id,
        repin=repin,
        repin_reason=repin_reason,
        extend_retries=extend_retries,
        extend_attempts=extend_attempts,
        max_walltime=max_walltime,
        walltime_reason=walltime_reason,
        unsafe_resume_reason=unsafe_resume_reason,
        server_spinup_timeout_seconds=server_spinup_timeout_seconds,
        spinup_timeout_reason=spinup_timeout_reason,
        judge_partition=judge_partition,
        judge_qos=judge_qos,
        auto_repin=auto_repin,
        rerun_masked_without_artifact=rerun_masked_without_artifact,
        rerun_reason=rerun_reason,
        acknowledge_failed_judge_jobs=acknowledge_failed_judge_jobs,
        acknowledge_reason=acknowledge_reason,
    )
    return outcome.model_copy(update={"reader_token": reader_token()})


def inspect_recovery(
    gym_job_id: str,
    benchmark: str,
    *,
    original_record: SubmissionRecord | None = None,
    score_key: list[str | int] | None = None,
    score_agent: str | None = None,
    score_environment: str | None = None,
    max_unhealthy_fraction: float | None = None,
    max_response_failure_fraction: float | None = None,
    max_trial_failure_fraction: float | None = None,
    policy_timeout_counts_as_model: bool | None = None,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    extend_retries: int = 0,
    extend_attempts: int = 0,
    repin: str | None = None,
    repin_reason: str | None = None,
    auto_repin: bool = True,
    rerun_masked_without_artifact: bool = False,
    rerun_reason: str | None = None,
    acknowledge_failed_judge_jobs: bool = False,
    acknowledge_reason: str | None = None,
) -> RecoveryOutcome:
    """Inspect the same saved evidence and acceptance gate as recovery, without dispatching any action.

    It plans as recovery would, on the installed Gym, and reports the same pin decision (``pin_plan``).

    With ``repin`` it reports the plan a recovery repinned to that commit would follow: the run's Store is read at the
    target commit, as the repinned resume would read it, and nothing is rendered, repinned or queued.

    The masked results that kept a saved answer but have no gradeable artifact and no grade are named in the decision
    (``masked_without_artifact``) and stay masked, with or without ``rerun_masked_without_artifact``: inspect never
    journals an approval.
    """
    outcome = _recover(
        gym_job_id,
        benchmark,
        inspect_only=True,
        **({"repin": repin, "repin_reason": repin_reason} if repin is not None else {}),
        original_record=original_record,
        extend_retries=extend_retries,
        extend_attempts=extend_attempts,
        score_key=score_key,
        score_agent=score_agent,
        score_environment=score_environment,
        max_unhealthy_fraction=max_unhealthy_fraction,
        max_response_failure_fraction=max_response_failure_fraction,
        max_trial_failure_fraction=max_trial_failure_fraction,
        policy_timeout_counts_as_model=policy_timeout_counts_as_model,
        outcome_policy=outcome_policy,
        mlflow_status=mlflow_status,
        mlflow_run_id=mlflow_run_id,
        auto_repin=auto_repin,
        rerun_masked_without_artifact=rerun_masked_without_artifact,
        rerun_reason=rerun_reason,
        acknowledge_failed_judge_jobs=acknowledge_failed_judge_jobs,
        acknowledge_reason=acknowledge_reason,
    )
    return outcome.model_copy(update={"reader_token": reader_token()})


def _spinup_timeout_raise(seconds: int | None, reason: str | None) -> dict[str, Any]:
    """The executor keywords of a spin-up timeout raise, none without one, so executors that predate it work."""
    if seconds is None:
        return {}
    return {"server_spinup_timeout_seconds": seconds, "spinup_timeout_reason": reason}


def _recover(
    gym_job_id: str,
    benchmark: str,
    *,
    inspect_only: bool = False,
    original_record: SubmissionRecord | None = None,
    score_key: list[str | int] | None = None,
    score_agent: str | None = None,
    score_environment: str | None = None,
    max_unhealthy_fraction: float | None = None,
    max_response_failure_fraction: float | None = None,
    max_trial_failure_fraction: float | None = None,
    policy_timeout_counts_as_model: bool | None = None,
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    repin: str | None = None,
    repin_reason: str | None = None,
    extend_retries: int = 0,
    extend_attempts: int = 0,
    max_walltime: str | None = None,
    walltime_reason: str | None = None,
    unsafe_resume_reason: str | None = None,
    server_spinup_timeout_seconds: int | None = None,
    spinup_timeout_reason: str | None = None,
    judge_partition: str | None = None,
    judge_qos: str | None = None,
    auto_repin: bool = True,
    rerun_masked_without_artifact: bool = False,
    rerun_reason: str | None = None,
    acknowledge_failed_judge_jobs: bool = False,
    acknowledge_reason: str | None = None,
) -> RecoveryOutcome:
    """Recover `benchmark` of submission `gym_job_id` from its persisted evidence (`gym eval resubmit --recover`).

    The executor collects identities, results, failures, the quality summary, the ``score_key`` metric and the job
    chain's stored budget; ``decide_recovery`` picks one action, which runs here: ``noop`` dispatches nothing;
    ``reverify`` re-judges saved answers through the executor (never the policy) or reports that the executor cannot;
    ``resubmit`` queues the run with the existing ``resubmit`` keeping its spent budget (retries, walltime and
    per-rollout attempts), and its Gym resume runs only missing and retryable identities; ``fresh-restart`` (no task
    inventory) queues the run again as a fresh run with the explicit, longer ``max_walltime``, and without one names
    the command; ``stop`` dispatches nothing and says why. Every outcome carries the acceptance gate task entry:
    per-rollout health is judged against ``max_unhealthy_fraction`` and reported in its warnings; response/trial
    health is evaluated separately, and MLflow delivery is checked by ``mlflow_errors``. Plain ``resubmit`` still
    starts a fresh retry and walltime budget.
    """
    if score_key is not None and (
        not isinstance(score_key, list)
        or not score_key
        or not all(isinstance(key, str) or (type(key) is int and key >= 0) for key in score_key)
    ):
        raise ValueError("score_key must be a nonempty list of object keys and nonnegative array indices")
    if rerun_masked_without_artifact and not (rerun_reason or "").strip():
        raise ValueError("--rerun-masked-without-artifact needs a nonempty --rerun-reason")
    if rerun_reason is not None and not rerun_masked_without_artifact:
        raise ValueError("--rerun-reason goes with --rerun-masked-without-artifact")
    if acknowledge_failed_judge_jobs and not (acknowledge_reason or "").strip():
        raise ValueError("--acknowledge-failed-judge-jobs needs a nonempty --acknowledge-reason")
    if acknowledge_reason is not None and not acknowledge_failed_judge_jobs:
        raise ValueError("--acknowledge-reason goes with --acknowledge-failed-judge-jobs")
    if max_unhealthy_fraction is not None and (
        isinstance(max_unhealthy_fraction, bool)
        or not isinstance(max_unhealthy_fraction, (int, float))
        or not math.isfinite(max_unhealthy_fraction)
        or not 0 < max_unhealthy_fraction <= 1
    ):
        raise ValueError("invalid response-health criterion: expected a finite number in (0, 1]")
    limits = {"max_response_failure_fraction": max_response_failure_fraction}
    limits["max_trial_failure_fraction"] = max_trial_failure_fraction
    if policy_timeout_counts_as_model is not None and not isinstance(policy_timeout_counts_as_model, bool):
        raise ValueError("invalid gate criteria: policy_timeout_counts_as_model expected true or false")
    limits["policy_timeout_counts_as_model"] = policy_timeout_counts_as_model
    try:
        criteria = GateCriteria(**{name: value for name, value in limits.items() if value is not None})
    except ValidationError as error:
        raise ValueError(f"invalid gate criteria: {error.errors()[0]['loc'][0]} expected a finite number in (0, 1]")
    for selector in (score_agent, score_environment):
        if selector is not None and (not isinstance(selector, str) or not selector):
            raise ValueError("score identity selectors must be nonempty strings")
    if (score_agent is not None or score_environment is not None) and not score_key:
        raise ValueError("score identity selectors require a metric path")
    record = original_record if original_record is not None else load_submission_record(gym_job_id)
    if original_record is not None:
        if record.gym_job_id != gym_job_id:
            raise ValueError("The supplied record belongs to another submission")
        record.benchmark_job(benchmark)
    executor_cls = _EXECUTORS_BY_NAME.get(record.executor)
    if executor_cls is None:
        raise ValueError(f"Submission {gym_job_id} was made by unknown executor {record.executor!r}.")
    executor = executor_cls()
    # Used only when the chain accounting is unknown: the walltime with its reason and N retries remaining from now.
    operator_budget = (
        OperatorBudget(
            max_walltime=max_walltime,
            reason=walltime_reason,
            retries=extend_retries or 1,
            user=getpass.getuser(),
        )
        if max_walltime is not None and (walltime_reason or "").strip() and not inspect_only
        else None
    )
    # A walltime raise the partition cannot take is refused at once, before the slow evidence read.
    plan_walltime = getattr(executor, "walltime_plan", None)
    walltime_plan = (
        plan_walltime(record, benchmark, max_walltime)
        if max_walltime is not None and not inspect_only and plan_walltime is not None
        else None
    )
    if walltime_plan is not None and walltime_plan.get("refused"):
        refused = RecoveryDecision(
            action=RecoveryAction.STOP,
            reason=f"walltime raise refused: {walltime_plan['refused']}",
            expected=0,
            counts={},
            identities={},
            evidence=[],
        )
        return RecoveryOutcome(decision=refused, ok=False, dispatched=None, walltime_plan=walltime_plan)
    segment = {"segment_walltime": walltime_plan["segment"]} if walltime_plan is not None else {}

    def collect(reader_ref: str | None):
        decision, collected = executor.recovery_evidence(
            record,
            benchmark,
            score_key=score_key,
            score_agent=score_agent,
            score_environment=score_environment,
            outcome_policy=outcome_policy,
            # A repinned run resumes on the target's Store: its reader decides what that Store would run.
            **({"reader_ref": reader_ref} if reader_ref is not None else {}),
            **({"extend_retries": extend_retries} if extend_retries else {}),
            **({"extend_attempts": extend_attempts} if extend_attempts else {}),
            # Inspect only reports the masked results the flag would reopen: it plans none of them as retries.
            **({"rerun_masked_without_artifact": True} if rerun_masked_without_artifact and not inspect_only else {}),
            # A walltime raise renders a larger chain cap: the plan checks the cap the resubmit will render.
            **({"max_walltime": max_walltime} if max_walltime is not None else {}),
            **({"operator_budget": operator_budget} if operator_budget is not None else {}),
            # The decision names a spin-up timeout raise the resubmit will render.
            **(
                spinup
                if (spinup := _spinup_timeout_raise(server_spinup_timeout_seconds, spinup_timeout_reason))
                else {}
            ),
        )
        if decision.action == RecoveryAction.FRESH_RESTART and max_walltime is None:
            command = fresh_restart_command(gym_job_id, benchmark)
            decision = decision.model_copy(
                update={"reason": f"{decision.reason}; --max-walltime is required: {command}"}
            )
        if decision.action == RecoveryAction.UNKNOWN:
            command = f"gym eval resubmit {gym_job_id} {benchmark} --recover --inspect"
            decision = decision.model_copy(update={"reason": f"{decision.reason}: {command}"})
        gate_task = (
            gate_summary_from_collected(
                decision,
                collected,
                max_unhealthy_fraction=max_unhealthy_fraction,
                mlflow_status=mlflow_status,
                mlflow_run_id=mlflow_run_id,
                criteria=criteria,
            )
            if collected is not None
            else gate_task_summary(
                decision,
                score=None,
                summary_valid=None,
                summary_reasons=["no evidence collected"],
                health_passed=None,
                health_reasons=["no evidence collected"],
            )
        )
        mlflow = {
            "local": collected.mlflow if collected is not None else None,
            "observed_status": mlflow_status,
            "observed_run_id": mlflow_run_id,
            "tracking_required": mlflow_tracking_required(collected),
        }
        if not mlflow["tracking_required"]:
            mlflow["note"] = MLFLOW_NOT_REQUIRED_NOTE
        return decision, collected, gate_task, mlflow

    # A run pinned at another commit is planned with the installed Gym's rules; its pin moves only when it must.
    pinned = executor.gym_pin(record) if repin is None and hasattr(executor, "gym_pin") else None
    installed = executor.installed_commit() if pinned is not None else None
    pin_plan: dict[str, Any] | None = None
    if (
        pinned is not None
        and installed is not None
        and re.fullmatch(r"[0-9a-f]{40}", installed)
        and installed != pinned
    ):
        decision, collected, gate_task, mlflow = collect(installed)
        planned = f"planned on {installed} for a run pinned at {pinned}"
        needs = None
        if decision.action == RecoveryAction.REVERIFY:
            needs = "its judge-only job runs the installed recovery source"
        elif decision.action == RecoveryAction.RESUBMIT:
            at_pin = collect(None)[0]
            if (at_pin.action, at_pin.counts) != (decision.action, decision.counts):
                needs = (
                    f"the pinned revision plans {at_pin.action.value} {at_pin.counts}, the installed one "
                    f"{decision.action.value} {decision.counts}"
                )
        if needs is None:
            pin_plan = {
                "installed": installed,
                "pinned": pinned,
                "repin": False,
                "reason": f"{planned}; the pin stays",
            }
        elif not auto_repin:
            pin_plan = {
                "installed": installed,
                "pinned": pinned,
                "repin": False,
                "refused": True,
                "reason": f"{planned}: {needs}; the pin was kept (--no-auto-repin): recover with --repin {installed} "
                "--repin-reason TEXT to move it",
            }
        else:
            repin, repin_reason = installed, f"automatic repin: {planned}: {needs}"
            pin_plan = {"installed": installed, "pinned": pinned, "repin": True, "reason": repin_reason}
        gate_task = {**gate_task, "warnings": [*gate_task["warnings"], pin_plan["reason"]]}
        plan_ref = installed
    else:
        decision, collected, gate_task, mlflow = collect(repin)
        plan_ref = repin
    if pin_plan is not None and pin_plan.get("refused") and not inspect_only:
        refused = decision.model_copy(update={"action": RecoveryAction.STOP, "reason": pin_plan["reason"]})
        return RecoveryOutcome(
            decision=refused,
            ok=False,
            dispatched=None,
            gate_task={**gate_task, "status": "FAILED", "errors": [*gate_task["errors"], pin_plan["reason"]]},
            mlflow=mlflow,
            pin_plan=pin_plan,
        )
    if inspect_only:
        if gate_task["status"] == "PLANNED":
            gate_task = {
                **gate_task,
                "warnings": [
                    *gate_task["warnings"],
                    f"PLANNED (NOT DISPATCHED): --inspect reports the {decision.action.value} decision and runs nothing",
                ],
            }
        if repin is not None and pin_plan is None:
            # The plan was read at the repin target, as a repinned recovery would read it; nothing was repinned.
            gate_task = {
                **gate_task,
                "warnings": [
                    *gate_task["warnings"],
                    f"INSPECTED AT REPIN {repin}: the plan of a recovery repinned to this commit; the run is not "
                    "repinned and nothing was dispatched",
                ],
            }
        return RecoveryOutcome(
            decision=decision,
            ok=decision.action not in (RecoveryAction.STOP, RecoveryAction.UNKNOWN),
            dispatched=None,
            gate_task=gate_task,
            mlflow=mlflow,
            pin_plan=pin_plan,
        )
    if decision.action == RecoveryAction.NOOP and _tracking_export_due(decision, collected, mlflow):
        export = executor.export_tracking(record, benchmark, collected)
        if export.get("state") == "unsupported":
            reason = f"MLflow post-run export not performed: {export.get('error')}"
        elif export.get("state") in ("succeeded", "skipped"):
            # The receipt changes the MLflow evidence and the gate: collect both again, as a later recovery would.
            decision, collected, gate_task, mlflow = collect(plan_ref)
            reason = f"MLflow post-run export performed: receipt written for run {export.get('run_id')}"
        else:
            reason = f"MLflow post-run export failed: {export.get('error')}"
        return RecoveryOutcome(
            decision=decision.model_copy(update={"reason": f"{decision.reason}; {reason}"}),
            ok=decision.action == RecoveryAction.NOOP and export.get("state") in ("succeeded", "skipped"),
            dispatched=None if export.get("state") == "unsupported" else "export",
            gate_task=gate_task,
            mlflow={**mlflow, "export": export},
        )
    if decision.action == RecoveryAction.NOOP:
        return RecoveryOutcome(decision=decision, ok=True, dispatched=None, gate_task=gate_task, mlflow=mlflow)
    if decision.action == RecoveryAction.REVERIFY:
        # Each option only when given, so an executor that predates it is called as before.
        options: dict[str, Any] = {}
        if judge_partition is not None:
            options["judge_partition"] = judge_partition
        if judge_qos is not None:
            options["judge_qos"] = judge_qos
        if repin is not None:
            options |= {"repin": repin, "repin_reason": repin_reason}
        if acknowledge_failed_judge_jobs and not inspect_only:
            options |= {
                "acknowledge_failed_judge_jobs": True,
                "acknowledge_reason": (acknowledge_reason or "").strip(),
            }
        outcome = executor.reverify_judge_failures(record, benchmark, decision, **options)
        if not outcome.ok or outcome.dispatched is None:
            gate_task = {
                **gate_task,
                "status": "FAILED",
                "errors": [*gate_task["errors"], outcome.unsupported or "saved-answer rejudgment was not dispatched"],
            }
        elif gate_task["status"] == "PLANNED":
            gate_task = {**gate_task, "status": "RESUMING"}
        if outcome.judge_job is not None:
            # The job just queued joins the run's judge-only jobs the evidence already reports.
            gate_task = {**gate_task, "judge_jobs": [*(gate_task.get("judge_jobs") or []), outcome.judge_job]}
        return outcome.model_copy(update={"gate_task": gate_task, "mlflow": mlflow, "pin_plan": pin_plan})
    if decision.action == RecoveryAction.RESUBMIT:
        operator = decision.operator_continuation
        try:
            job = executor.resubmit(
                record,
                benchmark,
                keep_budget=True,
                **(
                    {"ended_allocation_manifest": decision.ended_allocation_manifest}
                    if decision.ended_allocation_manifest
                    else {}
                ),
                accounted_walltime_s=collected.accounted_walltime_s if collected is not None else None,
                accounting_snapshot=collected.accounting_snapshot if collected is not None else None,
                # Only a reconciled chain passes the flag, so executors without requeue reconciliation keep working.
                **(
                    {"accounting_reconciled": True}
                    if collected is not None and collected.accounting_reconciled
                    else {}
                ),
                **({"repin": repin, "repin_reason": repin_reason} if repin is not None else {}),
                # Under an operator continuation, N is the allowance its record fills, not a recorded extension.
                **({"extend_retries": extend_retries} if extend_retries and operator is None else {}),
                **({"extend_attempts": extend_attempts} if extend_attempts else {}),
                # The approval of the masked results this plan runs again, journaled once the job is queued.
                **(
                    {
                        "rerun_masked": {
                            "reason": (rerun_reason or "").strip(),
                            "identities": [
                                {"identity": list(item.identity), "attempt": item.attempt, "attempts": item.attempts}
                                for item in decision.rerun_masked
                            ],
                        }
                    }
                    if decision.rerun_masked
                    else {}
                ),
                **(
                    {"max_walltime": max_walltime, "walltime_reason": walltime_reason}
                    if max_walltime is not None
                    else {}
                ),
                **({"unsafe_resume_reason": unsafe_resume_reason} if unsafe_resume_reason is not None else {}),
                **_spinup_timeout_raise(server_spinup_timeout_seconds, spinup_timeout_reason),
                **({"operator_continuation": operator} if operator is not None else {}),
                **segment,
            )
        except SchedulerRefusedError as error:
            return _refused(decision, gate_task, mlflow, walltime_plan, error)
        if gate_task["status"] == "PLANNED":
            gate_task = {**gate_task, "status": "RESUMING"}
        return RecoveryOutcome(
            decision=decision,
            ok=True,
            dispatched="resubmit",
            job_id=job.job_id,
            gate_task=gate_task,
            mlflow=mlflow,
            walltime_plan=walltime_plan,
            reverify=_reverify_first_note(decision, job.job_id),
            pin_plan=pin_plan,
        )
    if decision.action == RecoveryAction.FRESH_RESTART and max_walltime is not None:
        if repin is not None or extend_retries or extend_attempts:
            raise ValueError("a fresh restart takes only --max-walltime and --walltime-reason")
        try:
            job = executor.resubmit(
                record,
                benchmark,
                fresh_restart=True,
                accounting_snapshot=collected.accounting_snapshot if collected is not None else None,
                max_walltime=max_walltime,
                walltime_reason=walltime_reason,
                **_spinup_timeout_raise(server_spinup_timeout_seconds, spinup_timeout_reason),
                **segment,
            )
        except SchedulerRefusedError as error:
            return _refused(decision, gate_task, mlflow, walltime_plan, error)
        if gate_task["status"] == "PLANNED":
            gate_task = {**gate_task, "status": "RESUMING"}
        return RecoveryOutcome(
            decision=decision,
            ok=True,
            dispatched=RecoveryAction.FRESH_RESTART.value,
            job_id=job.job_id,
            gate_task=gate_task,
            mlflow=mlflow,
            walltime_plan=walltime_plan,
        )
    return RecoveryOutcome(decision=decision, ok=False, dispatched=None, gate_task=gate_task, mlflow=mlflow)


def _tracking_export_due(decision: Any, collected: Any, mlflow: dict[str, Any]) -> bool:
    """Whether a complete run lacks the MLflow export receipt its saved config requires.

    Only a run whose saved config configures MLflow (tracking URI, experiment and run name) and does not set
    ``mlflow_tracking_required`` to false, with no post-run receipt at all and no live job, is exported. A run that
    configured no exporter is never exported, and a receipt, current or not, is never replaced.
    """
    if collected is None or not decision.complete or decision.scheduler_live or not mlflow["tracking_required"]:
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


def _refused(
    decision: RecoveryDecision,
    gate_task: dict[str, Any],
    mlflow: dict[str, Any],
    walltime_plan: dict[str, Any] | None,
    error: SchedulerRefusedError,
) -> RecoveryOutcome:
    """The outcome of a resubmit the scheduler refused: nothing queued, its text verbatim, the job dir restored."""
    restored = ", ".join(error.detail.get("restored") or [])
    state = (
        f"nothing was queued and the job dir was restored ({restored})"
        if error.detail.get("job_dir_restored")
        else f"{restored}; whether a job was queued is unknown: {error.detail.get('chain_marker')}"
    )
    reason = f"{decision.reason}; the scheduler refused the job: {error.detail.get('scheduler_text')}; {state}"
    return RecoveryOutcome(
        decision=decision.model_copy(update={"reason": reason}),
        ok=False,
        dispatched=None,
        gate_task={**gate_task, "status": "FAILED"} if gate_task else gate_task,
        mlflow=mlflow,
        walltime_plan=walltime_plan,
        scheduler_refusal=error.detail,
    )


def _reverify_first_note(decision: RecoveryDecision, job_id: str | None) -> str | None:
    """How a RESUBMIT re-verifies the saved answers it names before the policy work, or None when it names none."""
    if not decision.reverify_first:
        return None
    return (
        f"{len(decision.reverify_first)} saved answer(s) are re-verified by resume job {job_id} first, while its "
        "servers are up, then its policy work runs; a saved answer whose verifier is not stateless, or whose "
        "re-verification does not grade it, stays as it was and recoverable later"
    )


def load_original_submission_record(path: str | Path) -> tuple[SubmissionRecord, str]:
    """Read an original submission file and its content digest without indexing or rewriting it."""
    content = Path(path).read_bytes()
    return SubmissionRecord.load(json.loads(content)), hashlib.sha256(content).hexdigest()
