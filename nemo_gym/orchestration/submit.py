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
from pathlib import Path

from pydantic import ValidationError

from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm import SlurmExecutor, reader_token
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, load_submission_record
from nemo_gym.orchestration.recovery import (
    MLFLOW_NOT_REQUIRED_NOTE,
    GateCriteria,
    OperatorBudget,
    RecoveryAction,
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
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    repin: str | None = None,
    repin_reason: str | None = None,
    extend_retries: int = 0,
    max_walltime: str | None = None,
    walltime_reason: str | None = None,
    unsafe_resume_reason: str | None = None,
) -> RecoveryOutcome:
    """Recover a saved submission, preserving its recorded retry and walltime budgets.

    ``repin`` (with ``repin_reason``) continues the run on another full Gym commit, which must be this launcher's.
    ``extend_retries`` records that many more infra retries for the run: explicit and journaled, never a reset.
    ``max_walltime`` (with ``walltime_reason``) raises the run's Slurm walltime, never below the recorded one.
    ``unsafe_resume_reason`` resumes the job with ``allow_unsafe_resume``, past a refused run identity check; the
    reason and the user are journaled.
    For a run whose chain accounting is unknown, the two are also an operator's explicit budget: the run continues
    as an OPERATOR-CONTINUATION with ``extend_retries`` (one when not given) infra retries remaining, recorded in the
    decision and the job dir's operator journal (see ``decide_from_collected``).
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
        outcome_policy=outcome_policy,
        mlflow_status=mlflow_status,
        mlflow_run_id=mlflow_run_id,
        repin=repin,
        repin_reason=repin_reason,
        extend_retries=extend_retries,
        max_walltime=max_walltime,
        walltime_reason=walltime_reason,
        unsafe_resume_reason=unsafe_resume_reason,
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
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    extend_retries: int = 0,
) -> RecoveryOutcome:
    """Inspect the same saved evidence and acceptance gate as recovery, without dispatching any action."""
    outcome = _recover(
        gym_job_id,
        benchmark,
        inspect_only=True,
        original_record=original_record,
        extend_retries=extend_retries,
        score_key=score_key,
        score_agent=score_agent,
        score_environment=score_environment,
        max_unhealthy_fraction=max_unhealthy_fraction,
        max_response_failure_fraction=max_response_failure_fraction,
        max_trial_failure_fraction=max_trial_failure_fraction,
        outcome_policy=outcome_policy,
        mlflow_status=mlflow_status,
        mlflow_run_id=mlflow_run_id,
    )
    return outcome.model_copy(update={"reader_token": reader_token()})


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
    outcome_policy: RecoveryOutcomePolicyConfig | None = None,
    mlflow_status: str | None = None,
    mlflow_run_id: str | None = None,
    repin: str | None = None,
    repin_reason: str | None = None,
    extend_retries: int = 0,
    max_walltime: str | None = None,
    walltime_reason: str | None = None,
    unsafe_resume_reason: str | None = None,
) -> RecoveryOutcome:
    """Recover `benchmark` of submission `gym_job_id` from its persisted evidence (`gym eval resubmit --recover`).

    The executor collects identities, results, failures, the quality summary, the ``score_key`` metric and the job
    chain's stored budget; ``decide_recovery`` picks one action, which runs here: ``noop`` dispatches nothing;
    ``reverify`` re-judges saved answers through the executor (never the policy) or reports that the executor cannot;
    ``resubmit`` queues the run with the existing ``resubmit`` keeping its spent budget (retries, walltime and
    per-rollout attempts), and its Gym resume runs only missing and retryable identities; ``stop`` dispatches nothing and says why. Every outcome carries the acceptance gate task entry: per-rollout
    health is judged against ``max_unhealthy_fraction`` and reported in its warnings; response/trial health is evaluated
    separately, and MLflow delivery is checked by ``mlflow_errors``. Plain ``resubmit`` is unchanged.
    """
    if score_key is not None and (
        not isinstance(score_key, list)
        or not score_key
        or not all(isinstance(key, str) or (type(key) is int and key >= 0) for key in score_key)
    ):
        raise ValueError("score_key must be a nonempty list of object keys and nonnegative array indices")
    if max_unhealthy_fraction is not None and (
        isinstance(max_unhealthy_fraction, bool)
        or not isinstance(max_unhealthy_fraction, (int, float))
        or not math.isfinite(max_unhealthy_fraction)
        or not 0 < max_unhealthy_fraction <= 1
    ):
        raise ValueError("invalid response-health criterion: expected a finite number in (0, 1]")
    limits = {"max_response_failure_fraction": max_response_failure_fraction}
    limits["max_trial_failure_fraction"] = max_trial_failure_fraction
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
    decision, collected = executor.recovery_evidence(
        record,
        benchmark,
        score_key=score_key,
        score_agent=score_agent,
        score_environment=score_environment,
        outcome_policy=outcome_policy,
        # A repinned run resumes on the target's Store: its reader decides what that Store would run.
        **({"reader_ref": repin} if repin is not None else {}),
        **({"extend_retries": extend_retries} if extend_retries else {}),
        # A walltime raise renders a larger chain cap: the plan checks the cap the resubmit will render.
        **({"max_walltime": max_walltime} if max_walltime is not None else {}),
        **({"operator_budget": operator_budget} if operator_budget is not None else {}),
    )
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
    if inspect_only:
        if gate_task["status"] == "PLANNED":
            gate_task = {
                **gate_task,
                "warnings": [
                    *gate_task["warnings"],
                    f"PLANNED (NOT DISPATCHED): --inspect reports the {decision.action.value} decision and runs nothing",
                ],
            }
        return RecoveryOutcome(
            decision=decision,
            ok=decision.action != RecoveryAction.STOP,
            dispatched=None,
            gate_task=gate_task,
            mlflow=mlflow,
        )
    if decision.action == RecoveryAction.NOOP:
        return RecoveryOutcome(decision=decision, ok=True, dispatched=None, gate_task=gate_task, mlflow=mlflow)
    if decision.action == RecoveryAction.REVERIFY:
        outcome = executor.reverify_judge_failures(record, benchmark, decision)
        if not outcome.ok or outcome.dispatched is None:
            gate_task = {
                **gate_task,
                "status": "FAILED",
                "errors": [*gate_task["errors"], outcome.unsupported or "saved-answer rejudgment was not dispatched"],
            }
        elif gate_task["status"] == "PLANNED":
            gate_task = {**gate_task, "status": "RESUMING"}
        return outcome.model_copy(update={"gate_task": gate_task, "mlflow": mlflow})
    if decision.action == RecoveryAction.RESUBMIT:
        operator = decision.operator_continuation
        job = executor.resubmit(
            record,
            benchmark,
            keep_budget=True,
            accounted_walltime_s=collected.accounted_walltime_s if collected is not None else None,
            accounting_snapshot=collected.accounting_snapshot if collected is not None else None,
            # Only a reconciled chain passes the flag, so executors without requeue reconciliation keep working.
            **({"accounting_reconciled": True} if collected is not None and collected.accounting_reconciled else {}),
            **({"repin": repin, "repin_reason": repin_reason} if repin is not None else {}),
            # Under an operator continuation, N is the allowance its record fills, not a recorded extension.
            **({"extend_retries": extend_retries} if extend_retries and operator is None else {}),
            **({"max_walltime": max_walltime, "walltime_reason": walltime_reason} if max_walltime is not None else {}),
            **({"unsafe_resume_reason": unsafe_resume_reason} if unsafe_resume_reason is not None else {}),
            **({"operator_continuation": operator} if operator is not None else {}),
        )
        if gate_task["status"] == "PLANNED":
            gate_task = {**gate_task, "status": "RESUMING"}
        return RecoveryOutcome(
            decision=decision, ok=True, dispatched="resubmit", job_id=job.job_id, gate_task=gate_task, mlflow=mlflow
        )
    return RecoveryOutcome(decision=decision, ok=False, dispatched=None, gate_task=gate_task, mlflow=mlflow)


def load_original_submission_record(path: str | Path) -> tuple[SubmissionRecord, str]:
    """Read an original submission file and its content digest without indexing or rewriting it."""
    content = Path(path).read_bytes()
    return SubmissionRecord.load(json.loads(content)), hashlib.sha256(content).hexdigest()
