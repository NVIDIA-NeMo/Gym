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

import hashlib
import json
import math
from pathlib import Path

from pydantic import ValidationError

from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, load_submission_record
from nemo_gym.orchestration.recovery import (
    GateCriteria,
    RecoveryAction,
    RecoveryOutcome,
    RecoveryOutcomePolicyConfig,
    gate_summary_from_collected,
    gate_task_summary,
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
) -> RecoveryOutcome:
    """Recover a saved submission, preserving its recorded retry and walltime budgets."""
    return _recover(
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
    )


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
) -> RecoveryOutcome:
    """Inspect the same saved evidence and acceptance gate as recovery, without dispatching any action."""
    return _recover(
        gym_job_id,
        benchmark,
        inspect_only=True,
        original_record=original_record,
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
) -> RecoveryOutcome:
    """Recover `benchmark` of submission `gym_job_id` from its persisted evidence (`gym eval resubmit --recover`).

    The executor collects identities, results, failures, the quality summary, the ``score_key`` metric and the job
    chain's stored budget; ``decide_recovery`` picks one action, which runs here: ``noop`` dispatches nothing;
    ``reverify`` re-judges saved answers through the executor (never the policy) or reports that the executor cannot;
    ``resubmit`` queues the run with the existing ``resubmit`` keeping its spent budget (retries, walltime and
    per-rollout attempts), and its Gym resume runs only missing and retryable identities; ``stop`` dispatches nothing and says why. Every outcome carries the Day-0 gate task entry: per-rollout
    health is judged against ``max_unhealthy_fraction`` and reported in its warnings, ModelOpt response/trial health
    stays unknown, and MLflow delivery is checked by ``mlflow_errors``. Plain ``resubmit`` is unchanged.
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
    decision, collected = executor.recovery_evidence(
        record,
        benchmark,
        score_key=score_key,
        score_agent=score_agent,
        score_environment=score_environment,
        outcome_policy=outcome_policy,
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
            fe_valid=None,
            fe_reasons=["no evidence collected"],
            health_passed=None,
            health_reasons=["no evidence collected"],
        )
    )
    mlflow = {
        "local": collected.mlflow if collected is not None else None,
        "observed_status": mlflow_status,
        "observed_run_id": mlflow_run_id,
    }
    if inspect_only:
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
        return outcome.model_copy(update={"gate_task": gate_task, "mlflow": mlflow})
    if decision.action == RecoveryAction.RESUBMIT:
        job = executor.resubmit(
            record,
            benchmark,
            keep_budget=True,
            accounted_walltime_s=collected.accounted_walltime_s if collected is not None else None,
            accounting_snapshot=collected.accounting_snapshot if collected is not None else None,
        )
        return RecoveryOutcome(
            decision=decision, ok=True, dispatched="resubmit", job_id=job.job_id, gate_task=gate_task, mlflow=mlflow
        )
    return RecoveryOutcome(decision=decision, ok=False, dispatched=None, gate_task=gate_task, mlflow=mlflow)


def load_original_submission_record(path: str | Path) -> tuple[SubmissionRecord, str]:
    """Read an original submission file and its content digest without indexing or rewriting it."""
    content = Path(path).read_bytes()
    return SubmissionRecord.load(json.loads(content)), hashlib.sha256(content).hexdigest()
