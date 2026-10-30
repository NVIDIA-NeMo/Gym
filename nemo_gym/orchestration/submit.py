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

from nemo_gym.decorators import experimental
from nemo_gym.orchestration import operations
from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.capabilities import Inspectable, Resubmittable, require
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.jobs import (
    BenchmarkJob,
    JobInfo,
    SubmissionRecord,
    SubmissionStatus,
    list_submissions,
    load_submission_record,
)


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


def _executor_for(record: SubmissionRecord) -> BaseExecutor:
    executor_cls = _EXECUTORS_BY_NAME.get(record.executor)
    if executor_cls is None:
        raise ValueError(f"Submission {record.gym_job_id} was made by unknown executor {record.executor!r}.")
    return executor_cls()


@experimental
def resubmit(gym_job_id: str, benchmark: str) -> BenchmarkJob:
    """Queue `benchmark` of submission `gym_job_id` again, in place, and return its updated record entry."""
    record = load_submission_record(gym_job_id)
    return require(_executor_for(record), Resubmittable, "resubmit").resubmit(record, benchmark)


@experimental
def status(
    ids: list[str] | None = None, *, benchmarks: list[str] | None = None, limit: int | None = 10
) -> list[SubmissionStatus]:
    """The live status of the submissions `ids`, or with no ids of the newest `limit` in the local index.

    `benchmarks` limits each submission to those benchmarks. With ids it is strict: a submission that lacks
    one fails the whole call. Without ids it selects: only submissions that have all of them are considered,
    and `limit` applies after that.
    """
    records = _records_to_inspect(ids, benchmarks, limit)
    by_executor: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_executor.setdefault(record.executor, []).append(index)

    results: list[SubmissionStatus | None] = [None] * len(records)
    for indices in by_executor.values():
        executor = require(_executor_for(records[indices[0]]), Inspectable, "status")
        answered = operations.status(executor, [records[index] for index in indices], benchmarks)
        for index, submission_status in zip(indices, answered, strict=True):
            results[index] = submission_status
    return [submission_status for submission_status in results if submission_status is not None]


@experimental
def info(gym_job_id: str, benchmark: str | None = None) -> JobInfo:
    """Describe submission `gym_job_id`, or only `benchmark` of it, and where its parts are."""
    record = load_submission_record(gym_job_id)
    return operations.info(require(_executor_for(record), Inspectable, "info"), record, benchmark)


def _records_to_inspect(
    ids: list[str] | None, benchmarks: list[str] | None, limit: int | None
) -> list[SubmissionRecord]:
    if ids:
        records, problems = [], []
        for gym_job_id in dict.fromkeys(ids):
            try:
                records.append(load_submission_record(gym_job_id))
            except ValueError as error:
                problems.append(str(error))
        if problems:
            raise ValueError(" ".join(problems))
        return records
    records = list_submissions()
    if benchmarks:
        records = [r for r in records if set(benchmarks) <= {job.benchmark for job in r.benchmarks}]
    return records if limit is None else records[:limit]
