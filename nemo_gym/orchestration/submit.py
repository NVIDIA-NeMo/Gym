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

from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.jobs import BenchmarkJob, SubmissionRecord, load_submission_record
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
