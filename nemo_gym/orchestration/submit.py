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

import json
from pathlib import Path

from nemo_gym.decorators import experimental
from nemo_gym.orchestration.api import KubernetesComputeConfig, SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.kubernetes import KubernetesExecutor
from nemo_gym.orchestration.executors.slurm import SlurmExecutor
from nemo_gym.orchestration.jobs import MANIFEST_NAME, BenchmarkJob, SubmissionRecord, local_index_dir


_EXECUTORS = {
    SlurmComputeConfig: SlurmExecutor,
    KubernetesComputeConfig: KubernetesExecutor,
}


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


def _find_benchmark_job(job_dir: str) -> tuple[SubmissionRecord, BenchmarkJob]:
    """The record naming `job_dir`: the run's manifest if it is on this machine, else this machine's job index."""
    target = Path(job_dir)
    manifest = target.parent / MANIFEST_NAME
    for path in [manifest] if manifest.is_file() else sorted(local_index_dir().glob("*.json")):
        record = SubmissionRecord.load(json.loads(path.read_text(encoding="utf-8")))
        for benchmark in record.benchmarks:
            if Path(benchmark.job_dir) == target:
                return record, benchmark
    raise ValueError(f"No submission record names job dir {job_dir}.")


@experimental
def resubmit(job_dir: str) -> str:
    """Queue a submitted benchmark's job.sh again, in place, and return the new Slurm job id."""
    record, benchmark = _find_benchmark_job(job_dir)
    if record.executor != "slurm":
        raise ValueError(f"Resubmit supports the Slurm executor only, not {record.executor!r}.")
    return SlurmExecutor().resubmit(record, benchmark)
