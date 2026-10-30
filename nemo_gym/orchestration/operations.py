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

"""What you do with a submission: status, info, cancel and logs.

This is where the policy lives -- guards, ordering, partial-failure handling, grouping and the mapping
to `JobState` -- so that an executor only answers questions about its own scheduler. Each function
takes an executor typed as the capability it needs (see `executors/capabilities.py`) and calls only
that capability's methods, so nothing here knows which executor it is talking to. This module imports
no executor: `submit.py`, which knows which executors exist, resolves one from a record and calls in.
"""

from collections.abc import Iterator

from nemo_gym.orchestration.executors.capabilities import Inspectable, LogReadable, Terminable
from nemo_gym.orchestration.jobs import JobInfo, JobStatus, LogSource, SubmissionRecord, SubmissionStatus


def status(
    executor: Inspectable, records: list[SubmissionRecord], benchmarks: list[str] | None = None
) -> list[SubmissionStatus]:
    """The live status of `records`, one `SubmissionStatus` each, in the order given.

    `benchmarks` limits each submission to those benchmarks. Records are grouped by `connection_key` and
    `query_states` is called once per group, so the cost grows with the number of connections, not
    submissions; the groups run concurrently. A group whose connection fails yields `UNKNOWN` for its
    records and does not hide the others. The executor's `state` is reconciled with the files the job
    wrote, and `CANCELLED` is never overridden by a file. Read-only: it never writes the record.
    """
    raise NotImplementedError


def info(executor: Inspectable, record: SubmissionRecord, benchmark: str | None = None) -> JobInfo:
    """Describe `record`, or only `benchmark` of it: its directories, the job ids observed, its log
    sources and, when `read_run_files` can read it, the resolved config. Read-only."""
    raise NotImplementedError


def cancel(executor: Terminable, record: SubmissionRecord, benchmark: str) -> JobStatus:
    """Stop `benchmark`'s job and return its status as observed afterwards.

    Re-queries the scheduler first, so it acts on the jobs that exist now and not on a stale record, and
    leaves a job that is already terminal alone. Calls `terminate` and writes nothing to the record. A
    refusal from the scheduler (permission denied, say) is raised for the caller to report against this
    benchmark.
    """
    raise NotImplementedError


def cancel_submission(executor: Terminable, record: SubmissionRecord, *, confirm: str) -> list[JobStatus]:
    """Stop every live benchmark of `record`, and return each one's status as observed afterwards.

    `confirm` has no default and must equal `record.gym_job_id`, or `ValueError` is raised: cancelling a
    whole submission by accident costs more than typing its id. A benchmark that cannot be stopped is
    reported on its own and the others proceed.
    """
    raise NotImplementedError


def logs(
    executor: LogReadable,
    record: SubmissionRecord,
    benchmark: str,
    *,
    source: str = "driver",
    instance: str | None = None,
    tail: int | None = None,
    follow: bool = False,
) -> Iterator[str]:
    """Yield the lines of one of `benchmark`'s logs.

    `source` and `instance` are resolved against `log_sources`, so `stream_logs` receives one `LogSource`
    and executors never validate names. A name that matches nothing, or a source with several instances
    and none chosen, raises `ValueError` listing what exists. The current job is found with
    `query_states`, and its id fills the location's `{job_id}`.
    """
    raise NotImplementedError


def log_sources(record: SubmissionRecord, benchmark: str) -> list[LogSource]:
    """The logs `benchmark`'s job was set up to write, as planned at submit time.

    Read from the record, so it needs no executor and costs no remote call. A record written before
    sources were planned has none. Raises `ValueError` naming the known benchmarks for one the record lacks.
    """
    return list(record.benchmark_job(benchmark).log_sources)
