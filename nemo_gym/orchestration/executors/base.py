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

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import yaml

from nemo_gym.orchestration.api import HOST_ENV_REFS, SubmitConfig
from nemo_gym.orchestration.jobs import (
    MANIFEST_NAME,
    RESOLVED_CONFIG_NAME,
    BenchmarkJob,
    JobInfo,
    JobKey,
    JobRef,
    JobStatus,
    LogSource,
    ObservedStatus,
    Progress,
    SubmissionRecord,
    SubmissionStatus,
)


class UnsupportedOperationError(RuntimeError):
    """The executor does not implement this operation."""


class BaseExecutor(ABC):
    # Whether this executor can auto-resubmit a benchmark that gets killed by
    # the scheduler (time limit, preemption, node failure). False means asking
    # for `resumable` on this executor is a config error, not a silent no-op.
    supports_resumable: ClassVar[bool] = False
    # What this executor writes to `SubmissionRecord.executor`, and what a resubmit dispatches on.
    name: ClassVar[str]

    @abstractmethod
    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        """Submit `config` and return the record describing it.

        Returns None on a dry run, which renders the scripts and stops before
        anything is submitted; every other path either returns a record or
        raises.
        """

    @abstractmethod
    def resubmit(self, record: SubmissionRecord, benchmark: str) -> BenchmarkJob:
        """Queue `benchmark` of `record` again, in place, and return its updated entry.

        Refuses while a job of that benchmark is still active, so two jobs never write one run.
        Persists the updated record before returning. An executor that cannot do this raises
        `UnsupportedOperationError`.
        """

    # The primitives job management is built on. Each one is the only place an executor talks to its
    # scheduler, and each raises `UnsupportedOperationError` until the executor implements it, so an
    # executor without them keeps working. They are not abstract: that would break every subclass that
    # only submits.

    def connection_key(self, record: SubmissionRecord) -> str:
        """Which connection serves `record`; records with the same key can be queried over one connection.

        Records are grouped by this key so that asking about many submissions costs one connection per
        scheduler, not one per submission.
        """
        raise self._unsupported("connection_key")

    def query_states(self, refs: list[JobRef]) -> dict[JobKey, ObservedStatus]:
        """What the scheduler says about each of `refs`: one `ObservedStatus` per `JobKey`.

        Every ref shares one `connection_key`, so an implementation opens one connection and asks the
        scheduler once for all of them. The record locates a job and the scheduler decides: look the job
        up by where it runs, not by the recorded id, so a job started by hand or resubmitted from another
        machine is still seen. When several jobs belong to one benchmark, aggregate them into one status
        whose `job_id` is the one observed. `state` is the executor's own mapping onto `JobState`; an
        unreachable scheduler yields `UNKNOWN` with the reason in `detail`, not an exception.
        """
        raise self._unsupported("query_states")

    def terminate(self, record: SubmissionRecord, job: BenchmarkJob) -> None:
        """Stop `job`, and everything that would start it again (a queued follow-up, say).

        Does only that: it neither checks the job's state nor touches the record. Raises when the
        scheduler refuses, so the caller can report it for this benchmark and go on with the others.
        """
        raise self._unsupported("terminate")

    def stream_logs(
        self, record: SubmissionRecord, job: BenchmarkJob, *, source: LogSource, tail: int | None, follow: bool
    ) -> Iterator[str]:
        """Yield the lines of `source`, a resolved `LogSource` whose location already names the job.

        `tail` keeps only the last lines. `follow` keeps yielding as the log grows.
        """
        raise self._unsupported("stream_logs")

    def read_run_files(self, record: SubmissionRecord, paths: list[Path]) -> dict[Path, str | None]:
        """The contents of files in the run directory, `None` for one that is missing or unreadable.

        The default reads nothing, which is right for an executor that cannot reach the run directory.
        """
        return {path: None for path in paths}

    def read_progress(self, record: SubmissionRecord, job: BenchmarkJob) -> Progress | None:
        """How far `job`'s run got, or `None` when this executor cannot tell."""
        return None

    def log_sources(self, record: SubmissionRecord, benchmark: str) -> list[LogSource]:
        """The logs `benchmark`'s job was set up to write, as planned at submit time.

        Read from the record, so it costs no remote call. A record written before sources were
        planned has none. Raises `ValueError` naming the known benchmarks for one the record lacks.
        """
        return list(record.benchmark_job(benchmark).log_sources)

    # The operations built on the primitives. They hold all the policy -- guards, ordering, partial
    # failure, the mapping to `JobState` -- so that every executor gets them by implementing the
    # primitives above. Executors do not override them.

    def status(self, records: list[SubmissionRecord], benchmarks: list[str] | None = None) -> list[SubmissionStatus]:
        """The live status of `records`, one `SubmissionStatus` each, in the order given.

        `benchmarks` limits each submission to those benchmarks. Records are grouped by
        `connection_key` and `query_states` is called once per group, so the cost grows with the number
        of connections, not submissions; the groups run concurrently. A group whose connection fails
        yields `UNKNOWN` for its records and does not hide the others. The executor's `state` is
        reconciled with the files the job wrote, and `CANCELLED` is never overridden by a file.
        Read-only: it never writes the record.
        """
        raise NotImplementedError

    def info(self, record: SubmissionRecord, benchmark: str | None = None) -> JobInfo:
        """Describe `record`, or only `benchmark` of it: its directories, the job ids observed, its log
        sources and, when `read_run_files` can read it, the resolved config. Read-only."""
        raise NotImplementedError

    def cancel(self, record: SubmissionRecord, benchmark: str) -> JobStatus:
        """Stop `benchmark`'s job and return its status as observed afterwards.

        Re-queries the scheduler first, so it acts on the jobs that exist now and not on a stale
        record, and leaves a job that is already terminal alone. Calls `terminate` and writes nothing to
        the record. A refusal from the scheduler (permission denied, say) is raised for the caller to
        report against this benchmark.
        """
        raise NotImplementedError

    def cancel_submission(self, record: SubmissionRecord, *, confirm: str) -> list[JobStatus]:
        """Stop every live benchmark of `record`, and return each one's status as observed afterwards.

        `confirm` has no default and must equal `record.gym_job_id`, or `ValueError` is raised: cancelling
        a whole submission by accident costs more than typing its id. A benchmark that cannot be stopped
        is reported on its own and the others proceed.
        """
        raise NotImplementedError

    def logs(
        self,
        record: SubmissionRecord,
        benchmark: str,
        *,
        source: str = "driver",
        instance: str | None = None,
        tail: int | None = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Yield the lines of one of `benchmark`'s logs.

        `source` and `instance` are resolved against `log_sources`, so `stream_logs` receives one
        `LogSource` and executors never validate names. A name that matches nothing, or a source with
        several instances and none chosen, raises `ValueError` listing what exists. The current job is
        found with `query_states`, and its id fills the location's `{job_id}`.
        """
        raise NotImplementedError

    def _unsupported(self, operation: str) -> UnsupportedOperationError:
        return UnsupportedOperationError(
            f"{getattr(self, 'name', type(self).__name__)} executor does not support {operation}"
        )

    def persist(
        self,
        record: SubmissionRecord,
        config: SubmitConfig,
        write_manifest: Callable[..., None],
    ) -> None:
        """Store the record and the resolved config, in the order that survives a partial failure.

        Shared by every executor because the ordering and the failure handling
        are policy rather than transport. The machine-local index goes first
        precisely because it cannot fail the submit, so if the manifest write
        does fail there is still a parseable record for the by-hand recovery the
        error asks for.

        The resolved config is written here, not by each executor's staging
        step, so that every executor -- today Slurm, tomorrow k8s or a local
        runner -- gets it for free from the one place all of them already call.

        Only the files' transport differs per executor, so it arrives as
        `write_manifest` -- `Connection.write_text` already has this signature --
        rather than this class owning a connection it cannot know how to open.
        Call it while that transport is still open: reopening one here would pay
        a second connection on every submit.

        The jobs are queued by the time this runs, so a failure has to name them
        or they are stranded with no record anywhere.
        """

        def write_resolved_config() -> None:
            # `host:` values are written as their references, so the record is safe for other readers
            # (e.g. certification under another account). The resolved secrets stay in job.sh only.
            record_config = config.model_dump(mode="json", context={HOST_ENV_REFS: True})
            write_manifest(Path(record.run_dir) / RESOLVED_CONFIG_NAME, yaml.safe_dump(record_config, sort_keys=False))

        self._store(record, write_manifest, "Submitted jobs", write_config=write_resolved_config)

    def persist_record(self, record: SubmissionRecord, write_manifest: Callable[..., None]) -> None:
        """Store an updated record in the same two places `persist()` does; the resolved config is unchanged."""
        self._store(record, write_manifest, "Queued jobs")

    @staticmethod
    def _store(
        record: SubmissionRecord,
        write_manifest: Callable[..., None],
        action: str,
        *,
        write_config: Callable[[], None] | None = None,
    ) -> None:
        """The ordering and failure handling `persist()` describes, for both of its callers."""
        record.write_local_index()
        written = "the manifest/resolved config" if write_config else "the manifest"
        try:
            if write_config is not None:
                write_config()
            write_manifest(Path(record.run_dir) / MANIFEST_NAME, record.dumps())
        except Exception as error:
            queued = ", ".join(f"{b.benchmark}={b.job_id}" for b in record.benchmarks if b.job_id)
            raise RuntimeError(
                f"{action} but could not write {written} under {record.run_dir}: {error}. The local index has "
                f"the record. Already queued: {queued or 'nothing'}. Record these by hand before collecting."
            ) from error
