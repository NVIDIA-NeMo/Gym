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

"""The opt-in interfaces an executor adds to `BaseExecutor`.

`BaseExecutor` needs only `run`. Each feature beyond submitting is an interface here, and an executor
lists the ones it supports in its bases:

    class SlurmExecutor(BaseExecutor, Resubmittable, Terminable, LogReadable): ...

Every method of a capability is abstract, so a capability cannot be half implemented; the only
exceptions are the optional hooks, which carry a default and say so. `Terminable` and `LogReadable`
extend `Inspectable` because both need `query_states`.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import TypeVar

from nemo_gym.orchestration.executors.base import UnsupportedOperationError
from nemo_gym.orchestration.jobs import (
    BenchmarkJob,
    JobKey,
    JobRef,
    LogSource,
    ObservedStatus,
    Progress,
    SubmissionRecord,
)


class Resubmittable(ABC):
    """An executor that can queue a benchmark of an existing submission again."""

    @abstractmethod
    def resubmit(self, record: SubmissionRecord, benchmark: str) -> BenchmarkJob:
        """Queue `benchmark` of `record` again, in place, and return its updated entry.

        Refuses while a job of that benchmark is still active, so two jobs never write one run.
        Persists the updated record before returning.
        """


class Inspectable(ABC):
    """An executor that can say what is happening to the jobs it queued."""

    @abstractmethod
    def connection_key(self, record: SubmissionRecord) -> str:
        """Which connection serves `record`; records with the same key can be queried over one connection.

        Records are grouped by this key so that asking about many submissions costs one connection per
        scheduler, not one per submission.
        """

    @abstractmethod
    def query_states(self, refs: list[JobRef]) -> dict[JobKey, ObservedStatus]:
        """What the scheduler says about each of `refs`: one `ObservedStatus` per `JobKey`.

        Every ref shares one `connection_key`, so an implementation opens one connection and asks the
        scheduler once for all of them. The record locates a job and the scheduler decides: look the job
        up by where it runs, not by the recorded id, so a job started by hand or resubmitted from another
        machine is still seen. When several jobs belong to one benchmark, aggregate them into one status
        whose `job_id` is the one observed. `state` is the executor's own mapping onto `JobState`; an
        unreachable scheduler yields `UNKNOWN` with the reason in `detail`, not an exception.
        """

    def read_run_files(self, record: SubmissionRecord, paths: list[Path]) -> dict[Path, str | None]:
        """Optional. The contents of files in the run directory, `None` for one that is missing or unreadable.

        The default reads nothing, which is right for an executor that cannot reach the run directory.
        """
        return {path: None for path in paths}

    def read_progress(self, record: SubmissionRecord, job: BenchmarkJob) -> Progress | None:
        """Optional. How far `job`'s run got, or `None` when this executor cannot tell."""
        return None


class Terminable(Inspectable):
    """An executor that can stop the jobs it queued."""

    @abstractmethod
    def terminate(self, record: SubmissionRecord, job: BenchmarkJob) -> None:
        """Stop `job`, and everything that would start it again (a queued follow-up, say).

        Does only that: it neither checks the job's state nor touches the record. Raises when the
        scheduler refuses, so the caller can report it for this benchmark and go on with the others.
        """


class LogReadable(Inspectable):
    """An executor that can read the logs of the jobs it queued."""

    @abstractmethod
    def stream_logs(
        self, record: SubmissionRecord, job: BenchmarkJob, *, source: LogSource, tail: int | None, follow: bool
    ) -> Iterator[str]:
        """Yield the lines of `source`, a resolved `LogSource` whose location already names the job.

        `tail` keeps only the last lines. `follow` keeps yielding as the log grows.
        """


_Capability = TypeVar("_Capability")


def require(executor: object, capability: type[_Capability], operation: str) -> _Capability:
    """`executor`, typed as `capability`, or `UnsupportedOperationError` naming what it cannot do."""
    if not isinstance(executor, capability):
        name = getattr(executor, "name", type(executor).__name__)
        raise UnsupportedOperationError(f"{name} executor does not support {operation}")
    return executor
