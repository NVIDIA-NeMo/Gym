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

"""The record a submission leaves behind.

`api.py` is the input schema (what to submit); this is the output schema (what
was submitted). Nothing here imports from `executors/`, so a reader — today
EFB's collect, tomorrow `gym eval status` — can load a record without pulling
Slurm, SSH and the sbatch templates into its import path.
"""

import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from nemo_gym import __version__


SCHEMA_VERSION = 1

# The manifest's name inside the remote run directory. Readers look for exactly this.
MANIFEST_NAME = "gym-job.json"


class BenchmarkJob(BaseModel):
    """One benchmark's submission. `job_id` is None exactly when its `sbatch`
    failed, in which case `error` says why; the other benchmarks in the same
    submission are unaffected."""

    benchmark: str
    job_dir: str
    job_id: str | None = None
    error: str | None = None


class SubmissionRecord(BaseModel):
    """Everything needed to find a submitted run again.

    `hostname` None means the submission ran on the login node itself (see
    `get_connection`), not that the host is unknown.
    """

    gym_job_id: str
    gym_version: str
    submitted_at: str
    run_dir: str
    cluster: str
    executor: Literal["slurm"]
    submitted_by: str
    benchmarks: list[BenchmarkJob]
    hostname: str | None = None
    config_path: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def failed(self) -> list[BenchmarkJob]:
        return [b for b in self.benchmarks if b.job_id is None]


def utc_timestamp(now: datetime) -> str:
    """ISO 8601, seconds, explicit Z. A run directory is read on a cluster whose
    timezone need not match the submitter's."""
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_gym_job_id(now: datetime) -> str:
    """The submission's primary key, and the run directory's name.

    The random suffix is what keeps two submits in the same second against the
    same `job.output_path` from sharing a directory — `SSHConnection.copy`
    rsyncs with `--delete`, so sharing one means the second silently erases the
    first's staged scripts.
    """
    return f"gym-job-{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def gym_version() -> str:
    return __version__


def local_index_dir() -> Path:
    """Where this machine remembers its own submissions."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "nemo-gym" / "jobs"


def dumps(record: SubmissionRecord) -> str:
    """The manifest's on-disk bytes; one spelling, so every store matches."""
    return record.model_dump_json(indent=2) + "\n"


def write_local_index(record: SubmissionRecord) -> Path | None:
    """Record the submission locally, or report why not and carry on.

    Best-effort by design: this runs after the jobs are queued, so raising here
    would report a failure for work that is really running. The durable copy is
    the manifest in the run directory.
    """
    path = local_index_dir() / f"{record.gym_job_id}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps(record), encoding="utf-8")
    except OSError as error:
        print(f"Could not write the local job index at {path}: {error}", file=sys.stderr)
        return None
    return path


def load_record(payload: dict) -> SubmissionRecord:
    """Parse a record, refusing one this version does not understand.

    A v1 reader silently accepting a v2 record would read fields that may have
    moved; failing loudly is the only honest option.
    """
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported job record schema_version {version!r}: this Gym understands {SCHEMA_VERSION}. "
            "Upgrade nemo-gym to read this record."
        )
    return SubmissionRecord.model_validate(payload)
