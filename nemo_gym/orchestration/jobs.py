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
was submitted).

This module is executor-agnostic on purpose, and that is a rule rather than a
coincidence: it must describe a submission made by ANY executor -- Slurm today,
k8s or a local runner later. Two consequences, both enforced by
`test_jobs_module_is_executor_agnostic`:

* Nothing here imports from `executors/`, so a reader -- today EFB's collect,
  `gym eval status` -- can load a record without pulling Slurm, SSH and
  the sbatch templates into its import path.
* No field is typed or named after one executor's vocabulary. Where a docstring
  explains a value by example, it says which executor the example comes from.

Compatibility. A record outlives the Gym that wrote it, so `schema_version` is
read in one direction only: a reader accepts anything at or below its own
version and refuses only what is newer than it understands. Upgrading nemo-gym
therefore never strands records already on disk.

That is only sound if the schema keeps its side of the bargain, so within one
schema version a field may only be ADDED WITH A DEFAULT, or removed. A field
must never be repurposed or have its meaning changed -- an older record would
then be read as if it meant the new thing. Anything of that kind needs a
`SCHEMA_VERSION` bump and an explicit migration, not a silent reinterpretation.
"""

import getpass
import hashlib
import json
import logging
import os
import re
import secrets
import socket
import sys
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, field_validator

from nemo_gym._moved_paths import moved_names_getattr


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# The manifest's name inside the run directory, wherever that directory lives.
# Readers look for exactly this.
MANIFEST_NAME = "gym-job.json"

# The fully-resolved submit config, written next to the manifest by every
# executor via `BaseExecutor.persist()`. This is what was actually submitted
# (interpolations resolved, overrides applied) rather than the config file on
# disk, which may have changed since, or the overrides, which are meaningless
# without the file they were applied to.
RESOLVED_CONFIG_NAME = "resolved-config.yaml"

# What a benchmark's directory (`<run dir>/<benchmark>/`) holds, by name. The executors create it, the job script
# writes into it and the exports read from it, so the names live in one place.
LOGS_DIRNAME = "logs"
ARTIFACTS_DIRNAME = "artifacts"
EXPORTS_DIRNAME = "exports"
ROLLOUTS_FILENAME = "rollouts.jsonl"
# The exit code of `gym eval run`, which is also the job's.
EVAL_EXIT_FILENAME = "eval.exit"
# Created by the export tail when the evaluation succeeded; one file per attempt, so `.<Slurm job id>` follows.
EVAL_SUCCEEDED_FILENAME = "eval.succeeded"
# `<exports dir>/<exporter><suffix>`: the outcome of one export.
EXPORT_RESULT_SUFFIX = ".result.json"


# State files a job writes into its job directory. An executor returns the ones it can read in
# `ObservedStatus.files`, and the status operation reads them. `.gym_chain_incomplete` holds a
# reason and is written when a chain of resumed jobs stops before the run finished; `eval.exit`
# holds the exit code of `gym eval run`.
CHAIN_INCOMPLETE_FILE = ".gym_chain_incomplete"
EVAL_EXIT_FILE = "eval.exit"


class LogSource(BaseModel):
    """One log a benchmark's job writes, as the executor planned it at submit time.

    `kind` is `driver` for the process that runs the benchmark and `service` for every other
    step of the job. `instance` names one instance of a service that runs several; it is None
    for a source that is not split by instance. `location` is where the log is, in the
    executor's terms and opaque to everyone else: for the Slurm executor a path relative to the
    job directory with a `{job_id}` placeholder, because the scheduler's id is not known at
    submit time and changes when a job is resumed; for a container-based executor, a container
    name. A reader fills in the placeholder with the id of the job it wants.
    """

    name: str
    kind: Literal["driver", "service"]
    instance: str | None = None
    location: str


class BenchmarkJob(BaseModel):
    """One benchmark's submission.

    `job_id` is None exactly when the executor failed to enqueue this benchmark
    (for the Slurm executor, a failed `sbatch`), in which case `error` says why.
    The other benchmarks in the same submission are unaffected -- one failure
    does not discard the record for the ones that did start.

    `log_sources` is the list of logs the job was set up to write, computed at
    submit time by the same code that renders the job's log flags, so the plan
    and the job cannot disagree. A plan is intent, not proof that a file exists.
    A record written before this field existed has none.
    """

    benchmark: str
    job_dir: str
    job_id: str | None = None
    error: str | None = None
    # Set by a repinned recovery: the Gym commit this benchmark's job runs, when it differs from the submission's.
    gym_commit: str | None = None
    # Set by a recovery that raised the walltime: the Slurm walltime this benchmark's job now runs with.
    walltime: str | None = None
    log_sources: list[LogSource] = []


class SubmissionTelemetry(BaseModel):
    """Resolved submission-time telemetry controls, not credentials or a delivery claim."""

    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool
    token_env: str
    remote_export: StrictBool

    @field_validator("token_env")
    @classmethod
    def validate_name(cls, name: str) -> str:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ValueError("Telemetry credential metadata must be an environment variable name")
        return name


class SubmissionEnvironment(BaseModel):
    """Names observed at submission, never values.

    required_env names submit-shell inputs (host: references and required forwarding).
    forwarded_env names variables actually exported by the submission transport. A
    runtime: alias is not a submit-shell requirement unless explicitly forwarded.
    None on the containing record means this observation was not recorded.
    """

    model_config = ConfigDict(extra="forbid")
    required_env: list[str]
    forwarded_env: list[str]
    telemetry: SubmissionTelemetry

    @field_validator("required_env", "forwarded_env")
    @classmethod
    def validate_names(cls, names: list[str]) -> list[str]:
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names):
            raise ValueError("Submission environment metadata must contain only variable names")
        return sorted(set(names))


class SubmissionRecord(BaseModel):
    """Everything needed to find a submitted run again.

    `executor` names the executor that produced this record, and is the field a
    reader branches on before interpreting the executor-shaped parts of the rest
    -- `job_id`, for instance, is a Slurm job ID under the Slurm executor and
    need not be numeric under another. It is deliberately a plain `str` rather
    than an enum of the executors that happen to exist today, so that adding one
    does not require a schema version bump on every reader.

    `hostname` None means the submission was made from the machine that runs the
    workload manager's client directly, rather than reaching it over SSH -- not
    that the host is unknown. Executors with no remote-submission concept at all
    leave it None.

    `gym_commit` is the commit of the Gym that wrote the record, when its install
    records one (see `nemo_gym.utils.gym_commit.installed_gym_commit`). A `-dirty` suffix marks a checkout
    with uncommitted changes: the commit alone does not reproduce that Gym.
    """

    gym_job_id: str
    gym_version: str
    gym_commit: str | None = None
    submitted_at: str
    run_dir: str
    cluster: str
    executor: str
    submitted_by: str
    benchmarks: list[BenchmarkJob]
    hostname: str | None = None
    # Whatever the executor needs to find this submission again that the fields
    # above cannot express -- a k8s namespace, say. Free-form because the shape
    # is the executor's business; a reader keys off `executor` before reading it.
    executor_metadata: dict[str, str] = {}
    environment: SubmissionEnvironment | None = Field(default=None, exclude_if=lambda value: value is None)
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def failed(self) -> list[BenchmarkJob]:
        return [b for b in self.benchmarks if b.job_id is None]

    def benchmark_job(self, benchmark: str) -> BenchmarkJob:
        """This record's entry for `benchmark`; an unknown name is an error that lists the known ones."""
        for job in self.benchmarks:
            if job.benchmark == benchmark:
                return job
        known = ", ".join(b.benchmark for b in self.benchmarks)
        raise ValueError(f"Submission {self.gym_job_id} has no benchmark {benchmark!r}; it has: {known}.")

    def with_benchmark(self, job: BenchmarkJob) -> "SubmissionRecord":
        """A copy of this record with the entry for `job.benchmark` replaced by `job`."""
        self.benchmark_job(job.benchmark)
        benchmarks = [job if b.benchmark == job.benchmark else b for b in self.benchmarks]
        return self.model_copy(update={"benchmarks": benchmarks})

    def dumps(self) -> str:
        """The manifest's on-disk bytes; one spelling, so every store matches."""
        return self.model_dump_json(indent=2) + "\n"

    def write_local_index(self) -> Path | None:
        """Record the submission on this machine, or report why not and carry on.

        Best-effort by design: this runs after the jobs are queued, so raising
        here would report a failure for work that is really running. The durable
        copy is the manifest in the run directory.
        """
        path = local_index_dir() / f"{self.gym_job_id}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.dumps(), encoding="utf-8")
        except OSError as error:
            print(f"Could not write the local job index at {path}: {error}", file=sys.stderr)
            return None
        return path

    @classmethod
    def load(cls, payload: dict) -> "SubmissionRecord":
        """Parse a record, refusing only one written by a NEWER Gym.

        Older records stay readable: fields added since take their defaults, which
        the module docstring's compatibility rule is there to guarantee. Refusing
        them instead would mean an upgrade silently orphaned every job already
        submitted.

        A newer record is the one case that cannot be read safely -- it may carry
        fields whose meaning this version does not know -- so it fails, and says
        which way round the mismatch is.
        """
        version = payload.get("schema_version")
        if not isinstance(version, int):
            raise ValueError(
                f"Job record has no usable schema_version (got {version!r}); it was not written by `gym eval submit`."
            )
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Job record schema_version {version} was written by a newer nemo-gym; this one understands "
                f"up to {SCHEMA_VERSION}. Upgrade nemo-gym to read it."
            )
        return cls.model_validate(payload)


# Each index entry rebuilt from a run directory's own record, one JSON line per restore, beside the entries.
INDEX_RESTORE_JOURNAL_NAME = "index-restores.jsonl"


def restore_index_entry(gym_job_id: str, run_dir: str | Path) -> SubmissionRecord:
    """Rebuild this machine's job index entry for `gym_job_id` from the run directory's own record.

    A recycled machine loses its index while the run directory keeps the `gym-job.json` that `persist()` wrote
    there. That record is taken only as this submission's and this directory's: its `gym_job_id` is the one asked
    for and names the directory, its `run_dir` is the directory, and every benchmark's `job_dir` lies inside it.
    An existing entry is kept as it is, with a warning when it names another run directory. Each restore is
    journaled in the index (`INDEX_RESTORE_JOURNAL_NAME`) with the source record's SHA-256, the user and the host.
    """
    directory = Path(run_dir).resolve()
    if (local_index_dir() / f"{gym_job_id}.json").is_file():
        record = load_submission_record(gym_job_id)
        if Path(record.run_dir).resolve() != directory:
            print(
                f"The job index already has submission {gym_job_id!r} with run_dir {record.run_dir}; "
                f"{directory} was not used.",
                file=sys.stderr,
            )
        return record
    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        raise ValueError(f"Cannot restore submission {gym_job_id!r}: no {MANIFEST_NAME} in {directory}.")
    content = manifest.read_bytes()
    record = SubmissionRecord.load(json.loads(content))
    problems = []
    if record.gym_job_id != gym_job_id:
        problems.append(f"the record is submission {record.gym_job_id!r}")
    if directory.name != gym_job_id:
        problems.append(f"the directory is named {directory.name!r}")
    if Path(record.run_dir).resolve() != directory:
        problems.append(f"the record's run_dir is {record.run_dir}")
    problems += [
        f"benchmark {job.benchmark!r} has its job_dir {job.job_dir} outside it"
        for job in record.benchmarks
        if not Path(job.job_dir).resolve().is_relative_to(directory)
    ]
    if problems:
        raise ValueError(f"Cannot restore submission {gym_job_id!r} from {manifest}: " + "; ".join(problems) + ".")
    if record.write_local_index() is None:
        raise ValueError(f"Cannot restore submission {gym_job_id!r}: the local job index is not writable.")
    entry = {
        "at": utc_timestamp(utc_now()),
        "gym_job_id": gym_job_id,
        "source": str(manifest),
        "sha256": hashlib.sha256(content).hexdigest(),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
    }
    with (local_index_dir() / INDEX_RESTORE_JOURNAL_NAME).open("a", encoding="utf-8") as journal:
        journal.write(json.dumps(entry, sort_keys=True) + "\n")
    return record


class JobState(StrEnum):
    """What is happening to one benchmark's job, in terms every executor shares.

    Each executor maps its own scheduler's vocabulary onto this; the native string
    survives only as a status's `detail`. `UNKNOWN` means the executor could not tell
    (a scheduler that is unreachable, or no longer remembers the job).
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


# Nothing is left to stop for a job in one of these states.
FINISHED_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})


class FailureCategory(StrEnum):
    """Which kind of thing went wrong, so a reader knows how far to trust a `Failure`."""

    ENQUEUE = "ENQUEUE"
    SCHEDULER = "SCHEDULER"
    INCOMPLETE = "INCOMPLETE"
    RUN = "RUN"
    EXPORT = "EXPORT"
    CANCELLED = "CANCELLED"


class Failure(BaseModel):
    """Why a benchmark's job failed, with where the reason came from.

    `source` says what produced `message` (the scheduler, the record, a file the job wrote),
    and `logs` is a command that shows the job's logs. The logs themselves are never parsed
    to guess a cause.
    """

    category: FailureCategory
    message: str
    exit_code: int | None = None
    source: str
    logs: str | None = None


class Progress(BaseModel):
    """How far a run got, in counts only: rollouts `done` of `total`, and how many `failed`."""

    done: int
    total: int
    failed: int


class ObservedStatus(BaseModel):
    """What an executor saw of one benchmark's job: the primitive behind every status.

    `state` is set by the executor from its own vocabulary, and `detail` keeps the native
    string (a Slurm `TIMEOUT`, say). `job_id` is the id actually observed, which can differ
    from the record's when a job was resumed or started outside Gym, and `job_ids` is every job
    id seen for the benchmark, oldest first (the whole chain when a run was resumed). `files` holds
    the state files the job wrote into its directory, by name, read in the same call as the
    scheduler's answer; a file that is missing or unreadable is absent.
    """

    state: JobState
    detail: str | None = None
    exit_code: int | None = None
    reason: str | None = None
    job_id: str | None = None
    job_ids: list[str] = []
    files: dict[str, str] = {}


class JobStatus(BaseModel):
    """The status of one benchmark's job as shown to a user.

    Built from an `ObservedStatus` and the files the job wrote. `progress` is None until an
    executor reports it.
    """

    benchmark: str
    state: JobState
    job_id: str | None = None
    detail: str | None = None
    exit_code: int | None = None
    failure: Failure | None = None
    progress: Progress | None = None


class SubmissionStatus(BaseModel):
    """The status of every selected benchmark of one submission.

    `state` rolls the benchmarks up into one value and `counts` says how many are in each
    state, so a mixed result is not hidden behind one word. Both describe only the benchmarks
    in `jobs`.
    """

    record: SubmissionRecord
    jobs: list[JobStatus]
    state: JobState
    counts: dict[JobState, int]


class BenchmarkInfo(BaseModel):
    """Where one benchmark's job lives and what it writes: its directory, the job ids observed for it
    (the whole chain when a run was resumed), and the logs it was set up to write."""

    benchmark: str
    job_dir: str
    job_ids: list[str] = []
    log_sources: list[LogSource] = []


class JobInfo(BaseModel):
    """A description of one submission and where its parts are, with no live state.

    `config` is the resolved config the submission was made with, or None when the executor
    cannot read it, in which case `config_unavailable` says why.
    """

    record: SubmissionRecord
    benchmarks: list[BenchmarkInfo]
    config: dict[str, object] | None = None
    config_unavailable: str | None = None


# One benchmark's job: the submission's id and the benchmark's name.
JobKey = tuple[str, str]


class JobRef(NamedTuple):
    """A record paired with one of its benchmark jobs, so one call can span many submissions."""

    record: SubmissionRecord
    job: BenchmarkJob

    @property
    def key(self) -> JobKey:
        return (self.record.gym_job_id, self.job.benchmark)


def utc_now() -> datetime:
    """The clock the run directory is named from.

    Here rather than in an executor because run-directory identity is this
    module's concern, and every executor needs the same answer. A seam: tests
    freeze it to prove two submits in the same second still get distinct
    directories.
    """
    return datetime.now(timezone.utc)


def utc_timestamp(now: datetime) -> str:
    """ISO 8601, seconds, explicit Z. A run directory is read on a cluster whose
    timezone need not match the submitter's."""
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_gym_job_id(now: datetime) -> str:
    """The submission's primary key, and the run directory's name.

    The random suffix is what keeps two submits in the same second against the
    same `job.output_path` from sharing a directory. Sharing one is not merely
    untidy: an executor that stages by mirroring a directory will delete what it
    does not recognise (the Slurm executor copies with `rsync --delete`), so the
    second submit silently erases the first's staged scripts.
    """
    return f"gym-job-{now.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def benchmark_artifact_dirs(target: str) -> dict[str, Path] | None:
    """The artifacts directory of each benchmark of the submission `target` names, or None when no record is found.

    `target` is a run directory holding the submission's manifest, or a Gym job ID in this machine's job index.
    """
    path = Path(target)
    record_path = path / MANIFEST_NAME if path.is_dir() else local_index_dir() / f"{target}.json"
    if not record_path.is_file():
        return None
    record = SubmissionRecord.load(json.loads(record_path.read_text()))
    return {benchmark.benchmark: Path(benchmark.job_dir) / "artifacts" for benchmark in record.benchmarks}


def local_index_dir() -> Path:
    """Where this machine remembers its own submissions."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "nemo-gym" / "jobs"


def list_submissions(*, limit: int | None = None) -> list[SubmissionRecord]:
    """The submissions in this machine's job index, newest first, at most `limit` of them.

    Reads the index only, so it needs no executor and no network. A file that cannot be read as a
    record is skipped with a warning: one bad file must not hide the rest.
    """
    index = local_index_dir()
    records: list[SubmissionRecord] = []
    for path in sorted(index.glob("*.json")) if index.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Job record must be a JSON object")
            records.append(SubmissionRecord.load(payload))
        except (OSError, ValueError) as error:
            logger.warning("Skipping %s in the job index: %s", path, error)
    records.sort(key=lambda record: record.submitted_at, reverse=True)
    return records if limit is None else records[:limit]


def load_submission_record(gym_job_id: str) -> SubmissionRecord:
    """The record of submission `gym_job_id`: the run's manifest when its run dir is on this machine, else
    this machine's job index. Both are written by `BaseExecutor.persist()`."""
    index = local_index_dir() / f"{gym_job_id}.json"
    if not index.is_file():
        raise ValueError(
            f"No submission {gym_job_id!r} in this machine's job index ({local_index_dir()}). If its run directory "
            "is here, `gym eval resubmit ... --run-dir RUN_DIR` restores the entry from the run's gym-job.json."
        )
    record = SubmissionRecord.load(json.loads(index.read_text(encoding="utf-8")))
    manifest = Path(record.run_dir) / MANIFEST_NAME
    if manifest.is_file():
        record = SubmissionRecord.load(json.loads(manifest.read_text(encoding="utf-8")))
    return record


__getattr__ = moved_names_getattr(__name__)
