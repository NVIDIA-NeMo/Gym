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
  tomorrow `gym eval status` -- can load a record without pulling Slurm, SSH and
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

import json
import logging
import os
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

from pydantic import BaseModel


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


# The nemo_gym package directory of the code that is running, however it got
# onto sys.path: an install, an editable checkout, or a plain source tree.
_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def installed_gym_commit() -> str | None:
    """Return the git commit of the nemo-gym that is running, or None when it cannot be determined.

    A git install (``pip install git+...``) records its commit in the distribution
    metadata. Otherwise the running package's own source tree is asked: a git
    checkout reports its HEAD, suffixed ``-dirty`` when it has uncommitted changes.
    """
    commit = _git_install_commit()
    if commit is not None:
        return commit
    return _source_checkout_commit(_PACKAGE_DIR)


def _git_install_commit() -> str | None:
    """The commit recorded in the distribution's PEP 610 ``direct_url.json``, or None."""
    try:
        raw = distribution("nemo-gym").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if raw is None:
        return None
    commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    return str(commit) if commit else None


def _source_checkout_commit(package_dir: Path) -> str | None:
    """HEAD of the git checkout ``package_dir`` is tracked in, ``-dirty`` when it has uncommitted changes."""
    try:
        tracked = _git(package_dir, "ls-files", "--error-unmatch", "__init__.py")
    except FileNotFoundError:
        return _unknown_commit("git is not available to inspect %s.", package_dir)
    if tracked.returncode != 0:
        return _unknown_commit("nemo-gym is not a git install and %s is not tracked in a git checkout.", package_dir)
    head = _git(package_dir, "rev-parse", "HEAD")
    status = _git(package_dir, "status", "--porcelain")
    if head.returncode != 0 or status.returncode != 0:
        return _unknown_commit("git failed in %s: %s", package_dir, (head.stderr or status.stderr).strip())
    commit = head.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True)


def _unknown_commit(reason: str, *args: object) -> None:
    logger.warning("Cannot determine the installed Gym commit: " + reason, *args)
    return None


class BenchmarkJob(BaseModel):
    """One benchmark's submission.

    `job_id` is None exactly when the executor failed to enqueue this benchmark
    (for the Slurm executor, a failed `sbatch`), in which case `error` says why.
    The other benchmarks in the same submission are unaffected -- one failure
    does not discard the record for the ones that did start.
    """

    benchmark: str
    job_dir: str
    job_id: str | None = None
    error: str | None = None


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
    records one (see `installed_gym_commit`). A `-dirty` suffix marks a checkout
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
    schema_version: int = SCHEMA_VERSION

    @property
    def failed(self) -> list[BenchmarkJob]:
        return [b for b in self.benchmarks if b.job_id is None]

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


def local_index_dir() -> Path:
    """Where this machine remembers its own submissions."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "nemo-gym" / "jobs"
