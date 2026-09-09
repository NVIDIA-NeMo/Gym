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

import getpass
import re
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from nemo_gym.orchestration.api import SlurmComputeConfig, SubmitConfig
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import Connection, LocalConnection, get_connection
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import (
    MANIFEST_NAME,
    BenchmarkJob,
    SubmissionRecord,
    dumps,
    gym_version,
    new_gym_job_id,
    utc_timestamp,
    write_local_index,
)


# Each sbatch reports its own result on a line that names its benchmark, so a
# failure cannot shift the benchmarks after it onto the wrong job ids.
_MARKER = "__GYM_JOB:"

# Benchmark names are interpolated into the marker line, so they must not carry
# its delimiter or anything the shell would act on.
_VALID_BENCHMARK_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

# Real `sbatch --parsable` output is a bare "<id>" or a federated "<id>;<cluster>".
# stderr is merged onto the same line (see `_sbatch_command`), and a *successful*
# sbatch can still print a warning (job_submit plugin notices, QOS/ntasks
# adjustments are routine on production Slurm) — so only the payload's last
# whitespace-delimited token is checked against this, not the whole line.
_JOB_ID_RE = re.compile(r"^\d+(;\S+)?$")


def _validate_benchmark_names(benchmarks: list[str]) -> None:
    """Fail before anything is staged or copied.

    `_sbatch_command` raises this same check, but only once submission is
    already underway — after staging, connecting, mount validation and the
    rsync. Checking here first means a bad name fails before any of that, and
    fails on `--dry-run` too, instead of the dry run printing a clean script
    listing for a benchmark that could never actually be submitted.
    """
    bad = [name for name in benchmarks if not _VALID_BENCHMARK_NAME.match(name)]
    if bad:
        raise ValueError(
            f"Invalid benchmark name(s) {', '.join(map(repr, bad))}: names must match "
            f"{_VALID_BENCHMARK_NAME.pattern} so they can be reported back from the submit script."
        )


def _sbatch_command(benchmark: str, script: Path) -> str:
    """One `sbatch` that reports its own benchmark, exit status and output.

    `rc` is captured immediately: `$?` after the `tr` pipeline would be *tr's*
    status, which is zero however badly sbatch failed. `tr` flattens sbatch's
    multi-line error messages so the whole result stays on one marker line.
    """
    if not _VALID_BENCHMARK_NAME.match(benchmark):
        raise ValueError(
            f"Invalid benchmark name {benchmark!r}: names must match {_VALID_BENCHMARK_NAME.pattern} "
            "so they can be reported back from the submit script."
        )
    return (
        f"out=$(sbatch --parsable {shlex.quote(str(script))} 2>&1); rc=$?; "
        "out=$(echo \"$out\" | tr '\\n' ' '); "
        f'echo "{_MARKER}{benchmark}:$rc:$out"'
    )


def _parse_sbatch_results(output: str) -> dict[str, tuple[str | None, str | None]]:
    """Benchmark name to `(job_id, error)`, exactly one of which is set."""
    results: dict[str, tuple[str | None, str | None]] = {}
    for line in output.splitlines():
        if not line.startswith(_MARKER):
            continue
        benchmark, _, rest = line[len(_MARKER) :].partition(":")
        status, _, payload = rest.partition(":")
        payload = payload.strip()
        if status == "0":
            tokens = payload.split()
            candidate = tokens[-1] if tokens else ""
            if _JOB_ID_RE.match(candidate):
                # A federated sbatch answers "jobid;cluster"; the ledger wants the id.
                results[benchmark] = (candidate.split(";")[0], None)
            else:
                # Exit 0 but the last token isn't an id: no output at all, or a
                # warning with nothing that looks like a job id after it. Either
                # way there is no id to trust, so this is a failure, not a
                # success with garbage (or an empty string) in job_id.
                results[benchmark] = (None, payload or "sbatch exited 0 with no output")
        else:
            results[benchmark] = (None, payload or f"sbatch exited {status}")
    return results


def _validate_mounts(config: SubmitConfig, conn: Connection) -> None:
    entries = [("driver", m) for m in config.driver.mounts]
    for name, service in config.services.items():
        entries += [(f"services.{name}", m) for m in service.mounts]
    srcs_by_label = [(label, mount.split(":")[0]) for label, mount in entries]
    if not srcs_by_label:
        return

    if isinstance(conn, LocalConnection):
        missing = {src for _, src in srcs_by_label if not Path(src).exists()}
    else:
        # SSHConnection.run() pipes commands as a bash script, so || works fine.
        checks = [f'test -e {shlex.quote(src)} || echo "__GYM_MISSING:{src}"' for _, src in srcs_by_label]
        output = conn.run(checks)
        missing = {line[len("__GYM_MISSING:") :] for line in output.splitlines() if line.startswith("__GYM_MISSING:")}

    if missing:
        bad = [(label, src) for label, src in srcs_by_label if src in missing]
        details = "\n".join(f"  {label}: {src!r}" for label, src in bad)
        raise ValueError(f"Mount src paths do not exist:\n{details}")


def _utc_now() -> datetime:
    """A seam: tests freeze this to prove two submits in the same second still
    get distinct run directories."""
    return datetime.now(timezone.utc)


class SlurmExecutor(BaseExecutor):
    """Slurm executor for Pyxis-enabled clusters (https://github.com/NVIDIA/pyxis).

    Every service and the driver are launched via `srun --container-image` so they
    run inside the container specified in their config. Health checks run as plain
    bash inside the sbatch script (no container needed — they just poll HTTP).
    """

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        compute = next(iter(config.compute.values()))
        cluster = next(iter(config.compute))
        benchmark_names = list(config.driver.benchmarks)
        _validate_benchmark_names(benchmark_names)
        now = _utc_now()
        gym_job_id = new_gym_job_id(now)
        remote_run_dir = Path(config.job.output_path) / gym_job_id

        if dry_run:
            self._dry_run(config, compute, remote_run_dir)
            return None

        with tempfile.TemporaryDirectory(prefix="gym-submit-") as staging_str:
            staging = self._stage(config, compute, remote_run_dir, Path(staging_str))
            with get_connection(compute.hostname) as conn:
                _validate_mounts(config, conn)
                conn.copy(staging, remote_run_dir)
                output = conn.run(
                    [_sbatch_command(name, remote_run_dir / name / "job.sh") for name in benchmark_names]
                )
                record = self._build_record(cluster, compute, gym_job_id, now, remote_run_dir, benchmark_names, output)
                # Best-effort local copy first: it cannot itself fail the submit,
                # and if the manifest write below fails, this is what preserves a
                # parseable record for the by-hand recovery that error asks for.
                write_local_index(record)
                # Inside the connection: the manifest is the durable copy, and a
                # run directory without one is the state this all exists to remove.
                # The jobs are already queued by now, so the error has to name them
                # or they are stranded with no record anywhere.
                try:
                    conn.write_text(remote_run_dir / MANIFEST_NAME, dumps(record))
                except Exception as error:
                    queued = ", ".join(f"{b.benchmark}={b.job_id}" for b in record.benchmarks if b.job_id)
                    raise RuntimeError(
                        f"Submitted jobs but could not write the manifest to "
                        f"{remote_run_dir / MANIFEST_NAME}: {error}. "
                        f"Already queued: {queued or 'nothing'}. Record these by hand before collecting."
                    ) from error

        return record

    def _build_record(
        self,
        cluster: str,
        compute: SlurmComputeConfig,
        gym_job_id: str,
        now: datetime,
        remote_run_dir: Path,
        benchmark_names: list[str],
        output: str,
    ) -> SubmissionRecord:
        results = _parse_sbatch_results(output)
        benchmarks = []
        for name in benchmark_names:
            # A benchmark absent from the output produced no marker line at all —
            # the shell died before reaching it, or the transport truncated. That
            # is a failure, not a success with a missing id.
            job_id, error = results.get(name, (None, "sbatch produced no result for this benchmark"))
            benchmarks.append(
                BenchmarkJob(
                    benchmark=name,
                    job_dir=str(remote_run_dir / name),
                    job_id=job_id,
                    error=error,
                )
            )
        return SubmissionRecord(
            gym_job_id=gym_job_id,
            gym_version=gym_version(),
            submitted_at=utc_timestamp(now),
            run_dir=str(remote_run_dir),
            cluster=cluster,
            executor="slurm",
            hostname=compute.hostname,
            submitted_by=getpass.getuser(),
            benchmarks=benchmarks,
        )

    def _dry_run(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path) -> None:
        print(f"[dry-run] remote run dir: {remote_run_dir}")
        for name, benchmark in config.driver.benchmarks.items():
            script = build_sbatch_script(config, name, benchmark, compute, remote_run_dir / name)
            print(f"\n{'=' * 60}")
            print(f"[dry-run] sbatch script for benchmark: {name}")
            print(f"{'=' * 60}")
            print(script)

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path, staging: Path) -> Path:
        for name, benchmark in config.driver.benchmarks.items():
            bench_dir = staging / name
            bench_dir.mkdir()
            (bench_dir / "logs").mkdir()
            (bench_dir / "artifacts").mkdir()
            script = build_sbatch_script(config, name, benchmark, compute, remote_run_dir / name)
            (bench_dir / "job.sh").write_text(script)
        return staging
