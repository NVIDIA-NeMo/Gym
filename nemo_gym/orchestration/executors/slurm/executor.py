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

"""``SlurmExecutor``: render a run's job scripts, submit them with sbatch and record the submission.

Its recovery of a run's chain (``chain``) and its judge-only jobs (``judge_job``) are mixed in.
"""

import getpass
import re
import shlex
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import __version__
from nemo_gym.eval.recovery.evidence import (
    CollectedEvidence,
)
from nemo_gym.orchestration.api import (
    SAVED_CONFIG,
    SlurmComputeConfig,
    SubmitConfig,
    VllmPDServiceConfig,
)
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import (
    Connection,
    get_connection,
)
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_CONFIG_NAME,
    COLLECTOR_DIR,
    otel_active,
    render_collector_config,
    resolve_token,
    validate_destination,
    validate_gym_telemetry,
)
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint
from nemo_gym.orchestration.executors.slurm.chain import (
    ChainRecovery,
)
from nemo_gym.orchestration.executors.slurm.judge_job import (
    JudgeJobs,
)
from nemo_gym.orchestration.executors.slurm.scheduler import (
    _RESUBMIT_LOCK,
    _VALID_BENCHMARK_NAME,
    _chain_live_check,
    _parse_sbatch_results,
    _sbatch_command,
    _submission_env_exports,
    _submission_exports,
)
from nemo_gym.orchestration.executors.slurm_script import (
    _resolve_env,
    build_sbatch_script,
)
from nemo_gym.orchestration.jobs import (
    ARTIFACTS_DIRNAME,
    LOGS_DIRNAME,
    RESOLVED_CONFIG_NAME,
    BenchmarkJob,
    SubmissionRecord,
    installed_gym_commit,
    new_gym_job_id,
    utc_now,
    utc_timestamp,
)


# How long a recovery's post-run MLflow export may take on the login host, its Gym install included.
_EXPORT_TIMEOUT_S = 3600


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


def _validate_mounts(config: SubmitConfig, conn: Connection) -> None:
    entries = [("driver", m) for m in config.driver.mounts]
    for name, service in config.services.items():
        entries += [(f"services.{name}", m) for m in service.mounts]
        if isinstance(service, VllmPDServiceConfig):
            for tier in ("prefill", "decode"):
                entries += [(f"services.{name}.{tier}", m) for m in getattr(service, tier).mounts]
    srcs_by_label = [(label, mount.split(":")[0]) for label, mount in entries]
    if not srcs_by_label:
        return

    # Both connections pipe commands to bash, so one shell program checks the
    # mounts wherever the submit is going -- and the local path exercises the
    # same code the SSH path runs.
    checks = [f'test -e {shlex.quote(src)} || echo "__GYM_MISSING:{src}"' for _, src in srcs_by_label]
    output = conn.run(checks)
    missing = {line[len("__GYM_MISSING:") :] for line in output.splitlines() if line.startswith("__GYM_MISSING:")}

    if missing:
        bad = [(label, src) for label, src in srcs_by_label if src in missing]
        details = "\n".join(f"  {label}: {src!r}" for label, src in bad)
        raise ValueError(f"Mount src paths do not exist:\n{details}")


class SlurmExecutor(ChainRecovery, JudgeJobs, BaseExecutor):
    """Slurm executor for Pyxis-enabled clusters (https://github.com/NVIDIA/pyxis).

    Every service and the driver are launched via `srun --container-image` so they
    run inside the container specified in their config. Health checks run as plain
    bash inside the sbatch script (no container needed — they just poll HTTP).
    """

    supports_resumable = True
    name = "slurm"

    def run(self, config: SubmitConfig, *, dry_run: bool = False) -> SubmissionRecord | None:
        compute = next(iter(config.compute.values()))
        cluster = next(iter(config.compute))
        submission_env = _submission_env_exports(compute) if not dry_run else []
        benchmark_names = list(config.driver.benchmarks)
        _validate_benchmark_names(benchmark_names)
        # The remote lookup overlaps the other validation, staging, the SSH connect and the mount
        # check; it is joined before anything is copied or queued, so a bad ref still fails first.
        pool = ThreadPoolExecutor(max_workers=1)
        ref_check = pool.submit(validate_gym_install_ref, config)
        pool.shutdown(wait=False)
        token = None
        if otel_active(config):
            token = resolve_token(config)
            validate_destination(config)
            validate_gym_telemetry(config)
        now = utc_now()
        gym_job_id = new_gym_job_id(now)
        remote_run_dir = Path(config.job.output_path) / gym_job_id

        if dry_run:
            ref_check.result()
            self._dry_run(config, compute, remote_run_dir)
            return None

        with tempfile.TemporaryDirectory(prefix="gym-submit-") as staging_str:
            staging = self._stage(config, compute, remote_run_dir, Path(staging_str))
            with get_connection(compute.hostname) as conn:
                _validate_mounts(config, conn)
                ref_check.result()
                conn.copy(staging, remote_run_dir)
                token_export = [f"export {config.otel.token_env}={shlex.quote(token)}"] if token is not None else []
                output = conn.run(
                    submission_env
                    + token_export
                    + [_sbatch_command(name, remote_run_dir / name / "job.sh") for name in benchmark_names]
                )
                record = self._build_record(cluster, compute, gym_job_id, now, remote_run_dir, benchmark_names, output)
                record.environment = self.submission_environment(
                    config,
                    required_env=compute.forward_env,
                    forwarded_env=[*compute.forward_env, *([config.otel.token_env] if token is not None else [])],
                )
                # Inside the connection, because that is the transport persist()
                # needs and reopening one would cost a second connection per
                # submit. Ordering and failure handling live in the base class.
                self.persist(record, config, conn.write_text)

        return record

    def export_tracking(
        self, record: SubmissionRecord, benchmark: str, evidence: CollectedEvidence | None
    ) -> dict[str, Any]:
        """Run `gym eval export mlflow` on the run's saved artifacts from the login host, and report its result.

        The export runs where the remote Store reader runs, with the run's saved pinned Gym install (and its MLflow
        extra) in a temporary checkout: it only reads the saved artifacts and talks to the tracking server, so it needs
        no allocation. It takes the benchmark's saved `mlflow_*` and `upload_rollouts` run settings, and its
        credentials as the driver did: the submission's forwarded environment and the driver's env. Gym's export takes
        the run lock, prepares the health metrics and writes the receipt bound to the artifact digests, and skips a run
        that already has one. The resubmit lock is held and no job of the chain may be live meanwhile.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        def failed(error: str) -> dict[str, Any]:
            return {"exporter": "mlflow", "host": "login", "state": "failed", "error": error}

        job_dir = Path(record.benchmark_job(benchmark).job_dir)
        try:
            with get_connection(record.hostname) as conn:
                config = self._saved_config(conn, Path(record.run_dir))
                install = config.driver.gym_install
                if install is None or re.fullmatch(r"[0-9a-fA-F]{40}", install.ref) is None:
                    return failed("the post-run export requires the run's saved immutable Gym ref")
                run = config.driver.benchmarks[benchmark].run
                settings = {k: v for k, v in run.items() if k.startswith("mlflow_") or k == "upload_rollouts"}
                rollouts = job_dir / "artifacts" / "rollouts.jsonl"
                command = " ".join(
                    ["gym eval export mlflow", shlex.quote(f"+rollouts_jsonl_fpath={rollouts}")]
                    + flatten_run_args(settings)
                )
                entrypoint = render_driver_entrypoint(
                    repo=install.repo,
                    ref=install.ref,
                    prepare_cmd=None,
                    command=command,
                    extras=("mlflow",),
                    cleanup=True,
                )
                chain = evidence.job_chain if evidence is not None else []
                commands = _submission_exports(config) + [
                    'fail() { echo "Error: $*" >&2; exit 1; }',
                    f"cd {shlex.quote(str(job_dir))} || fail 'job directory unavailable'",
                    f"(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail 'another recovery holds the lock'",
                    f"trap 'rm -f {_RESUBMIT_LOCK}' EXIT",
                    *(_chain_live_check(chain, "a job of the chain is still live") if chain else []),
                    "unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME",
                    f"_gym_out=$({_resolve_env(config.driver.env)}timeout --kill-after=10s {_EXPORT_TIMEOUT_S}s "
                    f"{entrypoint} 2>&1); _gym_rc=$?",
                    "printf '%s\n' \"$_gym_out\" | tail -n 40",
                    'echo "__GYM_EXPORT_RC:$_gym_rc"',
                ]
                output = conn.run(commands)
        except (RuntimeError, OSError, ValueError) as error:
            return failed(str(error))
        state = re.findall(r"^Export mlflow: (\w+)$", output, re.MULTILINE)
        run_id = re.findall(r"^  run_id: (\S+)$", output, re.MULTILINE)
        rc = re.findall(r"^__GYM_EXPORT_RC:(\d+)$", output, re.MULTILINE)
        result = {"exporter": "mlflow", "host": "login", "state": state[-1] if state else "failed"}
        if run_id:
            result["run_id"] = run_id[-1]
        if result["state"] not in ("succeeded", "skipped") or rc != ["0"]:
            tail = [line for line in output.splitlines() if not line.startswith("__GYM_EXPORT_RC:")][-5:]
            result |= {"state": "failed", "error": f"exit code {rc[-1] if rc else 'unknown'}: " + " | ".join(tail)}
        return result

    def installed_commit(self) -> str | None:
        """The Gym commit this launcher runs; the same value the judge-only job and a repin are checked against."""
        return installed_gym_commit()

    def gym_pin(self, record: SubmissionRecord) -> str | None:
        """The full Gym commit the run's saved config pins its driver to; None when there is none or it is unreadable."""
        try:
            with get_connection(record.hostname) as conn:
                install = self._saved_config(conn, Path(record.run_dir)).driver.gym_install
        except (OSError, RuntimeError, ValueError):
            return None
        ref = install.ref if install is not None else None
        return ref if ref is not None and re.fullmatch(r"[0-9a-f]{40}", ref) else None

    @staticmethod
    def _saved_config(conn: Connection, run_dir: Path) -> SubmitConfig:
        """The resolved config `persist()` saved for this run, so a resubmit repeats what submit did."""
        resolved = run_dir / RESOLVED_CONFIG_NAME
        text = conn.run([f"cat {shlex.quote(str(resolved))}"])
        return SubmitConfig.model_validate(yaml.safe_load(text), context={SAVED_CONFIG: True})

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
            gym_version=__version__,
            gym_commit=installed_gym_commit(),
            submitted_at=utc_timestamp(now),
            run_dir=str(remote_run_dir),
            cluster=cluster,
            executor=self.name,
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
            if otel_active(config):
                print(f"\n{'=' * 60}")
                print(f"[dry-run] {COLLECTOR_DIR}/{COLLECTOR_CONFIG_NAME} for benchmark: {name}")
                print(f"{'=' * 60}")
                print(render_collector_config(config, name, remote_run_dir / name))

    def _stage(self, config: SubmitConfig, compute: SlurmComputeConfig, remote_run_dir: Path, staging: Path) -> Path:
        # rsync -a copies these modes to the cluster. The temp dir is created 0700, which would
        # hide the run from other users; job.sh holds resolved secrets, so it stays owner-only.
        staging.chmod(0o755)
        for name, benchmark in config.driver.benchmarks.items():
            bench_dir = staging / name
            bench_dir.mkdir()
            (bench_dir / LOGS_DIRNAME).mkdir()
            (bench_dir / ARTIFACTS_DIRNAME).mkdir()
            script = build_sbatch_script(config, name, benchmark, compute, remote_run_dir / name)
            job_script = bench_dir / "job.sh"
            job_script.write_text(script)
            job_script.chmod(0o600)
            if otel_active(config):
                (bench_dir / COLLECTOR_DIR).mkdir()
                (bench_dir / COLLECTOR_DIR / COLLECTOR_CONFIG_NAME).write_text(
                    render_collector_config(config, name, remote_run_dir / name)
                )
        return staging
