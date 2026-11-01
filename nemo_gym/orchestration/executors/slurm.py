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
import inspect
import json
import math
import os
import re
import shlex
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from nemo_gym import __version__
from nemo_gym.orchestration.api import (
    HOST_ENV_REFS,
    SAVED_CONFIG,
    SlurmComputeConfig,
    SubmitConfig,
    VllmPDServiceConfig,
)
from nemo_gym.orchestration.executors.base import BaseExecutor
from nemo_gym.orchestration.executors.connection import Connection, LocalConnection, get_connection
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref
from nemo_gym.orchestration.executors.otel import (
    COLLECTOR_CONFIG_NAME,
    COLLECTOR_DIR,
    collector_config_path,
    otel_active,
    render_collector_config,
    resolve_token,
    validate_destination,
    validate_gym_telemetry,
)
from nemo_gym.orchestration.executors.resume_script import (
    ACCOUNTING_UNKNOWN_PREFIX,
    ALLOCATION_ACCOUNTING_MARKER,
    BOUNDED_BUDGET_MARKER,
    KEEP_BUDGET_ENV,
    PROLOGUE_MARKER,
    REQUEUE_RECONCILED_NOTE,
    REQUEUE_UNKNOWN_REASON,
    RESUBMIT_ENV,
    RESUBMIT_LOCK_NAME,
    RETRY_EXTENSION_MARKER,
    RETRY_EXTENSION_NAME,
    SCHEDULER_CAUSE_MARKER,
    _walltime_to_seconds,
)
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint
from nemo_gym.orchestration.executors.slurm_script import build_sbatch_script
from nemo_gym.orchestration.jobs import (
    RESOLVED_CONFIG_NAME,
    BenchmarkJob,
    SubmissionRecord,
    installed_gym_commit,
    new_gym_job_id,
    utc_now,
    utc_timestamp,
)
from nemo_gym.orchestration.recovery import (
    CHAIN_ACCOUNTING_FORMAT,
    COLLECT_SCRIPT,
    FOLLOWUP_ACCOUNTING_FORMAT,
    CollectedEvidence,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryOutcomePolicyConfig,
    _managed_trial_metadata,
    accounted_chain_walltime,
    accounted_requeued_chain_walltime,
    allocation_cut_from_accounting,
    chain_accounting,
    decide_from_collected,
    live_chain_job,
    select_managed_evidence,
    stopped_followup,
)


# Executed only inside the saved pinned install. The projection contains no saved answer text.
_REMOTE_MANAGED_READER = r"""
import json, subprocess, sys
from pathlib import Path
import nemo_gym.orchestration.recovery as recovery
job, expected_ref = Path(sys.argv[1]), sys.argv[2]
raw = json.load(sys.stdin)
root = Path.cwd().resolve()
def check_source():
    if not Path(recovery.__file__).resolve().is_relative_to(root / "nemo_gym"):
        raise ValueError("reader import is outside its pinned checkout")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != expected_ref:
        raise ValueError("reader checkout does not match the saved immutable ref")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("reader checkout changed")
def identity():
    stat = job.stat()
    return [str(job.resolve()), stat.st_dev, stat.st_ino]
check_source()
if not job.is_absolute() or raw.get("source_job_identity") != identity():
    raise ValueError("reader job path does not match the collected directory")
collected = recovery.CollectedEvidence.model_validate(raw)
selected = recovery.select_managed_evidence(collected, job)
projection = json.loads(selected.model_dump_json())
# A saved install whose selection already derives these facts is not read a second time (one Store parse and two
# fewer source snapshots). An older one may predate them: derive them with the caller's helper on its own APIs.
import inspect
derived = "_managed_trial_metadata(" in inspect.getsource(recovery.select_managed_evidence)
if not derived:
    projection["checkpoint_enabled"] = None
if not derived and selected.managed_selection is not None and selected.managed_reader_error is None:
    artifacts = job / "artifacts"
    def snapshot():
        return {name: recovery._managed_source_digest(artifacts / name) for name in collected.source_digests}
    # A changed source stops the reader; only a missing optional API leaves the metadata unknown.
    if snapshot() != collected.source_digests:
        raise ValueError("managed source changed after Store selection")
    try:
        from nemo_gym.rollout_store import RolloutStore
    except ImportError:
        RolloutStore = None
    if RolloutStore is not None:
        store = RolloutStore.read(artifacts / "rollouts.jsonl", import_legacy=False)
        projection["results"], projection["checkpoint_enabled"] = _managed_trial_metadata(
            artifacts, collected.source_digests, collected.quality, projection["results"], store
        )
    if snapshot() != collected.source_digests:
        raise ValueError("managed source changed while reading trial metadata")
check_source()
if raw.get("source_job_identity") != identity():
    raise ValueError("reader job directory changed")
print("__GYM_MANAGED:" + json.dumps(projection))
"""


# Appended in the job dir when `gym eval resubmit --recover --extend-retries` records an extension.
RETRY_EXTENSION_JOURNAL_NAME = "retry-extension-journal.jsonl"
# Appended in the run dir when `gym eval resubmit --recover --repin` moves a benchmark to another Gym commit.
REPIN_JOURNAL_NAME = "repin-journal.jsonl"
# Appended in the run dir when `gym eval resubmit --recover --max-walltime` raises a benchmark's walltime.
WALLTIME_JOURNAL_NAME = "walltime-extension-journal.jsonl"


def _repinned_config(config: SubmitConfig, benchmark: str, ref: str) -> SubmitConfig:
    """``config`` with the driver's Gym install at ``ref``, and the benchmark's MLflow pin tags that named the old one."""
    install = config.driver.gym_install
    old = install.ref if install is not None else None
    bench = config.driver.benchmarks[benchmark]
    run = dict(bench.run)
    tags = run.get("mlflow_run_tags")
    if isinstance(tags, Mapping) and old:
        # Saved tag values are Hydra-quoted strings ('<sha>'); keep the quoting, change only the old commit.
        run["mlflow_run_tags"] = {
            key: value.replace(old, ref) if key in ("gym_sha", "gym_commit") and isinstance(value, str) else value
            for key, value in tags.items()
        }
    driver = config.driver.model_copy(
        update={
            "gym_install": install.model_copy(update={"ref": ref}) if install is not None else None,
            "benchmarks": {**config.driver.benchmarks, benchmark: bench.model_copy(update={"run": run})},
        }
    )
    return config.model_copy(update={"driver": driver})


def _walltime_config(config: SubmitConfig, cluster: str, benchmark: str, walltime: str) -> SubmitConfig:
    """``config`` with the Slurm walltime ``walltime`` and a chain cap of at least that; every other field as saved."""
    bench = config.driver.benchmarks[benchmark]
    resume = bench.resume_config
    update = {}
    if resume is not None and resume.max_walltime is not None:
        cap = max(resume.max_walltime, walltime, key=_walltime_to_seconds)
        update["resumable"] = resume.model_copy(update={"max_walltime": cap})
    driver = config.driver.model_copy(
        update={"benchmarks": {**config.driver.benchmarks, benchmark: bench.model_copy(update=update)}}
    )
    compute = {**config.compute, cluster: config.compute[cluster].model_copy(update={"walltime": walltime})}
    return config.model_copy(update={"driver": driver, "compute": compute})


def _reconcile_requeued_chain(
    collected: CollectedEvidence, rows: list[tuple[str, str, int, int]]
) -> CollectedEvidence:
    """Know a requeued chain's budget again once every incarnation has a terminal scheduler record.

    Only the requeue marker is reconciled: the counters themselves stayed readable and the missing fact is the elapsed
    time of the requeued incarnations, which complete terminal records give. Any other unknown reason stays unknown.
    """
    states = collected.counter_states or {}
    if (
        collected.accounting_unknown_reason != REQUEUE_UNKNOWN_REASON
        or states.get("infra_retries") not in ("absent", "ok")
        or states.get("walltime") not in ("absent", "ok")
    ):
        return collected
    total = accounted_requeued_chain_walltime(
        collected.job_chain, rows, stored_walltime_s=collected.walltime_used_s or 0
    )
    if total is None:
        return collected
    return collected.model_copy(
        update={
            "infra_retries_state": states["infra_retries"],
            "walltime_state": states["walltime"],
            "accounted_walltime_s": total,
            "accounting_reconciled": True,
        }
    )


# The size-aware reader bound: bootstrap plus a floor, then time per GB of saved rollout files, up to a ceiling.
_READER_FLOOR_S = 900
_READER_S_PER_GB = 100
_READER_CEILING_S = 14400
_READER_RETRY_BACKOFF_S = 10


def _remote_reader_bound_s() -> int | None:
    """The explicit reader bound from ``NEMO_GYM_REMOTE_READER_TIMEOUT_S``, clamped to [300, 14400], else None.

    Without it the login host sizes the bound from the run's rollout files (``_reader_command``).
    """
    raw = os.environ.get("NEMO_GYM_REMOTE_READER_TIMEOUT_S", "")
    try:
        return min(max(int(raw), 300), _READER_CEILING_S) if raw else None
    except ValueError:
        return None


def _remote_reader_error_text(error: BaseException) -> str:
    """Name the reader failure with its cause, not only the exception type: the end of the message, where it is."""
    detail = " ".join(str(error).split())
    if not detail:
        return f"remote Store reader unavailable: {type(error).__name__}"
    if len(detail) > 300:
        detail = "..." + detail[-300:]
    return f"remote Store reader unavailable: {type(error).__name__}: {detail}"


def _reader_command(entrypoint: str, job_dir: Path, payload: str, *, bound_s: int | None, factor: int) -> list[str]:
    """The login-host steps: size the bound from the rollout files, run the reader, report its exit code and stderr.

    The reader's own failure is reported on stdout (``__GYM_READER_*`` lines) so a timeout, a MemoryError or a
    failed bootstrap reaches the decision with its exit code and the end of its stderr; only SSH itself fails the
    command.
    """
    artifacts = job_dir / "artifacts"
    sources = " ".join(shlex.quote(str(artifacts / name)) for name in ("rollouts.jsonl", "rollouts_failures.jsonl"))
    if bound_s is None:
        size_bound = f"$(( ({_READER_FLOOR_S} + _gym_bytes * {_READER_S_PER_GB} / 1073741824) * {factor} ))"
    else:
        size_bound = str(bound_s * factor)
    return [
        "unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME",
        f"_gym_bytes=$(stat -c %s -- {sources} 2>/dev/null | awk '{{s += $1}} END {{print s + 0}}')",
        f"_gym_bound={size_bound}",
        f"(( _gym_bound > {_READER_CEILING_S} )) && _gym_bound={_READER_CEILING_S}",
        'echo "__GYM_READER_BOUND:$_gym_bound $_gym_bytes"',
        "_gym_err=$(mktemp)",
        f'timeout --kill-after=10s "${{_gym_bound}}s" {entrypoint} 3<<\'GYM_MANAGED_INPUT\' 2>"$_gym_err"\n'
        + payload
        + "\nGYM_MANAGED_INPUT",
        "_gym_rc=$?",
        'echo "__GYM_READER_RC:$_gym_rc"',
        "printf '__GYM_READER_ERR:%s\\n' \"$(tail -c 2000 \"$_gym_err\" | tr '\\n' ' ')\"",
        'rm -f -- "$_gym_err"',
    ]


def _reader_failure(output: str, elapsed_s: float) -> tuple[str | None, bool]:
    """The reader failure named from its report, and whether it timed out; (None, False) when it exited 0."""
    facts = {}
    for line in output.splitlines():
        for name in ("BOUND", "RC", "ERR"):
            if line.startswith(f"__GYM_READER_{name}:"):
                facts[name] = line.split(":", 1)[1].strip()
    rc = int(facts["RC"]) if facts.get("RC", "").isdigit() else None
    if rc == 0 or (rc is None and sum(line.startswith("__GYM_MANAGED:") for line in output.splitlines()) == 1):
        return None, False
    bound, _, size = facts.get("BOUND", "").partition(" ")
    gb = int(size) / 2**30 if size.isdigit() else None
    scope = f"the {bound} s bound for {gb:.1f} GB of rollout files" if gb is not None else f"the {bound} s bound"
    timed_out = rc in (124, 137)
    cause = (
        f"timed out after {elapsed_s:.0f} s ({scope})"
        if timed_out
        else f"exited {rc if rc is not None else 'without an exit code'} after {elapsed_s:.0f} s ({scope})"
    )
    stderr = facts.get("ERR", "")
    if len(stderr) > 400:
        stderr = "..." + stderr[-400:]
    return f"remote Store reader unavailable: {cause}" + (f"; stderr: {stderr}" if stderr else ""), timed_out


def _remote_managed_reader_script() -> str:
    """The reader with the caller's metadata helper defined after its imports, sent as one script."""
    head, body = _REMOTE_MANAGED_READER.split("job, expected_ref", 1)
    return head + inspect.getsource(_managed_trial_metadata) + "job, expected_ref" + body


def _remote_managed_evidence(
    conn: Connection, collected: CollectedEvidence, job_dir: Path, config: SubmitConfig
) -> CollectedEvidence:
    """Use the saved pinned install over the existing transport; never import Gym into plain host Python."""
    if not job_dir.is_absolute() or collected.source_job_identity is None:
        return collected.model_copy(
            update={"managed_reader_error": "remote Store collection has no directory identity"}
        )
    install = config.driver.gym_install
    if install is None or re.fullmatch(r"[0-9a-fA-F]{40}", install.ref) is None:
        return collected.model_copy(
            update={"managed_reader_error": "remote Store reader requires a saved immutable Gym ref"}
        )
    try:
        command = f"python -c {shlex.quote(_remote_managed_reader_script())} {shlex.quote(str(job_dir))} {shlex.quote(install.ref)} <&3"
        entrypoint = render_driver_entrypoint(
            repo=install.repo, ref=install.ref, prepare_cmd=None, command=command, cleanup=True
        )
        # The login host need not contain Gym. Keep this isolated from an inherited caller environment,
        # and bound bootstrap as well as selection. FD 3 keeps large inventories out of argv and away from
        # bootstrap stdin. No service, prepare script or scheduler is invoked. One retry, with twice the bound,
        # follows a timeout or a failed SSH hop; any other failure is final and reported with its cause.
        payload = collected.model_dump_json()
        for attempt in (1, 2):
            started = time.monotonic()
            try:
                output = conn.run(
                    _reader_command(entrypoint, job_dir, payload, bound_s=_remote_reader_bound_s(), factor=attempt)
                )
            except RuntimeError as error:
                if "exit code 255" not in str(error):
                    raise
                if attempt == 2:
                    raise RuntimeError(f"SSH to the login host failed twice (exit code 255): {error}") from error
                time.sleep(_READER_RETRY_BACKOFF_S)
                continue
            failure, timed_out = _reader_failure(output, time.monotonic() - started)
            if failure is None:
                break
            if not timed_out or attempt == 2:
                retried = " (after one retry with twice the bound)" if attempt == 2 else ""
                return collected.model_copy(
                    update={"managed_selection": None, "managed_reader_error": failure + retried}
                )
        lines = [
            line.removeprefix("__GYM_MANAGED:") for line in output.splitlines() if line.startswith("__GYM_MANAGED:")
        ]
        if len(lines) != 1:
            raise ValueError("remote Store reader returned no unique projection")
        selected = CollectedEvidence.model_validate_json(lines[0])
        if selected.source_job_identity is not None and selected.source_job_identity != collected.source_job_identity:
            raise ValueError("remote Store reader changed job identity")
        if selected.source_digests != collected.source_digests:
            raise ValueError("remote Store reader changed the collection snapshot")
        # Selection may only replace selected outcomes and attach authoritative attempt facts.
        allowed = {
            "results",
            "failures",
            "managed_selection",
            "managed_reader_error",
            "source_job_identity",
            "checkpoint_enabled",
        }
        # A pinned reader older than these host-collected fields drops them; the host collection owns them either way.
        host_owned = ("accounting_snapshot", "coverage_allocation_id", "accounting_unknown_reason", "counter_states")
        compared = allowed | set(host_owned) | {"accounting_reconciled"}
        if selected.model_dump(exclude=compared) != collected.model_dump(exclude=compared):
            raise ValueError("remote Store reader changed collection evidence")
        return selected.model_copy(
            update={"source_job_identity": collected.source_job_identity}
            | {name: getattr(collected, name) for name in host_owned}
        )
    except (RuntimeError, OSError, ValueError) as error:
        return collected.model_copy(
            update={
                "managed_selection": None,
                "managed_reader_error": _remote_reader_error_text(error),
            }
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

# Held by `gym eval resubmit` from its scheduler check to its sbatch, so two resubmits cannot both queue a job.
_RESUBMIT_LOCK = RESUBMIT_LOCK_NAME


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


def _submission_env_exports(compute: SlurmComputeConfig) -> list[str]:
    """Forward explicitly selected values over SSH stdin, never into artifacts."""
    missing = [name for name in compute.forward_env if name not in os.environ]
    if missing:
        raise ValueError(f"compute.forward_env variables are unset: {', '.join(missing)}")
    return [f"export {name}={shlex.quote(os.environ[name])}" for name in sorted(set(compute.forward_env))]


def _submission_exports(config: SubmitConfig) -> list[str]:
    """What `run()` exports ahead of sbatch: the forwarded values, then the telemetry token when it is on."""
    exports = _submission_env_exports(next(iter(config.compute.values())))
    token = resolve_token(config)
    validate_destination(config)
    if token is not None:
        exports.append(f"export {config.otel.token_env}={shlex.quote(token)}")
    return exports


class SlurmExecutor(BaseExecutor):
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

    def recovery_evidence(
        self,
        record: SubmissionRecord,
        benchmark: str,
        *,
        score_key: list[str | int] | None = None,
        score_agent: str | None = None,
        score_environment: str | None = None,
        outcome_policy: RecoveryOutcomePolicyConfig | None = None,
        reader_ref: str | None = None,
        extend_retries: int = 0,
    ) -> tuple[RecoveryDecision, CollectedEvidence | None]:
        """Collect the job dir's identities, failures, quality summary, named score and chain budget over SSH.

        ``reader_ref`` runs the Store reader at that full commit instead of the saved one: a repinned run resumes there.
        """
        job_dir = Path(record.benchmark_job(benchmark).job_dir)
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
            resume = config.driver.benchmarks[benchmark].resume_config
            if resume is None:
                stop = RecoveryDecision(
                    action=RecoveryAction.STOP,
                    reason=f"{benchmark} stores no resume_config, so its infra-retry budget is unknown",
                    expected=0,
                    counts={},
                    identities={},
                    evidence=[],
                )
                return stop, None
            selector = {k: v for k, v in (("agent", score_agent), ("environment", score_environment)) if v is not None}
            args = [shlex.quote(str(job_dir))]
            if score_key:
                args += [shlex.quote(json.dumps(score_key)), shlex.quote(json.dumps(selector))]
            collected = CollectedEvidence.model_validate_json(
                conn.run([f"python3 -c {shlex.quote(COLLECT_SCRIPT)} {' '.join(args)}"])
            )
            if self._saved_config(conn, Path(record.run_dir)).model_dump() != config.model_dump():
                raise RuntimeError(
                    "saved submission config changed while collecting recovery evidence; collect it again"
                )
            if collected.managed_store:
                if isinstance(conn, LocalConnection):
                    collected = select_managed_evidence(collected, job_dir)
                else:
                    reader_config = _repinned_config(config, benchmark, reader_ref) if reader_ref else config
                    collected = _remote_managed_evidence(conn, collected, job_dir, reader_config)
                    try:
                        if self._saved_config(conn, Path(record.run_dir)).model_dump() != config.model_dump():
                            raise ValueError("saved submission config changed during remote selection")
                    except (RuntimeError, OSError, ValueError):
                        collected = collected.model_copy(
                            update={
                                "managed_selection": None,
                                "managed_reader_error": "saved submission config changed or could not be rechecked",
                            }
                        )
                    if collected.managed_selection is None:
                        return decide_from_collected(
                            collected,
                            max_infra_retries=resume.max_retries,
                            max_walltime_s=_walltime_to_seconds(resume.max_walltime)
                            if resume.max_walltime is not None
                            else None,
                            live_job=None,
                            outcome_policy=outcome_policy,
                            extend_retries=extend_retries,
                        ), collected
            workdirs = conn.run(["squeue --me --all --noheader --format=%Z"]).split()
            allocation_cut = False
            rows = None
            stopped = None
            if collected.job_chain:
                try:
                    chain = ",".join(collected.job_chain)
                    rows = chain_accounting(
                        conn.run([f"sacct -D -n -P -X -j {shlex.quote(chain)} --format={CHAIN_ACCOUNTING_FORMAT}"])
                    )
                except (RuntimeError, OSError):
                    rows = None
            # A continuation of the chain that still waits or runs is live, wherever the scheduler runs it from.
            live_job_name = live_chain_job(rows, collected.job_chain)
            if collected.job_chain and str(job_dir) not in workdirs and live_job_name is None:
                if rows is not None:
                    allocation_cut = allocation_cut_from_accounting(rows, collected.coverage_allocation_id)
                    collected = _reconcile_requeued_chain(collected, rows)
                if resume.max_walltime is not None and not collected.accounting_reconciled:
                    # Unavailable accounting cannot grant fresh walltime; a requeued id has two rows and stays unknown.
                    collected.accounted_walltime_s = (
                        accounted_chain_walltime(
                            collected.job_chain,
                            "\n".join(f"{job_id}|{state}|{elapsed}" for job_id, state, elapsed, _ in rows),
                            stored_walltime_s=None
                            if collected.walltime_state == "absent"
                            else collected.walltime_used_s or 0,
                        )
                        if rows is not None
                        else None
                    )
                # A follow-up the last allocation queued that ended before it recorded itself continued nothing.
                followup = collected.followup_claims.get(collected.job_chain[-1])
                if followup is not None and followup not in collected.job_chain:
                    try:
                        stopped = stopped_followup(
                            followup,
                            conn.run(
                                [f"sacct -D -n -P -X -j {shlex.quote(followup)} --format={FOLLOWUP_ACCOUNTING_FORMAT}"]
                            ),
                        )
                    except (RuntimeError, OSError):
                        stopped = None
        if collected.accounting_snapshot is not None:
            collected.accounting_snapshot = {
                **collected.accounting_snapshot,
                "resume_config": resume.model_dump_json(),
            }
        decision = decide_from_collected(
            collected,
            max_infra_retries=resume.max_retries,
            max_walltime_s=_walltime_to_seconds(resume.max_walltime) if resume.max_walltime is not None else None,
            live_job=str(job_dir) in workdirs or live_job_name is not None,
            outcome_policy=outcome_policy,
            allocation_cut=allocation_cut,
            extend_retries=extend_retries,
            live_job_name=live_job_name,
        )
        if stopped is not None:
            decision = decision.model_copy(update={"reason": f"{decision.reason}; {stopped}"})
        return decision, collected

    def recovery_decision(self, record: SubmissionRecord, benchmark: str) -> RecoveryDecision:
        """The decision of ``recovery_evidence`` without a score key."""
        return self.recovery_evidence(record, benchmark)[0]

    def reverify_judge_failures(
        self, record: SubmissionRecord, benchmark: str, decision: RecoveryDecision
    ) -> RecoveryOutcome:
        """Admit a saved-answer CPU rejudge using the original chain's remaining allowance.

        Only an already CPU-only submission with no deployed model services is supported. This never
        invents a new compute profile, starts a policy service, or turns an unavailable judge into a model run.
        Other deployments remain explicitly unsupported until they supply a judge-only execution capability.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        def stopped(reason: str) -> RecoveryOutcome:
            return RecoveryOutcome(
                decision=decision.model_copy(update={"action": RecoveryAction.STOP, "reason": reason}),
                ok=False,
                dispatched=None,
                unsupported=reason,
            )

        current, evidence = self.recovery_evidence(record, benchmark)
        if current.action != RecoveryAction.REVERIFY or evidence is None:
            return stopped("saved-answer rejudge admission changed; collect the current evidence")
        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
            original = config.driver.benchmarks[benchmark]
            resume = original.resume_config
            compute = config.compute[record.cluster]
            if not isinstance(compute, SlurmComputeConfig):
                return stopped("saved rejudge compute is not Slurm")
            if (
                config.services
                or config.driver.policy_model
                or not compute.node_pools
                or any(pool.gpus_per_node != 0 for pool in compute.node_pools.values())
            ):
                return stopped(
                    "this submission has no existing CPU-only judge execution profile; policy services are not restarted"
                )
            install = config.driver.gym_install
            if (
                install is None
                or install.ref != installed_gym_commit()
                or not re.fullmatch(r"[0-9a-f]{40}", install.ref)
            ):
                return stopped("saved judge driver and installed recovery source must be the same immutable revision")
            if resume is None or resume.max_walltime is None or evidence.accounting_snapshot is None:
                return stopped("saved rejudge walltime or accounting owner is unknown")
            used = evidence.accounted_walltime_s
            retries = evidence.infra_retries_used
            if (
                used is None
                or not math.isfinite(used)
                or used < 0
                or retries is None
                or not math.isfinite(retries)
                or retries < 0
            ):
                return stopped("actual chain walltime or retry accounting is unknown")
            if retries > resume.max_retries or not evidence.job_chain or evidence.chain_incomplete:
                return stopped("recorded chain is incomplete or exceeds its original retry allowance")
            cap = _walltime_to_seconds(resume.max_walltime)
            remaining = math.floor(cap - used)
            if compute.walltime is not None:
                remaining = min(remaining, _walltime_to_seconds(compute.walltime))
            if remaining < 1:
                return stopped("original walltime allowance is exhausted")
            if evidence.accounting_snapshot.get("resume_config") != resume.model_dump_json():
                return stopped("saved resume configuration changed since accounting")
            # Use the original runtime/configuration. Only operational output/cache controls change.
            run = dict(original.run)
            for name in (
                "output_jsonl_fpath",
                "materialized_inputs_jsonl_fpath",
                "resume_from_cache",
                "resubmit",
                "require_complete",
            ):
                run.pop(name, None)
            if evidence.source_digests is None:
                return stopped("saved source snapshot is unavailable")
            queued_snapshot = {k: v for k, v in evidence.source_digests.items() if Path(k).name == k}
            nonce = new_gym_job_id(utc_now())
            output = job_dir / "reverification" / nonce / "rollouts.jsonl"
            command = " ".join(
                [
                    "gym eval reverify",
                    "--judge-failed-only",
                    "--settle-into-source",
                    "--settle-source-snapshot",
                    shlex.quote(json.dumps(queued_snapshot, sort_keys=True)),
                    "--rollouts",
                    shlex.quote(str(job_dir / "artifacts/rollouts.jsonl")),
                    "--inputs",
                    shlex.quote(str(job_dir / "artifacts/rollouts_materialized_inputs.jsonl")),
                    "--output",
                    shlex.quote(str(output)),
                    *flatten_run_args(run),
                ]
            )
            # No preparation, fresh model run, reset prologue, or afternotok successor.
            bench = original.model_copy(update={"prepare": {}, "run": {}, "command": command, "resumable": False})
            bounded_compute = compute.model_copy(
                update={"walltime": f"{remaining // 3600:02}:{remaining % 3600 // 60:02}:{remaining % 60:02}"}
            )
            driver = config.driver.model_copy(update={"benchmarks": {benchmark: bench}})
            bound = config.model_copy(update={"driver": driver, "compute": {record.cluster: bounded_compute}})
            script = build_sbatch_script(bound, benchmark, bench, bounded_compute, job_dir)
            script = script.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --no-requeue", 1)
            script_path = job_dir / f"reverify-{nonce}.sh"
            expected = {k: v for k, v in evidence.accounting_snapshot.items() if k != "resume_config"}
            if evidence.source_digests is None:
                return stopped("saved source snapshot is unavailable")
            config_path = Path(record.run_dir) / RESOLVED_CONFIG_NAME
            config_digest = conn.run([f"sha256sum {shlex.quote(str(config_path))}"]).split()[0]
            if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
                return stopped("saved submission configuration digest is unknown")
            check = (
                "import hashlib,json,os,sys; from pathlib import Path\n"
                "expected=json.loads(sys.argv[1])\n"
                "def digest(n):\n"
                " p=Path(n)\n"
                " if not p.exists(): return None\n"
                " h=hashlib.sha256()\n"
                " with p.open('rb') as f:\n"
                "  for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)\n"
                " return h.hexdigest()\n"
                "names=set(expected)|{n for n in os.listdir('.') if n.startswith('.gym_followup_queued.')}\n"
                f"source=json.loads({json.dumps(json.dumps(evidence.source_digests))})\n"
                "if {n:digest('artifacts/'+n) for n in source} != source: raise ValueError('saved source changed')\n"
                f"if digest({str(config_path)!r}) != {config_digest!r}: raise ValueError('saved configuration changed')\n"
                "if {n:digest(n) for n in names} != expected: raise ValueError('stale accounting')\n"
                f"p=Path('.gym_accumulated_walltime'); total=max(float(p.read_text()) if p.exists() else 0, {used!r})\n"
                f"if not __import__('math').isfinite(total) or total < 0 or total >= {cap!r}: raise ValueError('spent walltime')\n"
                "tmp=p.with_suffix('.tmp'); tmp.write_text(str(total)+'\\n'); os.replace(tmp,p)\n"
            )
            commands = _submission_exports(config) + [
                'fail() { echo "Error: $*" >&2; exit 1; }',
                f"cd {shlex.quote(str(job_dir))} || fail 'job directory unavailable'",
                f"(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail 'another recovery holds the lock'",
                "_gym_submit_started=0; _gym_submit_confirmed=0",
                "_gym_release_rejudge_lock() { "
                'if [[ "$_gym_submit_started" == 1 && "$_gym_submit_confirmed" != 1 ]]; then '
                "printf '%s\\n' 'GYM_ACCOUNTING_UNKNOWN: rejudge acknowledgement unknown' > .gym_chain_incomplete; fi; "
                f"rm -f {_RESUBMIT_LOCK}; }}",
                "trap _gym_release_rejudge_lock EXIT",
                f"live=$(squeue --all --noheader -j {shlex.quote(','.join(evidence.job_chain))} --format=%A) || fail 'scheduler unknown'",
                '[ -z "$live" ] || fail "an original allocation is still live"',
                f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(expected))} || fail 'accounting changed or exhausted'",
                "umask 077",
                f"printf %s {shlex.quote(script)} > {shlex.quote(str(script_path))} || fail 'cannot stage judge driver'",
                "_gym_submit_started=1",
                _sbatch_command(benchmark, script_path).replace("sbatch --parsable", "sbatch --no-requeue --parsable"),
                'if [ "$rc" != 0 ]; then fail "judge allocation not admitted"; fi',
                "jid=$(printf '%s' \"$out\" | awk '{print $NF}' | cut -d';' -f1); [[ $jid =~ ^[0-9]+$ ]] || { printf '%s\\n' 'GYM_ACCOUNTING_UNKNOWN: rejudge submission acknowledgement unknown' > .gym_chain_incomplete; fail 'unknown job identity'; }",
                'printf "%s\\n" "$jid" >> .gym_job_chain || fail "cannot preserve judge allocation accounting"',
                "_gym_submit_confirmed=1",
            ]
            results = _parse_sbatch_results(conn.run(commands))
            job_id, error = results.get(benchmark, (None, "no scheduler acknowledgement"))
            if job_id is None:
                return stopped(f"saved-answer rejudge was not admitted: {error}")
            updated = job.model_copy(update={"job_id": job_id, "error": None})
            self.persist_record(record.with_benchmark(updated), conn.write_text)
        return RecoveryOutcome(decision=current, ok=True, dispatched="reverify", job_id=job_id)

    def resubmit(
        self,
        record: SubmissionRecord,
        benchmark: str,
        *,
        keep_budget: bool = False,
        accounted_walltime_s: float | None = None,
        accounting_snapshot: Mapping[str, str | None] | None = None,
        accounting_reconciled: bool = False,
        repin: str | None = None,
        repin_reason: str | None = None,
        extend_retries: int = 0,
        max_walltime: str | None = None,
        walltime_reason: str | None = None,
    ) -> BenchmarkJob:
        """Queue `benchmark`'s job.sh again from its job dir and record the new job id.

        With ``keep_budget`` (``--recover``) the job keeps the spent retry count, walltime and per-rollout attempts;
        a job.sh rendered before that option would reset them, so it is refused. ``accounting_snapshot`` is the
        collected evidence's ``accounting_snapshot``; it is required with ``keep_budget`` and rechecked under the
        resubmit lock before anything in the job dir changes, so a decision made on stale accounting is refused.
        ``accounting_reconciled`` (with ``keep_budget``) admits the requeue marker that the collected evidence
        reconciled from terminal scheduler records; it is replaced by a note just before the job is queued.
        ``repin`` (with ``keep_budget`` and ``repin_reason``) continues the run on that full Gym commit, which must be
        this launcher's: under the lock, after the live-job and snapshot checks, job.sh and resolved-config.yaml are
        rendered again with the new ref (and the MLflow pin tags that named the old one), the old files are kept
        beside them, and both are restored if the job is not queued. Rows, attempts and budget files stay as they are;
        the queued job's record carries the new commit and the run dir's repin journal records the move.
        ``extend_retries`` (with ``keep_budget``) adds that many infra retries to the recorded extension that the
        job's prologue adds to the configured cap; it is written just before sbatch, restored if the job is not
        queued, and journaled once queued. The spent retry count is never reset.
        ``max_walltime`` (with ``keep_budget`` and ``walltime_reason``) raises the Slurm walltime the same way a repin
        moves the commit, in the same render: ``compute.<cluster>.walltime`` becomes it, the chain cap becomes at least
        it, and a value below the recorded walltime is refused. The run dir's walltime journal records the raise.

        Refuses while any of the caller's jobs runs in that job dir, so two jobs never write one run, and
        refuses a run with rollouts whose `gym eval run` job.sh predates resubmit: it would start the run
        fresh and delete them. A `command` benchmark's job.sh runs its own command, so it is not refused.
        """
        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
            repin_commands, journals = (
                self._rerender_commands(
                    record,
                    benchmark,
                    config,
                    keep_budget=keep_budget,
                    repin=repin,
                    repin_reason=repin_reason,
                    max_walltime=max_walltime,
                    walltime_reason=walltime_reason,
                )
                if repin is not None or max_walltime is not None
                else ([], {})
            )
            exports = _submission_exports(config)
            # A resubmit can lose access to the remote token. Refresh only this benchmark's
            # collector config, inside the lock and after refusing any active job in its dir.
            collector_update = []
            if otel_active(config):
                collector_yaml = render_collector_config(config, benchmark, job_dir)
                collector_path = shlex.quote(str(collector_config_path(job_dir)))
                collector_update = [
                    f"printf %s {shlex.quote(collector_yaml)} > {collector_path} || fail 'cannot update collector config'"
                ]
            snapshot_check = []
            reconciled = keep_budget and accounting_reconciled and accounted_walltime_s is not None
            if keep_budget:
                stale = "run accounting changed since the recovery evidence was collected; collect it again"
                if accounting_snapshot is None:
                    raise RuntimeError(f"Resubmit of {benchmark} refused: no collected accounting snapshot")
                resume = config.driver.benchmarks[benchmark].resume_config
                files = {name: digest for name, digest in accounting_snapshot.items() if name != "resume_config"}
                if accounting_snapshot.get("resume_config") != (resume.model_dump_json() if resume else None):
                    raise RuntimeError(f"Resubmit of {benchmark} refused: {stale}")
                check = (
                    "import hashlib, json, os, sys\n"
                    "def digest(name):\n"
                    "    try:\n"
                    "        with open(name, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()\n"
                    "    except FileNotFoundError: return None\n"
                    "expected=json.loads(sys.argv[1])\n"
                    "names=set(expected)|{n for n in os.listdir('.') if n.startswith('.gym_followup_queued.')}\n"
                    "sys.exit(0 if {n: digest(n) for n in names}==expected else 1)\n"
                )
                snapshot_check = [
                    f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(files))} || fail {shlex.quote(stale)}"
                ]
            budget_update = []
            if keep_budget and accounted_walltime_s is not None:
                # This is terminal accounting of the complete recorded chain, not a new allowance.
                if not math.isfinite(accounted_walltime_s) or accounted_walltime_s < 0:
                    raise ValueError("accounted walltime must be finite and non-negative")
                resume = config.driver.benchmarks[benchmark].resume_config
                cap = _walltime_to_seconds(resume.max_walltime) if resume and resume.max_walltime else None
                update = (
                    "import math, os\nfrom pathlib import Path\n"
                    "p=Path('.gym_accumulated_walltime')\n"
                    "old=float(p.read_text()) if p.exists() else 0\n"
                    "if not math.isfinite(old) or old < 0: raise ValueError('malformed walltime counter')\n"
                    f"total=max(old, {accounted_walltime_s!r})\n"
                    f"cap={cap!r}\n"
                    "if cap is not None and total >= cap: raise ValueError('walltime budget exhausted')\n"
                    "p.with_suffix('.tmp').write_text(str(int(total) if total.is_integer() else total)+'\\n')\n"
                    "os.replace(p.with_suffix('.tmp'), p)\n"
                )
                budget_update = [f"python3 -c {shlex.quote(update)} || fail 'cannot preserve accounted walltime'"]
            # One script, so the lock spans the scheduler check and sbatch; any failed step stops it.
            # The job itself resets the run's retry and attempt state when it starts (see RESUBMIT_ENV).
            results = _parse_sbatch_results(
                conn.run(
                    exports
                    + [
                        'fail() { echo "Error: $*" >&2; exit 1; }',
                        f"cd {shlex.quote(str(job_dir))} || fail 'cannot enter the job dir'",
                        # bash's noclobber creates the lock with O_EXCL, which is atomic on shared filesystems
                        # that do not support flock. Not mkdir: uutils mkdir (Ubuntu 26.04's default) can
                        # report success to two concurrent callers.
                        f'(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail "another resubmit holds '
                        f'$PWD/{_RESUBMIT_LOCK}; remove it if no resubmit is running"',
                        "_gym_submit_started=0; _gym_submit_confirmed=0",
                        "_gym_release_resubmit_lock() { "
                        'if [[ "$_gym_submit_started" == 1 && "$_gym_submit_confirmed" != 1 ]]; then '
                        f"printf '%s\\n' {shlex.quote(ACCOUNTING_UNKNOWN_PREFIX + ' Resubmit acknowledgement was not confirmed.')} "
                        "> .gym_chain_incomplete || return; fi; "
                        f"rm -f {_RESUBMIT_LOCK}; }}",
                        "_gym_restore_repin() { :; }; _gym_restore_extension() { :; }",
                        "trap '_gym_restore_extension; _gym_restore_repin; _gym_release_resubmit_lock' EXIT",
                        # --all: without it squeue does not list jobs, ours included, in a hidden partition.
                        "workdirs=$(squeue --me --all --noheader --format=%Z) || fail 'squeue failed'",
                        'for d in $workdirs; do [ "$d" != "$PWD" ] || '
                        'fail "A job of this run is still pending or running in $PWD."; done',
                        # Before any job.sh, collector, counter or sbatch change, under the same lock.
                        *snapshot_check,
                        # The checks below then read the re-rendered job.sh; the exit trap restores it unless queued.
                        *repin_commands,
                        *(
                            [
                                f"grep -qxF -- {shlex.quote(RETRY_EXTENSION_MARKER)} job.sh || fail "
                                "'job.sh predates recorded retry extensions; repin the run or stage a new script'"
                            ]
                            if extend_retries
                            else []
                        ),
                        # An older chain retries FAILED, OUT_OF_MEMORY and DEADLINE as infrastructure failures.
                        # Refuse it before anything in the job dir changes.
                        f"if grep -qxF -- {shlex.quote(PROLOGUE_MARKER)} job.sh "
                        f"&& ! grep -qxF -- {shlex.quote(SCHEDULER_CAUSE_MARKER)} job.sh; then "
                        'fail "job.sh in $PWD predates scheduler-cause retry admission, so its chain would retry '
                        'FAILED, OUT_OF_MEMORY and DEADLINE jobs; stage a new job script"; fi',
                        f"if grep -qxF -- {shlex.quote(PROLOGUE_MARKER)} job.sh "
                        f"&& ! grep -qxF -- {shlex.quote(ALLOCATION_ACCOUNTING_MARKER)} job.sh; then "
                        'fail "job.sh predates durable allocation accounting; stage a new compatible script"; fi',
                        (
                            # Only the marker the evidence reconciled; its digest is also in the snapshot above.
                            f"grep -qxF -- {shlex.quote(ACCOUNTING_UNKNOWN_PREFIX + ' ' + REQUEUE_UNKNOWN_REASON)} "
                            ".gym_chain_incomplete || fail 'the reconciled requeue marker changed; collect it again'"
                        )
                        if reconciled
                        else f"if grep -q -- {shlex.quote('^' + ACCOUNTING_UNKNOWN_PREFIX)} .gym_chain_incomplete "
                        '2>/dev/null; then fail "allocation accounting is unknown; reconcile every allocation before '
                        'resubmission"; fi',
                        'for claim in .gym_followup_queued.*; do [ -e "$claim" ] || continue; '
                        'claimed=$(cat "$claim") || fail "follow-up claim unreadable"; '
                        '[[ "$claimed" =~ ^[0-9]+$ ]] || fail "follow-up submission identity unknown"; done',
                        # Only the built-in runner's fresh start deletes rollouts; its job.sh lists
                        # `gym eval run` on a line of its own (see `render_gym_cmd`).
                        "if [ -s artifacts/rollouts.jsonl ] && grep -qx -- '[[:space:]]*gym eval run' job.sh "
                        "&& ! grep -qF -- ++resubmit=True job.sh; then "
                        'fail "job.sh in $PWD predates resubmit, so it could start the run fresh and delete its '
                        'rollouts; submit the run again instead"; fi',
                        # Saved runners may predate normalization of the resume flag. Upgrade only the built-in
                        # resubmit command, under the same lock, before queueing the existing script.
                        "if grep -qx -- '[[:space:]]*gym eval run' job.sh "
                        "&& grep -qF -- 'GYM_CMD+=(++resubmit=True)' job.sh; then "
                        "sed -i 's/GYM_CMD+=(++resubmit=True)/GYM_CMD+=(++resubmit=True ++resume_from_cache=True)/g' "
                        "job.sh || fail 'cannot normalize resubmit flags in job.sh'; fi",
                        *collector_update,
                        *(
                            [
                                f"if {{ grep -qxF -- {shlex.quote(PROLOGUE_MARKER)} job.sh "
                                "|| grep -qx -- '[[:space:]]*gym eval run' job.sh; } "
                                f"&& ! grep -qF -- {KEEP_BUDGET_ENV} job.sh; then "
                                'fail "job.sh in $PWD predates budget-preserving recovery, so a resubmit would reset '
                                'its spent retries, walltime and attempts; use a plain resubmit or submit again"; fi',
                                f"if ! grep -qxF -- {shlex.quote(BOUNDED_BUDGET_MARKER)} job.sh; then "
                                'fail "job.sh in $PWD predates bounded recovery budgets; stage a new job script '
                                'before automatic recovery"; fi',
                                f"export {KEEP_BUDGET_ENV}=1",
                            ]
                            if keep_budget
                            else []
                        ),
                        *budget_update,
                        *(
                            [
                                f"printf '%s\\n' {shlex.quote(REQUEUE_RECONCILED_NOTE)} > .gym_chain_incomplete "
                                "|| fail 'cannot record the reconciled accounting'"
                            ]
                            if reconciled
                            else []
                        ),
                        *(self._extension_commands(extend_retries, keep_budget=keep_budget) if extend_retries else []),
                        f"export {RESUBMIT_ENV}=1",
                        "_gym_submit_started=1",
                        _sbatch_command(benchmark, job_dir / "job.sh"),
                        # Same accepted job-ID grammar as _parse_sbatch_results, evaluated while the lock is held.
                        "if [[ $rc == 0 ]] && python3 -c "
                        + shlex.quote(
                            "import re,sys; words=sys.argv[1].split(); "
                            f"sys.exit(0 if words and re.fullmatch({_JOB_ID_RE.pattern!r}, words[-1]) else 1)"
                        )
                        + ' "$out"; then _gym_submit_confirmed=1; fi',
                    ]
                )
            )
            job_id, error = results.get(benchmark, (None, "sbatch produced no result"))
            if job_id is None:
                raise RuntimeError(
                    f"Resubmit of {benchmark} failed: {error}"
                    + ("; the run stays on its recorded Gym commit" if repin is not None else "")
                    + ("; the run keeps its recorded walltime" if max_walltime is not None else "")
                )
            job = job.model_copy(
                update={"job_id": job_id, "error": None}
                | ({"gym_commit": repin} if repin is not None else {})
                | ({"walltime": max_walltime} if max_walltime is not None else {})
            )
            if extend_retries:
                entry = shlex.quote(json.dumps({"by": extend_retries, "at": utc_timestamp(utc_now())}))
                journal = shlex.quote(str(job_dir / RETRY_EXTENSION_JOURNAL_NAME))
                conn.run([f"printf '%s\\n' {entry} >> {journal}"])
            for name, entry in journals.items():
                journal = Path(record.run_dir) / name
                conn.run([f"printf '%s\\n' {shlex.quote(entry)} >> {shlex.quote(str(journal))}"])
            self.persist_record(record.with_benchmark(job), conn.write_text)
        return job

    @staticmethod
    def _extension_commands(extend_retries: int, *, keep_budget: bool) -> list[str]:
        """The locked steps that record ``extend_retries`` more infra retries, restored unless the job is queued."""
        if not keep_budget:
            raise RuntimeError("--extend-retries needs --recover: it extends the spent budget, never resets it")
        if type(extend_retries) is not int or not 0 < extend_retries <= 1000:
            raise ValueError(f"--extend-retries needs a whole number from 1 to 1000, not {extend_retries!r}")
        name = RETRY_EXTENSION_NAME
        return [
            f"_gym_old_extension=$(cat {name} 2>/dev/null || echo 0)",
            '[[ "$_gym_old_extension" =~ ^[0-9]{1,6}$ ]] || fail "the recorded retry extension is malformed"',
            f"[[ -e {name} ]] && _gym_had_extension=1 || _gym_had_extension=0",
            '_gym_restore_extension() { [[ "$_gym_submit_confirmed" == 1 ]] && return; '
            f'if [[ "$_gym_had_extension" == 1 ]]; then printf "%s\\n" "$_gym_old_extension" > {name}; '
            f"else rm -f {name}; fi; }}",
            f"printf '%s\\n' $((10#$_gym_old_extension + {extend_retries})) > {name}.tmp && mv -f {name}.tmp {name} "
            "|| fail 'cannot record the retry extension'",
        ]

    def _rerender_commands(
        self,
        record: SubmissionRecord,
        benchmark: str,
        config: SubmitConfig,
        *,
        keep_budget: bool,
        repin: str | None = None,
        repin_reason: str | None = None,
        max_walltime: str | None = None,
        walltime_reason: str | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """The locked shell steps that render one benchmark's job.sh and saved config again, with a repin, a walltime
        raise or both, and the journal entries to append once its job is queued."""
        install = config.driver.gym_install
        if len(record.benchmarks) != 1:
            raise RuntimeError("repin or walltime raise of a submission with several benchmarks is unsupported")
        target, journals = config, {}
        stamp = utc_timestamp(utc_now()).replace(":", "")
        if repin is not None:
            if not keep_budget:
                raise RuntimeError("--repin needs --recover: the run continues with its spent budgets")
            if re.fullmatch(r"[0-9a-f]{40}", repin) is None:
                raise ValueError(f"--repin needs a full 40-character commit hash, not {repin!r}")
            if not (repin_reason or "").strip():
                raise ValueError("--repin needs a nonempty --repin-reason")
            if install is None:
                raise RuntimeError(
                    "the submission installs no Gym revision (driver.gym_install), so it cannot be repinned"
                )
            if installed_gym_commit() != repin:
                raise RuntimeError(
                    f"this launcher is Gym {installed_gym_commit()}, not the repin target {repin}; "
                    "install that revision and resubmit with it"
                )
            target = _repinned_config(target, benchmark, repin)
            validate_gym_install_ref(target)
            journals[REPIN_JOURNAL_NAME] = json.dumps(
                {"benchmark": benchmark, "from": install.ref, "to": repin, "reason": repin_reason, "at": stamp},
                sort_keys=True,
            )
        if max_walltime is not None:
            if not keep_budget:
                raise RuntimeError("--max-walltime needs --recover: the run continues with its spent budgets")
            if not (walltime_reason or "").strip():
                raise ValueError("--max-walltime needs a nonempty --walltime-reason")
            try:
                seconds = _walltime_to_seconds(max_walltime)
            except ValueError as error:
                raise ValueError(
                    f"--max-walltime needs a Slurm duration such as 12:00:00, not {max_walltime!r}"
                ) from error
            recorded = config.compute[record.cluster].walltime
            if recorded is not None and seconds < _walltime_to_seconds(recorded):
                raise ValueError(f"--max-walltime {max_walltime} is below the recorded walltime {recorded}")
            resume = config.driver.benchmarks[benchmark].resume_config
            target = _walltime_config(target, record.cluster, benchmark, max_walltime)
            raised = target.driver.benchmarks[benchmark].resume_config
            journals[WALLTIME_JOURNAL_NAME] = json.dumps(
                {
                    "benchmark": benchmark,
                    "from": recorded,
                    "to": max_walltime,
                    "max_walltime_from": resume.max_walltime if resume else None,
                    "max_walltime_to": raised.max_walltime if raised else None,
                    "reason": walltime_reason,
                    "actor": getpass.getuser(),
                    "at": stamp,
                },
                sort_keys=True,
            )
        job_dir = Path(record.benchmark_job(benchmark).job_dir)
        run_dir = Path(record.run_dir)
        script = build_sbatch_script(
            target, benchmark, target.driver.benchmarks[benchmark], target.compute[record.cluster], job_dir
        )
        resolved = yaml.safe_dump(target.model_dump(mode="json", context={HOST_ENV_REFS: True}), sort_keys=False)
        old_config = shlex.quote(str(run_dir / RESOLVED_CONFIG_NAME))
        kinds = [
            kind for kind, wanted in (("repin", repin is not None), ("walltime", max_walltime is not None)) if wanted
        ]
        kept = [
            (f"job.sh.pre-{kind}-{stamp}", shlex.quote(str(run_dir / f"{RESOLVED_CONFIG_NAME}.pre-{kind}-{stamp}")))
            for kind in kinds
        ]
        restore = f"mv -f {kept[0][0]} job.sh 2>/dev/null; mv -f {kept[0][1]} {old_config} 2>/dev/null"
        commands = [
            *(
                command
                for kept_script, kept_config in kept
                for command in (
                    f"cp -p job.sh {kept_script} || fail 'cannot keep the job script before it is rendered again'",
                    f"cp -p {old_config} {kept_config} || fail 'cannot keep the resolved config before it is rendered again'",
                )
            ),
            # Restored by the exit trap below unless sbatch confirms the queued job.
            f'_gym_restore_repin() {{ [[ "$_gym_submit_confirmed" == 1 ]] || {{ {restore}; }}; }}',
            f"(umask 077; printf %s {shlex.quote(script)} > job.sh.repin) && mv -f job.sh.repin job.sh "
            "|| fail 'cannot write the rendered job script'",
            f"printf %s {shlex.quote(resolved)} > {old_config}.repin && mv -f {old_config}.repin {old_config} "
            "|| fail 'cannot write the rendered resolved config'",
        ]
        return commands, journals

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
            (bench_dir / "logs").mkdir()
            (bench_dir / "artifacts").mkdir()
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
