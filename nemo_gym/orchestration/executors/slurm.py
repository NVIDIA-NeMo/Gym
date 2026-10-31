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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml

from nemo_gym import __version__
from nemo_gym.orchestration.api import SAVED_CONFIG, SlurmComputeConfig, SubmitConfig, VllmPDServiceConfig
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
    RESUBMIT_ENV,
    RESUBMIT_LOCK_NAME,
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
    COLLECT_SCRIPT,
    CollectedEvidence,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcomePolicyConfig,
    _managed_trial_metadata,
    accounted_chain_walltime,
    decide_from_collected,
    select_managed_evidence,
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
# The saved install may predate these facts: derive them with the caller's helper on the install's own APIs.
projection["checkpoint_enabled"] = None
if selected.managed_selection is not None and selected.managed_reader_error is None:
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
        # bootstrap stdin. No service, prepare script or scheduler is invoked.
        output = conn.run(
            [
                "unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME",
                f"timeout --kill-after=10s 300s {entrypoint} 3<<'GYM_MANAGED_INPUT'\n"
                + collected.model_dump_json()
                + "\nGYM_MANAGED_INPUT",
            ]
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
        # A pinned reader older than the accounting snapshot drops it; the host collection owns it either way.
        compared = allowed | {"accounting_snapshot"}
        if selected.model_dump(exclude=compared) != collected.model_dump(exclude=compared):
            raise ValueError("remote Store reader changed collection evidence")
        return selected.model_copy(
            update={
                "source_job_identity": collected.source_job_identity,
                "accounting_snapshot": collected.accounting_snapshot,
            }
        )
    except (RuntimeError, OSError, ValueError) as error:
        return collected.model_copy(
            update={
                "managed_selection": None,
                "managed_reader_error": f"remote Store reader unavailable: {type(error).__name__}",
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
    ) -> tuple[RecoveryDecision, CollectedEvidence | None]:
        """Collect the job dir's identities, failures, quality summary, named score and chain budget over SSH."""
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
                    collected = _remote_managed_evidence(conn, collected, job_dir, config)
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
                        ), collected
            workdirs = conn.run(["squeue --me --all --noheader --format=%Z"]).split()
            if resume.max_walltime is not None and collected.job_chain and str(job_dir) not in workdirs:
                try:
                    chain = ",".join(collected.job_chain)
                    accounting = conn.run(
                        [f"sacct -n -P -X -j {shlex.quote(chain)} --format=JobIDRaw,State,ElapsedRaw"]
                    )
                    collected.accounted_walltime_s = accounted_chain_walltime(
                        collected.job_chain,
                        accounting,
                        stored_walltime_s=collected.walltime_used_s or 0,
                    )
                except (RuntimeError, OSError, ValueError):
                    # Unavailable accounting cannot grant fresh walltime.
                    collected.accounted_walltime_s = None
        if collected.accounting_snapshot is not None:
            collected.accounting_snapshot = {
                **collected.accounting_snapshot,
                "resume_config": resume.model_dump_json(),
            }
        decision = decide_from_collected(
            collected,
            max_infra_retries=resume.max_retries,
            max_walltime_s=_walltime_to_seconds(resume.max_walltime) if resume.max_walltime is not None else None,
            live_job=str(job_dir) in workdirs,
            outcome_policy=outcome_policy,
        )
        return decision, collected

    def recovery_decision(self, record: SubmissionRecord, benchmark: str) -> RecoveryDecision:
        """The decision of ``recovery_evidence`` without a score key."""
        return self.recovery_evidence(record, benchmark)[0]

    def resubmit(
        self,
        record: SubmissionRecord,
        benchmark: str,
        *,
        keep_budget: bool = False,
        accounted_walltime_s: float | None = None,
        accounting_snapshot: Mapping[str, str | None] | None = None,
    ) -> BenchmarkJob:
        """Queue `benchmark`'s job.sh again from its job dir and record the new job id.

        With ``keep_budget`` (``--recover``) the job keeps the spent retry count, walltime and per-rollout attempts;
        a job.sh rendered before that option would reset them, so it is refused. ``accounting_snapshot`` is the
        collected evidence's ``accounting_snapshot``; it is required with ``keep_budget`` and rechecked under the
        resubmit lock before anything in the job dir changes, so a decision made on stale accounting is refused.

        Refuses while any of the caller's jobs runs in that job dir, so two jobs never write one run, and
        refuses a run with rollouts whose `gym eval run` job.sh predates resubmit: it would start the run
        fresh and delete them. A `command` benchmark's job.sh runs its own command, so it is not refused.
        """
        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
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
                        "trap _gym_release_resubmit_lock EXIT",
                        # --all: without it squeue does not list jobs, ours included, in a hidden partition.
                        "workdirs=$(squeue --me --all --noheader --format=%Z) || fail 'squeue failed'",
                        'for d in $workdirs; do [ "$d" != "$PWD" ] || '
                        'fail "A job of this run is still pending or running in $PWD."; done',
                        # Before any job.sh, collector, counter or sbatch change, under the same lock.
                        *snapshot_check,
                        # An older chain retries FAILED, OUT_OF_MEMORY and DEADLINE as infrastructure failures.
                        # Refuse it before anything in the job dir changes.
                        f"if grep -qxF -- {shlex.quote(PROLOGUE_MARKER)} job.sh "
                        f"&& ! grep -qxF -- {shlex.quote(SCHEDULER_CAUSE_MARKER)} job.sh; then "
                        'fail "job.sh in $PWD predates scheduler-cause retry admission, so its chain would retry '
                        'FAILED, OUT_OF_MEMORY and DEADLINE jobs; stage a new job script"; fi',
                        f"if grep -qxF -- {shlex.quote(PROLOGUE_MARKER)} job.sh "
                        f"&& ! grep -qxF -- {shlex.quote(ALLOCATION_ACCOUNTING_MARKER)} job.sh; then "
                        'fail "job.sh predates durable allocation accounting; stage a new compatible script"; fi',
                        f"if grep -q -- {shlex.quote('^' + ACCOUNTING_UNKNOWN_PREFIX)} .gym_chain_incomplete 2>/dev/null; then "
                        'fail "allocation accounting is unknown; reconcile every allocation before resubmission"; fi',
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
                raise RuntimeError(f"Resubmit of {benchmark} failed: {error}")
            job = job.model_copy(update={"job_id": job_id, "error": None})
            self.persist_record(record.with_benchmark(job), conn.write_text)
        return job

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
