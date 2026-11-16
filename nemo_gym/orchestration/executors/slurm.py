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
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from nemo_gym import __version__
from nemo_gym.config.global_config import (
    DEFAULT_SERVER_SPINUP_TIMEOUT_SECONDS,
    MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME,
    SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME,
)
from nemo_gym.orchestration.api import (
    HOST_ENV_REFS,
    SAVED_CONFIG,
    NodePool,
    SlurmComputeConfig,
    SubmitConfig,
    VllmPDServiceConfig,
)
from nemo_gym.orchestration.executors.base import BaseExecutor, SchedulerRefusedError
from nemo_gym.orchestration.executors.connection import (
    READER_PROGRESS_PREFIX,
    Connection,
    LocalConnection,
    get_connection,
)
from nemo_gym.orchestration.executors.git_ref import validate_gym_install_ref
from nemo_gym.orchestration.executors.judge_profile import (
    JUDGE_POOL_NAME,
    NO_CPU_PARTITION,
    JudgeProfileUnsupported,
    choose_judge_partition,
    is_cpu_only_submission,
    judge_execution_profile,
    judge_job_config,
    placement_refusal,
    served_judge_placement,
    served_judge_profile,
)
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
    CHAIN_ACCOUNTING_FORMAT,
    INCOMPLETE_MARKER_NAME,
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
from nemo_gym.orchestration.executors.slurm_script import (
    _resolve_env,
    _with_default_capture_dir,
    build_sbatch_script,
)
from nemo_gym.orchestration.jobs import (
    ARTIFACTS_DIRNAME,
    LOGS_DIRNAME,
    RESOLVED_CONFIG_NAME,
    ROLLOUTS_FILENAME,
    BenchmarkJob,
    SubmissionRecord,
    new_gym_job_id,
    utc_now,
    utc_timestamp,
)
from nemo_gym.orchestration.recovery import (
    ACCUMULATED_WALLTIME_FILE,
    COLLECT_SCRIPT,
    FOLLOWUP_ACCOUNTING_FORMAT,
    INFRA_RETRIES_FILE,
    INFRA_RETRY_EXTENSION_FILE,
    JUDGE_JOB_ACK_JOURNAL_NAME,
    JUDGE_JOB_JOURNAL_NAME,
    LIVE_JOB_STATES,
    MAX_EXTEND_ATTEMPTS,
    NO_INVENTORY,
    OPERATOR_CONTINUATION_JOURNAL_NAME,
    OPERATOR_CONTINUATION_NOTE,
    READER_PLACEMENT_FIELDS,
    RERUN_MASKED_JOURNAL_NAME,
    CollectedEvidence,
    IdentityState,
    OperatorBudget,
    RecoveryAction,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryOutcomePolicyConfig,
    _managed_trial_metadata,
    accounted_chain_walltime,
    chain_accounting,
    decide_from_collected,
    effective_outcome_policy,
    failed_judge_jobs_blocking,
    failed_judge_jobs_stop,
    live_chain_job,
    reconcile_requeued_chain,
    requeue_note,
    select_managed_evidence,
    stopped_followup,
    store_writer_cut_from_accounting,
)
from nemo_gym.utils.gym_commit import installed_gym_commit
from nemo_gym.utils.path_utils import attempt_extension_path_for


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
        from nemo_gym.rollouts import RolloutStore
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


def _epoch_walltime(collected: CollectedEvidence, rows: list[tuple[str, str, int, int]] | None) -> float | None:
    """The used walltime of a chain after an operator continuation: its baseline plus every allocation since.

    The continuation filled or kept the counter as the baseline, so the allocations before it are not accounted again.
    None when the records since then are incomplete or the counter does not match them.
    """
    epoch = collected.operator_epoch
    stored = None if collected.walltime_state == "absent" else collected.walltime_used_s or 0
    if epoch is None or stored is None or stored < epoch.walltime_s:
        return None
    since = collected.job_chain[epoch.chain_length :]
    if not since:
        return epoch.walltime_s if stored == epoch.walltime_s else None
    if rows is None:
        return None
    accounted = accounted_chain_walltime(
        since,
        "\n".join(f"{job_id}|{state}|{elapsed}" for job_id, state, elapsed, _ in rows if job_id in since),
        stored_walltime_s=stored - epoch.walltime_s,
    )
    return None if accounted is None else epoch.walltime_s + accounted


# Appended in the job dir when `gym eval resubmit --recover --extend-retries` records an extension.
RETRY_EXTENSION_JOURNAL_NAME = "retry-extension-journal.jsonl"
# Appended in the run dir when `gym eval resubmit --recover --repin` moves a benchmark to another Gym commit.
REPIN_JOURNAL_NAME = "repin-journal.jsonl"
# Appended in the run dir when `gym eval resubmit --recover --max-walltime` raises a benchmark's walltime.
WALLTIME_JOURNAL_NAME = "walltime-extension-journal.jsonl"
# Appended in the run dir when `gym eval resubmit --recover --allow-unsafe-resume` overrides the run identity check.
UNSAFE_RESUME_JOURNAL_NAME = "unsafe-resume-journal.jsonl"
# Appended in the job dir when `gym eval resubmit --recover --server-spinup-timeout-seconds` raises the limit.
SPINUP_TIMEOUT_JOURNAL_NAME = "spinup-timeout-journal.jsonl"


class IncompatibleRerenderError(RuntimeError):
    """A job script would be rendered again by a launcher of another Gym revision than the one the job runs.

    The launcher's job.sh templates may need modules the pinned revision does not have, and the capability checks
    run later inspect only the newly rendered script, so the mismatch is refused before anything changes.
    """

    def __init__(self, pinned: str, launcher: str | None) -> None:
        super().__init__(
            f"the run is pinned to Gym {pinned}, but this launcher is Gym {launcher}: rendering job.sh again here "
            "would install this launcher's job templates into a job that runs the pinned revision; recover from a "
            "launcher installed at the pinned revision, or move the run here with --repin"
        )
        self.pinned = pinned
        self.launcher = launcher


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


def _unsafe_resume_config(config: SubmitConfig, benchmark: str) -> SubmitConfig:
    """``config`` with the benchmark's job resuming past a refused run identity check; every other field as saved."""
    bench = config.driver.benchmarks[benchmark]
    run = {**bench.run, "allow_unsafe_resume": True}
    driver = config.driver.model_copy(
        update={"benchmarks": {**config.driver.benchmarks, benchmark: bench.model_copy(update={"run": run})}}
    )
    return config.model_copy(update={"driver": driver})


def _recorded_spinup_timeout_seconds(config: SubmitConfig, benchmark: str) -> float:
    """The benchmark's server spin-up limit in seconds as the run starts its servers; 0 or less is no limit."""
    value = config.driver.benchmarks[benchmark].run.get(SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME)
    return float(DEFAULT_SERVER_SPINUP_TIMEOUT_SECONDS if value is None else value)


def _spinup_timeout_raise_refusal(recorded: float, seconds: int) -> str | None:
    """Why ``seconds`` is not a raise of the ``recorded`` limit (0 is no limit), or None when it is."""
    if seconds < 0:
        return f"--server-spinup-timeout-seconds is a number of seconds, 0 for no limit, not {seconds}"
    if recorded <= 0:
        return "the run's server spin-up has no limit already"
    if seconds != 0 and seconds <= recorded:
        return f"--server-spinup-timeout-seconds {seconds} does not raise the recorded limit of {recorded:g} s"
    return None


def _spinup_timeout_raise_text(recorded: float, seconds: int, reason: str | None) -> str:
    """The sentence a decision carries for a spin-up timeout raise."""
    to = "0 s (no limit)" if seconds == 0 else f"{seconds} s"
    return f"server spin-up timeout raised from {recorded:g} s to {to}: {reason}"


def _spinup_timeout_config(config: SubmitConfig, benchmark: str, seconds: int) -> SubmitConfig:
    """``config`` with the benchmark's server spin-up limit ``seconds``; every other field as saved."""
    bench = config.driver.benchmarks[benchmark]
    run = {**bench.run, SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME: seconds}
    driver = config.driver.model_copy(
        update={"benchmarks": {**config.driver.benchmarks, benchmark: bench.model_copy(update={"run": run})}}
    )
    return config.model_copy(update={"driver": driver})


def _walltime_config(
    config: SubmitConfig, cluster: str, benchmark: str, walltime: str, segment: str | None = None
) -> SubmitConfig:
    """``config`` with a chain cap of at least ``walltime`` and each Slurm segment requesting ``segment`` (by default
    ``walltime`` itself); every other field as saved. A chain without a cap takes ``walltime`` as its cap: a raise
    always bounds the total."""
    bench = config.driver.benchmarks[benchmark]
    resume = bench.resume_config
    update = {}
    if resume is not None:
        cap = _raised_cap(resume.max_walltime, walltime)
        update["resumable"] = resume.model_copy(update={"max_walltime": cap})
    driver = config.driver.model_copy(
        update={"benchmarks": {**config.driver.benchmarks, benchmark: bench.model_copy(update=update)}}
    )
    compute = {
        **config.compute,
        cluster: config.compute[cluster].model_copy(update={"walltime": segment or walltime}),
    }
    return config.model_copy(update={"driver": driver, "compute": compute})


def _raised_cap(cap: str | None, walltime: str) -> str:
    """The chain cap after a raise to ``walltime``: at least the raise, and the raise itself when there was none."""
    return walltime if cap is None else max(cap, walltime, key=_walltime_to_seconds)


def _slurm_time_limit_seconds(text: str) -> int | None:
    """A partition time limit as sinfo prints it (``%l``) in seconds; None for ``UNLIMITED`` or ``infinite``.

    Takes Slurm's duration spellings (``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM``, ``D-HH:MM:SS``) and
    raises ValueError for anything else, such as ``n/a``.
    """
    value = text.strip()
    if value.upper() in ("UNLIMITED", "INFINITE"):
        return None
    return _walltime_to_seconds(value)


def _slurm_duration(seconds: int) -> str:
    return f"{seconds // 3600:02}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def _partition_time_limit(conn: Connection, compute: SlurmComputeConfig) -> dict[str, Any]:
    """The partition a job of ``compute`` queues on and its MaxTime, read with sinfo over ``conn``.

    The partitions are the node pools' ones, else ``extra_args.partition`` (a list ``a,b`` allowed), else the
    cluster's default one. Every one must be listed, and the lowest limit applies: ``partition`` names it,
    ``max_time`` is sinfo's text and ``max_time_s`` its seconds (None when unlimited); ``error`` says why they are
    unknown.
    """
    pools = sorted({pool.partition for pool in compute.node_pools.values()})
    partition = ",".join(pools) if pools else compute.extra_args.get("partition")
    limit: dict[str, Any] = {"partition": partition, "max_time": None, "max_time_s": None, "error": None}
    try:
        if partition is not None:
            names = str(partition).split(",")
            rows = [
                line.split()
                for line in conn.run([f"sinfo -h -p {shlex.quote(str(partition))} -o '%P %l'"]).splitlines()
            ]
            listed = {row[0].rstrip("*"): row[1] for row in rows if len(row) == 2}
            missing = [name for name in names if name not in listed]
            if missing:
                raise ValueError(f"sinfo lists no partition {', '.join(missing)}")
            parsed = [(_slurm_time_limit_seconds(listed[name]), listed[name], name) for name in names]
        else:
            rows = [line.split() for line in conn.run(["sinfo -h -o '%P %l'"]).splitlines()]
            default = [row for row in rows if len(row) == 2 and row[0].endswith("*")]
            if not default:
                raise ValueError("sinfo names no default partition")
            name, text = default[0][0].rstrip("*"), default[0][1]
            parsed = [(_slurm_time_limit_seconds(text), text, name)]
    except (RuntimeError, OSError, ValueError) as error:
        return limit | {"error": str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__}
    bounded = [item for item in parsed if item[0] is not None]
    seconds, text, name = min(bounded) if bounded else (None, parsed[0][1], parsed[0][2])
    return limit | {"partition": name, "max_time": text, "max_time_s": seconds}


def _plan_walltime(
    config: SubmitConfig, cluster: str, benchmark: str, max_walltime: str, limit: dict[str, Any]
) -> dict[str, Any]:
    """How the raise ``max_walltime`` fits the partition ``limit``: each segment requests the smaller of the raise
    and the partition's MaxTime, and the raise is the chain cap the segments continue under. Refused when no segment
    fits: a run that does not chain segments needs the whole raise in one job, and a chained one at least its
    recorded segment walltime."""
    partition, max_time, max_s = limit["partition"], limit["max_time"], limit["max_time_s"]
    resume = config.driver.benchmarks[benchmark].resume_config
    plan = {
        "partition": partition,
        "max_time": max_time,
        "requested": max_walltime,
        "segment": max_walltime,
        "max_walltime_from": resume.max_walltime if resume else None,
        "max_walltime_to": None,
        "refused": None,
        "note": None,
    }
    try:
        requested_s = _walltime_to_seconds(max_walltime)
    except ValueError:
        return plan  # The raise itself is refused, with its own message, where it is rendered.
    plan["max_walltime_to"] = _raised_cap(resume.max_walltime, max_walltime) if resume else None
    if limit["error"] is not None:
        plan["note"] = (
            f"partition {partition or '(default)'} MaxTime unknown ({limit['error']}); each segment requests the "
            f"whole --max-walltime {max_walltime} as before"
        )
        return plan
    if max_s is None or requested_s <= max_s:
        return plan
    bench = config.driver.benchmarks[benchmark]
    recorded = config.compute[cluster].walltime
    if bench.resume_config is None:
        plan["segment"], plan["refused"] = (
            None,
            (
                f"--max-walltime {max_walltime} exceeds partition {partition} MaxTime {max_time}, and the run does not "
                "continue in segments (no resumable config), so no job can take it"
            ),
        )
    elif recorded is not None and _walltime_to_seconds(recorded) > max_s:
        plan["segment"], plan["refused"] = (
            None,
            (
                f"partition {partition} MaxTime {max_time} is below the run's segment walltime {recorded}, so not even one "
                f"segment of the --max-walltime {max_walltime} raise fits"
            ),
        )
    else:
        plan["segment"] = _slurm_duration(max_s)
        plan["note"] = (
            f"--max-walltime {max_walltime} exceeds partition {partition} MaxTime {max_time}: each segment requests "
            f"{plan['segment']} and the chain continues under the {max_walltime} cap"
        )
    return plan


# The size-aware reader bound: bootstrap plus a floor, then time per GB of saved rollout files, up to a ceiling.
_READER_FLOOR_S = 900
_READER_S_PER_GB = 100
_READER_CEILING_S = 14400
_READER_RETRY_BACKOFF_S = 10
# A watched reader reports its progress on stderr at this interval (see _READER_PROGRESS_SCRIPT).
_READER_PROGRESS_INTERVAL_S = 60
# Run as ``bash -c`` beside a watched reader: $1 is the reader's session, $2 the interval, $3 the rollout bytes, $4 the
# start time. Every interval it prints the bytes the session's processes have read (rchar) and the pass over the
# rollout files that implies. It reads /proc only, so it works for a pinned reader that prints nothing itself. Its
# sleep holds none of the shell's streams, so a loop that outlives its kill cannot keep the caller waiting.
_READER_PROGRESS_SCRIPT = r"""while sleep "$2" >/dev/null 2>&1; kill -0 "$1" 2>/dev/null; do
    b=0
    for p in /proc/[0-9]*; do
        [[ -O $p ]] || continue
        read -r s < "$p/stat" || continue
        read -ra f <<< "${s##*) }"
        [[ ${f[3]} == "$1" ]] || continue
        while read -r k v; do [[ $k == rchar: ]] && b=$((b + v)); done < "$p/io"
    done 2>/dev/null
    echo "gym-reader progress: pass=$(( $3 > 0 ? b / $3 + 1 : 1 )) bytes=$b elapsed_s=$(( $(date +%s) - $4 ))" >&2
done"""


def _remote_reader_bound_s() -> int | None:
    """The explicit reader bound from ``NEMO_GYM_REMOTE_READER_TIMEOUT_S``, clamped to [300, 14400], else None.

    Without it the login host sizes the bound from the run's rollout files (``_reader_command``).
    """
    raw = os.environ.get("NEMO_GYM_REMOTE_READER_TIMEOUT_S", "")
    try:
        return min(max(int(raw), 300), _READER_CEILING_S) if raw else None
    except ValueError:
        return None


# The heavy remote readers (the collector and the managed Store reader) can run in a Slurm step instead of on the
# login host, which may allow each user one core. Unset, they run on the login host exactly as before.
READER_SRUN_ENV = "NEMO_GYM_REMOTE_READER_SRUN"
# How long a remote reader's cold install of the pinned Gym waits for uv's lock: the caller's UV_LOCK_TIMEOUT, else the
# default a Gym run uses (global_config's uv_lock_timeout_seconds).
_READER_UV_LOCK_TIMEOUT_DEFAULT_S = 1800


def _reader_env_lines() -> list[str]:
    """The settings a remote reader takes from its caller: the reader's command starts from a cleared environment on
    another host, so without this a cold install there waits uv's own default for the lock and gives up early."""
    raw = os.environ.get("UV_LOCK_TIMEOUT", "").strip()
    lock_timeout = (
        raw if raw.isascii() and raw.isdecimal() and int(raw) > 0 else str(_READER_UV_LOCK_TIMEOUT_DEFAULT_S)
    )
    return [f"export UV_LOCK_TIMEOUT={lock_timeout}"]


READER_SRUN_WAIT_ENV = "NEMO_GYM_REMOTE_READER_SRUN_WAIT_S"
READER_MAX_ENV = "NEMO_GYM_REMOTE_READER_MAX"
# How long a reader waits for a slot when the user's reader steps are at the cap (0: it does not wait).
READER_QUEUE_WAIT_ENV = "NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S"
_READER_QUEUE_WAIT_DEFAULT_S = 600
# How often a queued reader counts the reader steps again.
_READER_QUEUE_POLL_S = 10
# A reader step srun refused outright (nothing was queued) is submitted again after this many seconds, doubling up
# to _READER_SRUN_RETRY_MAX_S, for as long as the queue wait allows.
_READER_SRUN_RETRY_S = 15
_READER_SRUN_RETRY_MAX_S = 60
# srun's words for a submission the scheduler refused without queueing it: a per-user submit or size limit, or no
# node free for an immediate start.
_SRUN_REFUSAL = (
    "Job violates accounting/QOS policy|Requested nodes are busy|"
    "Immediate execution impossible|Resources temporarily unavailable"
)
# With a reader step, whether a reader that cannot run in its step may read on the login host instead: "forbid" (the
# default) or "allow". Forbidden, such a read is unavailable: nothing is read on the login host or decided from it.
READER_LOGIN_FALLBACK_ENV = "NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK"
# The host a reader reports when its step did not run and the login-host fallback is forbidden.
READER_NOT_RUN_HOST = "none"
# After cancelling a reader step that did not start, how long to wait for it to leave the queue.
_READER_CANCEL_WAIT_S = 60
_READER_HOST_PREFIX = "__GYM_READER_HOST:"
_READER_FALLBACK_PREFIX = "__GYM_READER_FALLBACK:"


@dataclass(frozen=True)
class ReaderStep:
    """How a heavy reader runs in a Slurm step: the caller's srun arguments (each quoted), how long the step may take
    to start, how many reader steps the user may have running or pending at once, how long a reader waits for one of
    them to end when they are at that cap, and whether a reader that cannot run in its step may run on the login host
    instead."""

    args: str
    wait_s: int
    max_readers: int
    login_fallback: bool = False
    queue_wait_s: int = 0


def _env_seconds(name: str, default: int) -> int:
    """A nonnegative number of seconds from the environment; the default for an unset or invalid value."""
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_count(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value >= 1 else default


# The decision of a recovery whose evidence collector could not run in its step, with the fallback forbidden.
COLLECTOR_NOT_RUN = (
    "the evidence collector could not run in its Slurm step and the login-host fallback is forbidden, so nothing was "
    "read or decided"
)


def reader_step() -> ReaderStep | None:
    """The reader step from ``NEMO_GYM_REMOTE_READER_SRUN`` (extra srun arguments, for example ``-p cpu -c 4``),
    ``NEMO_GYM_REMOTE_READER_SRUN_WAIT_S`` (default 120), ``NEMO_GYM_REMOTE_READER_MAX`` (default 8),
    ``NEMO_GYM_REMOTE_READER_QUEUE_WAIT_S`` (default 600; 0 does not wait) and
    ``NEMO_GYM_REMOTE_READER_LOGIN_FALLBACK`` (``allow``; anything else forbids it); None when the arguments are unset
    or empty."""
    raw = os.environ.get(READER_SRUN_ENV, "").strip()
    if not raw:
        return None
    return ReaderStep(
        args=" ".join(shlex.quote(arg) for arg in shlex.split(raw)),
        wait_s=_env_count(READER_SRUN_WAIT_ENV, 120),
        max_readers=_env_count(READER_MAX_ENV, 8),
        login_fallback=os.environ.get(READER_LOGIN_FALLBACK_ENV, "").strip().lower() == "allow",
        queue_wait_s=_env_seconds(READER_QUEUE_WAIT_ENV, _READER_QUEUE_WAIT_DEFAULT_S),
    )


def _reader_step_lines(step: ReaderStep, token: str, *, script: str, stdin: str, watchdog: bool = False) -> list[str]:
    """Bash that runs ``script`` in a Slurm step, else leaves it to the login host.

    It sets ``_gym_host`` ("login" or "cpu:<job id>") and ``_gym_fallback`` (why the reader stays on the login host,
    empty when it ran in the step); after a step that ran, ``_gym_rc`` is its exit code and ``$_gym_step_out`` holds
    its output. Inside the step the input on stdin becomes FD 3, because srun forwards stdin but no other descriptor,
    and ``script`` reads the reader bound as ``$1``. The step is named gym-reader-<token>, and its time limit is the
    reader bound plus 60 s, so Slurm ends an orphan. A step that has not started after the wait is cancelled by name,
    and the reader runs on the login host. With ``watchdog`` (FD 4 is the shell's stdin), srun runs under the EOF
    watcher (``_watch_lines``), which also cancels the step by name. Without ``step.login_fallback``, a reader that
    would fall back runs nowhere: ``_gym_host`` is ``none``, ``_gym_rc`` 1, and ``_gym_fallback`` says why.

    When the user's reader steps are at ``step.max_readers``, the reader queues: it counts them again every
    ``_READER_QUEUE_POLL_S`` seconds for up to ``step.queue_wait_s``, saying on stderr how many it is queued behind
    (a progress line), and runs in its step once one ends. Only a cap still reached after that wait keeps it out.

    A step srun refused outright (``_SRUN_REFUSAL``: a submit or QOS limit, or no node free to start it) was never
    queued, so the reader submits it again after ``_READER_SRUN_RETRY_S`` seconds, doubling up to
    ``_READER_SRUN_RETRY_MAX_S``, with a progress line each time, for as long as ``step.queue_wait_s`` allows. Any
    other srun failure, or a refusal past that wait, falls through to the cancellation below as before.
    """
    name = shlex.quote(f"gym-reader-{token}")
    inside = shlex.quote(f'echo "{_READER_HOST_PREFIX}cpu:$SLURM_JOB_ID"; exec 3<&0 </dev/null; {script}')
    # The step's stderr (the reader's, through srun) goes where the login-host reader's goes.
    srun = (
        f'timeout --kill-after=10s "$(( _gym_bound + {step.wait_s} + 60 ))s" srun {step.args} --ntasks=1 '
        f'--immediate={step.wait_s} -t "$(( (_gym_bound + 119) / 60 ))" -J {name} bash -c {inside} gym-reader '
        f'"$_gym_bound" {stdin} >"$_gym_step_out" 2>>"$_gym_err"'
    )
    if watchdog:
        run_step = [f"setsid {srun} 4<&- &", "_gym_reader=$!", *_watch_lines(cancel=name)]
    else:
        run_step = [srun, "_gym_rc=$?"]
    return [
        "_gym_host=login",
        '_gym_fallback=""',
        "_gym_step_out=$(mktemp)",
        'if ! _gym_readers=$(squeue -h -u "$USER" -t RUNNING,PENDING -o %j 2>/dev/null); then',
        '    _gym_fallback="squeue failed, so the running reader steps cannot be counted"',
        "else",
        "    _gym_queued_s=0",
        f'    while _gym_n=$(printf "%s\\n" "$_gym_readers" | grep -c "^gym-reader-"); (( _gym_n >= {step.max_readers} )); do',
        f"        if (( _gym_queued_s >= {step.queue_wait_s} )); then",
        f'            _gym_fallback="{step.max_readers} reader steps are already running or pending"',
        f'            (( _gym_queued_s == 0 )) || _gym_fallback="$_gym_fallback (still $_gym_n after queueing '
        f'{step.queue_wait_s} s)"',
        "            break",
        "        fi",
        "        if (( _gym_queued_s % 60 == 0 )); then",
        f'            echo "{READER_PROGRESS_PREFIX} queued behind $_gym_n reader steps (cap {step.max_readers}), '
        f'waited $_gym_queued_s s of {step.queue_wait_s} s" >&2',
        "        fi",
        f"        sleep {_READER_QUEUE_POLL_S}",
        f"        _gym_queued_s=$(( _gym_queued_s + {_READER_QUEUE_POLL_S} ))",
        '        if ! _gym_readers=$(squeue -h -u "$USER" -t RUNNING,PENDING -o %j 2>/dev/null); then',
        '            _gym_fallback="squeue failed while queued, so the running reader steps cannot be counted"',
        "            break",
        "        fi",
        "    done",
        "fi",
        "if [[ -z $_gym_fallback ]]; then",
        "    _gym_retry_s=0",
        f"    _gym_backoff={_READER_SRUN_RETRY_S}",
        "    while :; do",
        *(f"        {line}" for line in run_step),
        f'        _gym_host=$(grep -m 1 "^{_READER_HOST_PREFIX}cpu:" "$_gym_step_out")',
        "        [[ -z $_gym_host ]] || break",
        '        _gym_srun_said=$(tail -n 1 "$_gym_err" | cut -c 1-300)',
        f'        printf "%s\\n" "$_gym_srun_said" | grep -qE {shlex.quote(_SRUN_REFUSAL)} || break',
        f"        (( _gym_retry_s + _gym_backoff <= {step.queue_wait_s} )) || break",
        # Refused at submission, so nothing was queued; the cancel only covers a race with a late start.
        f'        scancel --name {name} --user "$USER" 2>/dev/null',
        f'        echo "{READER_PROGRESS_PREFIX} the reader step was refused ($_gym_srun_said); retrying in '
        f'$_gym_backoff s, waited $_gym_retry_s s of {step.queue_wait_s} s" >&2',
        '        sleep "$_gym_backoff"',
        "        _gym_retry_s=$(( _gym_retry_s + _gym_backoff ))",
        f"        _gym_backoff=$(( _gym_backoff * 2 > {_READER_SRUN_RETRY_MAX_S} ? {_READER_SRUN_RETRY_MAX_S} : "
        "_gym_backoff * 2 ))",
        '        : >"$_gym_step_out"',
        "    done",
        "    if [[ -n $_gym_host ]]; then",
        f"        _gym_host=${{_gym_host#{_READER_HOST_PREFIX}}}",
        "    else",
        "        _gym_host=login",
        '        _gym_srun_said=$(tail -n 1 "$_gym_err" | cut -c 1-300)',
        f'        scancel --name {name} --user "$USER" 2>/dev/null',
        "        _gym_waited=0",
        "        _gym_gone=0",
        # Only a squeue that answers and no longer lists the step confirms the cancellation.
        f'        while _gym_queued=$(squeue -h -u "$USER" -n {name} -o %i 2>/dev/null); do',
        "            [[ -n $_gym_queued ]] || { _gym_gone=1; break; }",
        f"            (( _gym_waited < {_READER_CANCEL_WAIT_S} )) || break",
        "            sleep 2",
        "            _gym_waited=$(( _gym_waited + 2 ))",
        "        done",
        f'        _gym_fallback="the reader step did not start within {step.wait_s} s and was cancelled"',
        '        (( _gym_retry_s == 0 )) || _gym_fallback="$_gym_fallback after $_gym_retry_s s of resubmissions"',
        '        [[ -z $_gym_srun_said ]] || _gym_fallback="$_gym_fallback ($_gym_srun_said)"',
        # A step that may still start must not race a second reader on the login host: stop instead.
        "        if (( ! _gym_gone )); then",
        "            _gym_host=unconfirmed",
        "            _gym_rc=1",
        '            _gym_fallback=""',
        f'            echo "gym: the cancelled reader step {name} was not confirmed gone (still queued after '
        f'{_READER_CANCEL_WAIT_S} s, or squeue failed); no second reader was started on the login host" >>"$_gym_err"',
        "        fi",
        "    fi",
        "fi",
        *(
            []
            if step.login_fallback
            else [
                "if [[ $_gym_host == login ]]; then",
                f"    _gym_host={READER_NOT_RUN_HOST}",
                "    _gym_rc=1",
                f'    _gym_fallback="$_gym_fallback; the login-host fallback is forbidden ({READER_LOGIN_FALLBACK_ENV}='
                'forbid), so nothing was read"',
                '    echo "gym: the reader did not run: $_gym_fallback" >>"$_gym_err"',
                "fi",
            ]
        ),
    ]


def _watch_lines(*, cancel: str | None = None, progress: list[str] | None = None) -> list[str]:
    """Bash that waits for the background reader ``$_gym_reader``, started in its own session, and sets ``_gym_rc``.

    A watcher in its own session reads the shell's stdin through FD 4 and at EOF kills the reader's process group:
    TERM, then KILL after 10 s. With ``cancel``, a quoted Slurm job name, it cancels that job before the TERM: the step
    runs on another host, and srun may exit at the TERM, which ends the wait and stops the watcher. ``progress``
    starts a loop that sets ``$_gym_progress``; it is stopped after the wait.
    """
    cancel_first = ' scancel --name "$2" --user "$USER" 2>/dev/null;' if cancel is not None else ""
    name = "" if cancel is None else f" {cancel}"
    return [
        f'setsid bash -c \'cat <&4 >/dev/null;{cancel_first} kill -TERM -- -"$1" 2>/dev/null; sleep 10; '
        f'kill -KILL -- -"$1" 2>/dev/null\' _ "$_gym_reader"{name} >/dev/null 2>&1 &',
        "_gym_watcher=$!",
        *(progress or []),
        'wait "$_gym_reader"',
        "_gym_rc=$?",
        # Its sleep holds no stream; killing its pid covers a loop stopped before its setsid made the group.
        *(['kill -- -"$_gym_progress" 2>/dev/null || kill "$_gym_progress" 2>/dev/null'] if progress else []),
        'kill -- -"$_gym_watcher" 2>/dev/null',
    ]


def _report_placement_lines() -> list[str]:
    return [
        f'echo "{_READER_HOST_PREFIX}$_gym_host"',
        f'[[ -z $_gym_fallback ]] || echo "{_READER_FALLBACK_PREFIX}$_gym_fallback"',
    ]


def reader_placement(output: str) -> tuple[str | None, str | None]:
    """Where a reader ran ("login" or "cpu:<job id>") and why it fell back to the login host, from its report; both
    None for a reader that ran without a reader step."""
    host = reason = None
    for line in output.splitlines():
        if line.startswith(_READER_HOST_PREFIX):
            host = line.removeprefix(_READER_HOST_PREFIX).strip() or None
        elif line.startswith(_READER_FALLBACK_PREFIX):
            reason = line.removeprefix(_READER_FALLBACK_PREFIX).strip() or None
    return host, reason


def _bound_lines(job_dir: Path, *, bound_s: int | None, factor: int) -> list[str]:
    artifacts = job_dir / "artifacts"
    sources = " ".join(shlex.quote(str(artifacts / name)) for name in ("rollouts.jsonl", "rollouts_failures.jsonl"))
    if bound_s is None:
        size_bound = f"$(( ({_READER_FLOOR_S} + _gym_bytes * {_READER_S_PER_GB} / 1073741824) * {factor} ))"
    else:
        size_bound = str(bound_s * factor)
    return [
        f"_gym_bytes=$(stat -c %s -- {sources} 2>/dev/null | awk '{{s += $1}} END {{print s + 0}}')",
        f"_gym_bound={size_bound}",
        f"(( _gym_bound > {_READER_CEILING_S} )) && _gym_bound={_READER_CEILING_S}",
    ]


def collector_command(collect: str, job_dir: Path, step: ReaderStep, token: str) -> list[str]:
    """The collector in a reader step, else on the login host; its output keeps only the collector's own lines after
    the placement report is taken off (``reader_placement``, ``without_placement``). Its exit code is the
    collector's."""
    return [
        *_bound_lines(job_dir, bound_s=_remote_reader_bound_s(), factor=1),
        "_gym_err=$(mktemp)",
        *_reader_step_lines(step, token, script=f"exec {collect}", stdin="</dev/null"),
        *_report_placement_lines(),
        'if [[ $_gym_host == login ]]; then rm -f -- "$_gym_err" "$_gym_step_out"; exec ' + collect + "; fi",
        # A collector that ran nowhere reports only its placement: the caller decides nothing from it.
        f'if [[ $_gym_host == {READER_NOT_RUN_HOST} ]]; then cat "$_gym_err" >&2; rm -f -- "$_gym_err" "$_gym_step_out"; '
        "exit 0; fi",
        f'grep -v "^{_READER_HOST_PREFIX}" "$_gym_step_out"',
        '[[ $_gym_rc == 0 ]] || cat "$_gym_err" >&2',
        'rm -f -- "$_gym_err" "$_gym_step_out"',
        'exit "$_gym_rc"',
    ]


def without_placement(output: str) -> str:
    """The output without the reader placement report."""
    return "\n".join(
        line for line in output.splitlines() if not line.startswith((_READER_HOST_PREFIX, _READER_FALLBACK_PREFIX))
    )


def _remote_reader_error_text(error: BaseException) -> str:
    """Name the reader failure with its cause, not only the exception type: the end of the message, where it is."""
    detail = " ".join(str(error).split())
    if not detail:
        return f"remote Store reader unavailable: {type(error).__name__}"
    if len(detail) > 300:
        detail = "..." + detail[-300:]
    return f"remote Store reader unavailable: {type(error).__name__}: {detail}"


READER_TOKEN_ENV = "GYM_READER_TOKEN"
_reader_token: str | None = None


def reader_token() -> str:
    """The token this process's login-host Store readers carry in their environment as ``GYM_READER_TOKEN``: the
    caller's own value when it sets one, else one generated uuid for the process. Recovery reports it as
    ``reader_token``, so whoever stops a recovery can find its reader."""
    global _reader_token
    if _reader_token is None:
        _reader_token = os.environ.get(READER_TOKEN_ENV) or uuid.uuid4().hex
    return _reader_token


class RemoteReaderInterrupted(KeyboardInterrupt):
    """A signal stopped recovery while the login-host Store reader ran; the reader was stopped by its token.

    A ``KeyboardInterrupt``, so the reader's ordinary error handling does not turn it into evidence and the open
    connection unwinds and closes its master.
    """


# The bash script that stops a login-host reader by its token. The match runs in bash itself, so no helper process
# carries the token in its own arguments; the script's own environment does not carry it either.
_KILL_READER_SCRIPT = r"""_gym_token=@TOKEN@
_gym_matches() {
    local proc pid environ cmdline
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        [[ -O $proc && $pid != "$$" && $pid != "$BASHPID" ]] || continue
        environ=$(tr '\0' '\n' < "$proc/environ" 2>/dev/null) || continue
        cmdline=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null) || continue
        if [[ $'\n'$environ$'\n' == *$'\n'"@ENV@=$_gym_token"$'\n'* || $cmdline == *"$_gym_token"* ]]; then
            echo "$pid"
        fi
    done
}
_gym_pids=$(_gym_matches)
_gym_termed=$(echo $_gym_pids | wc -w)
[[ -z $_gym_pids ]] || kill -TERM $_gym_pids 2>/dev/null
_gym_waited=0
while [[ -n $_gym_pids && $_gym_waited -lt 10 ]]; do
    sleep 1
    _gym_waited=$((_gym_waited + 1))
    _gym_pids=$(_gym_matches)
done
_gym_killed=$(echo $_gym_pids | wc -w)
[[ -z $_gym_pids ]] || kill -KILL $_gym_pids 2>/dev/null
echo "__GYM_READER_KILL:term=$_gym_termed kill=$_gym_killed"
"""


def _kill_reader_command(token: str) -> list[str]:
    """Stop this user's login-host processes whose environment or command line carries ``token``: TERM, then KILL
    for any still running after 10 s. It prints ``__GYM_READER_KILL:term=<n> kill=<n>``."""
    return [_KILL_READER_SCRIPT.replace("@TOKEN@", shlex.quote(token)).replace("@ENV@", READER_TOKEN_ENV)]


def _run_reader(conn: Connection, command: list[str], token: str) -> str:
    """Run the reader. A signal on the client (raised as ``KeyboardInterrupt``) leaves the remote reader running,
    since ``ssh bash -s`` has no terminal to hang up, so it is stopped by its token before the error unwinds.

    A connection that ``holds_input`` keeps the remote shell's stdin open while the reader runs, and the command's
    watcher (``_reader_command(..., watchdog=True)``) stops the reader when that stdin ends.
    """
    try:
        return conn.run_with_input_held(command) if getattr(conn, "holds_input", False) else conn.run(command)
    except KeyboardInterrupt as interrupt:
        try:
            report = conn.run(_kill_reader_command(token)).strip()
        except (RuntimeError, OSError) as error:
            report = f"stopping it failed: {error}"
        cause = str(interrupt) or "an interrupt"
        raise RemoteReaderInterrupted(
            f"remote Store reader interrupted by {cause}; its login-host processes with {READER_TOKEN_ENV}={token} "
            f"were stopped ({report})"
        ) from None


def _reader_command(
    entrypoint: str,
    job_dir: Path,
    payload: str,
    *,
    bound_s: int | None,
    factor: int,
    token: str | None = None,
    watchdog: bool = False,
    progress_interval_s: int | None = None,
    step: ReaderStep | None = None,
) -> list[str]:
    """The login-host steps: size the bound from the rollout files, run the reader, report its exit code and stderr.

    The reader's own failure is reported on stdout (``__GYM_READER_*`` lines) so a timeout, a MemoryError or a
    failed bootstrap reaches the decision with its exit code and the end of its stderr; only SSH itself fails the
    command. With ``token``, the reader runs with it as ``GYM_READER_TOKEN`` in its environment. With ``step``, it
    runs in a Slurm step when it can (``_reader_step_lines``) and reports where it ran.

    With ``watchdog``, for a connection that holds the shell's stdin open while it runs: the reader runs in its own
    session (``setsid``), and a watcher in its own session waits for EOF on the shell's stdin, which comes when the
    client process or its SSH connection ends, then kills the reader's process group (TERM, KILL after 10 s). It is
    an EOF loop, not a hangup: without a terminal nothing sends SIGHUP. The steps run as one function that exits the
    shell, so the watcher never reads a line of the script itself. In a step, the watched process is srun, and the
    watcher also cancels the step by name. On the login host, a progress loop in its own session prints
    ``gym-reader progress: pass=<n> bytes=<b> elapsed_s=<s>`` on stderr every ``progress_interval_s`` (60 s by
    default) while the reader runs; a reader in a step has none. Stdout and its ``__GYM_READER_*`` lines are
    unchanged.
    """
    if step is not None:
        return _reader_command_in_step(
            entrypoint, job_dir, payload, bound_s=bound_s, factor=factor, token=token, step=step, watchdog=watchdog
        )
    artifacts = job_dir / "artifacts"
    sources = " ".join(shlex.quote(str(artifacts / name)) for name in ("rollouts.jsonl", "rollouts_failures.jsonl"))
    if bound_s is None:
        size_bound = f"$(( ({_READER_FLOOR_S} + _gym_bytes * {_READER_S_PER_GB} / 1073741824) * {factor} ))"
    else:
        size_bound = str(bound_s * factor)
    if watchdog:
        run_reader = [
            # A background command's stdin is /dev/null, so the watcher reads the shell's stdin through FD 4.
            "exec 4<&0",
            f'setsid timeout --kill-after=10s "${{_gym_bound}}s" {entrypoint} 3<<\'GYM_MANAGED_INPUT\' 2>"$_gym_err" </dev/null 4<&- &\n'
            + payload
            + "\nGYM_MANAGED_INPUT",
            "_gym_reader=$!",
            *_watch_lines(
                progress=[
                    f"setsid bash -c {shlex.quote(_READER_PROGRESS_SCRIPT)} "
                    f'_ "$_gym_reader" {progress_interval_s or _READER_PROGRESS_INTERVAL_S} "$_gym_bytes" "$(date +%s)" '
                    "</dev/null >/dev/null 4<&- &",
                    "_gym_progress=$!",
                ]
            ),
        ]
    else:
        run_reader = [
            f'timeout --kill-after=10s "${{_gym_bound}}s" {entrypoint} 3<<\'GYM_MANAGED_INPUT\' 2>"$_gym_err"\n'
            + payload
            + "\nGYM_MANAGED_INPUT",
            "_gym_rc=$?",
        ]
    steps = [
        "unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME",
        *_reader_env_lines(),
        *([f"export {READER_TOKEN_ENV}={shlex.quote(token)}"] if token is not None else []),
        f"_gym_bytes=$(stat -c %s -- {sources} 2>/dev/null | awk '{{s += $1}} END {{print s + 0}}')",
        f"_gym_bound={size_bound}",
        f"(( _gym_bound > {_READER_CEILING_S} )) && _gym_bound={_READER_CEILING_S}",
        'echo "__GYM_READER_BOUND:$_gym_bound $_gym_bytes"',
        "_gym_err=$(mktemp)",
        *run_reader,
        'echo "__GYM_READER_RC:$_gym_rc"',
        "printf '__GYM_READER_ERR:%s\\n' \"$(tail -c 2000 \"$_gym_err\" | tr '\\n' ' ')\"",
        'rm -f -- "$_gym_err"',
    ]
    if not watchdog:
        return steps
    # One function, called by the last line: bash has read the whole script before the watcher reads its stdin.
    return ["_gym_read() {", *steps, "exit 0", "}", "_gym_read"]


def _reader_command_in_step(
    entrypoint: str,
    job_dir: Path,
    payload: str,
    *,
    bound_s: int | None,
    factor: int,
    token: str | None,
    step: ReaderStep,
    watchdog: bool = False,
) -> list[str]:
    """``_reader_command`` with the reader in a Slurm step: the payload is saved once, sent on the step's stdin, and
    read again from the file by a login-host reader that runs instead. With ``watchdog``, srun and that login-host
    reader each run under the EOF watcher, as the reader of ``_reader_command(..., watchdog=True)`` does."""
    token = token if token is not None else uuid.uuid4().hex
    login = f'timeout --kill-after=10s "${{_gym_bound}}s" {entrypoint} 3<"$_gym_payload" 2>"$_gym_err"'
    if watchdog:
        run_login = [f"setsid {login} </dev/null 4<&- &", "_gym_reader=$!", *_watch_lines()]
    else:
        run_login = [login, "_gym_rc=$?"]
    steps = [
        "unset VIRTUAL_ENV UV_PYTHON PYTHONPATH PYTHONHOME",
        # Exported before srun, so the reader in its step gets them too.
        *_reader_env_lines(),
        f"export {READER_TOKEN_ENV}={shlex.quote(token)}",
        *_bound_lines(job_dir, bound_s=bound_s, factor=factor),
        'echo "__GYM_READER_BOUND:$_gym_bound $_gym_bytes"',
        "_gym_err=$(mktemp)",
        "_gym_payload=$(mktemp)",
        "cat >\"$_gym_payload\" <<'GYM_MANAGED_INPUT'\n" + payload + "\nGYM_MANAGED_INPUT",
        # A background command's stdin is /dev/null, so the watcher reads the shell's stdin through FD 4.
        *(["exec 4<&0"] if watchdog else []),
        # In the step as on the login host, the reader stops at its bound with exit code 124 (retried once).
        *_reader_step_lines(
            step,
            token,
            script=f'timeout --kill-after=10s "${{1}}s" {entrypoint}',
            stdin='<"$_gym_payload"',
            watchdog=watchdog,
        ),
        "if [[ $_gym_host == login ]]; then",
        *(f"    {line}" for line in run_login),
        "else",
        f'    grep -v "^{_READER_HOST_PREFIX}" "$_gym_step_out"',
        "fi",
        'rm -f -- "$_gym_step_out" "$_gym_payload"',
        *_report_placement_lines(),
        'echo "__GYM_READER_RC:$_gym_rc"',
        "printf '__GYM_READER_ERR:%s\\n' \"$(tail -c 2000 \"$_gym_err\" | tr '\\n' ' ')\"",
        'rm -f -- "$_gym_err"',
    ]
    if not watchdog:
        return steps
    # As in _reader_command: bash has read the whole script before a watcher reads its stdin.
    return ["_gym_read() {", *steps, "exit 0", "}", "_gym_read"]


def _reader_failure(output: str, elapsed_s: float) -> tuple[str | None, bool]:
    """The reader failure named from its report with its exit code and stderr, and whether it reached its deadline
    (the only failure retried with twice the bound); (None, False) when it exited 0."""
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
    # timeout(1) exits 124 at the deadline, and 137 when it SIGKILLs the reader after the grace period. A 137 before
    # the deadline is no timeout: something else killed the reader, so another attempt is not retried.
    timed_out = rc == 124 or (rc == 137 and bound.isdigit() and elapsed_s >= int(bound))
    if timed_out:
        cause = f"timed out after {elapsed_s:.0f} s ({scope}), exit code {rc}"
    elif rc == 137:
        cause = f"killed (SIGKILL; OOM or external kill) after {elapsed_s:.0f} s, below {scope}, exit code 137"
    else:
        cause = f"exited {rc if rc is not None else 'without an exit code'} after {elapsed_s:.0f} s ({scope})"
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
    placement: dict[str, str | None] = {}
    try:
        command = f"python -c {shlex.quote(_remote_managed_reader_script())} {shlex.quote(str(job_dir))} {shlex.quote(install.ref)} <&3"
        entrypoint = render_driver_entrypoint(
            repo=install.repo,
            ref=install.ref,
            prepare_cmd=None,
            command=command,
            cleanup=True,
            private_uv=True,
            # The reader asks the tracking server for the exported run's status (`query_mlflow_status`), as the
            # login-host export does; without the client that read fails and delivery stays unverified.
            extras=("mlflow",),
        )
        # The login host need not contain Gym. Keep this isolated from an inherited caller environment,
        # and bound bootstrap as well as selection. FD 3 keeps large inventories out of argv and away from
        # bootstrap stdin. No service, prepare script or scheduler is invoked. One retry, with twice the bound,
        # follows a timeout or a failed SSH hop; any other failure is final and reported with its cause.
        payload = collected.model_dump_json(exclude=set(READER_PLACEMENT_FIELDS))
        for attempt in (1, 2):
            started = time.monotonic()
            try:
                output = _run_reader(
                    conn,
                    _reader_command(
                        entrypoint,
                        job_dir,
                        payload,
                        bound_s=_remote_reader_bound_s(),
                        factor=attempt,
                        token=reader_token(),
                        watchdog=getattr(conn, "holds_input", False),
                        step=reader_step(),
                    ),
                    reader_token(),
                )
                placement = dict(zip(("reader_host", "reader_fallback_reason"), reader_placement(output)))
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
                failed = {"managed_selection": None, "managed_reader_error": failure + retried}
                # The wrapper reports the exit code of a reader that crashed, timed out or was killed; output
                # without one is a malformed answer, refused like any other.
                crashed = any(line.startswith("__GYM_READER_RC:") for line in output.splitlines())
                return collected.model_copy(update=failed | {"managed_reader_unavailable": crashed} | placement)
        lines = [
            line.removeprefix("__GYM_MANAGED:") for line in output.splitlines() if line.startswith("__GYM_MANAGED:")
        ]
        if len(lines) != 1:
            raise ValueError("remote Store reader returned no unique projection")
        projection = json.loads(lines[0])
        if not isinstance(projection, dict):
            raise ValueError("remote Store reader returned no evidence object")
        selected = CollectedEvidence.model_validate(projection)
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
            "managed_reader_unavailable",
            "source_job_identity",
            "checkpoint_enabled",
            # The attempt extension the Store applied (--extend-attempts); an older reader leaves it unset.
            "attempt_extension_applied",
            # True from a reader whose Store applies the rerun-masked approvals; an older one leaves it unset.
            "rerun_masked_applied",
            # The admitted re-judgments without a saved outcome the Store found (a pass killed after admitting).
            "unresolved_rejudgments",
        }
        # A pinned reader older than a host-collected field drops it on its own model (the reader's projection then
        # lacks the key); the host collection owns such fields, and these named ones, either way.
        host_owned = (
            "accounting_snapshot",
            "coverage_allocation_id",
            "coverage_writer_unrecorded",
            "accounting_unknown_reason",
            "counter_states",
            *READER_PLACEMENT_FIELDS,
            *sorted(set(CollectedEvidence.model_fields) - set(projection)),
        )
        compared = allowed | set(host_owned) | {"accounting_reconciled"}
        if selected.model_dump(exclude=compared) != collected.model_dump(exclude=compared):
            raise ValueError("remote Store reader changed collection evidence")
        return selected.model_copy(
            update={"source_job_identity": collected.source_job_identity}
            | {name: getattr(collected, name) for name in host_owned}
            | placement
        )
    except (RuntimeError, OSError, ValueError) as error:
        return collected.model_copy(
            update={
                "managed_selection": None,
                "managed_reader_error": _remote_reader_error_text(error),
                # A transport or host failure read nothing; a ValueError is a projection the reader returned and
                # this host refused.
                "managed_reader_unavailable": not isinstance(error, ValueError),
            }
            | placement
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
# How long a recovery's post-run MLflow export may take on the login host, its Gym install included.
_EXPORT_TIMEOUT_S = 3600


def _chain_live_check(chain: list[str], live_error: str) -> list[str]:
    """Shell lines that fail unless the scheduler confirms that no job of `chain` waits or runs.

    Slurm drops a finished job from the controller after MinJobAge, and `squeue -j` then exits 1 with "Invalid job
    id specified" once it knows none of the ids. That is no answer about the jobs, so sacct, which keeps finished
    jobs, decides: a job whose latest incarnation (highest Restarts; sacct -D lists every requeue) is in one of the
    `LIVE_JOB_STATES` is live, and one it does not know is not. Any other squeue failure, or a failing sacct, leaves
    the scheduler unknown.
    """
    ids = shlex.quote(",".join(chain))
    live_states = "|".join(sorted(LIVE_JOB_STATES))
    return [
        f"if _gym_queue=$(squeue --all --noheader -j {ids} --format=%A 2>&1); then "
        "live=$(printf '%s\n' \"$_gym_queue\" | grep -E '^[0-9]' || true); "
        "elif printf '%s' \"$_gym_queue\" | grep -q 'Invalid job id specified'; then "
        f"_gym_accounting=$(sacct -D -n -P -X -j {ids} --format=JobIDRaw,State,Restarts) || fail 'scheduler unknown'; "
        "live=$(printf '%s\n' \"$_gym_accounting\" | awk -F'|' "
        '\'NF >= 2 {split($2, s, " "); state = s[1]; sub(/[+]$/, "", state); '
        "if (!($1 in restarts) || $3 + 0 >= restarts[$1]) {restarts[$1] = $3 + 0; latest[$1] = state}} "
        f"END {{for (id in latest) if (latest[id] ~ /^({live_states})$/) print id}}'); "
        "else fail 'scheduler unknown'; fi",
        f'[ -z "$live" ] || fail "{live_error}"',
    ]


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


# Settings of the run's own collection that a judge-only job's `gym eval reverify` replaces with its own.
_COLLECTION_ONLY_RUN_KEYS = (
    "limit",
    "output_jsonl_fpath",
    "materialized_inputs_jsonl_fpath",
    "resume_from_cache",
    "resubmit",
    "require_complete",
)
# Appends one judge-job journal entry (argv: the journal, the entry as JSON, the job id or empty, an optional error).
_JOURNAL_JUDGE_JOB = (
    "import json,sys; entry=json.loads(sys.argv[2]); entry['job_id']=sys.argv[3] or None; "
    "error=sys.argv[4] if len(sys.argv) > 4 else ''; entry.update({'error': error} if error else {}); "
    "f=open(sys.argv[1],'a'); f.write(json.dumps(entry, sort_keys=True)+chr(10)); f.close()"
)

# Sets the job ID of the last operator journal entry from sbatch's reply (its last word), replacing the file atomically.
_NAME_OPERATOR_JOB = (
    "import json,os,sys; p=sys.argv[1]; lines=open(p).read().splitlines(); entry=json.loads(lines[-1]); "
    "entry['job_id']=sys.argv[2].split()[-1]; lines[-1]=json.dumps(entry, sort_keys=True); "
    "open(p+'.tmp','w').write(chr(10).join(lines)+chr(10)); os.replace(p+'.tmp', p)"
)


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
        extend_attempts: int = 0,
        max_walltime: str | None = None,
        operator_budget: OperatorBudget | None = None,
        server_spinup_timeout_seconds: int | None = None,
        spinup_timeout_reason: str | None = None,
        rerun_masked_without_artifact: bool = False,
    ) -> tuple[RecoveryDecision, CollectedEvidence | None]:
        """Collect the job dir's identities, failures, quality summary, named score and chain budget over SSH.

        ``reader_ref`` runs the Store reader at that full commit instead of the saved one: a repinned run resumes there.
        ``max_walltime`` is a walltime raise the resubmit will render: the plan checks the chain cap it renders.
        ``server_spinup_timeout_seconds`` (with ``spinup_timeout_reason``) is a spin-up limit raise the resubmit
        will render: the decision names it.
        ``rerun_masked_without_artifact`` plans the masked results that kept a saved answer but have no gradeable
        artifact and no grade as retries; the resubmit journals them (see ``resubmit``).
        ``operator_budget`` lets a run whose chain accounting is unknown continue (see ``decide_from_collected``); with
        it, a failed scheduler query is a stop that names it instead of an error.
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
            # The policy is read from the run's saved config, so every recovery and resubmit applies the one the run
            # was started with. An explicit Python argument wins over that section, which wins over the defaults.
            outcome_policy, policy_source = effective_outcome_policy(outcome_policy, resume.recovery_outcome_policy)
            cap_s = _walltime_to_seconds(resume.max_walltime) if resume.max_walltime is not None else None
            raised_cap = None
            if max_walltime is not None:
                try:
                    # The raise renders the chain cap as at least the new walltime, or as the raise itself on a chain
                    # without one (see _walltime_config).
                    if cap_s is None or _walltime_to_seconds(max_walltime) > cap_s:
                        raised_cap, cap_s = max_walltime, _walltime_to_seconds(max_walltime)
                except ValueError:
                    pass  # the resubmit refuses the value with its own reason before anything changes
            selector = {k: v for k, v in (("agent", score_agent), ("environment", score_environment)) if v is not None}
            args = [shlex.quote(str(job_dir))]
            if score_key:
                args += [shlex.quote(json.dumps(score_key)), shlex.quote(json.dumps(selector))]
            collect = f"python3 -c {shlex.quote(COLLECT_SCRIPT)} {' '.join(args)}"
            step = reader_step()
            if step is None:
                collected = CollectedEvidence.model_validate_json(conn.run([collect]))
            else:
                output = conn.run(collector_command(collect, job_dir, step, reader_token()))
                host, reason = reader_placement(output)
                if host == READER_NOT_RUN_HOST:
                    unread = RecoveryDecision(
                        action=RecoveryAction.UNKNOWN,
                        reason=f"{COLLECTOR_NOT_RUN}: {reason or 'no cause reported'}; "
                        "inspect again once the reader step can start",
                        expected=0,
                        counts={},
                        identities={},
                        evidence=[],
                    )
                    return unread, None
                collected = CollectedEvidence.model_validate_json(without_placement(output)).model_copy(
                    update={"collector_host": host, "collector_fallback_reason": reason}
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
                                "managed_reader_unavailable": False,
                            }
                        )
                    # A reader that could not run decides nothing, so the scheduler is asked whether the run is live.
                    if collected.managed_selection is None and not collected.managed_reader_unavailable:
                        return decide_from_collected(
                            collected,
                            max_infra_retries=resume.max_retries,
                            max_walltime_s=cap_s,
                            live_job=None,
                            outcome_policy=outcome_policy,
                            outcome_policy_source=policy_source,
                            extend_retries=extend_retries,
                            extend_attempts=extend_attempts,
                            rerun_masked_without_artifact=rerun_masked_without_artifact,
                        ), collected
            try:
                workdirs = conn.run(["squeue --me --all --noheader --format=%Z"]).split()
            except (RuntimeError, OSError):
                if operator_budget is None:
                    raise
                workdirs = None  # the operator continuation stops and names the failed query
            allocation_cut = False
            rows = None
            stopped = None
            # A queued follow-up and its sentinel are jobs of the run before they append themselves to the chain.
            followups = [*collected.followup_sentinels.values(), *collected.followup_claims.values()]
            watched = list(dict.fromkeys([*collected.job_chain, *followups]))
            scheduled = None
            if watched:
                try:
                    ids = shlex.quote(",".join(watched))
                    scheduled = chain_accounting(
                        conn.run([f"sacct -D -n -P -X -j {ids} --format={CHAIN_ACCOUNTING_FORMAT}"])
                    )
                except (RuntimeError, OSError):
                    pass
            if collected.job_chain and scheduled is not None:
                rows = [row for row in scheduled if row[0] in collected.job_chain]
            # A job of the run that still waits or runs is live, wherever the scheduler runs it from.
            live_job_name = live_chain_job(scheduled, [*followups, *collected.job_chain])
            # So is a judge-only job of the run that has no receipt yet; one that ended is reported as it ended.
            judge_live = self._judge_job_states(conn, collected)
            if live_job_name is None:
                live_job_name = judge_live
            if collected.job_chain and workdirs is not None and str(job_dir) not in workdirs and live_job_name is None:
                if rows is not None:
                    allocation_cut = store_writer_cut_from_accounting(rows, collected)
                    collected = reconcile_requeued_chain(collected, rows)
                if resume.max_walltime is not None and not collected.accounting_reconciled:
                    # Unavailable accounting cannot grant fresh walltime; a requeued id has two rows and stays unknown.
                    collected.accounted_walltime_s = (
                        _epoch_walltime(collected, rows)
                        if collected.operator_epoch is not None
                        else accounted_chain_walltime(
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
        collected = collected.model_copy(update={"infra_retries_configured": resume.max_retries})
        if collected.accounting_snapshot is not None:
            collected.accounting_snapshot = {
                **collected.accounting_snapshot,
                "resume_config": resume.model_dump_json(),
            }
        decision = decide_from_collected(
            collected,
            max_infra_retries=resume.max_retries,
            max_walltime_s=cap_s,
            live_job=True if live_job_name is not None else None if workdirs is None else str(job_dir) in workdirs,
            outcome_policy=outcome_policy,
            outcome_policy_source=policy_source,
            allocation_cut=allocation_cut,
            extend_retries=extend_retries,
            extend_attempts=extend_attempts,
            rerun_masked_without_artifact=rerun_masked_without_artifact,
            live_job_name=live_job_name,
            operator_budget=operator_budget,
        )
        if stopped is not None:
            decision = decision.model_copy(update={"reason": f"{decision.reason}; {stopped}"})
        if (
            collected.accounting_reconciled
            and rows is not None
            and (requeued := requeue_note(collected.job_chain, rows))
        ):
            decision = decision.model_copy(update={"reason": f"{decision.reason}; {requeued}"})
        if raised_cap is not None:
            raised = (
                f"walltime cap raised from {resume.max_walltime} to {raised_cap} for this plan"
                if resume.max_walltime is not None
                else f"walltime cap set to {raised_cap} for this plan (the run had none)"
            )
            decision = decision.model_copy(update={"reason": f"{decision.reason}; {raised}"})
        if server_spinup_timeout_seconds is not None:
            recorded = _recorded_spinup_timeout_seconds(config, benchmark)
            raised = _spinup_timeout_raise_text(recorded, server_spinup_timeout_seconds, spinup_timeout_reason)
            decision = decision.model_copy(update={"reason": f"{decision.reason}; {raised}"})
        return decision, collected

    def recovery_decision(self, record: SubmissionRecord, benchmark: str) -> RecoveryDecision:
        """The decision of ``recovery_evidence`` without a score key."""
        return self.recovery_evidence(record, benchmark)[0]

    def reverify_judge_failures(
        self,
        record: SubmissionRecord,
        benchmark: str,
        decision: RecoveryDecision,
        *,
        judge_partition: str | None = None,
        judge_qos: str | None = None,
        repin: str | None = None,
        repin_reason: str | None = None,
        acknowledge_failed_judge_jobs: bool = False,
        acknowledge_reason: str | None = None,
    ) -> RecoveryOutcome:
        """Queue a judge-only job that grades the run's selected saved answers again, never the policy.

        A CPU-only submission (no service, no policy model, CPU node pools only) runs it on its own compute with the
        chain's remaining walltime allowance, as before. Any other submission, one that served its policy on GPUs,
        runs it on a judge execution profile derived from its saved config (``_queue_judge_job``): the run's own judge
        and verifier servers on one node of a CPU partition (``judge_partition``, a saved CPU pool or the launcher's
        reader placement, with ``judge_qos`` or a QOS derived for it), or, for a run that serves its judge on a node
        pool of its own, on that saved pool with those services started and the run's QOS, for the run's segment
        walltime bounded by the partition's MaxTime, at the run's pinned Gym commit or ``repin``.
        ``acknowledge_failed_judge_jobs`` (with ``acknowledge_reason``) lets one more judge-only job queue when the last
        ones ended without settling a saved answer (see ``_queue_judge_job``). Either way the job settles through
        ``gym eval reverify --judge-failed-only --settle-into-source``. This never starts a policy service or turns an
        unavailable judge into a model run.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        def stopped(reason: str) -> RecoveryOutcome:
            return RecoveryOutcome(
                decision=decision.model_copy(update={"action": RecoveryAction.STOP, "reason": reason}),
                ok=False,
                dispatched=None,
                unsupported=reason,
            )

        current, evidence = self.recovery_evidence(
            record, benchmark, **({"reader_ref": repin} if repin is not None else {})
        )
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
            if not is_cpu_only_submission(config):
                # A run that served its policy on GPUs: a judge-only job on a derived CPU-only profile instead.
                return self._queue_judge_job(
                    conn,
                    record,
                    benchmark,
                    current,
                    evidence,
                    config,
                    judge_partition=judge_partition,
                    judge_qos=judge_qos,
                    repin=repin,
                    repin_reason=repin_reason,
                    acknowledge_failed_judge_jobs=acknowledge_failed_judge_jobs,
                    acknowledge_reason=acknowledge_reason,
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
                "limit",
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
            # Observability is on by default, and it needs a capture directory: the job's own folder, as a submitted
            # evaluation captures into its benchmark directory.
            run = _with_default_capture_dir(run, output.parent)
            command = " ".join(
                [
                    "gym eval reverify",
                    "--judge-failed-only",
                    "--settle-into-source",
                    # The dispatch below runs only when no allocation of the run is live.
                    "--withdraw-unresolved-admissions",
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
                *_chain_live_check(evidence.job_chain, "an original allocation is still live"),
                f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(expected))} || fail 'accounting changed or exhausted'",
                # Only the staged driver script is owner-only. The umask of the rest of the dispatch is the caller's:
                # sbatch hands it to the job, so the job's output directory and files, and the journals and markers
                # written below, are readable by the run's other readers like the run's other artifacts.
                f"(umask 077; printf %s {shlex.quote(script)} > {shlex.quote(str(script_path))}) "
                "|| fail 'cannot stage judge driver'",
                "_gym_submit_started=1",
                _sbatch_command(benchmark, script_path).replace("sbatch --parsable", "sbatch --no-requeue --parsable"),
                'if [ "$rc" != 0 ]; then fail "judge allocation not admitted: $out"; fi',
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

    @staticmethod
    def _judge_job_states(conn: Connection, collected: CollectedEvidence) -> str | None:
        """Ask the scheduler about the run's judge-only jobs that have no receipt (``CollectedEvidence.judge_jobs``,
        state ``queued``), set each one's ``state`` and ``scheduler_state`` from its record, and name the newest live
        one as ``judge-only job <id> (STATE)``; None when none is live, or when the query failed or knows none."""
        pending = [job for job in collected.judge_jobs if job.get("state") == "queued" and job.get("job_id")]
        if not pending:
            return None
        ids = ",".join(str(job["job_id"]) for job in pending)
        try:
            output = conn.run([f"sacct -D -n -P -X -j {shlex.quote(ids)} --format={FOLLOWUP_ACCOUNTING_FORMAT}"])
        except (RuntimeError, OSError):
            return None
        records: dict[str, tuple[str, str]] = {}
        for line in output.splitlines():
            parts = line.strip().split("|")
            if len(parts) != 4 or not parts[0]:
                continue
            state = parts[1].split()[0].rstrip("+") if parts[1].split() else ""
            records[parts[0]] = (state, parts[3])
        live = None
        for job in pending:
            record = records.get(str(job["job_id"]))
            if record is None:
                continue
            state, exit_code = record
            if state in LIVE_JOB_STATES:
                job["state"] = "running" if state == "RUNNING" else "pending"
                job["scheduler_state"] = state
                live = f"judge-only job {job['job_id']} ({state})"
            elif state:
                # It ended and wrote no receipt: a failure of the job, with the scheduler's word on it.
                job["state"] = "failed"
                job["scheduler_state"] = f"{state} (exit {exit_code})"
        return live

    def _queue_judge_job(
        self,
        conn: Connection,
        record: SubmissionRecord,
        benchmark: str,
        decision: RecoveryDecision,
        evidence: CollectedEvidence,
        config: SubmitConfig,
        *,
        judge_partition: str | None,
        judge_qos: str | None = None,
        repin: str | None,
        repin_reason: str | None,
        acknowledge_failed_judge_jobs: bool = False,
        acknowledge_reason: str | None = None,
    ) -> RecoveryOutcome:
        """Queue the judge-only job of a run that is not CPU-only, on the judge execution profile derived from its
        saved config (``judge_profile``), and journal it: the served-judge profile when the run serves services beside
        its policy (its judge on a node pool of its own), else the CPU profile.

        The job's command is the same ``gym eval reverify --judge-failed-only --settle-into-source`` as a CPU-only
        run's, with the run's saved settings, the policy's ``--model-type`` so the run's config composes as it did
        (the policy server it names is never called) and a receipt the job writes when it ends. Under the resubmit
        lock, no job of the run may wait or run in its job dir, no allocation of the chain may be live, and the saved
        sources, resolved config and chain accounting must be as the evidence saw them; then the script is staged
        beside the run's and queued with ``--no-requeue``, and its id, partition, walltime, rows and receipt path are
        appended to the job dir's judge-job journal. The job is not a segment of the policy chain: nothing is added
        to ``.gym_job_chain`` and no retry or walltime budget is spent or needed. When the last
        ``JUDGE_JOB_FAILURE_BOUND`` judge-only jobs ended without settling a saved answer, nothing is queued and the
        stop names them and the way on: ``acknowledge_failed_judge_jobs`` with a nonempty ``acknowledge_reason`` appends
        one line (time, user, reason, those jobs' ids) to the acknowledgement journal once the new job is accepted by the
        scheduler, so exactly one more job queues; the acknowledgement covers only the jobs it names.
        """
        from nemo_gym.orchestration.executors.utils import flatten_run_args

        def stopped(reason: str, refusal: str | None = None) -> RecoveryOutcome:
            return RecoveryOutcome(
                decision=decision.model_copy(
                    update={"action": RecoveryAction.STOP, "reason": reason, "judge_refusal": refusal}
                ),
                ok=False,
                dispatched=None,
                unsupported=reason,
            )

        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        original = config.driver.benchmarks[benchmark]
        blocked = failed_judge_jobs_blocking(evidence.judge_jobs, evidence.judge_job_acknowledgements)
        acknowledgement = None
        if blocked is not None:
            if not acknowledge_failed_judge_jobs or None in blocked:
                return stopped(failed_judge_jobs_stop(blocked, str(job_dir / "reverification")))
            if not (acknowledge_reason or "").strip():
                raise ValueError("--acknowledge-failed-judge-jobs needs a nonempty --acknowledge-reason")
            acknowledgement = {
                "at": utc_timestamp(utc_now()),
                "user": getpass.getuser(),
                "reason": (acknowledge_reason or "").strip(),
                "job_ids": blocked,
                "gym_job_id": record.gym_job_id,
                "benchmark": benchmark,
            }
        try:
            # A run that serves its judge on a node pool of its own takes that pool; any other, a CPU partition.
            placement = served_judge_placement(config, record.cluster)
            if placement is None:
                partition, source = choose_judge_partition(
                    config, record.cluster, partition=judge_partition, reader_srun=os.environ.get(READER_SRUN_ENV)
                )
        except JudgeProfileUnsupported as error:
            return stopped(
                str(error), "cpu_partition_unknown" if NO_CPU_PARTITION in str(error) else "unsupported_profile"
            )
        install = config.driver.gym_install
        ref = repin if repin is not None else (install.ref if install is not None else None)
        if install is None or ref is None or ref != installed_gym_commit() or not re.fullmatch(r"[0-9a-f]{40}", ref):
            return stopped(
                "saved judge driver and installed recovery source must be the same immutable revision"
                + (
                    ""
                    if repin is not None
                    else "; --repin <this launcher's full commit> --repin-reason TEXT runs the judge-only job at it"
                )
            )
        if repin is not None and not (repin_reason or "").strip():
            raise ValueError("--repin needs a nonempty --repin-reason")
        compute = config.compute[record.cluster]
        if placement is not None:
            probe = compute.model_copy(update={"node_pools": placement[1]})
            profile = served_judge_profile(
                config, record.cluster, _partition_time_limit(conn, probe), partition=judge_partition, qos=judge_qos
            )
        else:
            probe = compute.model_copy(update={"node_pools": {JUDGE_POOL_NAME: NodePool(partition=partition)}})
            profile = judge_execution_profile(
                config,
                record.cluster,
                partition,
                source,
                _partition_time_limit(conn, probe),
                qos=judge_qos,
                reader_srun=os.environ.get(READER_SRUN_ENV),
            )
        if evidence.source_digests is None or evidence.accounting_snapshot is None:
            return stopped("saved source or accounting snapshot is unavailable; collect the evidence again")
        queued_snapshot = {k: v for k, v in evidence.source_digests.items() if Path(k).name == k}
        run = {k: v for k, v in original.run.items() if k not in _COLLECTION_ONLY_RUN_KEYS}
        nonce = new_gym_job_id(utc_now())
        folder = job_dir / "reverification" / nonce
        output, receipt = folder / "rollouts.jsonl", folder / "receipt.json"
        # Observability is on by default, and it needs a capture directory: the job's own folder, as a submitted
        # evaluation captures into its benchmark directory.
        run = _with_default_capture_dir(run, folder)
        # The job only judges saved answers: it never calls the policy model, whose served endpoint is gone with the
        # run's allocation, so waiting for that endpoint's readiness would only time the job out.
        run = {key: value for key, value in run.items() if key != MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}
        policy_type = config.driver.policy_model_type
        command = " ".join(
            [
                "gym eval reverify",
                "--judge-failed-only",
                "--settle-into-source",
                # The judge-only job is queued only when no job of the run is live, so an admission a pass that died
                # left without an outcome is withdrawn and judged again instead of refusing the job.
                "--withdraw-unresolved-admissions",
                "--settle-source-snapshot",
                shlex.quote(json.dumps(queued_snapshot, sort_keys=True)),
                "--rollouts",
                shlex.quote(str(job_dir / "artifacts/rollouts.jsonl")),
                "--inputs",
                shlex.quote(str(job_dir / "artifacts/rollouts_materialized_inputs.jsonl")),
                "--output",
                shlex.quote(str(output)),
                "--settle-receipt",
                shlex.quote(str(receipt)),
                *([f"--model-type {shlex.quote(policy_type)}"] if config.driver.policy_model and policy_type else []),
                # A judge failure whose saved attempts are spent keeps its masked row; the rest are judged.
                "+skip_spent_judge_failures=True",
                *flatten_run_args(run),
                f"++{MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}=0",
            ]
        )
        bound = judge_job_config(config, record.cluster, benchmark, profile, command, gym_ref=repin)
        if repin is not None:
            validate_gym_install_ref(bound)
        bench = bound.driver.benchmarks[benchmark]
        script = build_sbatch_script(bound, benchmark, bench, bound.compute[record.cluster], job_dir)
        script = script.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --no-requeue", 1)
        script_path = job_dir / f"reverify-{nonce}.sh"
        config_path = Path(record.run_dir) / RESOLVED_CONFIG_NAME
        config_digest = conn.run([f"sha256sum {shlex.quote(str(config_path))}"]).split()[0]
        if not re.fullmatch(r"[0-9a-f]{64}", config_digest):
            return stopped("saved submission configuration digest is unknown")
        expected = {k: v for k, v in evidence.accounting_snapshot.items() if k != "resume_config"}
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
        )
        entry = {
            "at": utc_timestamp(utc_now()),
            "job_id": None,
            "nonce": nonce,
            # Its kind, partition, pools, services, QOS and walltime, each with where it came from.
            **profile.report(),
            "rows": decision.counts.get(IdentityState.JUDGE_RECOVERABLE.value, 0),
            "output": str(output),
            "receipt": str(receipt),
            "gym_commit": ref,
            "actor": getpass.getuser(),
            **({"repin_reason": repin_reason} if repin is not None else {}),
        }
        # Only once the scheduler has accepted the job: an acknowledgement of a job that was not queued is never written.
        acknowledge = (
            f"printf '%s\\n' {shlex.quote(json.dumps(acknowledgement, sort_keys=True))} "
            f">> {shlex.quote(JUDGE_JOB_ACK_JOURNAL_NAME)} || fail 'cannot journal the acknowledgement'; "
            if acknowledgement is not None
            else ""
        )
        journal = (
            f"python3 -c {shlex.quote(_JOURNAL_JUDGE_JOB)} {shlex.quote(JUDGE_JOB_JOURNAL_NAME)} "
            f"{shlex.quote(json.dumps(entry, sort_keys=True))}"
        )
        commands = _submission_exports(config) + [
            'fail() { echo "Error: $*" >&2; exit 1; }',
            f"cd {shlex.quote(str(job_dir))} || fail 'job directory unavailable'",
            f"(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail 'another recovery holds the lock'",
            f"trap 'rm -f {_RESUBMIT_LOCK}' EXIT",
            # No job of this run, a chain segment or an earlier judge-only job, may wait or run in its job dir.
            "workdirs=$(squeue --me --all --noheader --format=%Z) || fail 'scheduler unknown'",
            'for d in $workdirs; do [ "$d" != "$PWD" ] || '
            'fail "a job of this run is still pending or running in $PWD"; done',
            *(
                _chain_live_check(evidence.job_chain, "an original allocation is still live")
                if evidence.job_chain
                else []
            ),
            f"python3 -c {shlex.quote(check)} {shlex.quote(json.dumps(expected))} "
            "|| fail 'saved sources or accounting changed since the evidence was collected; collect it again'",
            # Only the staged driver script is owner-only. The umask of the rest of the dispatch is the caller's: sbatch
            # hands it to the job, so the job's output directory and files (the reverification folder, its receipt), and
            # the journals written below, are readable by the run's other readers like the run's other artifacts.
            f"(umask 077; printf %s {shlex.quote(script)} > {shlex.quote(str(script_path))}) "
            "|| fail 'cannot stage the judge driver'",
            _sbatch_command(benchmark, script_path).replace("sbatch --parsable", "sbatch --no-requeue --parsable"),
            'if [ "$rc" != 0 ]; then fail "judge allocation not admitted: $out"; fi',
            "jid=$(printf '%s' \"$out\" | awk '{print $NF}' | cut -d';' -f1)",
            # The queued job is journaled by id; an acknowledgement without one is journaled as such, so inspect
            # reports it and the next recovery's scheduler check finds the job by its working directory.
            f'if [[ $jid =~ ^[0-9]+$ ]]; then {acknowledge}{journal} "$jid" || fail "cannot journal the judge-only job"; '
            f'else {journal} "" "acknowledgement unknown: $out"; fail "unknown job identity"; fi',
        ]
        try:
            results = _parse_sbatch_results(conn.run(commands))
        except (RuntimeError, OSError) as error:
            text = str(error).strip()
            text = text.splitlines()[-1] if text else type(error).__name__
            return stopped(f"judge-only job was not queued: {text}", placement_refusal(text, placement is not None))
        job_id, error = results.get(benchmark, (None, "no scheduler acknowledgement"))
        if job_id is None:
            return stopped(
                f"judge-only job was not admitted: {error}", placement_refusal(str(error), placement is not None)
            )
        updated = job.model_copy(update={"job_id": job_id, "error": None})
        self.persist_record(record.with_benchmark(updated), conn.write_text)
        return RecoveryOutcome(
            decision=decision, ok=True, dispatched="reverify", job_id=job_id, judge_job={**entry, "job_id": job_id}
        )

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

    def walltime_plan(self, record: SubmissionRecord, benchmark: str, max_walltime: str) -> dict[str, Any]:
        """Check a ``--max-walltime`` raise against the partition's MaxTime, read with sinfo on the cluster, before
        any evidence is read (see ``_plan_walltime``). A failed query keeps the whole raise per segment and says so."""
        with get_connection(record.hostname) as conn:
            config = self._saved_config(conn, Path(record.run_dir))
            limit = _partition_time_limit(conn, config.compute[record.cluster])
        return _plan_walltime(config, record.cluster, benchmark, max_walltime, limit)

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
        extend_attempts: int = 0,
        max_walltime: str | None = None,
        walltime_reason: str | None = None,
        operator_continuation: Mapping[str, Any] | None = None,
        unsafe_resume_reason: str | None = None,
        server_spinup_timeout_seconds: int | None = None,
        spinup_timeout_reason: str | None = None,
        fresh_restart: bool = False,
        segment_walltime: str | None = None,
        rerun_masked: Mapping[str, Any] | None = None,
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
        ``segment_walltime`` (from ``walltime_plan``) is what each Slurm segment requests instead, when the partition's
        MaxTime is below the raise: the raise stays the chain cap.
        ``rerun_masked`` (with ``keep_budget``) is the operator's approval to run again the masked results that kept a
        saved answer but have no gradeable artifact and no grade: its ``reason`` and ``identities`` (identity, attempt
        id and attempts counted) are appended as one line to the run's rerun-masked journal in artifacts/, which the
        Store reads. The line is written just before sbatch and taken back if the job is not queued.
        A job the scheduler refuses raises ``SchedulerRefusedError`` with sbatch's text and the restored job dir.
        ``unsafe_resume_reason`` (with ``keep_budget``) renders the job with ``allow_unsafe_resume`` the same way, so
        its resume proceeds past a refused run identity check; the run dir's unsafe-resume journal records the reason
        and the user once the job is queued.
        ``server_spinup_timeout_seconds`` (with ``spinup_timeout_reason``) renders the job with that server
        spin-up limit the same way; it only raises the recorded limit (0 removes it), and the job dir's
        spin-up timeout journal records both limits, the reason and the user once the job is queued.
        ``operator_continuation`` (with ``keep_budget``) is a decision's OPERATOR-CONTINUATION record for a run whose
        accounting is unknown: under the lock it fills the counters the record names, replaces the unknown-accounting
        marker with a note and keeps the readable counters; each step is undone unless the job is queued, and the job
        dir's operator journal records the queued continuation.
        ``fresh_restart`` (with ``max_walltime``, ``walltime_reason`` and ``accounting_snapshot``, without
        ``keep_budget``) queues a run that has no task inventory again as a fresh run: a fresh retry and walltime
        budget, a walltime longer than the recorded one, the snapshot rechecked under the lock. Its walltime journal
        entry, the receipt, carries ``plan`` FRESH-RESTART and ``why``; ``.gym_chain_incomplete`` names the restart
        and both walltimes until the queued job continues the chain.

        Refuses while any of the caller's jobs runs in that job dir, so two jobs never write one run, and
        refuses a run with rollouts whose `gym eval run` job.sh predates resubmit: it would start the run
        fresh and delete them. A `command` benchmark's job.sh runs its own command, so it is not refused.
        """
        if fresh_restart and (keep_budget or max_walltime is None):
            raise ValueError("a fresh restart starts a fresh budget and needs --max-walltime")
        job = record.benchmark_job(benchmark)
        job_dir = Path(job.job_dir)
        if operator_continuation is not None and (not keep_budget or extend_retries):
            raise RuntimeError("an operator continuation needs --recover and sets its own infra-retry allowance")
        operator_commands = (
            self._operator_commands(operator_continuation, at=utc_timestamp(utc_now()))
            if operator_continuation is not None
            else []
        )
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
                    unsafe_resume_reason=unsafe_resume_reason,
                    server_spinup_timeout_seconds=server_spinup_timeout_seconds,
                    spinup_timeout_reason=spinup_timeout_reason,
                    fresh_restart=fresh_restart,
                    segment_walltime=segment_walltime,
                )
                if repin is not None
                or max_walltime is not None
                or unsafe_resume_reason is not None
                or server_spinup_timeout_seconds is not None
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
            if keep_budget or fresh_restart:
                stale = "run accounting changed since the recovery evidence was collected; collect it again"
                if accounting_snapshot is None:
                    raise RuntimeError(f"Resubmit of {benchmark} refused: no collected accounting snapshot")
                resume = config.driver.benchmarks[benchmark].resume_config
                # resume_config includes recovery_outcome_policy. A policy that changed since the evidence was
                # collected changes which rows run again, so like semantic drift it is refused, never resumed across.
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
            if keep_budget and accounted_walltime_s is not None and operator_continuation is None:
                # This is terminal accounting of the complete recorded chain, not a new allowance.
                if not math.isfinite(accounted_walltime_s) or accounted_walltime_s < 0:
                    raise ValueError("accounted walltime must be finite and non-negative")
                resume = config.driver.benchmarks[benchmark].resume_config
                cap = _walltime_to_seconds(resume.max_walltime) if resume and resume.max_walltime else None
                if resume is not None and max_walltime is not None:
                    # The raise renders the chain cap the way _walltime_config does.
                    cap = _walltime_to_seconds(_raised_cap(resume.max_walltime, max_walltime))
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
            restart_note = (
                [
                    f"printf '%s\\n' {shlex.quote(self._fresh_restart_note(config, record, max_walltime))} "
                    "> .gym_chain_incomplete || fail 'cannot record the fresh restart'"
                ]
                if fresh_restart
                else []
            )
            # One script, so the lock spans the scheduler check and sbatch; any failed step stops it.
            # The job itself resets the run's retry and attempt state when it starts, unless the resubmit keeps its
            # budget (see RESUBMIT_ENV and KEEP_BUDGET_ENV).
            results = _parse_sbatch_results(
                output := conn.run(
                    exports
                    + [
                        'fail() { echo "Error: $*" >&2; exit 1; }',
                        f"cd {shlex.quote(str(job_dir))} || fail 'cannot enter the job dir'",
                        # bash's noclobber creates the lock with O_EXCL, which is atomic on shared filesystems
                        # that do not support flock. Not mkdir: uutils mkdir (Ubuntu 26.04's default) can
                        # report success to two concurrent callers.
                        f'(set -o noclobber; : > {_RESUBMIT_LOCK}) 2>/dev/null || fail "another resubmit holds '
                        f'$PWD/{_RESUBMIT_LOCK}; remove it if no resubmit is running"',
                        "_gym_submit_started=0; _gym_submit_confirmed=0; _gym_submit_refused=0",
                        "_gym_release_resubmit_lock() { "
                        'if [[ "$_gym_submit_started" == 1 && "$_gym_submit_confirmed" != 1 '
                        '&& "$_gym_submit_refused" != 1 ]]; then '
                        f"printf '%s\\n' {shlex.quote(ACCOUNTING_UNKNOWN_PREFIX + ' Resubmit acknowledgement was not confirmed.')} "
                        "> .gym_chain_incomplete || return; fi; "
                        f"rm -f {_RESUBMIT_LOCK}; }}",
                        "_gym_restore_repin() { :; }; _gym_restore_extension() { :; }; _gym_restore_operator() { :; }; "
                        "_gym_restore_attempts() { :; }; _gym_restore_rerun_masked() { :; }",
                        "trap '_gym_restore_operator; _gym_restore_extension; _gym_restore_attempts; "
                        "_gym_restore_rerun_masked; _gym_restore_repin; "
                        "_gym_release_resubmit_lock' EXIT",
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
                        *(
                            # Only the marker the evidence reconciled; its digest is also in the snapshot above.
                            [
                                f"grep -qxF -- {shlex.quote(ACCOUNTING_UNKNOWN_PREFIX + ' ' + REQUEUE_UNKNOWN_REASON)} "
                                ".gym_chain_incomplete || fail 'the reconciled requeue marker changed; collect it again'"
                            ]
                            if reconciled
                            # An operator continuation replaces the unknown marker below; the snapshot pins it.
                            else []
                            if operator_continuation is not None
                            else [
                                f"if grep -q -- {shlex.quote('^' + ACCOUNTING_UNKNOWN_PREFIX)} .gym_chain_incomplete "
                                '2>/dev/null; then fail "allocation accounting is unknown; reconcile every allocation '
                                'before resubmission"; fi'
                            ]
                        ),
                        'for claim in .gym_followup_queued.*; do [ -e "$claim" ] || continue; '
                        'claimed=$(cat "$claim") || fail "follow-up claim unreadable"; '
                        '[[ "$claimed" =~ ^[0-9]+$ ]] || fail "follow-up submission identity unknown"; done',
                        # Only the built-in runner's fresh start deletes rollouts; its job.sh lists
                        # `gym eval run` on a line of its own (see `render_gym_cmd`).
                        f"if [ -s {ARTIFACTS_DIRNAME}/{ROLLOUTS_FILENAME} ] && grep -qx -- '[[:space:]]*gym eval run' job.sh "
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
                        *operator_commands,
                        *(self._extension_commands(extend_retries, keep_budget=keep_budget) if extend_retries else []),
                        *(
                            self._attempt_extension_commands(extend_attempts, keep_budget=keep_budget)
                            if extend_attempts
                            else []
                        ),
                        *(
                            self._rerun_masked_commands(rerun_masked, record, benchmark, keep_budget=keep_budget)
                            if rerun_masked is not None
                            else []
                        ),
                        *restart_note,
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
                        # The scheduler's own refusal, not a lost reply, and the queue confirms nothing of this run is
                        # in it: no job was queued, so the chain's accounting stays known.
                        'if [[ $rc != 0 ]] && [[ $out == *"Batch job submission failed"* ]] '
                        "&& ! grep -qiE 'timed out|unable to contact|socket|zero bytes|connection' <<< \"$out\" "
                        "&& _gym_after=$(squeue --me --all --noheader --format=%Z); then _gym_submit_refused=1; "
                        'for d in $_gym_after; do [ "$d" != "$PWD" ] || _gym_submit_refused=0; done; fi',
                        'echo "__GYM_SUBMIT_REFUSED:$_gym_submit_refused"',
                        *(
                            # Still under the lock: name the queued job in the epoch record written before sbatch.
                            # A failure here leaves the record without its job ID, never without its epoch.
                            [
                                'if [[ "$_gym_submit_confirmed" == 1 ]]; then python3 -c '
                                + shlex.quote(_NAME_OPERATOR_JOB)
                                + f' {OPERATOR_CONTINUATION_JOURNAL_NAME} "$out" || :; fi'
                            ]
                            if operator_continuation is not None
                            else []
                        ),
                    ]
                )
            )
            job_id, error = results.get(benchmark, (None, "sbatch produced no result"))
            if job_id is None:
                message = (
                    f"Resubmit of {benchmark} failed: {error}"
                    + ("; the run stays on its recorded Gym commit" if repin is not None else "")
                    + ("; the run keeps its recorded walltime" if max_walltime is not None else "")
                )
                if benchmark not in results:
                    raise RuntimeError(message)
                # sbatch answered: the exit trap put back everything this resubmit changed before releasing the lock.
                marker = conn.run(
                    [f"cat {shlex.quote(str(job_dir / INCOMPLETE_MARKER_NAME))} 2>/dev/null || true"]
                ).strip()
                raise SchedulerRefusedError(
                    message,
                    {
                        "benchmark": benchmark,
                        "scheduler_text": error,
                        # Only a refusal the queue confirmed leaves the chain marker as it was; otherwise the
                        # trap records that the acknowledgement is unknown, as before.
                        "job_dir_restored": "__GYM_SUBMIT_REFUSED:1" in output,
                        "restored": [
                            *(["job.sh", RESOLVED_CONFIG_NAME] if repin_commands else []),
                            *(["infra-retry extension"] if extend_retries else []),
                            *(["attempt extension"] if extend_attempts else []),
                            *(["rerun-masked approval"] if rerun_masked is not None else []),
                            *(["operator continuation counters"] if operator_continuation is not None else []),
                            "resubmit lock released",
                        ],
                        "chain_marker": marker or None,
                    },
                )
            job = job.model_copy(
                update={"job_id": job_id, "error": None}
                | ({"gym_commit": repin} if repin is not None else {})
                | ({"walltime": segment_walltime or max_walltime} if max_walltime is not None else {})
            )
            # One journal for both extensions, one line per kind: the run's infra-retry cap and every rollout's
            # attempt cap.
            journal = shlex.quote(str(job_dir / RETRY_EXTENSION_JOURNAL_NAME))
            for kind, by in (("infra", extend_retries), ("attempts", extend_attempts)):
                if by:
                    entry = shlex.quote(
                        json.dumps({"kind": kind, "by": by, "at": utc_timestamp(utc_now()), "user": getpass.getuser()})
                    )
                    conn.run([f"printf '%s\\n' {entry} >> {journal}"])
            for name, entry in journals.items():
                # A name is in the run dir; a job-dir journal is keyed by its absolute path.
                journal = Path(record.run_dir) / name
                conn.run([f"printf '%s\\n' {shlex.quote(entry)} >> {shlex.quote(str(journal))}"])
            self.persist_record(record.with_benchmark(job), conn.write_text)
        return job

    @staticmethod
    def _operator_commands(continuation: Mapping[str, Any], *, at: str) -> list[str]:
        """The locked steps of an operator continuation: fill the counters it names, replace the unknown marker and
        journal the continuation with its epoch before sbatch, so a queued continuation always has its epoch.

        Each file is kept aside first and put back unless the job is queued. A readable counter is never named, except
        the retry extension an explicit ``--extend-retries`` raises when the spent count is readable.
        """
        writes = continuation.get("writes") or {}
        if not set(writes) <= {INFRA_RETRIES_FILE, ACCUMULATED_WALLTIME_FILE, INFRA_RETRY_EXTENSION_FILE} or any(
            type(write.get("value")) is not int or write["value"] < 0 for write in writes.values()
        ):
            raise ValueError("an operator continuation fills only the chain's counters, each with a whole number")
        names = [*sorted(writes), INCOMPLETE_MARKER_NAME, OPERATOR_CONTINUATION_JOURNAL_NAME]
        kept = " ".join(f"{name}.operator-previous" for name in names)
        restore = "; ".join(
            f"if [[ -e {name}.operator-previous ]]; then mv -f {name}.operator-previous {name}; else rm -f {name}; fi"
            for name in names
        )
        unknown = shlex.quote("^" + ACCOUNTING_UNKNOWN_PREFIX)
        entry = {**continuation, "job_id": None, "at": at}
        return [
            f"rm -f {kept}",
            *(
                f"if [[ -e {name} ]]; then cp -p {name} {name}.operator-previous || fail 'cannot keep {name}'; fi"
                for name in names
            ),
            f'_gym_restore_operator() {{ if [[ "$_gym_submit_confirmed" == 1 ]]; then rm -f {kept}; else {restore}; fi; }}',
            *(
                f"printf '%s\\n' {writes[name]['value']} > {name}.tmp && mv -f {name}.tmp {name} "
                f"|| fail 'cannot fill {name}'"
                for name in sorted(writes)
            ),
            f"if grep -q -- {unknown} {INCOMPLETE_MARKER_NAME} 2>/dev/null; then "
            f"printf '%s\\n' {shlex.quote(OPERATOR_CONTINUATION_NOTE)} > {INCOMPLETE_MARKER_NAME} "
            "|| fail 'cannot record the operator continuation'; fi",
            f"printf '%s\\n' {shlex.quote(json.dumps(entry, sort_keys=True))} >> {OPERATOR_CONTINUATION_JOURNAL_NAME} "
            "|| fail 'cannot journal the operator continuation'",
        ]

    @staticmethod
    def _fresh_restart_note(config: SubmitConfig, record: SubmissionRecord, max_walltime: str) -> str:
        """The chain marker a fresh restart writes: why, the recorded walltime and the new one."""
        recorded = config.compute[record.cluster].walltime or "the cluster default"
        return f"FRESH-RESTART: {NO_INVENTORY}; walltime {recorded} -> {max_walltime}"

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

    @staticmethod
    def _attempt_extension_commands(extend_attempts: int, *, keep_budget: bool) -> list[str]:
        """The locked steps that record ``extend_attempts`` more attempts for every rollout, restored unless queued.

        The Store reads the record (artifacts/rollouts_attempt_extension) as an addition to the run's saved
        ``max_rollout_attempts``; a rollout that ended with a model failure is never run again.
        """
        if not keep_budget:
            raise RuntimeError("--extend-attempts needs --recover: it extends the spent attempts, never resets them")
        if type(extend_attempts) is not int or not 0 < extend_attempts <= MAX_EXTEND_ATTEMPTS:
            raise ValueError(
                f"--extend-attempts needs a whole number from 1 to {MAX_EXTEND_ATTEMPTS}, not {extend_attempts!r}"
            )
        name = "artifacts/" + attempt_extension_path_for(Path("rollouts.jsonl")).name
        return [
            f"_gym_old_attempts=$(cat {name} 2>/dev/null || echo 0)",
            '[[ "$_gym_old_attempts" =~ ^[0-9]{1,6}$ ]] || fail "the recorded attempt extension is malformed"',
            f"[[ -e {name} ]] && _gym_had_attempts=1 || _gym_had_attempts=0",
            '_gym_restore_attempts() { [[ "$_gym_submit_confirmed" == 1 ]] && return; '
            f'if [[ "$_gym_had_attempts" == 1 ]]; then printf "%s\\n" "$_gym_old_attempts" > {name}; '
            f"else rm -f {name}; fi; }}",
            f"printf '%s\\n' $((10#$_gym_old_attempts + {extend_attempts})) > {name}.tmp && mv -f {name}.tmp {name} "
            "|| fail 'cannot record the attempt extension'",
        ]

    @staticmethod
    def _rerun_masked_commands(
        approval: Mapping[str, Any], record: SubmissionRecord, benchmark: str, *, keep_budget: bool
    ) -> list[str]:
        """The locked steps that journal the approval to run masked results again, taken back unless the job is queued.

        The line goes to the rerun-masked journal beside the rollouts file, where the Store reads it when the job starts:
        its time, user, reason and each approved identity with the attempt it covers and the attempts counted for it.
        """
        if not keep_budget:
            raise RuntimeError(
                "--rerun-masked-without-artifact needs --recover: the run continues with its spent budgets"
            )
        reason = approval.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("--rerun-masked-without-artifact needs a nonempty --rerun-reason")
        identities = approval.get("identities")
        if not isinstance(identities, list) or not identities:
            raise ValueError("--rerun-masked-without-artifact approves at least one identity")
        entry = {
            "at": utc_timestamp(utc_now()),
            "user": getpass.getuser(),
            "reason": reason.strip(),
            "gym_job_id": record.gym_job_id,
            "benchmark": benchmark,
            "identities": identities,
        }
        name = "artifacts/" + RERUN_MASKED_JOURNAL_NAME
        return [
            f"_gym_rerun_size=$(stat -c %s {name} 2>/dev/null || echo -1)",
            '_gym_restore_rerun_masked() { [[ "$_gym_submit_confirmed" == 1 ]] && return; '
            f'if [[ "$_gym_rerun_size" == -1 ]]; then rm -f {name}; else truncate -s "$_gym_rerun_size" {name}; fi; }}',
            f"printf '%s\\n' {shlex.quote(json.dumps(entry, sort_keys=True))} >> {name} "
            "|| fail 'cannot journal the rerun-masked approval'",
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
        unsafe_resume_reason: str | None = None,
        server_spinup_timeout_seconds: int | None = None,
        spinup_timeout_reason: str | None = None,
        fresh_restart: bool = False,
        segment_walltime: str | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        """The locked shell steps that render one benchmark's job.sh and saved config again, with a repin, a walltime
        raise or both, and the journal entries to append once its job is queued. A fresh restart raises the walltime
        above the recorded one without ``keep_budget``."""
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
            if not keep_budget and not fresh_restart:
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
            if fresh_restart and recorded is not None and seconds <= _walltime_to_seconds(recorded):
                raise ValueError(f"--max-walltime {max_walltime} is not longer than the recorded walltime {recorded}")
            if recorded is not None and seconds < _walltime_to_seconds(recorded):
                raise ValueError(f"--max-walltime {max_walltime} is below the recorded walltime {recorded}")
            resume = config.driver.benchmarks[benchmark].resume_config
            target = _walltime_config(target, record.cluster, benchmark, max_walltime, segment_walltime)
            raised = target.driver.benchmarks[benchmark].resume_config
            journals[WALLTIME_JOURNAL_NAME] = json.dumps(
                {
                    "benchmark": benchmark,
                    "from": recorded,
                    "to": max_walltime,
                    "max_walltime_from": resume.max_walltime if resume else None,
                    "max_walltime_to": raised.max_walltime if raised else None,
                    "segment": segment_walltime or max_walltime,
                    "reason": walltime_reason,
                    "actor": getpass.getuser(),
                    "at": stamp,
                }
                | ({"plan": "FRESH-RESTART", "why": NO_INVENTORY} if fresh_restart else {}),
                sort_keys=True,
            )
        if unsafe_resume_reason is not None:
            if not keep_budget:
                raise RuntimeError("--allow-unsafe-resume needs --recover: the run continues with its spent budgets")
            if not unsafe_resume_reason.strip():
                raise ValueError("--allow-unsafe-resume needs a nonempty --unsafe-resume-reason")
            target = _unsafe_resume_config(target, benchmark)
            journals[UNSAFE_RESUME_JOURNAL_NAME] = json.dumps(
                {"benchmark": benchmark, "reason": unsafe_resume_reason, "actor": getpass.getuser(), "at": stamp},
                sort_keys=True,
            )
        job_dir = Path(record.benchmark_job(benchmark).job_dir)
        if server_spinup_timeout_seconds is not None:
            if not keep_budget and not fresh_restart:
                raise RuntimeError(
                    "--server-spinup-timeout-seconds needs --recover: the run continues with its spent budgets"
                )
            if not (spinup_timeout_reason or "").strip():
                raise ValueError("--server-spinup-timeout-seconds needs a nonempty --spinup-timeout-reason")
            recorded = _recorded_spinup_timeout_seconds(config, benchmark)
            refusal = _spinup_timeout_raise_refusal(recorded, server_spinup_timeout_seconds)
            if refusal is not None:
                raise ValueError(refusal)
            target = _spinup_timeout_config(target, benchmark, server_spinup_timeout_seconds)
            journals[str(job_dir / SPINUP_TIMEOUT_JOURNAL_NAME)] = json.dumps(
                {
                    "ts": stamp,
                    "user": getpass.getuser(),
                    "from_seconds": recorded,
                    "to_seconds": server_spinup_timeout_seconds,
                    "reason": spinup_timeout_reason,
                    "gym_job_id": record.gym_job_id,
                    "benchmark": benchmark,
                },
                sort_keys=True,
            )
        # This full render writes this launcher's job templates, so the launcher must be the revision the job runs,
        # whatever setting is raised. A repin has already made its target that revision.
        pinned = target.driver.gym_install.ref if target.driver.gym_install is not None else None
        if pinned is not None and installed_gym_commit() != pinned:
            raise IncompatibleRerenderError(pinned, installed_gym_commit())
        run_dir = Path(record.run_dir)
        script = build_sbatch_script(
            target, benchmark, target.driver.benchmarks[benchmark], target.compute[record.cluster], job_dir
        )
        resolved = yaml.safe_dump(target.model_dump(mode="json", context={HOST_ENV_REFS: True}), sort_keys=False)
        old_config = shlex.quote(str(run_dir / RESOLVED_CONFIG_NAME))
        kinds = [
            kind
            for kind, wanted in (
                ("repin", repin is not None),
                ("walltime", max_walltime is not None),
                ("unsafe-resume", unsafe_resume_reason is not None),
                ("spinup-timeout", server_spinup_timeout_seconds is not None),
            )
            if wanted
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
