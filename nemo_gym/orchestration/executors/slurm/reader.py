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

"""The remote Store reader: runs the evidence collector and the run's own Store selection on the cluster, in a
Slurm step when one is configured, else on the login host, and bounds, watches and stops it.
"""

import inspect
import json
import os
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from nemo_gym.eval.recovery.evidence import (
    READER_PLACEMENT_FIELDS,
    CollectedEvidence,
    _managed_trial_metadata,
)
from nemo_gym.orchestration.api import (
    SubmitConfig,
)
from nemo_gym.orchestration.executors.connection import (
    READER_PROGRESS_PREFIX,
    Connection,
)
from nemo_gym.orchestration.executors.script_templates import render_driver_entrypoint


# Executed only inside the saved pinned install. The projection contains no saved answer text. That install may
# predate the move to nemo_gym.eval.recovery, so the reader falls back to the old planner path.
_REMOTE_MANAGED_READER = r"""
import json, subprocess, sys
from pathlib import Path
try:
    import nemo_gym.eval.recovery.evidence as recovery
except ImportError:
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
