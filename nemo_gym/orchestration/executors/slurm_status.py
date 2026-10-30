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

"""How the Slurm executor looks at the jobs it queued.

One script per host asks Slurm about every job directory of every record at once: a single `squeue` for
the whole batch, a single `sacct` over every job id any of them has had, and the state files each job
directory holds. The cost grows with the number of hosts, not with the number of submissions. Jobs are
found by where they run, not by the recorded id, so a job started by hand, a resumed one, or one
resubmitted from another machine is still seen.
"""

import base64
import re
import shlex
from collections.abc import Iterable
from pathlib import Path

from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME
from nemo_gym.orchestration.jobs import CHAIN_INCOMPLETE_FILE, EVAL_EXIT_FILE, JobKey, JobRef, JobState, ObservedStatus


# Every line of the script's output that carries an answer starts with this, so a warning from a shell
# profile or from Slurm cannot be mistaken for one.
_MARK = "__GYM_STATE:"

# Read at most this much of a state file; they hold a short reason or an exit code.
_STATE_FILE_BYTES = 4096

# The state files read from each job directory, in the order they are asked for.
STATE_FILES = (CHAIN_INCOMPLETE_FILE, EVAL_EXIT_FILE)

_PENDING_STATES = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD"}
_ACTIVE_STATES = {"RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING", "SIGNALING", "STAGE_OUT", "STOPPED"}
_FAILED_STATES = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
    "SPECIAL_EXIT",
}


def render_status_script(refs: Iterable[JobRef]) -> list[str]:
    """The commands that report the state of every job in `refs`, on one connection."""
    lines = [
        "set +e",
        "squeue_out=$(squeue --me --all --noheader --format='%i|%T|%Z' 2>&1); squeue_rc=$?",
        f'if [ "$squeue_rc" -ne 0 ]; then echo "{_MARK}squeue-error:$(echo "$squeue_out" | tr "\\n" " ")"; squeue_out=""; fi',
        'ids=""',
    ]
    for index, ref in enumerate(refs):
        tag = f"r{index}"
        # Only digits reach the sacct id list: the recorded id is data from a file on disk.
        recorded = ref.job.job_id if ref.job.job_id and ref.job.job_id.isdigit() else ""
        lines += [
            f"dir={shlex.quote(ref.job.job_dir)}",
            f'chain=$(tr "\\n" "," < "$dir/{JOB_CHAIN_NAME}" 2>/dev/null)',
            f'echo "{_MARK}chain:{tag}:$chain"',
            f'ids="$ids,$chain,{recorded}"',
            # squeue's %Z is the --chdir path as given while the directory here may be the resolved one,
            # so the two are compared as the same file (-ef), never as strings.
            'while IFS="|" read -r _id _state _cwd; do',
            f'  if [ -n "$_cwd" ] && [ "$_cwd" -ef "$dir" ]; then echo "{_MARK}active:{tag}:$_id|$_state"; fi',
            'done <<< "$squeue_out"',
            f"for _f in {' '.join(STATE_FILES)}; do",
            f'  if [ -f "$dir/$_f" ]; then echo "{_MARK}file:{tag}:$_f:'
            f'$(head -c {_STATE_FILE_BYTES} "$dir/$_f" | base64 | tr -d "\\n")"; fi',
            "done",
        ]
    lines += [
        'ids=$(echo "$ids" | tr "," "\\n" | grep -E "^[0-9]+$" | sort -un | paste -sd, -)',
        'if [ -n "$ids" ]; then',
        '  acct=$(sacct -j "$ids" -X -P -n -o JobID,State,ExitCode 2>&1); acct_rc=$?',
        f'  if [ "$acct_rc" -eq 0 ]; then echo "$acct" | sed "s/^/{_MARK}acct:/";'
        f' else echo "{_MARK}acct-error:$(echo "$acct" | tr "\\n" " ")"; fi',
        "fi",
        "true",
    ]
    return lines


def parse_status_output(output: str, refs: list[JobRef]) -> dict[JobKey, ObservedStatus]:
    """What `render_status_script` reported, as one `ObservedStatus` per job in `refs`."""
    chains: dict[str, list[str]] = {}
    active: dict[str, list[tuple[str, str]]] = {}
    files: dict[str, dict[str, str]] = {}
    accounting: dict[str, tuple[str, int | None]] = {}
    squeue_error = acct_error = None

    for line in output.splitlines():
        if not line.startswith(_MARK):
            continue
        kind, _, rest = line[len(_MARK) :].partition(":")
        if kind == "squeue-error":
            squeue_error = rest.strip()
        elif kind == "acct-error":
            acct_error = rest.strip()
        elif kind == "chain":
            tag, _, ids = rest.partition(":")
            chains[tag] = [job_id for job_id in ids.split(",") if job_id.isdigit()]
        elif kind == "active":
            tag, _, job = rest.partition(":")
            job_id, _, state = job.partition("|")
            active.setdefault(tag, []).append((job_id, state.strip()))
        elif kind == "file":
            tag, _, rest = rest.partition(":")
            name, _, encoded = rest.partition(":")
            files.setdefault(tag, {})[name] = base64.b64decode(encoded).decode("utf-8", errors="replace")
        elif kind == "acct":
            job_id, _, rest = rest.partition("|")
            state, _, exit_code = rest.partition("|")
            if job_id.isdigit():
                accounting[job_id] = (state.strip(), _exit_status(exit_code))

    return {
        ref.key: _observe(
            ref,
            chain=chains.get(f"r{index}", []),
            active=active.get(f"r{index}", []),
            files=files.get(f"r{index}", {}),
            accounting=accounting,
            squeue_error=squeue_error,
            acct_error=acct_error,
        )
        for index, ref in enumerate(refs)
    }


def render_read_files_script(paths: Iterable[Path]) -> list[str]:
    """The commands that print each readable file of `paths`, encoded so a file cannot break the output."""
    lines = ["set +e"]
    for index, path in enumerate(paths):
        quoted = shlex.quote(str(path))
        lines.append(
            f'if [ -f {quoted} ] && [ -r {quoted} ]; then echo "{_MARK}file:{index}:$(base64 < {quoted} | tr -d "\\n")"; fi'
        )
    lines.append("true")
    return lines


def parse_read_files_output(output: str, paths: list[Path]) -> dict[Path, str | None]:
    """The contents `render_read_files_script` printed, `None` for a file it did not find."""
    contents: dict[Path, str | None] = {path: None for path in paths}
    for line in output.splitlines():
        if not line.startswith(f"{_MARK}file:"):
            continue
        index, _, encoded = line[len(f"{_MARK}file:") :].partition(":")
        if index.isdigit() and int(index) < len(paths):
            contents[paths[int(index)]] = base64.b64decode(encoded).decode("utf-8", errors="replace")
    return contents


def _observe(
    ref: JobRef,
    *,
    chain: list[str],
    active: list[tuple[str, str]],
    files: dict[str, str],
    accounting: dict[str, tuple[str, int | None]],
    squeue_error: str | None,
    acct_error: str | None,
) -> ObservedStatus:
    recorded = [ref.job.job_id] if ref.job.job_id and ref.job.job_id.isdigit() else []
    job_ids = sorted({*chain, *recorded, *(job_id for job_id, _ in active)}, key=int)
    common = {"job_ids": job_ids, "files": files}

    running = [job_id for job_id, state in active if state not in _PENDING_STATES]
    if running or active:
        job_id = max(running or [job_id for job_id, _ in active], key=int)
        state = JobState.RUNNING if running else JobState.QUEUED
        native = next(native for found, native in active if found == job_id)
        return ObservedStatus(state=state, detail=native, job_id=job_id, **common)

    # Nothing of this run is queued or running, so the most recent job of the run says how it ended.
    for job_id in sorted(job_ids, key=int, reverse=True):
        if job_id in accounting:
            native, exit_status = accounting[job_id]
            state = state_from_sacct(native)
            reason = f"The Slurm job ended as {native}." if state is JobState.FAILED else None
            return ObservedStatus(
                state=state, detail=native, exit_code=exit_status, reason=reason, job_id=job_id, **common
            )

    if squeue_error:
        return ObservedStatus(state=JobState.UNKNOWN, detail=f"squeue failed: {squeue_error}", **common)
    if acct_error:
        return ObservedStatus(state=JobState.UNKNOWN, detail=f"sacct failed: {acct_error}", **common)
    if ref.job.job_id is None and ref.job.error:
        return ObservedStatus(state=JobState.FAILED, detail=ref.job.error, **common)
    return ObservedStatus(state=JobState.UNKNOWN, detail="Slurm has no record of this job", **common)


def state_from_sacct(native: str) -> JobState:
    """Slurm's own state name as a `JobState`; a name this does not know is `UNKNOWN`."""
    name = native.split()[0].rstrip("+") if native.strip() else ""
    if name == "COMPLETED":
        return JobState.SUCCEEDED
    if name.startswith("CANCELLED"):
        return JobState.CANCELLED
    if name in _PENDING_STATES:
        return JobState.QUEUED
    if name in _ACTIVE_STATES:
        return JobState.RUNNING
    if name in _FAILED_STATES:
        return JobState.FAILED
    return JobState.UNKNOWN


def _exit_status(text: str) -> int | None:
    """The exit code of sacct's `code:signal` pair."""
    match = re.match(r"\s*(\d+):\d+\s*$", text)
    return int(match.group(1)) if match else None
