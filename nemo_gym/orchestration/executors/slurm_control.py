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

"""How the Slurm executor stops the jobs it queued and reads their logs.

Stopping a benchmark is more than `scancel` on its latest job. Under auto-resume every job queues its own
follow-up, so a follow-up that is still pending would start after its parent is cancelled. The script finds
everything of the benchmark that is queued or running -- by the directory it runs in, and by the job ids
the chain and the follow-up claim files name -- and cancels the pending jobs before the running ones, so no
follow-up can start in between.
"""

import shlex
from pathlib import Path

from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME
from nemo_gym.orchestration.jobs import BenchmarkJob


# Each job writes this file, named by its own job id, holding the id of the follow-up it queued.
FOLLOWUP_CLAIM_PREFIX = ".gym_followup_queued."

# Every line of the script's output that carries an answer starts with this, so a warning from a shell
# profile or from Slurm cannot be mistaken for one.
_MARK = "__GYM_CONTROL:"


def render_terminate_script(job: BenchmarkJob) -> list[str]:
    """The commands that cancel every queued or running job of `job`: the pending ones first, then the rest."""
    # Only digits reach the id list: the recorded id is data from a file on disk.
    recorded = job.job_id if job.job_id and job.job_id.isdigit() else ""
    return [
        "set +e",
        f"dir={shlex.quote(job.job_dir)}",
        f'chain=$(tr "\\n" "," < "$dir/{JOB_CHAIN_NAME}" 2>/dev/null)',
        'followups=""',
        f'for _f in "$dir"/{FOLLOWUP_CLAIM_PREFIX}*; do [ -f "$_f" ] && followups="$followups,$(tr "\\n" "," < "$_f")"; done',
        f'wanted=",$chain,$followups,{recorded},"',
        "squeue_out=$(squeue --me --all --noheader --format='%i|%T|%Z' 2>&1); squeue_rc=$?",
        f'if [ "$squeue_rc" -ne 0 ]; then echo "{_MARK}error:squeue failed: $(echo "$squeue_out" | tr "\\n" " ")"; exit 0; fi',
        'pending=""; others=""',
        'while IFS="|" read -r _id _state _cwd; do',
        '  [ -n "$_id" ] || continue',
        # squeue's %Z is the --chdir path as given while the directory here may be the resolved one, so the
        # two are compared as the same file (-ef), never as strings.
        '  if { [ -n "$_cwd" ] && [ "$_cwd" -ef "$dir" ]; } || [[ "$wanted" == *",$_id,"* ]]; then',
        '    if [ "$_state" = PENDING ]; then pending="$pending $_id"; else others="$others $_id"; fi',
        "  fi",
        'done <<< "$squeue_out"',
        'for _group in "$pending" "$others"; do',
        '  [ -n "$_group" ] || continue',
        "  out=$(scancel $_group 2>&1); rc=$?",
        f'  if [ "$rc" -eq 0 ]; then echo "{_MARK}cancelled:$(echo $_group | tr " " ",")";'
        f' else echo "{_MARK}error:scancel$_group: $(echo "$out" | tr "\\n" " ")"; fi',
        "done",
        f'[ -z "$pending$others" ] && echo "{_MARK}none"',
        "true",
    ]


def parse_terminate_output(output: str) -> tuple[list[str], list[str]]:
    """The job ids `render_terminate_script` cancelled, and what went wrong, if anything."""
    cancelled: list[str] = []
    errors: list[str] = []
    for line in output.splitlines():
        if not line.startswith(_MARK):
            continue
        kind, _, rest = line[len(_MARK) :].partition(":")
        if kind == "cancelled":
            cancelled += [job_id for job_id in rest.split(",") if job_id]
        elif kind == "error":
            errors.append(rest.strip())
    return cancelled, errors


def render_read_log_script(path: Path, tail: int | None) -> list[str]:
    """The commands that print `path`, or only its last `tail` lines, after a line saying whether it exists."""
    quoted = shlex.quote(str(path))
    reader = f"cat -- {quoted}" if tail is None else f"tail -n {int(tail)} -- {quoted}"
    return ["set +e", f'if [ -f {quoted} ]; then echo "{_MARK}ok"; {reader}; else echo "{_MARK}missing"; fi', "true"]


def parse_log_output(output: str, path: Path) -> list[str]:
    """The lines of the log `render_read_log_script` printed; a file that does not exist is `FileNotFoundError`."""
    lines = output.split("\n")
    for index, line in enumerate(lines):
        if line == f"{_MARK}missing":
            raise FileNotFoundError(f"{path} does not exist: the job may not have reached the step that writes it.")
        if line == f"{_MARK}ok":
            return "\n".join(lines[index + 1 :]).splitlines()
    raise RuntimeError(f"Reading {path} gave no answer: {output.strip()[:200]!r}")
