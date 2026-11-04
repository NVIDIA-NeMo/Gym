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

"""How the Slurm executor stops the jobs it queued.

Stopping a benchmark is more than `scancel` on its latest job. Under auto-resume every job queues its own
follow-up, so a follow-up that is still pending would start after its parent is cancelled. The script finds
everything of the benchmark that is queued or running -- by the directory it runs in, and by the job ids
the chain and the follow-up claim files name -- and cancels the pending jobs before the running ones, so no
follow-up can start in between.

A job that is being cancelled can still queue its follow-up in the moments before it dies, after the queue
was read. So the queue is read again until nothing of the benchmark is left in it, and whatever appeared in
between is cancelled too.
"""

import shlex

from nemo_gym.orchestration.executors.resume_script import JOB_CHAIN_NAME
from nemo_gym.orchestration.jobs import BenchmarkJob


# Each job writes this file, named by its own job id, holding the id of the follow-up it queued.
FOLLOWUP_CLAIM_PREFIX = ".gym_followup_queued."

# Every line of the script's output that carries an answer starts with this, so a warning from a shell
# profile or from Slurm cannot be mistaken for one.
_MARK = "__GYM_CONTROL"
_SEP = ":"

# What an answer line is about: `<_MARK><_SEP><kind><_SEP><payload>`.
_KIND_ERROR = "error"
_KIND_CANCELLED = "cancelled"
_KIND_NONE = "none"

# The queue is read at most this many times, this many seconds apart, while jobs already asked to stop
# are still leaving it.
_DRAIN_PASSES = 6
_DRAIN_WAIT_S = 5


# Shell that turns what a command printed into one line, so it fits in an answer line.
_FLAT_OUT = '$(echo "$out" | tr "\\n" " ")'
_FLAT_QUEUE = '$(echo "$squeue_out" | tr "\\n" " ")'
_FLAT_GROUP = '$(echo $_group | tr " " ",")'


def _line(kind: str, payload: str) -> str:
    return f"{_MARK}{_SEP}{kind}{_SEP}{payload}"


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
        'asked=" "; cancelled_any=0',
        f"for _pass in $(seq 1 {_DRAIN_PASSES}); do",
        "  squeue_out=$(squeue --me --all --noheader --format='%i|%T|%Z' 2>&1); squeue_rc=$?",
        f'  if [ "$squeue_rc" -ne 0 ]; then echo "{_line(_KIND_ERROR, f"squeue failed: {_FLAT_QUEUE}")}"; exit 0; fi',
        '  pending=""; others=""; present=0',
        '  while IFS="|" read -r _id _state _cwd; do',
        '    [ -n "$_id" ] || continue',
        # squeue's %Z is the --chdir path as given while the directory here may be the resolved one, so the
        # two are compared as the same file (-ef), never as strings.
        '    if { [ -n "$_cwd" ] && [ "$_cwd" -ef "$dir" ]; } || [[ "$wanted" == *",$_id,"* ]]; then',
        "      present=1",
        '      case "$asked" in *" $_id "*) continue ;; esac',
        '      if [ "$_state" = PENDING ]; then pending="$pending $_id"; else others="$others $_id"; fi',
        "    fi",
        '  done <<< "$squeue_out"',
        '  if [ -z "$pending$others" ]; then',
        '    [ "$present" -eq 0 ] && break',
        f"    sleep {_DRAIN_WAIT_S}; continue",
        "  fi",
        '  for _group in "$pending" "$others"; do',
        '    [ -n "$_group" ] || continue',
        "    out=$(scancel $_group 2>&1); rc=$?",
        '    asked="$asked$_group "; cancelled_any=1',
        f'    if [ "$rc" -eq 0 ]; then echo "{_line(_KIND_CANCELLED, _FLAT_GROUP)}";'
        f' else echo "{_line(_KIND_ERROR, f"scancel$_group: {_FLAT_OUT}")}"; fi',
        "  done",
        "  sleep 2",
        "done",
        f'[ "$cancelled_any" -eq 0 ] && echo "{_line(_KIND_NONE, "")}"',
        "true",
    ]


def parse_terminate_output(output: str) -> tuple[list[str], list[str]]:
    """The job ids `render_terminate_script` cancelled, and what went wrong, if anything."""
    cancelled: list[str] = []
    errors: list[str] = []
    for line in output.splitlines():
        if not line.startswith(f"{_MARK}{_SEP}"):
            continue
        kind, _, rest = line[len(_MARK) + len(_SEP) :].partition(_SEP)
        if kind == _KIND_CANCELLED:
            cancelled += [job_id for job_id in rest.split(",") if job_id]
        elif kind == _KIND_ERROR:
            errors.append(rest.strip())
    return cancelled, errors
