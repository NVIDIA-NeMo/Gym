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

"""The auto-resume prologue: a self-chaining Slurm job.

Rendered at the top of a benchmark's sbatch script, ahead of every
service/driver command, when `BenchmarkRunConfig.resumable` is set. Each job
in the chain inspects the PREVIOUS job (passed as `$1`) via `sacct`, decides
whether to continue, and unconditionally queues its own successor with an
`afternotok` dependency before doing any real work. A job that COMPLETEs
leaves its queued successor with a dependency that can never be satisfied, so
Slurm cancels it on its own -- the chain needs no explicit "we're done" step
on the success path.
"""

from nemo_gym.orchestration.api import ResumeConfig


# `_stage` in slurm.py always writes this benchmark's script to exactly this
# filename inside its job dir, which `#SBATCH --chdir` makes `$PWD` at runtime.
_JOB_SCRIPT_NAME = "job.sh"
# The first line of the prologue, so a resubmit can tell a resumable job.sh from one without the chain.
PROLOGUE_MARKER = "# --- Auto-resume chain ---"
JOB_CHAIN_NAME = ".gym_job_chain"
RETRY_FILE_NAME = ".gym_infra_retries"
# Written when the chain stops before the run finished (the walltime cap, or a cancel), so a reader of the
# job dir, not only of sacct, sees that rollouts are missing. A chain that continues removes it.
INCOMPLETE_MARKER_NAME = ".gym_chain_incomplete"
# Set to 1 in the environment of a job queued by `gym eval resubmit`.
RESUBMIT_ENV = "GYM_RESUBMIT"
# Set to 1 with RESUBMIT_ENV by `gym eval resubmit --recover`: the job resumes and keeps the spent retry count,
# walltime and per-rollout attempts, so recovery stays within the budget it read.
KEEP_BUDGET_ENV = "GYM_RESUBMIT_KEEP_BUDGET"
# Capability of the generated script, not evidence that a particular run completed.
BOUNDED_BUDGET_MARKER = "# --- Bounded resume budgets v1 ---"

_PROLOGUE_TEMPLATE = """\
# --- Auto-resume chain ---
OUTPUT_DIR="$PWD"
_this_script="$OUTPUT_DIR/{job_script_name}"
_prev_slurm_job_id="${{1:-}}"
_walltime_file="$OUTPUT_DIR/.gym_accumulated_walltime"
_retry_file="$OUTPUT_DIR/{retry_file_name}"
_incomplete_file="$OUTPUT_DIR/{incomplete_marker_name}"

# The chain only stops here after a job that did not finish the run: a finished run's job COMPLETEs,
# and its follow-up never starts. So the stop exits non-zero, and sacct does not read the run as done.
_gym_stop_incomplete() {{
    echo "$1" > "$_incomplete_file"
    echo "$1"
    exit 1
}}

# A job queued by `gym eval resubmit` starts a fresh retry count and a fresh walltime budget: a chain
# stopped at the walltime cap says "resubmit to continue". One queued by `--recover` keeps both. The variables
# are read once and unset, so the follow-up queued below (which inherits this environment) is an ordinary hop.
_gym_resubmit="${{{resubmit_env}:-}}"
_gym_keep_budget="${{{keep_budget_env}:-}}"
unset {resubmit_env} {keep_budget_env}
if [[ "$_gym_resubmit" == 1 && "$_gym_keep_budget" != 1 ]]; then
    rm -f "$_retry_file" "$_walltime_file"
fi

{bounded_budget_marker}
# Counters are data, never shell arithmetic expressions. An absent counter denotes a fresh chain.
_gym_accumulated=0
_retries=0
if [[ -e "$_walltime_file" ]]; then
    _gym_accumulated=$(cat "$_walltime_file") || _gym_stop_incomplete "Cannot read stored walltime."
fi
if [[ -e "$_retry_file" ]]; then
    _retries=$(cat "$_retry_file") || _gym_stop_incomplete "Cannot read stored retries."
fi
[[ "$_gym_accumulated" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_incomplete "Stored walltime is unknown or malformed."
[[ "$_retries" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_incomplete "Stored retries are unknown or malformed."
_gym_accumulated=$((10#$_gym_accumulated))
_retries=$((10#$_retries))
_gym_needs_retry=0

if [[ "$_prev_slurm_job_id" != "" ]]; then
    _accounting=""
    for _sacct_try in 1 2 3 4 5 6; do
        if _accounting=$(sacct -X -P -n -j "$_prev_slurm_job_id" --format=State,ElapsedRaw); then
            [[ -n "$_accounting" && "$_accounting" != "|"* ]] && break
        else
            _accounting=""
        fi
        sleep 10
    done
    [[ "$_accounting" != *$'\n'* ]] || _gym_stop_incomplete "Previous job accounting has multiple rows."
    IFS='|' read -r _prev_state _prev_elapsed _extra <<< "$_accounting"
    _prev_state=${{_prev_state%% *}}
    _prev_state=${{_prev_state%+}}
    case "$_prev_state" in
        COMPLETED)
            echo "Previous job $_prev_slurm_job_id completed successfully. Exiting."
            exit 0 ;;
        CANCELLED)
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id was cancelled. Chain stopped with rollouts missing; resubmit to continue." ;;
        TIMEOUT)
            # Scheduler allocation timeout continues the planned segments. This is not a model timeout.
            ;;
        FAILED|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
            _gym_needs_retry=1 ;;
        *)
            _gym_stop_incomplete "Previous job state is unknown or nonterminal; refusing to resume." ;;
    esac
    [[ -z "$_extra" && "$_prev_elapsed" =~ ^[0-9]{{1,15}}$ ]] || \
        _gym_stop_incomplete "Previous job elapsed accounting is unknown or malformed."
    _gym_accumulated=$((_gym_accumulated + 10#$_prev_elapsed))
    echo "$_gym_accumulated" > "$_walltime_file" || _gym_stop_incomplete "Cannot preserve elapsed walltime."
elif [[ "$_gym_resubmit" == 1 && "$_gym_keep_budget" == 1 ]]; then
    # Recovery is another allocation, not a fresh allowance. A plain manual resubmit reset above is explicit.
    _gym_needs_retry=1
fi

if [[ "$_gym_needs_retry" == 1 && $_retries -ge {max_retries} ]]; then
    _gym_stop_incomplete "Infra retry limit ({max_retries}) reached. Stopping."
fi
{max_walltime_check}
if [[ "$_gym_needs_retry" == 1 ]]; then
    _retries=$((_retries + 1))
    echo "$_retries" > "$_retry_file" || _gym_stop_incomplete "Cannot preserve the spent retry."
    echo "Infra retry $_retries/{max_retries}..."
fi

# This job continues the run, so an earlier stop no longer describes it.
rm -f "$_incomplete_file"
echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/{job_chain_name}" || _gym_stop_incomplete "Cannot record the allocation."
# A requeued job keeps its job id and runs this script again. One follow-up per job id: the marker is a
# claim on it, taken with an O_EXCL create (bash noclobber, not mkdir: uutils coreutils 0.10.0's mkdir tells
# concurrent callers that both succeeded). A claim with no follow-up pending on this job was left by a shell
# that died before sbatch answered, or by an sbatch that failed; replace it.
_guard_marker="$OUTPUT_DIR/.gym_followup_queued.$SLURM_JOB_ID"
_gym_claim_followup() {{ (set -C; : > "$_guard_marker") 2>/dev/null; }}
_gym_queue_followup=1
if ! _gym_claim_followup; then
    _gym_queue_followup=0
    # squeue's %E lists the dependencies, e.g. "afternotok:123(unfulfilled)", and %Z the job's working directory.
    # The follow-up runs in this job dir (--chdir), so a dependent job of another run does not count. %Z is the
    # --chdir path as given, while $PWD in the job can be the resolved one, so the two are compared with -ef
    # (same file), never as strings.
    # --all: without it squeue does not list jobs, ours included, in a hidden partition.
    if ! _pending=$(squeue --me --all --noheader --states=PENDING --format='%i|%E|%Z' 2>&1); then
        # Unknown is not none: a second follow-up would run the job dir twice, so keep the claim.
        echo "WARNING: squeue failed ($_pending); cannot tell whether job $SLURM_JOB_ID has a follow-up, so none is queued."
    elif _followup=$(
            while IFS='|' read -r _id _deps _cwd; do
                [[ -n "$_cwd" && "$_cwd" -ef "$OUTPUT_DIR" ]] || continue
                if grep -qE "(^|[ ,])afternotok:$SLURM_JOB_ID([(,]|$)" <<< "$_deps"; then
                    echo "$_id"
                    break
                fi
            done <<< "$_pending"
        ) && [[ -n "$_followup" ]]; then
        echo "Auto-resume follow-up for job $SLURM_JOB_ID is already queued: $_followup (requeued job). Not queueing another."
    else
        echo "Follow-up claim of job $SLURM_JOB_ID has no pending follow-up (submission interrupted or failed). Queueing one."
        rm -rf -- "$_guard_marker"
        _gym_claim_followup && _gym_queue_followup=1
    fi
fi
if [[ $_gym_queue_followup == 1 ]]; then
    # kill-on-invalid-dep: without it, clusters lacking kill_invalid_depend keep the successor pending forever.
    if _next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes "$_this_script" $SLURM_JOB_ID 2>&1); then
        _next_id=$(echo "$_next_output" | grep -oE '[0-9]+')
        if [[ -n "$_next_id" ]]; then
            echo "$_next_id" > "$_guard_marker"
            echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
        fi
    else
        # Keep the claim: sbatch can fail after the job was created (a lost reply). A requeue of this job asks
        # squeue, and queues the follow-up only if none is pending.
        echo "WARNING: Failed to submit auto-resume follow-up. The chain may not continue on failure; a requeue of this job queues one if squeue shows none pending."
    fi
fi
"""

_MAX_WALLTIME_CHECK_TEMPLATE = """\
if (( _gym_accumulated >= {seconds} )); then
    _gym_stop_incomplete "Accumulated walltime (${{_gym_accumulated}}s) reached max_walltime cap ({raw} = {seconds}s). Chain stopped at the walltime cap with rollouts missing; resubmit to continue."
fi
# Slurm rounds time limits up to minutes. Round the remaining allowance down instead, never granting extra time.
_gym_remaining_minutes=$((({seconds} - _gym_accumulated) / 60))
(( _gym_remaining_minutes > 0 )) || _gym_stop_incomplete "Less than one minute remains in the walltime budget."
_gym_allocation_limit_seconds() {{
    _gym_job_info=$(scontrol show job "$SLURM_JOB_ID" -o) || return 1
    [[ "$_gym_job_info" =~ (^|[[:space:]])TimeLimit=([^[:space:]]+) ]] || return 1
    _gym_limit=${{BASH_REMATCH[2]}}
    if [[ "$_gym_limit" == UNLIMITED ]]; then
        echo $((_gym_remaining_minutes * 60 + 1))
    elif [[ "$_gym_limit" =~ ^(([0-9]{{1,8}})-)?([0-9]{{1,8}}):([0-5][0-9]):([0-5][0-9])$ ]]; then
        _gym_days=${{BASH_REMATCH[2]:-0}}
        echo $((10#$_gym_days * 86400 + 10#${{BASH_REMATCH[3]}} * 3600 + 10#${{BASH_REMATCH[4]}} * 60 + 10#${{BASH_REMATCH[5]}}))
    else
        return 1
    fi
}}
_gym_limit_seconds=$(_gym_allocation_limit_seconds) || \
    _gym_stop_incomplete "Current allocation time limit is unknown."
if (( _gym_limit_seconds > _gym_remaining_minutes * 60 )); then
    scontrol update JobId="$SLURM_JOB_ID" TimeLimit="$_gym_remaining_minutes" || \
        _gym_stop_incomplete "Cannot enforce the remaining allocation walltime."
    _gym_limit_seconds=$(_gym_allocation_limit_seconds) || \
        _gym_stop_incomplete "Updated allocation time limit is unknown."
    (( _gym_limit_seconds <= _gym_remaining_minutes * 60 )) || \
        _gym_stop_incomplete "Updated allocation time limit exceeds the remaining budget."
fi

"""


def _walltime_to_seconds(duration: str) -> int:
    """Convert a Slurm-style duration string to seconds.

    Covers the subset of Slurm's `--time` grammar that `max_walltime` accepts:
    `MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, `D-HH:MM:SS` -- the same
    spellings Slurm itself takes, so a value can be copied straight from
    `compute.walltime`.
    """
    text = duration.strip()
    days = 0
    rest = text
    if "-" in text:
        day_str, _, rest = text.partition("-")
        if not day_str.isdigit():
            raise ValueError(f"Invalid max_walltime {duration!r}: day component must be an integer.")
        days = int(day_str)

    parts = rest.split(":")
    if not (1 <= len(parts) <= 3) or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Invalid max_walltime {duration!r}: expected Slurm duration syntax, e.g. '48:00:00' or '2-00:00:00'."
        )
    values = [int(p) for p in parts]

    if days:
        # D-HH, D-HH:MM, or D-HH:MM:SS
        values += [0] * (3 - len(values))
        hours, minutes, seconds = values
    elif len(values) == 1:
        hours, minutes, seconds = 0, values[0], 0
    elif len(values) == 2:
        hours, minutes, seconds = 0, values[0], values[1]
    else:
        hours, minutes, seconds = values

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def render_resume_prologue(resume: ResumeConfig) -> str:
    """Render the auto-resume prologue for one benchmark's sbatch script."""
    max_walltime_check = ""
    if resume.max_walltime is not None:
        seconds = _walltime_to_seconds(resume.max_walltime)
        max_walltime_check = _MAX_WALLTIME_CHECK_TEMPLATE.format(seconds=seconds, raw=resume.max_walltime)

    return _PROLOGUE_TEMPLATE.format(
        job_script_name=_JOB_SCRIPT_NAME,
        job_chain_name=JOB_CHAIN_NAME,
        retry_file_name=RETRY_FILE_NAME,
        incomplete_marker_name=INCOMPLETE_MARKER_NAME,
        resubmit_env=RESUBMIT_ENV,
        keep_budget_env=KEEP_BUDGET_ENV,
        max_retries=resume.max_retries,
        bounded_budget_marker=BOUNDED_BUDGET_MARKER,
        max_walltime_check=max_walltime_check,
    )
