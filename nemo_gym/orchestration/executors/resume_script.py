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
# Written when the chain stops before the run finished (no eligible retries, walltime cap, or cancel), so a reader of the
# job dir, not only of sacct, sees that rollouts are missing. A chain that continues removes it.
INCOMPLETE_MARKER_NAME = ".gym_chain_incomplete"
# Set to 1 in the environment of a job queued by `gym eval resubmit`.
RESUBMIT_ENV = "GYM_RESUBMIT"
# Set to 1 with RESUBMIT_ENV by `gym eval resubmit --recover`: the job resumes and keeps the spent retry count,
# walltime and per-rollout attempts, so recovery stays within the budget it read.
KEEP_BUDGET_ENV = "GYM_RESUBMIT_KEEP_BUDGET"
# Capability of the generated script, not evidence that a particular run completed.
BOUNDED_BUDGET_MARKER = "# --- Bounded resume budgets v1 ---"
# Capability: only PREEMPTED, NODE_FAIL and BOOT_FAIL admit an infra retry; FAILED, OUT_OF_MEMORY and DEADLINE stop.
SCHEDULER_CAUSE_MARKER = "# --- Scheduler-cause retry admission v1 ---"
RESUBMIT_LOCK_NAME = ".gym_resubmit.lock"
ACCOUNTING_UNKNOWN_PREFIX = "GYM_ACCOUNTING_UNKNOWN:"
ALLOCATION_ACCOUNTING_MARKER = "# --- Durable allocation accounting v1 ---"
RUNTIME_ADMISSION_MARKER = "# --- Installed-runtime continuation admission v1 ---"

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
    if ! grep -q '^{accounting_unknown_prefix}' "$_incomplete_file" 2>/dev/null; then
        echo "$1" > "$_incomplete_file"
    fi
    echo "$1"
    exit 1
}}

_gym_stop_accounting_unknown() {{
    echo "{accounting_unknown_prefix} $1" > "$_incomplete_file"
    _gym_stop_incomplete "$1"
}}

{allocation_accounting_marker}
# Every started allocation is spent history, including one that stops in a guard below.
# Keep it before any reset or accounting query; a later reader must account for this allocation too.
_gym_existing_chain=""
if [[ -e "$OUTPUT_DIR/{job_chain_name}" ]]; then
    _gym_existing_chain=$(cat "$OUTPUT_DIR/{job_chain_name}") || _gym_stop_accounting_unknown "Recorded allocation history is unreadable."
fi
[[ "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || _gym_stop_accounting_unknown "Current allocation identity is unknown."
if ! grep -qxF "$SLURM_JOB_ID" "$OUTPUT_DIR/{job_chain_name}" 2>/dev/null; then
    echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/{job_chain_name}" || _gym_stop_accounting_unknown "Cannot record the allocation."
fi
# A manually submitted worker can start before sbatch returns to its caller. The existing submission
# lock keeps it from resetting counters until the caller has either confirmed the reply or recorded uncertainty.
for _gym_submit_wait in {{1..30}}; do
    [[ -e "$OUTPUT_DIR/{resubmit_lock_name}" ]] || break
    sleep 1
done
[[ ! -e "$OUTPUT_DIR/{resubmit_lock_name}" ]] || \
    _gym_stop_accounting_unknown "Submission acknowledgement is still unresolved."
if grep -q '^{accounting_unknown_prefix}' "$_incomplete_file" 2>/dev/null; then
    _gym_stop_incomplete "Allocation accounting remains unknown; reconcile the complete queue and terminal chain before recovery."
fi
# An unresolved earlier claim can name a job whose sbatch reply was lost. Its absence from squeue
# does not prove that it never ran. The current allocation's own claim is checked below.
for _claim in "$OUTPUT_DIR"/.gym_followup_queued.*; do
    [[ -e "$_claim" ]] || continue
    [[ "$_claim" == "$OUTPUT_DIR/.gym_followup_queued.$SLURM_JOB_ID" ]] && continue
    _claimed_id=$(cat "$_claim" 2>/dev/null) || _gym_stop_accounting_unknown "A predecessor follow-up claim is unreadable."
    [[ "$_claimed_id" =~ ^[0-9]+$ ]] || _gym_stop_accounting_unknown "A predecessor follow-up submission is unobserved."
done
if [[ -n "$_gym_existing_chain" && -z "$_prev_slurm_job_id" && "${{{resubmit_env}:-}}" != 1 ]]; then
    [[ "${{_gym_existing_chain##*$'\n'}}" == "$SLURM_JOB_ID" ]] || \
        _gym_stop_accounting_unknown "A manual follower has no predecessor accounting."
fi

# A Slurm requeue restarts the same job id and may restart its allocation clock. The persisted counters
# do not identify elapsed time of that prior incarnation. Stop before resets/accounting/work instead of
# charging the predecessor twice or granting an unaccounted allocation.
_gym_restart_count="${{SLURM_RESTART_COUNT:-0}}"
[[ "$_gym_restart_count" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_accounting_unknown "Slurm restart accounting is unknown."
if (( 10#$_gym_restart_count > 0 )); then
    _gym_stop_accounting_unknown "Requeued allocation elapsed time is unknown; recover only after complete terminal accounting."
fi

# A job queued by `gym eval resubmit` starts a fresh retry count and a fresh walltime budget: a chain
# stopped at the walltime cap says "resubmit to continue". One queued by `--recover` keeps both. The variables
# are read once and unset, so the follow-up queued below (which inherits this environment) is an ordinary hop.
_gym_resubmit="${{{resubmit_env}:-}}"
_gym_keep_budget="${{{keep_budget_env}:-}}"
unset {resubmit_env} {keep_budget_env}

{bounded_budget_marker}
{scheduler_cause_marker}
# Counters are data, never shell arithmetic expressions. An absent counter denotes a fresh chain.
_gym_accumulated=0
_retries=0
if [[ -e "$_walltime_file" ]]; then
    _gym_accumulated=$(cat "$_walltime_file") || _gym_stop_accounting_unknown "Cannot read stored walltime."
fi
if [[ -e "$_retry_file" ]]; then
    _retries=$(cat "$_retry_file") || _gym_stop_accounting_unknown "Cannot read stored retries."
fi
[[ "$_gym_accumulated" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_accounting_unknown "Stored walltime is unknown or malformed."
[[ "$_retries" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_accounting_unknown "Stored retries are unknown or malformed."
_gym_accumulated=$((10#$_gym_accumulated))
_retries=$((10#$_retries))
# An explicit reset is permitted only after the old counters are readable and known.
if [[ "$_gym_resubmit" == 1 && "$_gym_keep_budget" != 1 ]]; then
    rm -f "$_retry_file" "$_walltime_file"
    _gym_accumulated=0
    _retries=0
fi
_gym_needs_retry=0
_gym_stop_reason=""

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
    [[ "$_accounting" != *$'\n'* ]] || _gym_stop_accounting_unknown "Previous job accounting has multiple rows."
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
        PREEMPTED|NODE_FAIL|BOOT_FAIL)
            # The scheduler names the infrastructure cause: the node was taken, died, or never booted.
            _gym_needs_retry=1 ;;
        FAILED|OUT_OF_MEMORY|DEADLINE)
            # These states do not establish a retryable infrastructure cause. A FAILED exit code can come
            # from invalid configuration, an OOM may be workload-dependent, and a deadline has passed.
            # Stop after the elapsed time is counted below; a typed recovery goes through "gym eval resubmit --recover".
            _gym_stop_reason="Previous job $_prev_slurm_job_id ended $_prev_state without an infrastructure cause. Chain stopped with rollouts missing; it does not retry." ;;
        *)
            _gym_stop_accounting_unknown "Previous job state is unknown or nonterminal; refusing to resume." ;;
    esac
    [[ -z "$_extra" && "$_prev_elapsed" =~ ^[0-9]{{1,15}}$ ]] || \
        _gym_stop_accounting_unknown "Previous job elapsed accounting is unknown or malformed."
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
if [[ -n "$_gym_stop_reason" || ( -n "$_prev_slurm_job_id" && -e "$OUTPUT_DIR/artifacts/rollouts_manifest.json" ) ]]; then
    if [[ "$_prev_state" == FAILED || "$_prev_state" == TIMEOUT || "$_prev_state" == PREEMPTED || "$_prev_state" == NODE_FAIL || "$_prev_state" == BOOT_FAIL ]]; then
        _gym_exit_code=$(sacct -X -P -n -j "$_prev_slurm_job_id" --format=ExitCode) || \
            _gym_stop_accounting_unknown "Previous exit accounting is unknown."
        [[ "$_gym_exit_code" =~ ^[0-9]+:[0-9]+$ ]] || _gym_stop_accounting_unknown "Previous exit accounting is malformed."
        {runtime_admission_marker}
        _gym_admission_output=$({admission_command}) || _gym_stop_incomplete "Installed recovery runtime failed."
        _gym_admission=$(printf '%s\n' "$_gym_admission_output" | grep '^__GYM_CONTINUATION:' || true)
        case "$_gym_admission" in
            __GYM_CONTINUATION:segment) _gym_needs_retry=0 ;;
            __GYM_CONTINUATION:retry) _gym_needs_retry=1 ;;
            *) _gym_stop_incomplete "Saved recovery evidence has no eligible retries; $_gym_stop_reason" ;;
        esac
    else
        _gym_stop_incomplete "$_gym_stop_reason"
    fi
fi
# Admission can identify a retry only after reading the authoritative saved outcomes.
if [[ "$_gym_needs_retry" == 1 && $_retries -ge {max_retries} ]]; then
    _gym_stop_incomplete "Infra retry limit ({max_retries}) reached. Stopping."
fi
if [[ "$_gym_needs_retry" == 1 ]]; then
    _retries=$((_retries + 1))
    echo "$_retries" > "$_retry_file" || _gym_stop_incomplete "Cannot preserve the spent retry."
    echo "Infra retry $_retries/{max_retries}..."
fi

# This job continues the run, so an earlier stop no longer describes it.
rm -f "$_incomplete_file"
# A requeued job keeps its job id and runs this script again. One follow-up per job id: the marker is a
# claim on it, taken with an O_EXCL create (bash noclobber, not mkdir: uutils coreutils 0.10.0's mkdir tells
# concurrent callers that both succeeded). A claim with no follow-up pending on this job was left by a shell
# that died before sbatch answered, or by an sbatch that failed. Absence from the queue is not proof it never ran.
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
        _gym_stop_accounting_unknown "squeue failed; existing follow-up accounting is unknown."
    elif _followup=$(
            while IFS='|' read -r _id _deps _cwd; do
                [[ -n "$_cwd" && "$_cwd" -ef "$OUTPUT_DIR" ]] || continue
                if grep -qE "(^|[ ,])afternotok:$SLURM_JOB_ID([(,]|$)" <<< "$_deps"; then
                    echo "$_id"
                fi
            done <<< "$_pending"
        ) && [[ "$_followup" =~ ^[0-9]+$ ]]; then
        echo "$_followup" > "$_guard_marker" || _gym_stop_accounting_unknown "Cannot preserve the observed follow-up."
        echo "Auto-resume follow-up for job $SLURM_JOB_ID is already queued: $_followup (requeued job). Not queueing another."
    else
        _gym_stop_accounting_unknown "The existing follow-up claim has no unique observed pending job; it may already have run."
    fi
fi
if [[ $_gym_queue_followup == 1 ]]; then
    # kill-on-invalid-dep: without it, clusters lacking kill_invalid_depend keep the successor pending forever.
    if _next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes "$_this_script" $SLURM_JOB_ID 2>&1); then
        _next_id="${{_next_output##* }}"
        if [[ "$_next_id" =~ ^[0-9]+$ ]]; then
            echo "$_next_id" > "$_guard_marker" || _gym_stop_accounting_unknown "Cannot preserve the queued follow-up identity."
            echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
        else
            echo "{accounting_unknown_prefix} Follow-up submission returned no unique job identity." > "$_incomplete_file"
        fi
    else
        # The current allocation can finish, but a successor must not erase this uncertainty or reset budgets.
        echo "{accounting_unknown_prefix} Follow-up submission may have succeeded without an observed reply." > "$_incomplete_file"
        echo "WARNING: Failed to submit auto-resume follow-up. Inspect the complete queue and terminal accounting; automatic continuation is stopped."
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


def render_resume_prologue(resume: ResumeConfig, *, admission_command: str = "false") -> str:
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
        scheduler_cause_marker=SCHEDULER_CAUSE_MARKER,
        max_walltime_check=max_walltime_check,
        resubmit_lock_name=RESUBMIT_LOCK_NAME,
        accounting_unknown_prefix=ACCOUNTING_UNKNOWN_PREFIX,
        allocation_accounting_marker=ALLOCATION_ACCOUNTING_MARKER,
        runtime_admission_marker=(
            RUNTIME_ADMISSION_MARKER if admission_command != "false" else "# No installed-runtime continuation reader"
        ),
        admission_command=admission_command,
    )
