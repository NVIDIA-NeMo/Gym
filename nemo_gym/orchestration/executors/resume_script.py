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
`afternotok` dependency before doing any real work, and a sentinel that
continues the chain if that successor dies before its prologue. A job that
COMPLETEs leaves its queued successor with a dependency that can never be
satisfied, so Slurm would cancel it on its own, but the sentinel would then
start on a full allocation only to exit. So the epilogue, rendered after the
driver step, cancels both when the job finished the run.
"""

import logging
import os
from pathlib import Path

from nemo_gym.core.failure_kinds import (
    COLLECTION_RESUME_REFUSED,
    DRIVER_INSTALL_FAILED,
    ENVIRONMENT_PREPARE_FAILED,
    ENVIRONMENT_PROTOCOL_VIOLATION,
    PROVIDER_CREDENTIALS_REJECTED,
    PROVIDER_QUOTA_EXHAUSTED,
    SERVER_SPINUP_TIMEOUT,
    SERVER_START_FAILED,
)
from nemo_gym.orchestration.api import ResumeConfig
from nemo_gym.orchestration.executors.script_templates import DRIVER_FAILURE_ENV


logger = logging.getLogger(__name__)


# `_stage` in slurm.py always writes this benchmark's script to exactly this
# filename inside its job dir, which `#SBATCH --chdir` makes `$PWD` at runtime.
_JOB_SCRIPT_NAME = "job.sh"
# The first line of the prologue, so a resubmit can tell a resumable job.sh from one without the chain.
PROLOGUE_MARKER = "# --- Auto-resume chain ---"
JOB_CHAIN_NAME = ".gym_job_chain"
RETRY_FILE_NAME = ".gym_infra_retries"
# Written by a segment that restarted the run fresh after a server failed to start before any rollout was saved,
# named by that segment's job id: the next segment restarts it no more, so one such restart is the bound.
FRESH_RESTART_MARKER_NAME = ".gym_fresh_restart"
# Written only by `gym eval resubmit --recover --extend-retries`: infra retries added to the configured cap.
RETRY_EXTENSION_NAME = ".gym_infra_retry_extension"
# Capability of the generated script: it reads the recorded extension.
RETRY_EXTENSION_MARKER = "# --- Recorded retry extension v1 ---"
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
# The line prefix the installed continuation reader prints its decision reason with.
CONTINUATION_REASON_PREFIX = "__GYM_CONTINUATION_REASON:"
# The unknown-accounting reason a requeued allocation writes; recovery reconciles it from complete terminal accounting.
REQUEUE_UNKNOWN_REASON = (
    "Requeued allocation elapsed time is unknown; recover only after complete terminal accounting."
)
# What a reconciled recovery writes over that marker before it queues the run again.
REQUEUE_RECONCILED_NOTE = (
    "Requeued allocation accounting was reconciled from terminal scheduler records; recovery resubmitted."
)
# How a sentinel waits for every incarnation of a requeued chain to have a terminal scheduler record before it
# reconciles the chain: it polls every REQUEUE_SETTLE_POLL_S seconds for up to REQUEUE_SETTLE_BOUND_S seconds. The
# sentinel is the job script, so it holds the run's allocation while it waits; the bound keeps that short.
REQUEUE_SETTLE_POLL_S = 30
REQUEUE_SETTLE_BOUND_S = 300
# The sacct fields of every incarnation of every allocation of a chain (see recovery.chain_accounting).
CHAIN_ACCOUNTING_FORMAT = "JobIDRaw,State,ElapsedRaw,Restarts"
# Appended in the job dir: one JSON row per allocation incarnation of the run's jobs (chain segments, follow-ups and
# sentinels), from sacct, for GPU-hour accounting. Each job records every incarnation that has ended, and a job that
# ends without a successor to record it (a stop, a complete run, an idle sentinel) also records its own as of its exit
# (``"ended": false``).
JOB_ALLOCATIONS_NAME = "job-allocations.jsonl"
# Read with SLURM_TIME_FORMAT=%s, so Start and End are Unix seconds.
JOB_ALLOCATIONS_FORMAT = "JobIDRaw,State,Start,End,ElapsedRaw,NNodes,NodeList,AllocTRES,Restarts"
# Recording is best-effort: its sacct queries and the waits between them end within this many seconds.
JOB_ALLOCATIONS_BUDGET_S = 30
JOB_ALLOCATIONS_RETRY_S = 5
# Held only while a job reads, deduplicates and appends rows (never during sacct); older than a minute it is stale.
JOB_ALLOCATIONS_LOCK_NAME = ".gym_job_allocations.lock"
# The previous state a sentinel gives the installed continuation reader to reconcile a requeued chain, with the
# chain's records in CHAIN_ACCOUNTING_ENV; the reader prints the reconciled walltime after ACCOUNTED_WALLTIME_PREFIX.
REQUEUE_RECONCILE_STATE = "REQUEUED"
CHAIN_ACCOUNTING_ENV = "GYM_CHAIN_ACCOUNTING"
ACCOUNTED_WALLTIME_PREFIX = "__GYM_ACCOUNTED_WALLTIME:"
ALLOCATION_ACCOUNTING_MARKER = "# --- Durable allocation accounting v1 ---"
RUNTIME_ADMISSION_MARKER = "# --- Installed-runtime continuation admission v1 ---"
# The first words of the line a continuation writes to the incomplete marker as soon as it records itself, before
# it has queued its own follow-up. It stays only when the job ended in between, before its guard.
PROVISIONAL_MARKER_PREFIX = "Continuation "
# The words of the line a sentinel writes when its follow-up was cancelled before it started and the parent did
# not finish the run: the chain stopped there, and the line says why.
CANCELLED_FOLLOWUP_MARKER = "was cancelled before it started"
# The scheduler answered the follow-up's sbatch with a refusal (a QOS submit limit, an invalid account): nothing was
# queued, and the accounting is known. The chain stops; `gym eval resubmit --recover` continues it.
FOLLOW_UP_SUBMISSION_REFUSED = "follow_up_submission_refused"
# Why the chain ends after a segment that timed out before Gym wrote the task inventory.
PRE_INVENTORY_REASON = "the segment ended before the task inventory was written; nothing can be resumed"
# Where a resumed segment records how many progress rows the Store held when it started (one file per job id).
ROWS_AT_START_NAME = ".gym_rows_at_start"
# Why the chain ends after a resumed segment that timed out without saving a rollout: its walltime went to setup.
NO_PROGRESS_SEGMENT_REASON = (
    "the segment saved no rollout: its walltime is shorter than this run's setup (Gym install and server start-up)"
    " plus a rollout, so the next segment would repeat it; the chain does not retry it; raise the walltime"
)
# Where the driver step's DRIVER_FAILURE_ENV points: the next segment's guard ends the chain with the kind in it.
DRIVER_FAILURE_NAME = ".gym_driver_failure"
# The driver failure kinds the next segment's prologue turns into its own stop and remedy (its guard). A segment whose
# driver exits 76 with one of them keeps its follow-up for that stop; the guard handles each (see the prologue's case).
GUARDED_DRIVER_FAILURE_KINDS = (
    SERVER_SPINUP_TIMEOUT,
    DRIVER_INSTALL_FAILED,
    COLLECTION_RESUME_REFUSED,
    PROVIDER_CREDENTIALS_REJECTED,
    ENVIRONMENT_PREPARE_FAILED,
    PROVIDER_QUOTA_EXHAUSTED,
    ENVIRONMENT_PROTOCOL_VIOLATION,
)
SPINUP_TIMEOUT_REASON = (
    f"{SERVER_SPINUP_TIMEOUT}: the Gym servers did not become ready within server_spinup_timeout_seconds; "
    "the chain does not retry it; raise the limit with gym eval resubmit --recover --server-spinup-timeout-seconds N --spinup-timeout-reason TEXT"
)
CREDENTIALS_REJECTED_REASON = (
    f"{PROVIDER_CREDENTIALS_REJECTED}: the provider refused the credentials; the chain does not retry it; "
    "fix the credentials, then continue with gym eval resubmit --recover"
)
PREPARE_FAILED_REASON = (
    f"{ENVIRONMENT_PREPARE_FAILED}: EVAL STOPPED: data preparation failed; the chain does not retry it; "
    "fix the benchmark's preparation, then continue with gym eval resubmit --recover"
)
QUOTA_EXHAUSTED_REASON = (
    f"{PROVIDER_QUOTA_EXHAUSTED}: the grading provider's quota is exhausted; the chain does not retry it; "
    "after the reset continue with gym eval resubmit --recover (re-judges the saved answers, no model calls)"
)
PROTOCOL_VIOLATION_REASON = (
    f"{ENVIRONMENT_PROTOCOL_VIOLATION}: an environment server answered with a reply the collector cannot use; "
    "the chain does not retry it; fix the server or the benchmark pairing, then gym eval submit again"
)
RESUME_REFUSED_REASON = (
    f"{COLLECTION_RESUME_REFUSED}: the saved rollouts do not verify against this run's inputs and configuration; "
    "the chain does not retry it; start a fresh run or set allow_unsafe_resume=true"
)
# The installer's first error line (else its last line) is cut to this many characters.
INSTALL_ERROR_MAX_CHARS = 200
# The typed line that ends the chain when the installed runtime has no recovery reader (``nemo_gym.orchestration.recovery``
# or its ``continuation_admission``): the runtime predates the recovery module, and retrying installs the same one.
RECOVERY_RUNTIME_OUTDATED = "recovery_runtime_outdated"
RECOVERY_RUNTIME_OUTDATED_REASON = (
    f"{RECOVERY_RUNTIME_OUTDATED}: the Gym runtime this run installs predates nemo_gym.orchestration.recovery, which "
    "decides whether a chain may continue; the chain does not retry it; point the run's Gym runtime at a version "
    "that contains that module, then continue with gym eval resubmit --recover"
)
# The typed line that ends the chain of a complete run whose required MLflow export has no current receipt.
EXPORT_RECEIPT_MISSING = "run_complete_export_receipt_missing"
EXPORT_RECEIPT_MISSING_REASON = (
    f"{EXPORT_RECEIPT_MISSING}: the run is complete, but its MLflow export receipt "
    "(artifacts/rollouts_mlflow_export_done) for the saved artifacts is missing; the chain does not export the run; "
    "continue with gym eval resubmit --recover"
)
# Appended in the job dir by a job whose driver step finished the run: how the chain ended there and the sentinel and
# follow-up it had queued and then cancelled, each with its job id and whether scancel took it.
CHAIN_END_JOURNAL_NAME = "chain-end-journal.jsonl"
# The driver's exit when nothing is eligible for another attempt (gym eval run's stop exit code), and what a job
# that exits with it writes and journals when it ends its own chain.
NO_ELIGIBLE_RETRY_EXIT = 76
NO_ELIGIBLE_RETRY = "no_eligible_retry"
NO_ELIGIBLE_RETRY_REASON = (
    "No tasks are eligible for retry: the job ended its chain itself, with nothing queued after it; inspect "
    "terminal failures, omissions and exhausted budgets, then continue with gym eval resubmit --recover."
)
# The files an export receipt binds (`_artifact_version` in the exporters), by receipt key, in the job's artifacts.
_RECEIPT_ARTIFACTS = (
    ("rollouts_jsonl_fpath", "rollouts.jsonl"),
    ("resolved_config_yaml_fpath", "rollouts_resolved_config.yaml"),
    ("aggregate_metrics_json_fpath", "rollouts_aggregate_metrics.json"),
    ("failures_jsonl_fpath", "rollouts_failures.jsonl"),
    ("materialized_inputs_jsonl_fpath", "rollouts_materialized_inputs.jsonl"),
)

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
    # The run's standing in one line: how many rollouts have no final result, from the Store's coverage report
    # (its ``open`` count), so the stop never reads as a finished run.
    local _open _expected _report="$OUTPUT_DIR/artifacts/rollouts_coverage.json"
    _open=$(_gym_report_value open "$_report")
    _expected=$(_gym_report_value expected "$_report")
    if [[ "$_open" =~ ^[0-9]+$ && "$_expected" =~ ^[0-9]+$ ]]; then
        echo "Run incomplete: $_open of $_expected rollouts open (no final result); see $_report."
    else
        echo "Run incomplete: the open rollout count is unknown (no readable coverage report at $_report)."
    fi
    _gym_record_own_allocation
    exit 1
}}

# The marker keeps the reason alone (readers match it); the log line also names the operator's next step.
_gym_stop_accounting_unknown() {{
    echo "{accounting_unknown_prefix} $1" > "$_incomplete_file"
    _gym_stop_incomplete "$1 Reconcile the queue and the job chain, then continue with gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT, adding --extend-retries N when the retries are spent."
}}

# A run that is complete and exported needs no further segment, whatever the scheduler says about the last one: a
# segment can finish its last rollout and its export seconds before its walltime and still end TIMEOUT.
_gym_artifacts="$OUTPUT_DIR/artifacts"
# A top-level scalar of a Store report: atomic_write_json writes two-space indented JSON, so a nested key never matches.
_gym_report_value() {{
    grep -m 1 '^  "'"$1"'": ' "$2" 2>/dev/null | cut -d: -f2- | tr -d ' ,"'
}}
# Collection's coverage report says complete only when every rollout of the inventory has a saved result or a settled
# final zero. The Store writes it from the rows it has already appended, removes it when a run starts fresh and rewrites
# it whenever it opens, so it must name this run (the manifest's run_id) and its inventory (one line per rollout).
_gym_run_complete() {{
    local _report="$_gym_artifacts/rollouts_coverage.json" _expected _inventory _run
    [[ "$(_gym_report_value complete "$_report")" == true ]] || return 1
    _expected=$(_gym_report_value expected "$_report")
    [[ "$_expected" =~ ^[0-9]{{1,15}}$ ]] || return 1
    _inventory=$(wc -l < "$_gym_artifacts/rollouts_materialized_inputs.jsonl" 2>/dev/null) || return 1
    [[ "$_inventory" =~ ^[0-9]{{1,15}}$ ]] && (( 10#$_expected == 10#$_inventory )) || return 1
    _run=$(_gym_report_value run_id "$_report")
    [[ "$_run" =~ ^[0-9a-f]{{32}}$ && "$_run" == "$(_gym_report_value run_id "$_gym_artifacts/rollouts_manifest.json")" ]] || return 1
    _gym_complete_rollouts=$((10#$_expected))
}}
# The saved config asks for an MLflow export unless it sets mlflow_tracking_required to false or leaves one of the three
# MLflow names an exporter needs unset. An unreadable config, or an interpolated value, still asks for one.
_gym_export_required() {{
    local _config="$_gym_artifacts/rollouts_resolved_config.yaml" _key _value
    [[ -r "$_config" ]] || return 0
    if grep -qxE 'mlflow_tracking_required:[[:space:]]*false' "$_config" 2>/dev/null; then
        _gym_export_note="it exports nothing by design (mlflow_tracking_required is false)"
        return 1
    fi
    for _key in mlflow_tracking_uri mlflow_experiment_name mlflow_run_name; do
        _value=$(grep -m 1 "^$_key:" "$_config" 2>/dev/null)
        _value=${{_value#*:}}
        _value=${{_value//[[:space:]]/}}
        case "$_value" in
            ''|null|'~'|"''"|'""')
                _gym_export_note="it configures no MLflow export ($_key is unset)"
                return 1 ;;
        esac
    done
    return 0
}}
# The export receipt is current only when every file it binds still has the digest it records.
_gym_receipt_current() {{
    local _receipt _pair _re _have
    _receipt=$(cat "$_gym_artifacts/rollouts_mlflow_export_done" 2>/dev/null) || return 1
    for _pair in {receipt_artifacts}; do
        _re='"'"${{_pair%%:*}}"'":"([0-9a-f]{{64}})"'
        [[ "$_receipt" =~ $_re ]] || return 1
        _have=$(sha256sum < "$_gym_artifacts/${{_pair#*:}}" 2>/dev/null) || return 1
        [[ "${{_have%% *}}" == "${{BASH_REMATCH[1]}}" ]] || return 1
    done
    _gym_export_note="its MLflow export receipt is current"
}}
# Ends the chain of a complete run before any retry or budget decision. A deliberate stop already in the marker says
# why the chain ended and stays; only a missing marker or a provisional line gives way to the run's completion.
_gym_end_if_complete() {{
    [[ ! -s "$_incomplete_file" ]] || grep -q '^{provisional_marker_prefix}' "$_incomplete_file" 2>/dev/null || return 0
    _gym_run_complete || return 0
    if _gym_export_required && ! _gym_receipt_current; then
        _gym_stop_incomplete "{export_receipt_missing_reason}"
    fi
    rm -f "$_incomplete_file"
    echo "Run complete: the Store covers all $_gym_complete_rollouts expected rollouts and $_gym_export_note; job ${{SLURM_JOB_ID:-}} ends the chain with nothing to recover."
    _gym_record_own_allocation
    exit 0
}}

# A Slurm requeue stops the requeued allocation and then its follow-up on the requeue marker: inside the job, the
# elapsed time of the earlier incarnation is unknown. The follow-up's sentinel starts after both have ended. Once every
# incarnation of the chain has a terminal record, it reconciles the chain's walltime from them through the installed
# continuation reader, as gym eval resubmit --recover does, and when the budget allows it queues the recovery the same
# way: under the resubmit lock, with the reconciled note written over the marker before sbatch, so no later sentinel
# or operator queues it again. Otherwise the chain stays stopped on the marker, exactly as before. The sentinel runs no
# work, so it stays out of the chain.
_gym_chain_settled() {{
    awk -F'|' -v chain="$1" '
        BEGIN {{ n = split(chain, ids, ","); for (i = 1; i <= n; i++) want[ids[i]] = 1 }}
        NF != 4 || !($1 in want) || $3 !~ /^[0-9]+$/ || $4 !~ /^[0-9]+$/ {{ bad = 1; next }}
        {{
            split($2, words, " "); state = words[1]; sub(/[+]$/, "", state)
            # REQUEUED is what sacct keeps for an incarnation that a later one followed: it has ended. It is terminal
            # only then (checked below); the last incarnation still REQUEUED waits to run again.
            if (state == "REQUEUED") requeued_at[$1, $4 + 0] = 1
            else if (state !~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED)$/) bad = 1
            rows[$1]++; seen[$1, $4 + 0]++
            if (($4 + 0) > last[$1]) last[$1] = $4 + 0
        }}
        END {{
            if (bad) exit 1
            for (key in requeued_at) {{
                split(key, part, SUBSEP)
                if (part[2] + 0 >= last[part[1]]) exit 1
            }}
            for (id in want) {{
                if (!(id in rows)) exit 1
                for (r = 0; r < rows[id]; r++) if (seen[id, r] != 1) exit 1
                if (rows[id] > 1) requeued = 1
            }}
            exit requeued ? 0 : 1
        }}' <<< "$2"
}}
_gym_requeue_stop() {{
    echo "Requeued chain: $1; sentinel ${{SLURM_JOB_ID:-}} leaves the chain stopped. Continue with gym eval resubmit --recover."
    _gym_record_own_allocation
    exit 0
}}
_gym_requeue_unlock() {{
    rm -f "$OUTPUT_DIR/{resubmit_lock_name}"
    _gym_requeue_stop "$1"
}}
_gym_reconcile_requeue() {{
    local _ids _waited=0 _output _total _reason _old _pending _id _deps _cwd _claim _claimed _out _queued
    _ids=$(paste -sd, "$OUTPUT_DIR/{job_chain_name}" 2>/dev/null) || _ids=""
    [[ "$_ids" =~ ^[0-9]+(,[0-9]+)*$ ]] || _gym_requeue_stop "the recorded chain is unreadable"
    _gym_requeue_poll() {{
        _gym_requeue_rows=$(sacct -D -n -P -X -j "$_ids" --format={chain_accounting_format} 2>/dev/null) || _gym_requeue_rows=""
    }}
    _gym_requeue_poll
    until _gym_chain_settled "$_ids" "$_gym_requeue_rows"; do
        (( _waited < {requeue_settle_bound_s} )) || \
            _gym_requeue_stop "its terminal scheduler records did not settle within {requeue_settle_bound_s}s"
        sleep {requeue_settle_poll_s}
        _waited=$((_waited + {requeue_settle_poll_s}))
        _gym_requeue_poll
    done
    # The installed reader decides as a recovery would, on the saved rows and the reconciled budget.
    _prev_state={requeue_reconcile_state}
    _gym_exit_code=""
    export {chain_accounting_env}="$_gym_requeue_rows"
    _output=$({admission_command}) || _output=""
    unset {chain_accounting_env}
    _reason=$(printf '%s\n' "$_output" | grep -m 1 '^{continuation_reason_prefix}' || true)
    _reason=${{_reason#{continuation_reason_prefix}}}
    _total=$(printf '%s\n' "$_output" | grep -m 1 '^{accounted_walltime_prefix}' || true)
    _total=${{_total#{accounted_walltime_prefix}}}
    if ! printf '%s\n' "$_output" | grep -qx '__GYM_CONTINUATION:retry' || [[ ! "$_total" =~ ^[0-9]{{1,15}}$ ]]; then
        _gym_requeue_stop "${{_reason:-the installed reader did not reconcile it}}"
    fi
    echo "Requeued chain reconciled: $_reason"
    # The resubmit lock, as gym eval resubmit takes it; one that is held belongs to someone else and stays.
    (set -C; : > "$OUTPUT_DIR/{resubmit_lock_name}") 2>/dev/null || _gym_requeue_stop "another resubmit holds the lock"
    grep -qxF -- "{requeue_unknown_marker}" "$_incomplete_file" 2>/dev/null || \
        _gym_requeue_unlock "the requeue marker changed: another recovery already handled it"
    _pending=$(squeue --me --all --noheader --format='%i|%E|%Z' 2>/dev/null) || _gym_requeue_unlock "squeue failed"
    while IFS='|' read -r _id _deps _cwd; do
        [[ -n "$_cwd" && "$_cwd" -ef "$OUTPUT_DIR" && "$_id" != "${{SLURM_JOB_ID:-}}" ]] || continue
        _gym_requeue_unlock "job $_id of this run is still pending or running"
    done <<< "$_pending"
    for _claim in "$OUTPUT_DIR"/.gym_followup_queued.*; do
        [[ -e "$_claim" ]] || continue
        _claimed=$(cat "$_claim" 2>/dev/null) || _claimed=""
        [[ "$_claimed" =~ ^[0-9]+$ ]] || _gym_requeue_unlock "a follow-up submission identity is unknown"
    done
    # Terminal accounting of the complete recorded chain, never less than what is stored.
    _old=0
    if [[ -e "$_walltime_file" ]]; then _old=$(cat "$_walltime_file") || _gym_requeue_unlock "the stored walltime is unreadable"; fi
    [[ "$_old" =~ ^[0-9]{{1,15}}$ ]] || _gym_requeue_unlock "the stored walltime is malformed"
    (( 10#$_total >= 10#$_old )) || _total=$((10#$_old))
    {{ echo "$((10#$_total))" > "$_walltime_file.tmp" && mv -f "$_walltime_file.tmp" "$_walltime_file"; }} || \
        _gym_requeue_unlock "cannot preserve the accounted walltime"
    echo "{requeue_reconciled_note}" > "$_incomplete_file" || _gym_requeue_unlock "cannot record the reconciled accounting"
    _queued=""
    if _out=$({resubmit_env}=1 {keep_budget_env}=1 sbatch "$_this_script" 2>&1); then
        _queued="${{_out##* }}"
    fi
    if [[ "$_queued" =~ ^[0-9]+$ ]]; then
        echo "Requeued chain: sentinel ${{SLURM_JOB_ID:-}} queued recovery $_queued; the walltime spent is ${{_total}}s."
    else
        echo "{accounting_unknown_prefix} Resubmit acknowledgement was not confirmed." > "$_incomplete_file"
        echo "WARNING: the recovery submission may have succeeded without an observed reply: $_out"
    fi
    rm -f "$OUTPUT_DIR/{resubmit_lock_name}"
    _gym_record_own_allocation
    exit 0
}}

# Every job of the run the job dir names (the chain, each claimed follow-up and each queued sentinel), and $1.
_gym_run_job_ids() {{
    {{ cat "$OUTPUT_DIR/{job_chain_name}" "$OUTPUT_DIR"/.gym_followup_queued.* "$OUTPUT_DIR"/.gym_followup_sentinel.* 2>/dev/null
       echo "${{1:-}}"; }} | grep -xE '[0-9]{{1,15}}' | sort -u | paste -sd, -
}}
# A command cut off after $1 seconds, where coreutils timeout exists.
_gym_bounded() {{
    if command -v timeout >/dev/null 2>&1; then timeout "$@"; else shift; "$@"; fi
}}
# Appends to {job_allocations_name} one row per allocation incarnation (job id, restarts) of those jobs that has ended
# and has no row yet; with $1, also the running incarnation of that job, once (ended false). Best-effort: the queries
# end within the budget, and accounting that does not answer records nothing (a later job records what is missing).
# The read, deduplication and append hold a lock, so jobs that run at once never write the same row twice.
_gym_record_allocations() {{
    local _ids _rows="" _new _left _sacct_try _try _deadline=$((SECONDS + {job_allocations_budget_s}))
    local _file="$OUTPUT_DIR/{job_allocations_name}" _lock="$OUTPUT_DIR/{job_allocations_lock_name}"
    _ids=$(_gym_run_job_ids "${{1:-}}")
    [[ -n "$_ids" ]] || return 0
    for _sacct_try in 1 2 3 4 5 6; do
        _left=$((_deadline - SECONDS))
        (( _left > 0 )) || break
        _rows=$(SLURM_TIME_FORMAT=%s _gym_bounded "$_left" sacct -D -X -P -n -j "$_ids" --format={job_allocations_format} 2>/dev/null) && \
            [[ -n "$_rows" ]] && break
        _rows=""
        (( _deadline - SECONDS > {job_allocations_retry_s} )) || break
        sleep {job_allocations_retry_s}
    done
    if [[ -z "$_rows" ]]; then
        echo "WARNING: scheduler accounting of jobs $_ids is unavailable; {job_allocations_name} records none of them now."
        return 0
    fi
    # Up to about 15 s; the jitter keeps writers that wait together from waking together.
    for _try in {{1..50}}; do
        find "$_lock" -mmin +1 -exec rm -f {{}} + 2>/dev/null
        (set -C; : > "$_lock") 2>/dev/null && break
        _try=""
        sleep "0.$((RANDOM % 5 + 1))"
    done
    if [[ -z "$_try" ]]; then
        echo "WARNING: {job_allocations_lock_name} stayed held; {job_allocations_name} records none of jobs $_ids now."
        return 0
    fi
    _new=$(awk -F'|' -v want="$_ids" -v self="${{1:-}}" -v file="$_file" -v q='"' \
        -v at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
        function text(s) {{ gsub("[^][A-Za-z0-9_.,=:/+()* -]", "", s); return q s q }}
        function number(s) {{ return s ~ /^[0-9]+$/ ? s + 0 : "null" }}
        BEGIN {{
            n = split(want, ids, ","); for (i = 1; i <= n; i++) wanted[ids[i]] = 1
            while ((getline line < file) > 0) {{
                if (!match(line, q "job_id" q ":" q "[0-9]+" q)) continue
                id = substr(line, RSTART + 10, RLENGTH - 11)
                if (!match(line, q "restarts" q ":[0-9]+")) continue
                seen[id, substr(line, RSTART + 11) + 0, index(line, q "ended" q ":true") > 0] = 1
            }}
        }}
        NF != 9 || !($1 in wanted) || $5 !~ /^[0-9]+$/ || $6 !~ /^[0-9]+$/ || $9 !~ /^[0-9]+$/ {{ next }}
        {{
            ended = $4 ~ /^[0-9]+$/
            if ((!ended && $1 != self) || seen[$1, $9 + 0, ended]) next
            seen[$1, $9 + 0, ended] = 1
            split($2, words, " "); state = words[1]; sub(/[+]$/, "", state)
            gpus = "null"; k = split($8, tres, ",")
            for (i = 1; i <= k; i++) if (tres[i] ~ "^gres/gpu=[0-9]+$") gpus = substr(tres[i], 10) + 0
            print "{{" q "alloc_tres" q ":" text($8) "," q "elapsed_s" q ":" ($5 + 0) "," q "end_epoch_s" q ":" number($4) \
                "," q "ended" q ":" (ended ? "true" : "false") "," q "gpus" q ":" gpus "," q "job_id" q ":" q $1 q \
                "," q "node_list" q ":" text($7) "," q "nodes" q ":" ($6 + 0) "," q "recorded_at" q ":" q at q \
                "," q "restarts" q ":" ($9 + 0) "," q "source" q ":" q "sacct" q "," q "start_epoch_s" q ":" number($3) \
                "," q "state" q ":" text(state) "}}"
        }}' <<< "$_rows")
    [[ -z "$_new" ]] || printf '%s\\n' "$_new" >> "$_file" || echo "WARNING: cannot append to {job_allocations_name}."
    rm -f "$_lock"
    return 0
}}
# Before a job ends with nothing after it to record it: its own allocation as of now, once.
_gym_record_own_allocation() {{
    [[ -z "${{_gym_own_allocation_recorded:-}}" && "${{SLURM_JOB_ID:-}}" =~ ^[0-9]+$ ]] || return 0
    _gym_own_allocation_recorded=1
    _gym_record_allocations "$SLURM_JOB_ID"
}}

# The previous job queues a sentinel next to its follow-up F (afterany:F). F records itself in the chain before
# anything else, so a sentinel only acts when F started and ended unrecorded: F died before its prologue ran.
# The variable is read once and unset, so the follow-up and sentinel queued below are ordinary hops.
_gym_sentinel_for="${{GYM_FOLLOWUP_SENTINEL:-}}"
unset GYM_FOLLOWUP_SENTINEL
_gym_sentinel_takeover=""
_gym_sentinel_debit=0
_gym_sentinel_recorded=""
if [[ -n "$_gym_sentinel_for" ]]; then
    if grep -qxF -- "$_gym_sentinel_for" "$OUTPUT_DIR/{job_chain_name}" 2>/dev/null; then
        # The follow-up recorded itself and stopped on a requeue marker: reconcile the requeued chain (see above).
        if grep -qxF -- "{requeue_unknown_marker}" "$_incomplete_file" 2>/dev/null; then
            _gym_reconcile_requeue
        fi
        # A follow-up that recorded itself died before its guard only when it made no follow-up claim and its own
        # provisional line is still the marker: a deliberate stop replaced that line with its reason.
        if [[ -e "$OUTPUT_DIR/.gym_followup_queued.$_gym_sentinel_for" ]] || \
            ! grep -qF -- "{provisional_marker_prefix}$_gym_sentinel_for started" "$_incomplete_file" 2>/dev/null; then
            echo "Follow-up $_gym_sentinel_for recorded itself in the chain; sentinel ${{SLURM_JOB_ID:-}} has nothing to do."
            _gym_record_own_allocation
            exit 0
        fi
        _gym_sentinel_recorded=1
    fi
    # The follow-up did not get to ask whether the run is complete: ask before taking its place.
    _gym_end_if_complete
    _gym_sentinel_elapsed=""
    for _sacct_try in 1 2 3 4 5 6; do
        _gym_sentinel_elapsed=$(sacct -X -P -n -j "$_gym_sentinel_for" --format=ElapsedRaw 2>/dev/null) || _gym_sentinel_elapsed=""
        [[ "$_gym_sentinel_elapsed" =~ ^[0-9]{{1,15}}$ ]] && break
        sleep 10
    done
    [[ "$_gym_sentinel_elapsed" =~ ^[0-9]{{1,15}}$ ]] || \
        _gym_stop_accounting_unknown "Follow-up $_gym_sentinel_for accounting is unknown; its sentinel cannot tell whether it ran."
    if [[ -z "$_gym_sentinel_recorded" ]] && (( 10#$_gym_sentinel_elapsed == 0 )); then
        # Slurm cancels the afternotok follow-up of a segment that COMPLETED: the run finished, and nothing is recorded.
        _gym_parent_state=$(sacct -X -P -n -j "$_prev_slurm_job_id" --format=State 2>/dev/null | head -n 1)
        _gym_parent_state=${{_gym_parent_state%% *}}
        _gym_parent_state=${{_gym_parent_state%+}}
        if [[ "$_gym_parent_state" == COMPLETED ]]; then
            echo "Follow-up $_gym_sentinel_for never started: job $_prev_slurm_job_id completed, so Slurm cancelled it; sentinel ${{SLURM_JOB_ID:-}} leaves the chain as it is."
            _gym_record_own_allocation
            exit 0
        fi
        _gym_cancel_note="Follow-up $_gym_sentinel_for {cancelled_followup_marker} (job $_prev_slurm_job_id ended ${{_gym_parent_state:-in an unknown state}}); the chain stopped. Continue with gym eval resubmit --recover."
        # A deliberate stop already says why the chain ended: only a missing marker or a provisional line is replaced.
        if [[ ! -s "$_incomplete_file" ]] || grep -q '^{provisional_marker_prefix}' "$_incomplete_file" 2>/dev/null; then
            echo "$_gym_cancel_note" > "$_incomplete_file"
        fi
        echo "Follow-up $_gym_sentinel_for never started (it was cancelled); sentinel ${{SLURM_JOB_ID:-}} leaves the chain as it is."
        echo "$_gym_cancel_note"
        _gym_record_own_allocation
        exit 0
    fi
    if [[ -n "$_gym_sentinel_recorded" ]]; then
        echo "Follow-up $_gym_sentinel_for ended after ${{_gym_sentinel_elapsed}}s, after it recorded itself and before it queued its follow-up; sentinel ${{SLURM_JOB_ID:-}} continues the chain in its place."
    else
        echo "Follow-up $_gym_sentinel_for ended after ${{_gym_sentinel_elapsed}}s before it recorded itself; sentinel ${{SLURM_JOB_ID:-}} continues the chain in its place."
    fi
    _gym_sentinel_takeover=1
    # The dead follow-up is spent history: record it before this allocation (unless it recorded itself), so a reader
    # rebuilds the chain from disk, and debit its elapsed time once, with the parent's below. Nothing else names it as
    # a predecessor.
    [[ -n "$_gym_sentinel_recorded" ]] || echo "$_gym_sentinel_for" >> "$OUTPUT_DIR/{job_chain_name}" || \
        _gym_stop_accounting_unknown "Cannot record the dead follow-up."
    _gym_sentinel_debit=$((10#$_gym_sentinel_elapsed))
fi

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
# Until this job has queued its follow-up, the marker says so. A stop below replaces the line with its reason; the
# line goes once the follow-up is claimed or the run turns out complete. If the job dies in between (a node failure
# seconds in), the line stays, so the job dir and recovery say what happened, and this job's sentinel takes over.
_gym_provisional="{provisional_marker_prefix}$SLURM_JOB_ID started and had not queued its follow-up; if this line remains, it ended before its guard and nothing continued the chain: continue with gym eval resubmit --recover."
if ! grep -q '^{accounting_unknown_prefix}' "$_incomplete_file" 2>/dev/null; then
    echo "$_gym_provisional" > "$_incomplete_file" || _gym_stop_accounting_unknown "Cannot record the continuation."
fi
_gym_clear_provisional() {{
    if grep -qxF -- "$_gym_provisional" "$_incomplete_file" 2>/dev/null; then rm -f "$_incomplete_file"; fi
    return 0
}}
# A manually submitted worker can start before sbatch returns to its caller. The existing submission
# lock keeps it from resetting counters until the caller has either confirmed the reply or recorded uncertainty.
for _gym_submit_wait in {{1..30}}; do
    [[ -e "$OUTPUT_DIR/{resubmit_lock_name}" ]] || break
    sleep 1
done
[[ ! -e "$OUTPUT_DIR/{resubmit_lock_name}" ]] || \
    _gym_stop_accounting_unknown "Submission acknowledgement is still unresolved."
if grep -q '^{accounting_unknown_prefix}' "$_incomplete_file" 2>/dev/null; then
    _gym_stop_incomplete "Allocation accounting remains unknown; reconcile the complete queue and terminal chain, then continue with gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT, adding --extend-retries N when the retries are spent."
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
    _gym_stop_accounting_unknown "{requeue_unknown_reason}"
fi
# A follow-up first asks whether the run is already complete, before the previous job's state decides anything.
[[ -z "$_prev_slurm_job_id" ]] || _gym_end_if_complete

# A job queued by `gym eval resubmit` starts a fresh retry count and a fresh walltime budget. One queued by
# `--recover` keeps both, so a stopped chain points to `--recover`, which carries the budgets over. The variables
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
{retry_extension_marker}
# The configured cap plus any extension an explicit recovery recorded; never a reset of the spent count.
_gym_max_retries={max_retries}
if [[ -e "$OUTPUT_DIR/{retry_extension_name}" ]]; then
    _gym_extension=$(cat "$OUTPUT_DIR/{retry_extension_name}") || _gym_stop_accounting_unknown "Cannot read the recorded retry extension."
    [[ "$_gym_extension" =~ ^[0-9]{{1,6}}$ ]] || _gym_stop_accounting_unknown "The recorded retry extension is malformed."
    _gym_max_retries=$(( {max_retries} + 10#$_gym_extension ))
fi
# An explicit reset is permitted only after the old counters are readable and known.
if [[ "$_gym_resubmit" == 1 && "$_gym_keep_budget" != 1 ]]; then
    rm -f "$_retry_file" "$_walltime_file"
    _gym_accumulated=0
    _retries=0
fi
_gym_needs_retry=0
_gym_stop_reason=""
_gym_fresh_restart=""

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
            _gym_clear_provisional
            echo "Previous job $_prev_slurm_job_id completed successfully. Exiting."
            _gym_record_own_allocation
            exit 0 ;;
        CANCELLED)
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id was cancelled. Chain stopped with rollouts missing; continue with gym eval resubmit --recover, which carries the retry and walltime budgets over." ;;
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
    # The previous job is debited exactly once. Before the counter changes, the value it had before that debit is kept
    # beside it, named by the previous job. A follow-up can write the counter and then die; the sentinel that takes
    # its place has the same previous job, so it starts again from the kept value instead of debiting that job twice.
    _gym_before_debit="$OUTPUT_DIR/.gym_walltime_before.$_prev_slurm_job_id"
    if [[ -e "$_gym_before_debit" ]]; then
        _gym_accumulated=$(cat "$_gym_before_debit") || _gym_stop_accounting_unknown "Cannot read the walltime before the previous job's debit."
        [[ "$_gym_accumulated" =~ ^[0-9]{{1,15}}$ ]] || _gym_stop_accounting_unknown "The walltime before the previous job's debit is malformed."
        _gym_accumulated=$((10#$_gym_accumulated))
    else
        {{ echo "$_gym_accumulated" > "$_gym_before_debit.tmp" && mv -f "$_gym_before_debit.tmp" "$_gym_before_debit"; }} ||             _gym_stop_incomplete "Cannot preserve elapsed walltime."
    fi
    _gym_accumulated=$((_gym_accumulated + 10#$_prev_elapsed + _gym_sentinel_debit))
    echo "$_gym_accumulated" > "$_walltime_file" || _gym_stop_incomplete "Cannot preserve elapsed walltime."
    # A driver that failed before collection ran anything left its kind; the next segment would fail the same way.
    _gym_failure_kind=""
    _gym_failure_file="$OUTPUT_DIR/{driver_failure_name}.$_prev_slurm_job_id"
    [[ ! -e "$_gym_failure_file" ]] || _gym_failure_kind=$(head -n 1 "$_gym_failure_file") || _gym_failure_kind=""
    case "$_gym_failure_kind" in
        {server_spinup_timeout})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {spinup_timeout_reason}" ;;
        {driver_install_failed})
            # The install ran before Gym, so its output is all the driver log holds.
            _gym_driver_log="$OUTPUT_DIR/logs/driver-$_prev_slurm_job_id.log"
            _gym_install_error=$(grep -m 1 -iE 'error|fatal' "$_gym_driver_log" 2>/dev/null || tail -n 1 "$_gym_driver_log" 2>/dev/null)
            _gym_install_error=$(printf '%s' "$_gym_install_error" | tr -d '\\000-\\037')
            [[ -n "$_gym_install_error" ]] || _gym_install_error="the install printed nothing"
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {driver_install_failed}: ${{_gym_install_error:0:{install_error_max_chars}}}; the chain does not retry it; see logs/driver-$_prev_slurm_job_id.log" ;;
        {collection_resume_refused})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {resume_refused_reason}" ;;
        {provider_credentials_rejected})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {credentials_rejected_reason}" ;;
        {environment_prepare_failed})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {prepare_failed_reason}" ;;
        {provider_quota_exhausted})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {quota_exhausted_reason}" ;;
        {environment_protocol_violation})
            _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {protocol_violation_reason}" ;;
        {server_start_failed})
            # A server that did not start (its port taken, for instance) before the run saved anything: there is
            # nothing to resume, and the failure is the infrastructure's, so the segment starts the run again fresh,
            # once, as an infra retry. A run with saved rollouts keeps the saved-evidence path below.
            if [[ "$_prev_state" == FAILED && ! -e "$OUTPUT_DIR/artifacts/rollouts_manifest.json" ]]; then
                _gym_run_dir="${{OUTPUT_DIR%/*}}"
                if [[ -e "$OUTPUT_DIR/{fresh_restart_marker_name}.$_prev_slurm_job_id" ]] || (( _retries >= _gym_max_retries )); then
                    _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended $_prev_state: {server_start_failed} on job $_prev_slurm_job_id before any rollout was saved; fresh restart exhausted; resubmit fresh: gym eval resubmit ${{_gym_run_dir##*/}} ${{OUTPUT_DIR##*/}}"
                fi
                _gym_stop_reason=""
                _gym_needs_retry=1
                _gym_fresh_restart=1
                echo "Previous job $_prev_slurm_job_id ended $_prev_state: {server_start_failed} before any rollout was saved; restarting the run fresh as an infra retry."
            fi ;;
    esac
    # A segment that timed out before Gym wrote the task inventory saved nothing to resume, and the next one would
    # repeat the same startup under the same walltime: the chain ends and names the fresh restart instead.
    if [[ "$_prev_state" == TIMEOUT && ! -e "$OUTPUT_DIR/artifacts/rollouts_materialized_inputs.jsonl" ]]; then
        _gym_run_dir="${{OUTPUT_DIR%/*}}"
        _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended TIMEOUT: {pre_inventory_reason}; fresh restart: {fresh_restart_command}"
    fi
elif [[ "$_gym_resubmit" == 1 && "$_gym_keep_budget" == 1 ]]; then
    # Recovery is another allocation, not a fresh allowance. A plain manual resubmit reset above is explicit.
    _gym_needs_retry=1
fi

if [[ "$_gym_needs_retry" == 1 && $_retries -ge $_gym_max_retries ]]; then
    _gym_stop_incomplete "Infra retry limit ($_gym_max_retries) reached. Stopping."
fi
{max_walltime_check}
if [[ -n "$_gym_stop_reason" || ( -n "$_prev_slurm_job_id" && -e "$OUTPUT_DIR/artifacts/rollouts_manifest.json" ) ]]; then
    if [[ "$_prev_state" == FAILED || "$_prev_state" == TIMEOUT || "$_prev_state" == PREEMPTED || "$_prev_state" == NODE_FAIL || "$_prev_state" == BOOT_FAIL ]]; then
        _gym_exit_code=$(sacct -X -P -n -j "$_prev_slurm_job_id" --format=ExitCode) || \
            _gym_stop_accounting_unknown "Previous exit accounting is unknown."
        [[ "$_gym_exit_code" =~ ^[0-9]+:[0-9]+$ ]] || _gym_stop_accounting_unknown "Previous exit accounting is malformed."
        {runtime_admission_marker}
        # The installer's stderr goes to a file while it runs; on failure its first error or fatal line joins
        # the stop text. Either way the whole stderr is copied to the job log through the saved descriptor.
        _gym_admission_err=$(mktemp 2>/dev/null) || _gym_admission_err=/dev/null
        _gym_admission_error() {{
            cat "$_gym_admission_err" >&9 2>/dev/null
            local _gym_error
            _gym_error=$(grep -m 1 -iE 'error|fatal' "$_gym_admission_err" 2>/dev/null || tail -n 1 "$_gym_admission_err" 2>/dev/null)
            _gym_error=$(printf '%s' "$_gym_error" | tr -d '\\000-\\037')
            [[ -z "$_gym_error" ]] || printf ': %s' "${{_gym_error:0:{install_error_max_chars}}}"
            printf '; the chain does not retry it'
        }}
        # A runtime without the recovery reader fails on its import, whatever else it prints: name the cause and the fix.
        _gym_admission_failure() {{
            if grep -qE "No module named .nemo_gym\\.orchestration(\\.recovery)?.|cannot import name .continuation_admission." "$_gym_admission_err" 2>/dev/null; then
                cat "$_gym_admission_err" >&9 2>/dev/null
                printf '%s' "{recovery_runtime_outdated_reason}"
            else
                printf 'Installed recovery runtime failed%s' "$(_gym_admission_error)"
            fi
        }}
        exec 9>&2 2>"$_gym_admission_err"
        _gym_admission_output=$({admission_command}) || _gym_stop_incomplete "$(_gym_admission_failure)"
        exec 2>&9 9>&-
        cat "$_gym_admission_err" >&2 2>/dev/null
        [[ "$_gym_admission_err" == /dev/null ]] || rm -f -- "$_gym_admission_err"
        _gym_admission=$(printf '%s\n' "$_gym_admission_output" | grep '^__GYM_CONTINUATION:' || true)
        # The installed reader's own decision, when it gives one, replaces the scheduler-state guess above.
        _gym_admission_reason=$(printf '%s\n' "$_gym_admission_output" | grep -m 1 '^{continuation_reason_prefix}' || true)
        _gym_admission_reason=${{_gym_admission_reason#{continuation_reason_prefix}}}
        case "$_gym_admission" in
            __GYM_CONTINUATION:segment|__GYM_CONTINUATION:retry)
                [[ "$_gym_admission" == __GYM_CONTINUATION:retry ]] && _gym_needs_retry=1 || _gym_needs_retry=0
                [[ -z "$_gym_admission_reason" ]] || echo "Saved recovery evidence: $_gym_admission_reason" ;;
            # The installed reader could not run: no verdict on the saved artifacts, which an inspection reads again.
            __GYM_CONTINUATION:unknown)
                _gym_stop_incomplete "Saved recovery evidence was not read; ${{_gym_admission_reason}} (previous job $_prev_slurm_job_id ended $_prev_state)" ;;
            *) _gym_stop_incomplete "Saved recovery evidence has no eligible retries; ${{_gym_admission_reason:-$_gym_stop_reason}}${{_gym_admission_reason:+ (previous job $_prev_slurm_job_id ended $_prev_state)}}" ;;
        esac
    else
        _gym_stop_incomplete "$_gym_stop_reason"
    fi
fi
# The rows the Store counts as progress (RolloutStore.record_outcome): every saved result, and every recorded failure
# but an interruption (kill_shaped, not sent, refused at admission) that is neither terminal nor an omission. Only
# newline-terminated lines count: the Store drops a cut tail and terminates a whole one, so the count never shrinks.
_gym_progress_rows() {{
    local _results="$OUTPUT_DIR/artifacts/rollouts.jsonl" _failures="$OUTPUT_DIR/artifacts/rollouts_failures.jsonl"
    local _rows=0 _lines
    if [[ -e "$_results" ]]; then
        _lines=$(wc -l < "$_results") || return 1
        _rows=$((_rows + _lines))
    fi
    if [[ -e "$_failures" ]]; then
        _lines=$(wc -l < "$_failures") || return 1
        _lines=$(head -n "$_lines" "$_failures" | awk -v q='"' '
            index($0, q "_ng_failure_terminal" q ":true") || index($0, q "_ng_omitted" q ":") {{ n++; next }}
            index($0, q "_ng_failure_class" q ":" q "kill_shaped" q) {{ next }}
            index($0, q "delivery" q ":" q "not_sent" q) || index($0, q "stage" q ":" q "admission" q) {{ next }}
            {{ n++ }}
            END {{ print n + 0 }}') || return 1
        _rows=$((_rows + _lines))
    fi
    echo "$_rows"
}}
# A resumed segment that timed out without saving a rollout spent its walltime on setup (or on one rollout longer
# than what was left): the next segment would do the same. The previous segment recorded the Store's progress rows
# when it started; only an unchanged count stops, and a missing or unreadable record continues. Nothing here depends
# on the Store's coverage report, which a segment killed at its walltime may not have flushed. It runs after
# admission, so a stop the saved evidence explains (unknown reservations, no retries) keeps its own reason.
if [[ "$_prev_state" == TIMEOUT && -n "$_prev_slurm_job_id" ]]; then
    _gym_rows_then=$(head -n 1 "$OUTPUT_DIR/{rows_at_start_name}.$_prev_slurm_job_id" 2>/dev/null) || _gym_rows_then=""
    if [[ "$_gym_rows_then" =~ ^[0-9]+$ ]] && _gym_rows_now=$(_gym_progress_rows) && \
        (( _gym_rows_now == 10#$_gym_rows_then )); then
        _gym_run_dir="${{OUTPUT_DIR%/*}}"
        _gym_stop_incomplete "Previous job $_prev_slurm_job_id ended TIMEOUT after $((10#$_prev_elapsed))s: {no_progress_segment_reason}: {fresh_restart_command}"
    fi
fi
# A resumed segment (a continuation or a recovery) records its own start for the next one. The first segment does not:
# its successor reuses the prepared data and the installed toolchain, so it is not the same segment again.
if [[ -n "$_prev_slurm_job_id" || "$_gym_resubmit" == 1 ]] && _gym_rows_now=$(_gym_progress_rows); then
    echo "$_gym_rows_now" > "$OUTPUT_DIR/{rows_at_start_name}.$SLURM_JOB_ID" || true
fi
# The infrastructure ended the follow-up this sentinel replaces, so the takeover is an infra retry: dying
# follow-ups spend the chain's retry budget and stop at its cap instead of continuing without end.
[[ -z "$_gym_sentinel_takeover" ]] || _gym_needs_retry=1
# Admission can identify a retry only after reading the authoritative saved outcomes.
if [[ "$_gym_needs_retry" == 1 && $_retries -ge $_gym_max_retries ]]; then
    _gym_stop_incomplete "Infra retry limit ($_gym_max_retries) reached. Stopping."
fi
if [[ "$_gym_needs_retry" == 1 ]]; then
    _retries=$((_retries + 1))
    echo "$_retries" > "$_retry_file" || _gym_stop_incomplete "Cannot preserve the spent retry."
    if [[ -n "$_gym_fresh_restart" ]]; then
        : > "$OUTPUT_DIR/{fresh_restart_marker_name}.$SLURM_JOB_ID" || _gym_stop_incomplete "Cannot record the fresh restart."
    fi
    echo "Infra retry $_retries/$_gym_max_retries..."
fi

# This job continues the run, so an earlier stop no longer describes it: its provisional line replaced that stop when
# it recorded itself, and goes once the follow-up below is claimed.
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
        _gym_clear_provisional
    else
        _gym_stop_accounting_unknown "The existing follow-up claim has no unique observed pending job; it may already have run."
    fi
fi
# Whether sbatch's reply is the scheduler's definite refusal (an invalid request, a policy or limit it violates, a
# resource or dependency that is not available), with no sign that the request or its reply was lost on the way. Any
# other error leaves the submission's outcome unknown: the job may exist.
_gym_submission_refused() {{
    [[ "$1" == *"sbatch: error:"* ]] || return 1
    grep -qiE 'invalid|violates|policy|limit|not available|dependency|qos|account' <<< "$1" || return 1
    ! grep -qiE 'timed out|time out|socket|unable to contact|connection|communication|zero bytes|try again|temporarily|broken pipe|not connected|unreachable|no route|resolve|standby|i/o error' <<< "$1"
}}
if [[ $_gym_queue_followup == 1 ]]; then
    # kill-on-invalid-dep: without it, clusters lacking kill_invalid_depend keep the successor pending forever.
    if _next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes "$_this_script" $SLURM_JOB_ID 2>&1); then
        _next_id="${{_next_output##* }}"
        if [[ "$_next_id" =~ ^[0-9]+$ ]]; then
            echo "$_next_id" > "$_guard_marker" || _gym_stop_accounting_unknown "Cannot preserve the queued follow-up identity."
            _gym_clear_provisional
            echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
            # Its sentinel continues the chain if the follow-up dies before its prologue records it (see the top).
            if _sentinel_output=$(GYM_FOLLOWUP_SENTINEL=$_next_id sbatch --dependency=afterany:$_next_id --kill-on-invalid-dep=yes "$_this_script" $SLURM_JOB_ID 2>&1); then
                _sentinel_id="${{_sentinel_output##* }}"
                if [[ "$_sentinel_id" =~ ^[0-9]+$ ]]; then
                    echo "$_sentinel_id" > "$OUTPUT_DIR/.gym_followup_sentinel.$SLURM_JOB_ID" || true
                fi
                echo "Follow-up sentinel queued: $_sentinel_id (afterany:$_next_id)"
            else
                echo "WARNING: Failed to queue the follow-up sentinel; a follow-up that dies before its prologue ends the chain."
            fi
        else
            echo "{accounting_unknown_prefix} Follow-up submission returned no unique job identity." > "$_incomplete_file"
        fi
    else
        _gym_submit_rc=$?
        if _gym_submission_refused "$_next_output"; then
            # An observed refusal queued nothing: the accounting is known, and the chain stops here without a requeue.
            # Its claim names no job, so it goes, or a later reader would take it for an unobserved follow-up.
            rm -f "$_guard_marker"
            _gym_refusal=$(grep -m 3 'sbatch: error:' <<< "$_next_output" | paste -sd ' ' - | cut -c1-300)
            echo "{follow_up_submission_refused}: sbatch exited $_gym_submit_rc: $_gym_refusal" > "$_incomplete_file"
            echo "WARNING: The scheduler refused the auto-resume follow-up (sbatch exited $_gym_submit_rc: $_gym_refusal). Automatic continuation is stopped; this allocation still runs. Once the scheduler accepts the job, continue with gym eval resubmit --recover."
        else
            # The current allocation can finish, but a successor must not erase this uncertainty or reset budgets.
            echo "{accounting_unknown_prefix} Follow-up submission may have succeeded without an observed reply." > "$_incomplete_file"
            echo "WARNING: Failed to submit auto-resume follow-up (sbatch exited $_gym_submit_rc: $(paste -sd ' ' - <<< "$_next_output" | cut -c1-300)). Inspect the complete queue and terminal accounting; automatic continuation is stopped."
        fi
    fi
fi
# With its follow-up queued, the job records the run's ended allocations; recording never delays the continuation.
_gym_record_allocations
"""

_MAX_WALLTIME_CHECK_TEMPLATE = """\
if (( _gym_accumulated >= {seconds} )); then
    _gym_stop_incomplete "Accumulated walltime (${{_gym_accumulated}}s) reached max_walltime cap ({raw} = {seconds}s). Chain stopped at the walltime cap with rollouts missing; continue with gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT, with HH:MM:SS above the cap: the retry and walltime budgets carry over."
fi
# Slurm rounds time limits up to minutes. Round the remaining allowance down instead, never granting extra time.
_gym_remaining_minutes=$((({seconds} - _gym_accumulated) / 60))
(( _gym_remaining_minutes > 0 )) || _gym_stop_incomplete "Less than one minute remains in the walltime budget (${{_gym_accumulated}}s of {seconds}s); continue with gym eval resubmit --recover --max-walltime HH:MM:SS --walltime-reason TEXT, with HH:MM:SS above the cap: the retry and walltime budgets carry over."
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

# Rendered right after the driver step; it uses the prologue's functions and files. Its first line keeps the driver
# step's status (a function definition resets `$?`), and the function returns it: the job's state, and what a later
# step reads, is the driver's.
_EPILOGUE_TEMPLATE = """\
# --- A job that finishes the run ends its own chain ---
_gym_driver_rc=$?
_gym_end_own_chain() {{
    local _rc=$1 _ended=run_complete _role _file _id _result _cancelled="" _kind="" _superseded=""
    # A failure kind the follow-up's prologue turns into its own stop and remedy (its guard): that job keeps its
    # follow-up. Any other kind, or none, gets no stop from the follow-up.
    _file="$OUTPUT_DIR/{driver_failure_name}.$SLURM_JOB_ID"
    [[ ! -e "$_file" ]] || _kind=$(head -n 1 "$_file" 2>/dev/null) || _kind=""
    case "$_kind" in
        {guarded_failure_kinds})
            _kind=guarded ;;
    esac
    if [[ $_rc == {no_eligible_retry_exit} && "$_kind" != guarded ]]; then
        # Nothing is eligible for another attempt: the follow-up would start on a full allocation only to write this
        # stop and exit, so the job writes it and cancels what it queued. A deliberate stop already there stays.
        _ended={no_eligible_retry}
        if [[ ! -s "$_incomplete_file" ]] || grep -q '^{provisional_marker_prefix}' "$_incomplete_file" 2>/dev/null; then
            echo "{no_eligible_retry_reason}" > "$_incomplete_file"
            echo "{no_eligible_retry_reason}"
        fi
    # A job whose driver failed otherwise has its follow-up start, and that prologue decides; an incomplete run
    # continues.
    elif [[ $_rc != 0 ]] || ! _gym_run_complete; then
        # Its follow-up records it; with none queued (a refused or unobserved submission), nothing else would.
        _id=$(head -n 1 "$OUTPUT_DIR/.gym_followup_queued.$SLURM_JOB_ID" 2>/dev/null) || _id=""
        [[ "$_id" =~ ^[0-9]+$ ]] || _gym_record_own_allocation
        return $_rc
    # This job COMPLETEs: Slurm would cancel its afternotok follow-up, and the follow-up's sentinel would then start on
    # a full allocation only to write the record below and exit. A stop the chain recorded earlier (a refused
    # follow-up, unknown accounting, a cap) described a run with rollouts missing; the complete run supersedes it.
    else
        if [[ -s "$_incomplete_file" ]] && ! grep -q '^{provisional_marker_prefix}' "$_incomplete_file" 2>/dev/null; then
            _superseded=$(head -n 1 "$_incomplete_file" 2>/dev/null | cut -c1-300)
        fi
        if _gym_export_required && ! _gym_receipt_current; then
            _ended={export_receipt_missing}
            echo "{export_receipt_missing_reason}" > "$_incomplete_file"
            echo "{export_receipt_missing_reason}"
        else
            rm -f "$_incomplete_file"
            echo "Run complete: the Store covers all $_gym_complete_rollouts expected rollouts and $_gym_export_note; job $SLURM_JOB_ID ends the chain with nothing to recover."
            if [[ -n "$_superseded" ]]; then
                _ended=run_complete_stop_superseded
                echo "The chain's earlier stop no longer applies: $_superseded"
            fi
        fi
    fi
    # The sentinel first: once its follow-up is gone, its afterany dependency is met and it could start.
    for _role in sentinel follow_up; do
        _file="$OUTPUT_DIR/.gym_followup_queued.$SLURM_JOB_ID"
        [[ $_role == follow_up ]] || _file="$OUTPUT_DIR/.gym_followup_sentinel.$SLURM_JOB_ID"
        _id=$(head -n 1 "$_file" 2>/dev/null) || _id=""
        [[ "$_id" =~ ^[0-9]{{1,15}}$ ]] || continue
        if scancel "$_id"; then
            _result=cancelled
            echo "Cancelled this chain's queued ${{_role/_/-}} $_id: the run is complete."
        else
            _result=scancel_failed
            echo "WARNING: could not cancel this chain's queued ${{_role/_/-}} $_id; if it starts, its prologue ends it."
        fi
        _cancelled+="${{_cancelled:+,}}\\"$_role\\":{{\\"job_id\\":\\"$_id\\",\\"result\\":\\"$_result\\"}}"
    done
    # Nothing follows this job to record it: it records its own allocation as of now, and any ended one still missing.
    _gym_record_own_allocation
    printf '{{"at":"%s","cancelled":{{%s}},"ended":"%s","job_id":"%s"}}\\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \\
        "$_cancelled" "$_ended" "$SLURM_JOB_ID" >> "$OUTPUT_DIR/{chain_end_journal_name}" || \\
        echo "WARNING: cannot journal the end of the chain in {chain_end_journal_name}."
    return $_rc
}}
_gym_end_own_chain "$_gym_driver_rc\""""


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


def fresh_restart_command(gym_job_id: str, benchmark: str) -> str:
    """The command that queues a run with no task inventory again, as a fresh run with a longer walltime.

    Submission ids and benchmark names are shell-safe by construction (see `new_gym_job_id` and the executor's
    benchmark name check), so the command needs no quoting, and the prologue renders it with shell expansions of both.
    """
    return f"gym eval resubmit {gym_job_id} {benchmark} --recover --max-walltime HH:MM:SS --walltime-reason TEXT"


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
        provisional_marker_prefix=PROVISIONAL_MARKER_PREFIX,
        cancelled_followup_marker=CANCELLED_FOLLOWUP_MARKER,
        follow_up_submission_refused=FOLLOW_UP_SUBMISSION_REFUSED,
        allocation_accounting_marker=ALLOCATION_ACCOUNTING_MARKER,
        runtime_admission_marker=(
            RUNTIME_ADMISSION_MARKER if admission_command != "false" else "# No installed-runtime continuation reader"
        ),
        admission_command=admission_command,
        continuation_reason_prefix=CONTINUATION_REASON_PREFIX,
        retry_extension_marker=RETRY_EXTENSION_MARKER,
        retry_extension_name=RETRY_EXTENSION_NAME,
        requeue_unknown_reason=REQUEUE_UNKNOWN_REASON,
        requeue_unknown_marker=f"{ACCOUNTING_UNKNOWN_PREFIX} {REQUEUE_UNKNOWN_REASON}",
        requeue_reconciled_note=REQUEUE_RECONCILED_NOTE,
        requeue_settle_poll_s=REQUEUE_SETTLE_POLL_S,
        requeue_settle_bound_s=REQUEUE_SETTLE_BOUND_S,
        chain_accounting_format=CHAIN_ACCOUNTING_FORMAT,
        job_allocations_name=JOB_ALLOCATIONS_NAME,
        job_allocations_format=JOB_ALLOCATIONS_FORMAT,
        job_allocations_budget_s=JOB_ALLOCATIONS_BUDGET_S,
        job_allocations_retry_s=JOB_ALLOCATIONS_RETRY_S,
        job_allocations_lock_name=JOB_ALLOCATIONS_LOCK_NAME,
        requeue_reconcile_state=REQUEUE_RECONCILE_STATE,
        chain_accounting_env=CHAIN_ACCOUNTING_ENV,
        accounted_walltime_prefix=ACCOUNTED_WALLTIME_PREFIX,
        pre_inventory_reason=PRE_INVENTORY_REASON,
        rows_at_start_name=ROWS_AT_START_NAME,
        no_progress_segment_reason=NO_PROGRESS_SEGMENT_REASON,
        # The job dir is <output_path>/<gym_job_id>/<benchmark>: the command names both from the directory it runs in.
        server_start_failed=SERVER_START_FAILED,
        fresh_restart_marker_name=FRESH_RESTART_MARKER_NAME,
        fresh_restart_command=fresh_restart_command("${_gym_run_dir##*/}", "${OUTPUT_DIR##*/}"),
        driver_failure_name=DRIVER_FAILURE_NAME,
        server_spinup_timeout=SERVER_SPINUP_TIMEOUT,
        spinup_timeout_reason=SPINUP_TIMEOUT_REASON,
        driver_install_failed=DRIVER_INSTALL_FAILED,
        install_error_max_chars=INSTALL_ERROR_MAX_CHARS,
        recovery_runtime_outdated_reason=RECOVERY_RUNTIME_OUTDATED_REASON,
        collection_resume_refused=COLLECTION_RESUME_REFUSED,
        resume_refused_reason=RESUME_REFUSED_REASON,
        provider_credentials_rejected=PROVIDER_CREDENTIALS_REJECTED,
        credentials_rejected_reason=CREDENTIALS_REJECTED_REASON,
        environment_prepare_failed=ENVIRONMENT_PREPARE_FAILED,
        prepare_failed_reason=PREPARE_FAILED_REASON,
        provider_quota_exhausted=PROVIDER_QUOTA_EXHAUSTED,
        environment_protocol_violation=ENVIRONMENT_PROTOCOL_VIOLATION,
        protocol_violation_reason=PROTOCOL_VIOLATION_REASON,
        quota_exhausted_reason=QUOTA_EXHAUSTED_REASON,
        export_receipt_missing_reason=EXPORT_RECEIPT_MISSING_REASON,
        receipt_artifacts=" ".join(f"{key}:{name}" for key, name in _RECEIPT_ARTIFACTS),
    )


def render_resume_epilogue() -> str:
    """Render the step after the driver of a resumable job: a job that finished the run ends its own chain.

    When the driver step succeeded and the run is complete, it writes the record the follow-up or its sentinel would
    have written, cancels the sentinel and then the follow-up it queued, and journals both in
    ``CHAIN_END_JOURNAL_NAME``, so no allocation starts for a finished run. A driver that exits
    ``NO_ELIGIBLE_RETRY_EXIT`` (nothing is eligible for another attempt) ends its chain the same way, with the
    ``NO_ELIGIBLE_RETRY_REASON`` stop, unless it left in ``DRIVER_FAILURE_NAME`` a failure kind the follow-up's
    prologue turns into its own stop (spin-up timeout, install failed, resume refused, credentials rejected). The job keeps the driver step's status.
    """
    return _EPILOGUE_TEMPLATE.format(
        no_eligible_retry_exit=NO_ELIGIBLE_RETRY_EXIT,
        no_eligible_retry=NO_ELIGIBLE_RETRY,
        no_eligible_retry_reason=NO_ELIGIBLE_RETRY_REASON,
        driver_failure_name=DRIVER_FAILURE_NAME,
        guarded_failure_kinds="|".join(GUARDED_DRIVER_FAILURE_KINDS),
        provisional_marker_prefix=PROVISIONAL_MARKER_PREFIX,
        export_receipt_missing=EXPORT_RECEIPT_MISSING,
        export_receipt_missing_reason=EXPORT_RECEIPT_MISSING_REASON,
        chain_end_journal_name=CHAIN_END_JOURNAL_NAME,
    )


def record_failure_kind(error: BaseException) -> None:
    """Leave the registered kind ``error`` carries in the failure file the driver step names, for the next guard.

    The driver step of a resumable job names the file in ``DRIVER_FAILURE_ENV``; elsewhere this does nothing.
    """
    kind = getattr(error, "failure_kind", None)
    path = os.environ.get(DRIVER_FAILURE_ENV)
    if not isinstance(kind, str) or not path:
        return
    try:
        Path(path).write_text(kind + "\n")
    except OSError as write_error:
        # The next segment then reads the saved state as it would without the kind.
        logger.warning("Cannot record failure kind %s in %s: %s", kind, path, write_error)
