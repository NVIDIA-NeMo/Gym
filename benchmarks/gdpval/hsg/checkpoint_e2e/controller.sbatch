#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=gdpval-e2e-controller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1

: "${RUN_DIR:?set RUN_DIR}"
: "${E2E_DIR:?set E2E_DIR}"

# shellcheck disable=SC1090
source "$RUN_DIR/settings.env"
# shellcheck disable=SC1090
source "$E2E_DIR/slurm_receipts.sh"
# shellcheck disable=SC1090
source "$E2E_DIR/judge_process_group.sh"
CAMPAIGN_PY="$E2E_DIR/campaign.py"
JUDGE_STATE_PY="$E2E_DIR/judge_state.py"
JUDGE_CACHE_ROOT="${JUDGE_DELIVERABLES:-$DELIVERABLES}"
PYTHON_BIN="${CHECKPOINT_E2E_PYTHON:-python3}"
AUTHORIZE="${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}"
MAX_RECOVERY_ROUNDS="${CHECKPOINT_E2E_MAX_RECOVERY_ROUNDS:-6}"
MAX_PRECONVERT_ATTEMPTS="${CHECKPOINT_E2E_MAX_PRECONVERT_ATTEMPTS:-4}"
MAX_TRANSPORT_PREBUILD_ATTEMPTS="${CHECKPOINT_E2E_MAX_TRANSPORT_PREBUILD_ATTEMPTS:-2}"
MAX_JUDGE_ATTEMPTS=4
JUDGE_CONCURRENCIES=(16 16 8 4)
CONTROLLER_ROTATION="${SLURM_RESTART_COUNT:-0}"
MAX_CONTROLLER_REQUEUES="${CHECKPOINT_E2E_MAX_CONTROLLER_REQUEUES:-2}"
JUDGE_NO_PROGRESS_SECONDS="${CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS:-1800}"
JUDGE_CANCEL_GRACE_SECONDS="${CHECKPOINT_E2E_JUDGE_CANCEL_GRACE_SECONDS:-300}"
JUDGE_TASK_TIMEOUT_SECONDS="${JUDGE_TASK_TIMEOUT_SECONDS:-1500}"
POLL_SECONDS="${CHECKPOINT_E2E_POLL_SECONDS:-60}"

[[ $CONTROLLER_ROTATION =~ ^[0-9]+$ && $MAX_CONTROLLER_REQUEUES =~ ^[0-9]+$ \
    && $MAX_RECOVERY_ROUNDS =~ ^[1-9][0-9]*$ && $MAX_PRECONVERT_ATTEMPTS =~ ^[1-9][0-9]*$ \
    && $MAX_TRANSPORT_PREBUILD_ATTEMPTS =~ ^[1-9][0-9]*$ ]] \
    || { echo "invalid controller rotation contract" >&2; exit 64; }
for seconds in "$JUDGE_NO_PROGRESS_SECONDS" "$JUDGE_CANCEL_GRACE_SECONDS" \
    "$JUDGE_TASK_TIMEOUT_SECONDS" "$POLL_SECONDS"; do
    [[ $seconds =~ ^[1-9][0-9]*$ ]] || { echo "invalid positive timeout: $seconds" >&2; exit 64; }
done
(( MAX_CONTROLLER_REQUEUES <= 4 )) \
    || { echo "controller requeue bound exceeds 4" >&2; exit 64; }
(( MAX_RECOVERY_ROUNDS <= 12 && MAX_PRECONVERT_ATTEMPTS <= 12 )) \
    || { echo "recovery attempt bound exceeds 12" >&2; exit 64; }
(( MAX_TRANSPORT_PREBUILD_ATTEMPTS <= 4 )) \
    || { echo "transport prebuild attempt bound exceeds 4" >&2; exit 64; }
(( JUDGE_NO_PROGRESS_SECONDS <= 7200 && JUDGE_CANCEL_GRACE_SECONDS <= 900 && POLL_SECONDS <= 300 )) \
    || { echo "judge watchdog timeout exceeds its safety bound" >&2; exit 64; }
(( JUDGE_NO_PROGRESS_SECONDS >= JUDGE_TASK_TIMEOUT_SECONDS + 300 )) \
    || { echo "judge watchdog must allow the internal task timeout plus persistence grace" >&2; exit 64; }

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
block() {
    log "BLOCKED: $*"
    printf '%s\n' "$*" > "$RUN_DIR/CONTROLLER_BLOCKED"
    exit 1
}

exec 9> "$RUN_DIR/controller.lock"
flock -n 9 || block "another controller owns this campaign"
rm -f "$RUN_DIR/CONTROLLER_BLOCKED"

on_term() {
    trap - TERM
    if (( CONTROLLER_ROTATION >= MAX_CONTROLLER_REQUEUES )); then
        block "controller walltime requeue limit reached at rotation=$CONTROLLER_ROTATION"
    fi
    log "walltime signal; requeueing controller rotation=$CONTROLLER_ROTATION/$MAX_CONTROLLER_REQUEUES"
    scontrol requeue "$SLURM_JOB_ID" || block "controller requeue request failed"
    exit 0
}
trap on_term TERM

ids_from_tsv() {
    awk -F '\t' '
        NF == 0 {next}
        NF < 2 || $2 !~ /^[1-9][0-9]*$/ {exit 1}
        {print $2}
    ' "$1" | paste -sd, -
}

validate_job_ids() {
    [[ $1 =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]
}

jobs_all_terminal() {
    local ids=$1 accounting
    validate_job_ids "$ids" || return 1
    accounting=$(sacct -X -j "$ids" --format=JobIDRaw,State -n -P 2>/dev/null) || return 1
    awk -F '|' -v wanted="$ids" '
        BEGIN {
            count = split(wanted, values, ",")
            for (i = 1; i <= count; i++) required[values[i]] = 1
        }
        $1 in required {
            state = $2
            sub(/\+.*/, "", state)
            sub(/ .*/, "", state)
            if (state ~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)$/) {
                terminal[$1] = 1
            }
        }
        END {
            for (job in required) if (!(job in terminal)) exit 1
        }
    ' <<<"$accounting"
}

wait_jobs() {
    local label=$1 ids=$2 live queue
    validate_job_ids "$ids" || block "$label has invalid or missing recorded jobs: $ids"
    while :; do
        if ! queue=$(squeue -h -j "$ids" -o '%A' 2>/dev/null); then
            if jobs_all_terminal "$ids"; then
                log "$label is absent from squeue and terminal in accounting"
                break
            fi
            log "$label queue query failed; retaining ownership and retrying"
            sleep "$POLL_SECONDS"
            continue
        fi
        live=$(printf '%s\n' "$queue" | awk 'NF {n++} END {print n+0}')
        log "$label live=$live coverage=$(fast_coverage_count)/$EXPECTED_TASKS"
        if (( live == 0 )); then
            if jobs_all_terminal "$ids"; then
                break
            fi
            log "$label is absent from squeue but not fully terminal in accounting; retrying"
        fi
        sleep "$POLL_SECONDS"
    done
    sacct -X -j "$ids" --format=JobIDRaw,State,ExitCode,Elapsed,Account,Partition,QOS -n -P \
        | awk -F '|' '$1 !~ /\./ && NF {print}' || true
}

judge_progress_signature() {
    local output="$RUN_DIR/judge_e2e/gdpval_aav2.jsonl"
    local journal="$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl"
    local output_bytes=0 journal_bytes=0 cache_count=0
    [[ ! -f $output ]] || output_bytes=$(wc -c < "$output")
    [[ ! -f $journal ]] || journal_bytes=$(wc -c < "$journal")
    if [[ -d $JUDGE_CACHE_ROOT ]]; then
        cache_count=$(find "$JUDGE_CACHE_ROOT" -type f -name 'repeat_*_verify_response_*.json' -print 2>/dev/null \
            | awk 'END {print NR + 0}') || cache_count=0
    fi
    printf '%s:%s:%s\n' "$output_bytes" "$journal_bytes" "$cache_count"
}

job_is_terminal() {
    local jid=$1 state
    state=$(sacct -X -j "$jid" --format=JobIDRaw,State -n -P 2>/dev/null \
        | awk -F '|' -v wanted="$jid" '$1 == wanted {print $2; exit}') || return 1
    state=${state%%+*}
    state=${state%% *}
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

latest_judge_attempt() {
    local path name suffix expected latest=0
    for path in "$RUN_DIR"/JUDGE_JOB_[0-9]*; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || block "judge receipt is not a regular file: $path"
        name=${path##*/}
        suffix=${name#JUDGE_JOB_}
        [[ $suffix =~ ^[1-9][0-9]*$ ]] || block "invalid judge receipt name: $name"
        (( suffix <= MAX_JUDGE_ATTEMPTS )) \
            || block "judge receipt count exceeds campaign-global maximum: $suffix/$MAX_JUDGE_ATTEMPTS"
        (( suffix > latest )) && latest=$suffix
    done
    for (( expected=1; expected<=latest; expected++ )); do
        [[ -f $RUN_DIR/JUDGE_JOB_${expected} && ! -L $RUN_DIR/JUDGE_JOB_${expected} ]] \
            || block "judge receipt sequence has a gap before attempt $expected"
    done
    printf '%s\n' "$latest"
}

latest_preconvert_attempt() {
    local path name suffix latest=0
    for path in "$RUN_DIR"/PRECONVERT_JOB_[0-9]*; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || block "preconvert receipt is not a regular file: $path"
        name=${path##*/}
        suffix=${name#PRECONVERT_JOB_}
        [[ $suffix =~ ^[1-9][0-9]*$ ]] || block "invalid preconvert receipt name: $name"
        (( suffix > latest )) && latest=$suffix
    done
    printf '%s\n' "$latest"
}

latest_transport_prebuild_attempt() {
    local path name suffix expected latest=0
    for path in "$RUN_DIR"/TRANSPORT_PREBUILD_JOB_[0-9]*; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || block "transport prebuild receipt is not a regular file: $path"
        name=${path##*/}
        suffix=${name#TRANSPORT_PREBUILD_JOB_}
        [[ $suffix =~ ^[1-9][0-9]*$ ]] || block "invalid transport prebuild receipt name: $name"
        (( suffix <= MAX_TRANSPORT_PREBUILD_ATTEMPTS )) \
            || block "transport prebuild receipt count exceeds campaign-global maximum: $suffix/$MAX_TRANSPORT_PREBUILD_ATTEMPTS"
        (( suffix > latest )) && latest=$suffix
    done
    for (( expected=1; expected<=latest; expected++ )); do
        [[ -f $RUN_DIR/TRANSPORT_PREBUILD_JOB_${expected} \
            && ! -L $RUN_DIR/TRANSPORT_PREBUILD_JOB_${expected} ]] \
            || block "transport prebuild receipt sequence has a gap before attempt $expected"
    done
    printf '%s\n' "$latest"
}

read_judge_job_id() {
    slurm_read_job_receipt "$1" || block "invalid judge job id receipt: $1"
}

read_preconvert_job_id() {
    slurm_read_job_receipt "$1" || block "invalid preconvert job id receipt: $1"
}

read_transport_prebuild_job_id() {
    slurm_read_job_receipt "$1" || block "invalid transport prebuild job id receipt: $1"
}

judge_job_requires_wait() {
    local jid=$1 states
    if states=$(squeue -h -j "$jid" -o '%T' 2>/dev/null); then
        [[ -n $states ]] && return 0
        job_is_terminal "$jid" && return 1
        log "judge job $jid is not yet visible in queue or terminal accounting; reattaching safely"
        return 0
    fi
    job_is_terminal "$jid" && return 1
    log "judge job $jid state query failed without terminal accounting; reattaching safely"
    return 0
}

wait_judge_job() {
    local label=$1 jid=$2 states signature last_signature now last_progress live
    local running_seen=0 cancelled=0
    last_signature=$(judge_progress_signature)
    last_progress=$(date +%s)
    while :; do
        if ! states=$(squeue -h -j "$jid" -o '%T' 2>/dev/null); then
            job_is_terminal "$jid" && break
            log "$label state query failed; retaining ownership and retrying"
            sleep "$POLL_SECONDS"
            continue
        fi
        if [[ -z $states ]]; then
            job_is_terminal "$jid" && break
            log "$label is not yet visible in squeue or terminal accounting; retrying"
            sleep "$POLL_SECONDS"
            continue
        fi
        now=$(date +%s)
        signature=$(judge_progress_signature)
        if grep -Eq '^(RUNNING|COMPLETING)$' <<<"$states"; then
            if (( running_seen == 0 )) || [[ $signature != "$last_signature" ]]; then
                running_seen=1
                last_signature=$signature
                last_progress=$now
            elif (( now - last_progress >= JUDGE_NO_PROGRESS_SECONDS )); then
                log "$label made no scientific progress for ${JUDGE_NO_PROGRESS_SECONDS}s; stopping job $jid"
                scancel --signal=TERM --batch "$jid" 2>/dev/null || scancel "$jid" 2>/dev/null || true
                cancelled=1
                break
            fi
        fi
        log "$label state=$(tr '\n' ',' <<<"$states") progress=$signature idle=$((now - last_progress))s"
        sleep "$POLL_SECONDS"
    done

    if (( cancelled )); then
        local deadline=$(( $(date +%s) + JUDGE_CANCEL_GRACE_SECONDS ))
        while (( $(date +%s) < deadline )); do
            if ! live=$(squeue -h -j "$jid" -o '%A' 2>/dev/null); then
                sleep "$POLL_SECONDS"
                continue
            fi
            [[ -n $live ]] || break
            sleep "$POLL_SECONDS"
        done
        if ! live=$(squeue -h -j "$jid" -o '%A' 2>/dev/null); then
            job_is_terminal "$jid" && live="" \
                || block "$label state query failed after cancellation grace: $jid"
        elif [[ -z $live ]] && ! job_is_terminal "$jid"; then
            block "$label disappeared without a terminal accounting state: $jid"
        fi
        if [[ -n $live ]]; then
            scancel "$jid" 2>/dev/null || true
            sleep "$POLL_SECONDS"
            if ! live=$(squeue -h -j "$jid" -o '%A' 2>/dev/null); then
                job_is_terminal "$jid" && live="" \
                    || block "$label state query failed after forced cancellation: $jid"
            elif [[ -z $live ]] && ! job_is_terminal "$jid"; then
                block "$label disappeared after forced cancellation without terminal accounting: $jid"
            fi
            [[ -z $live ]] || block "$label remained live after bounded cancellation: $jid"
        fi
        log "$label stopped after bounded no-progress interval; advancing recovery concurrency"
    fi
    sacct -X -j "$jid" --format=JobIDRaw,State,ExitCode,Elapsed,Account,Partition,QOS -n -P \
        | awk -F '|' '$1 !~ /\./ && NF {print}' || true
}

fast_coverage_count() {
    local count
    [[ -d $DELIVERABLES ]] || { echo 0; return; }
    count=$(find "$DELIVERABLES" -mindepth 3 -maxdepth 3 -type f -name finish_params.json -print 2>/dev/null \
        | awk -F/ 'NF >= 3 && !seen[$(NF-2)]++ {n++} END {print n+0}') || count=0
    printf '%s\n' "$count"
}

coverage_exact() {
    "$PYTHON_BIN" "$CAMPAIGN_PY" coverage --dataset "$DATASET" --deliverables "$DELIVERABLES" \
        --expected-tasks "$EXPECTED_TASKS"
}

remaining_produced() {
    "$PYTHON_BIN" "$E2E_DIR/preconvert_closure.py" inventory --root "$DELIVERABLES" --json \
        | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["remaining_produced"])'
}

validate_preconvert_pass() {
    local current="$RUN_DIR/.preconvert_inventory_current.$$"
    [[ -s $RUN_DIR/PRECONVERT_PASS ]] || return 1
    "$PYTHON_BIN" "$E2E_DIR/preconvert_closure.py" inventory --root "$DELIVERABLES" --json > "$current"
    "$PYTHON_BIN" - "$RUN_DIR/PRECONVERT_PASS" "$current" <<'PY'
import json, sys
saved, current = (json.load(open(path)) for path in sys.argv[1:])
assert saved["status"] == current["status"] == "CLOSED"
assert saved["remaining_produced"] == current["remaining_produced"] == 0
assert saved["closure_fingerprint"] == current["closure_fingerprint"]
assert saved["produced_pairs"] == current["produced_pairs"]
PY
    local rc=$?
    rm -f "$current"
    return "$rc"
}

publish_preconvert_pass() {
    local temporary="$RUN_DIR/PRECONVERT_PASS.tmp.$$"
    "$PYTHON_BIN" "$E2E_DIR/preconvert_closure.py" inventory --root "$DELIVERABLES" --json > "$temporary"
    "$PYTHON_BIN" - "$temporary" <<'PY'
import json, sys
document = json.load(open(sys.argv[1]))
assert document["status"] == "CLOSED" and document["remaining_produced"] == 0
PY
    chmod 0400 "$temporary"
    mv -f "$temporary" "$RUN_DIR/PRECONVERT_PASS"
}

validate_transport_prebuild_pass() {
    local receipt="$RUN_DIR/TRANSPORT_PREBUILD_PASS"
    local manifest="$TRANSPORT_VIEW_ROOT/manifest.json" expected actual
    [[ -f $receipt && ! -L $receipt && -s $receipt ]] || return 1
    [[ -f $manifest && ! -L $manifest && -s $manifest ]] || return 1
    [[ $(sed -n '1p' "$receipt") == schema=gdpval.transport-prebuild.v1 ]] || return 1
    [[ $(sed -n '2p' "$receipt") == "manifest=$manifest" ]] || return 1
    expected=$(sed -n 's/^manifest_sha256=//p' "$receipt")
    [[ $expected =~ ^[0-9a-f]{64}$ && $(awk 'END {print NR + 0}' "$receipt") == 3 ]] || return 1
    actual=$(sha256sum "$manifest" | awk '{print $1}') || return 1
    [[ $actual == "$expected" ]] || return 1
    configure_gdpval_container_python \
        "$APPTAINER_BIN/apptainer" "$GDPVAL_SIF" "$E2E_DIR" "$RUN_DIR" "$REFERENCE_OVERLAY" \
        || return 1
    "${GDPVAL_CONTAINER_PYTHON[@]}" "$TRANSPORT_VIEWS_PY" validate \
        --output-root "$TRANSPORT_VIEW_ROOT" >/dev/null
}

strict_result() {
    "$PYTHON_BIN" "$CAMPAIGN_PY" result \
        --output "$RUN_DIR/judge_e2e/gdpval_aav2.jsonl" \
        --journal "$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl" \
        --dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS" --json
}

repair_judge_state() {
    local output="$RUN_DIR/judge_e2e/gdpval_aav2.jsonl"
    local journal="$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl"
    local report status post
    [[ -e $output || -e $journal ]] || return 0
    [[ -s $journal && -f $journal && ! -L $journal ]] \
        || block "judge output exists without a usable multistage journal: $journal"
    if [[ ! -e $output && ! -L $output ]]; then
        : > "$output"
        chmod 0600 "$output"
        log "created empty judge output beside an existing journal for safe resume"
    fi
    [[ -f $output && ! -L $output ]] || block "judge output is not a regular file: $output"

    # `sanitize` also recovers an interrupted prior transaction. A clean run is
    # scanned once; only a repaired/recovered run receives the second audit.
    report=$("$PYTHON_BIN" "$JUDGE_STATE_PY" sanitize \
        --output "$output" --journal "$journal" --deliverables "$JUDGE_CACHE_ROOT" \
        --state-root "$RUN_DIR/judge_e2e/.checkpoint_e2e_judge_state" \
        --max-attempts 3) || block "judge-state repair failed"
    status=$(printf '%s\n' "$report" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])') \
        || block "judge-state repair emitted an invalid report"
    case "$status" in
        CLEAN)
            return 0
            ;;
        SANITIZED|RECOVERED)
            log "judge state $status; invalid cached evidence was quarantined for exact retry"
            post=$("$PYTHON_BIN" "$JUDGE_STATE_PY" audit \
                --output "$output" --journal "$journal" --deliverables "$JUDGE_CACHE_ROOT" \
                --max-attempts 3) || block "post-repair judge-state audit failed"
            [[ $(printf '%s\n' "$post" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])') == CLEAN ]] \
                || block "judge-state repair did not leave a clean active cache"
            ;;
        *)
            block "unexpected judge-state repair status: $status"
            ;;
    esac
}

"$E2E_DIR/run_checkpoint_e2e.sh" _compute-preflight "$RUN_DIR" \
    || block "exact compute preflight did not pass"
[[ -s $RUN_DIR/FLEET_JOBS.tsv ]] || block "initial fleet receipt is missing"

initial_ids="$(ids_from_tsv "$RUN_DIR/FLEET_JOBS.tsv")"
wait_jobs initial_fleet "$initial_ids"

round=1
while ! coverage_exact; do
    (( round <= MAX_RECOVERY_ROUNDS )) \
        || block "rollout coverage remained incomplete after $MAX_RECOVERY_ROUNDS recovery rounds"
    recovery="$RUN_DIR/recovery_r${round}"
    if [[ ! -s $recovery/JOBS.tsv && ! -e $recovery/NOT_NEEDED ]]; then
        "$E2E_DIR/run_checkpoint_e2e.sh" _submit-recovery "$RUN_DIR" "$round" \
            | tee "$RUN_DIR/recovery_r${round}_submit.log"
    fi
    if [[ -e $recovery/NOT_NEEDED ]]; then
        coverage_exact || block "recovery reported no residue but coverage is incomplete"
        break
    fi
    [[ -s $recovery/JOBS.tsv ]] || block "recovery round $round has no job receipt"
    wait_jobs "recovery_r${round}" "$(ids_from_tsv "$recovery/JOBS.tsv")"
    round=$((round + 1))
done

coverage_exact
if [[ -e $RUN_DIR/ROLLOUT_COVERAGE_PASS || -L $RUN_DIR/ROLLOUT_COVERAGE_PASS ]]; then
    [[ -f $RUN_DIR/ROLLOUT_COVERAGE_PASS && ! -L $RUN_DIR/ROLLOUT_COVERAGE_PASS ]] \
        || block "rollout coverage receipt is not a regular file"
else
    : > "$RUN_DIR/ROLLOUT_COVERAGE_PASS"
    chmod 0400 "$RUN_DIR/ROLLOUT_COVERAGE_PASS"
fi
log "rollout coverage PASS 220/220"

if [[ -e $RUN_DIR/PRECONVERT_PASS ]] && ! validate_preconvert_pass; then
    block "Office conversion receipt no longer matches the produced deliverables"
fi
if [[ ! -e $RUN_DIR/PRECONVERT_PASS ]]; then
    before=$(remaining_produced)
    no_progress=0
    attempt=$(latest_preconvert_attempt)
    (( attempt <= MAX_PRECONVERT_ATTEMPTS )) \
        || block "preconvert receipt count exceeds configured maximum: $attempt/$MAX_PRECONVERT_ATTEMPTS"
    if (( attempt > 0 )); then
        job_file="$RUN_DIR/PRECONVERT_JOB_${attempt}"
        jid=$(read_preconvert_job_id "$job_file")
        if judge_job_requires_wait "$jid"; then
            log "reattaching newest preconvert attempt=$attempt job=$jid before submitting another attempt"
            wait_jobs "preconvert_${attempt}_reattach" "$jid"
            after=$(remaining_produced)
            log "preconvert attempt=$attempt remaining=$after before=$before"
            (( after < before )) || no_progress=1
            before=$after
        fi
    fi
    while (( before > 0 && attempt < MAX_PRECONVERT_ATTEMPTS )); do
        attempt=$((attempt + 1))
        job_file="$RUN_DIR/PRECONVERT_JOB_${attempt}"
        [[ ! -e $job_file && ! -L $job_file ]] \
            || block "refusing to replace existing preconvert receipt: $job_file"
        job_name="gdp-${RUN_ID:0:22}-pc${attempt}"
        jid=$(slurm_submit_or_adopt "$job_file" "preconvert-a${attempt}" "$job_name" \
            --parsable -J "$job_name" \
            -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
            -o "$RUN_DIR/logs/%j_preconvert.out" -e "$RUN_DIR/logs/%j_preconvert.err" \
            --export=ALL,RUN_DIR="$RUN_DIR",E2E_DIR="$E2E_DIR",PRECONVERT_ATTEMPT="$attempt" \
            "$E2E_DIR/preconvert_closure.sbatch") \
            || block "could not submit or adopt preconvert attempt=$attempt"
        wait_jobs "preconvert_${attempt}" "$jid"
        after=$(remaining_produced)
        log "preconvert attempt=$attempt remaining=$after before=$before"
        (( after > 0 )) || break
        if (( after >= before )); then
            no_progress=$((no_progress + 1))
        else
            no_progress=0
        fi
        (( no_progress < 2 )) || block "Office conversion made no progress twice; remaining=$after"
        before=$after
    done
    [[ $(remaining_produced) == 0 ]] || block "model-produced Office files remain unrendered"
    publish_preconvert_pass
fi
log "Office conversion PASS"

if [[ -e $RUN_DIR/TRANSPORT_PREBUILD_PASS || -L $RUN_DIR/TRANSPORT_PREBUILD_PASS ]]; then
    validate_transport_prebuild_pass \
        || block "transport prebuild pass receipt or immutable view no longer validates"
else
    attempt=$(latest_transport_prebuild_attempt)
    if (( attempt > 0 )); then
        job_file="$RUN_DIR/TRANSPORT_PREBUILD_JOB_${attempt}"
        jid=$(read_transport_prebuild_job_id "$job_file")
        log "reattaching newest transport prebuild attempt=$attempt job=$jid before submitting another attempt"
        wait_jobs "transport_prebuild_${attempt}_reattach" "$jid"
    fi
    while ! validate_transport_prebuild_pass && (( attempt < MAX_TRANSPORT_PREBUILD_ATTEMPTS )); do
        attempt=$((attempt + 1))
        job_file="$RUN_DIR/TRANSPORT_PREBUILD_JOB_${attempt}"
        [[ ! -e $job_file && ! -L $job_file ]] \
            || block "refusing to replace existing transport prebuild receipt: $job_file"
        job_name="gdp-${RUN_ID:0:20}-tv${attempt}"
        jid=$(slurm_submit_or_adopt "$job_file" "transport-a${attempt}" "$job_name" \
            --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
            -o "$RUN_DIR/logs/%j_transport.out" -e "$RUN_DIR/logs/%j_transport.err" \
            --export=RUN_DIR="$RUN_DIR",E2E_DIR="$E2E_DIR" \
            "$E2E_DIR/transport_prebuild.sbatch") \
            || block "could not submit or adopt transport prebuild attempt=$attempt"
        wait_jobs "transport_prebuild_${attempt}" "$jid"
    done
    validate_transport_prebuild_pass \
        || block "transport views failed validation after $MAX_TRANSPORT_PREBUILD_ATTEMPTS attempts"
fi
log "transport prebuild PASS"

if [[ $AUTHORIZE != true ]]; then
    : > "$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED"
    chmod 0400 "$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED"
    log "judge authorization required; resume with CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true"
    exit 0
fi
rm -f "$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED"

mkdir -p "$RUN_DIR/judge_e2e"
stages_file="$RUN_DIR/judge_e2e/stages.txt"
if [[ ! -e $stages_file ]]; then
    printf '%s' '[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1, waivable_failure_classes: [timeout_exceeded, transient]}}, {num_tasks: 220, num_models: 4}]' > "$stages_file"
    chmod 0400 "$stages_file"
fi

attempt=$(latest_judge_attempt)
result_pass=0
if (( attempt > 0 )); then
    newest_receipt="$RUN_DIR/JUDGE_JOB_${attempt}"
    newest_jid=$(read_judge_job_id "$newest_receipt")
    if judge_job_requires_wait "$newest_jid"; then
        log "reattaching newest judge attempt=$attempt job=$newest_jid before submitting another attempt"
        wait_judge_job "judge_attempt${attempt}_reattach" "$newest_jid"
    fi
fi

repair_judge_state
if strict_result > "$RUN_DIR/final_receipt.tmp" 2>/dev/null; then
    result_pass=1
fi

if (( result_pass == 0 )); then
    while (( attempt < MAX_JUDGE_ATTEMPTS )); do
        attempt=$((attempt + 1))
        concurrency=${JUDGE_CONCURRENCIES[attempt - 1]}
        job_file="$RUN_DIR/JUDGE_JOB_${attempt}"
        [[ ! -e $job_file && ! -L $job_file ]] \
            || block "refusing to replace existing judge receipt: $job_file"
        job_name="gdp-${RUN_ID:0:22}-j${concurrency}"
        jid=$(slurm_submit_or_adopt "$job_file" "judge-a${attempt}" "$job_name" \
            --parsable -J "$job_name" \
            -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
            -t "$JUDGE_WALL" --cpus-per-task=32 --mem=184G \
            -o "$RUN_DIR/logs/%j_judge.out" -e "$RUN_DIR/logs/%j_judge.err" \
            --export=ALL,RUN_DIR="$RUN_DIR",JUDGE_DIR_SUFFIX=e2e,STAGES_FILE="$stages_file",DATASET="$DATASET",CONCURRENCY="$concurrency",NUM_COMPARISON_TRIALS=4,NUM_REPEATS=1,MULTISTAGE_SEED=42,MODEL_NAME="$MODEL_NAME" \
            "$JUDGE_SBATCH") \
            || block "could not submit or adopt judge attempt=$attempt"
        wait_judge_job "judge_attempt${attempt}_c${concurrency}" "$jid"
        repair_judge_state
        if strict_result > "$RUN_DIR/final_receipt.tmp" 2>/dev/null; then
            result_pass=1
            break
        fi
    done
fi

if (( result_pass == 0 )); then
    strict_result > "$RUN_DIR/final_receipt.tmp" \
        || block "accepted Stage 0 plus exact 220-task/880-vote Stage 1 failed after concurrency 16->16->8->4"
fi
CHECKPOINT_E2E_REHASH_REFERENCE_ASSETS=true \
    "$E2E_DIR/run_checkpoint_e2e.sh" _compute-preflight "$RUN_DIR" \
    || block "final immutable-input preflight did not pass"
coverage_exact || block "final rollout coverage recheck did not pass"
chmod 0400 "$RUN_DIR/final_receipt.tmp"
mv -f "$RUN_DIR/final_receipt.tmp" "$RUN_DIR/final_receipt.json"
sha256sum "$RUN_DIR/final_receipt.json" > "$RUN_DIR/final_receipt.json.sha256"
chmod 0400 "$RUN_DIR/final_receipt.json.sha256"
: > "$RUN_DIR/CAMPAIGN_COMPLETE"
chmod 0400 "$RUN_DIR/CAMPAIGN_COMPLETE"
rm -f "$RUN_DIR/CONTROLLER_BLOCKED"
log "CAMPAIGN_COMPLETE $(<"$RUN_DIR/final_receipt.json")"
