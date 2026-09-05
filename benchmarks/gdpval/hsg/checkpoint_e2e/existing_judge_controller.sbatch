#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CPU-only controller for a frozen external-deliverables import.
#SBATCH --job-name=gdpval-existing-judge
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
export PYTHONDONTWRITEBYTECODE=1
SAFE_PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=$SAFE_PATH

: "${RUN_DIR:?set RUN_DIR}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"
[[ -f $RUN_DIR/settings.env && ! -L $RUN_DIR/settings.env \
    && -f $RUN_DIR/existing_judge.env && ! -L $RUN_DIR/existing_judge.env ]] || {
    echo "EXISTING_CONTROLLER_FAIL: prepared settings are missing" >&2
    exit 64
}
RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] || {
    echo "EXISTING_CONTROLLER_FAIL: settings contain an invalid run id" >&2
    exit 64
}
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "EXISTING_CONTROLLER_FAIL: invalid Slurm user" >&2; exit 64;
}
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-controller
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]] || {
    echo "EXISTING_CONTROLLER_FAIL: bootstrap root resolves to Lustre" >&2
    exit 64
}
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" controller "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
mars_stage_file "$RUN_DIR/existing_judge.env" "$MARS_JOB_ROOT/existing_judge.env"
# Generated owner-only settings copied to node-local storage before sourcing.
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
CAMPAIGN_E2E_SCRIPT=$E2E_SCRIPT
CAMPAIGN_E2E_DIR=$E2E_DIR
CAMPAIGN_JUDGE_RUNTIME_OVERLAY=$JUDGE_RUNTIME_OVERLAY
CAMPAIGN_TRANSPORT_VIEW_ROOT=$TRANSPORT_VIEW_ROOT
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/existing_judge.env"
DURABLE_ACTIVE_PACKAGE=$ACTIVE_PACKAGE
DURABLE_CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY
mars_stage_gym "$GYM_ROOT" "$GYM_REVISION"
mars_stage_runtime
mars_stage_file "$REFERENCE_OVERLAY" "$MARS_RUN_ROOT/config/reference_paths.yaml"
mars_stage_file "$ENV_FILE" "$MARS_RUN_ROOT/config/aav2.env"
E2E_DIR=$MARS_PACKAGE
PYTHON_BIN=$MARS_PYTHON
GYM_ROOT=$MARS_GYM
CORRECTED_RUNTIME_OVERLAY=$MARS_RUNTIME
REFERENCE_OVERLAY=$MARS_RUN_ROOT/config/reference_paths.yaml
ENV_FILE=$MARS_RUN_ROOT/config/aav2.env
LOCAL_COMPONENT_VENVS=$MARS_RUN_ROOT/component_venvs
LOCAL_DATASET_OVERLAY=$MARS_RUN_ROOT/config/dataset_overlay.yaml
LOCAL_REFERENCE_VIEW_OVERLAY=$MARS_RUN_ROOT/config/reference_views.yaml
CAMPAIGN_E2E_SCRIPT=$E2E_DIR/run_checkpoint_e2e.sh
export CHECKPOINT_E2E_LOCAL_PACKAGE=$E2E_DIR
export CHECKPOINT_E2E_LOCAL_GYM=$GYM_ROOT
export CHECKPOINT_E2E_LOCAL_RUNTIME=$CORRECTED_RUNTIME_OVERLAY
export CHECKPOINT_E2E_LOCAL_REFERENCE_OVERLAY=$REFERENCE_OVERLAY
export CHECKPOINT_E2E_LOCAL_ENV_FILE=$ENV_FILE
export CHECKPOINT_E2E_LOCAL_COMPONENT_VENVS=$LOCAL_COMPONENT_VENVS
export CHECKPOINT_E2E_LOCAL_DATASET_OVERLAY=$LOCAL_DATASET_OVERLAY
export CHECKPOINT_E2E_LOCAL_REFERENCE_VIEW_OVERLAY=$LOCAL_REFERENCE_VIEW_OVERLAY
export CHECKPOINT_E2E_LOCAL_SETTINGS=$MARS_JOB_ROOT/settings.env
export CHECKPOINT_E2E_LOCAL_EXISTING_SETTINGS=$MARS_JOB_ROOT/existing_judge.env
# shellcheck disable=SC1090
source "$E2E_DIR/slurm_receipts.sh"

AUTHORIZE=${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}
POLL_SECONDS=${CHECKPOINT_E2E_POLL_SECONDS:-60}
JUDGE_NO_PROGRESS_SECONDS=${CHECKPOINT_E2E_JUDGE_NO_PROGRESS_SECONDS:-1800}
JUDGE_CANCEL_GRACE_SECONDS=${CHECKPOINT_E2E_JUDGE_CANCEL_GRACE_SECONDS:-300}
PERSISTENT_JUDGE_SESSION=${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:-false}
ROTATION=${SLURM_RESTART_COUNT:-0}
MAX_ROTATIONS=${CHECKPOINT_E2E_MAX_CONTROLLER_REQUEUES:-2}
MAX_PRECONVERT_ATTEMPTS=4
MAX_TRANSPORT_ATTEMPTS=2
MAX_JUDGE_ATTEMPTS=8
# Attempts 1-4 are the original bounded ladder. A second identical ladder is
# reserved for an explicitly corrected immutable execution package.
JUDGE_CONCURRENCIES=(16 8 4 1 16 8 4 1)
JUDGE_DIR=$RUN_DIR/judge_$JUDGE_DIR_SUFFIX
OUTPUT=$JUDGE_DIR/gdpval_aav2.jsonl
JOURNAL=$JUDGE_DIR/gdpval_aav2_multistage_state.jsonl
JUDGE_CACHE_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT/candidate
STATE_ROOT=$JUDGE_DIR/.checkpoint_e2e_judge_state
FINAL_RECEIPT=$RUN_DIR/final_receipt_$JUDGE_DIR_SUFFIX.json
FINAL_SHA=$FINAL_RECEIPT.sha256
FINAL_MARKER=$RUN_DIR/CAMPAIGN_${JUDGE_DIR_SUFFIX}_COMPLETE

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
block() {
    log "BLOCKED: $*"
    temporary="$RUN_DIR/.EXISTING_JUDGE_BLOCKED.$$"
    printf '%s\n' "$*" > "$temporary"
    chmod 0400 "$temporary"
    mv -f "$temporary" "$RUN_DIR/EXISTING_JUDGE_BLOCKED"
    exit 1
}

[[ $AUTHORIZE == true || $AUTHORIZE == false ]] || block "invalid provider authorization value"
[[ ${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-} =~ ^[1-4]$ ]] \
    || block "Gemini concurrency must be an integer from 1 through 4"
export GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS
[[ $PERSISTENT_JUDGE_SESSION == true || $PERSISTENT_JUDGE_SESSION == false ]] \
    || block "invalid persistent judge-session opt-in"
[[ $ACCOUNT == nemotron_n3_post ]] || block "campaign account is not nemotron_n3_post: $ACCOUNT"
[[ $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]] \
    || block "unexpected CPU routing: $CPU_PARTITION/$CPU_QOS"
[[ $DURABLE_ACTIVE_PACKAGE == "$RUN_DIR/existing_judge_package" \
    && -d $DURABLE_ACTIVE_PACKAGE && ! -L $DURABLE_ACTIVE_PACKAGE \
    && $DURABLE_ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR" ]] || block "run-owned active package is invalid"
[[ $DURABLE_CORRECTED_RUNTIME_OVERLAY == "$RUN_DIR"/* \
    && -d $DURABLE_CORRECTED_RUNTIME_OVERLAY && ! -L $DURABLE_CORRECTED_RUNTIME_OVERLAY \
    && $DURABLE_CORRECTED_RUNTIME_OVERLAY != "$CAMPAIGN_JUDGE_RUNTIME_OVERLAY" ]] \
    || block "corrected runtime is invalid"
[[ $E2E_DIR == /raid/scratch/* && $GYM_ROOT == /raid/scratch/* \
    && $CORRECTED_RUNTIME_OVERLAY == /raid/scratch/* ]] \
    || block "execution code was not staged node-locally"
[[ $CORRECTED_TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* \
    && ! -L $CORRECTED_TRANSPORT_VIEW_ROOT \
    && $CORRECTED_TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT" ]] \
    || block "corrected transport root is invalid"
[[ $JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && ${#JUDGE_DIR_SUFFIX} -le 32 && $JUDGE_DIR_SUFFIX != e2e ]] \
    || block "judge suffix is invalid"
[[ $POLL_SECONDS =~ ^[1-9][0-9]*$ && $POLL_SECONDS -le 300 \
    && $JUDGE_NO_PROGRESS_SECONDS =~ ^[1-9][0-9]*$ && $JUDGE_NO_PROGRESS_SECONDS -le 7200 \
    && $JUDGE_CANCEL_GRACE_SECONDS =~ ^[1-9][0-9]*$ && $JUDGE_CANCEL_GRACE_SECONDS -le 900 \
    && $ROTATION =~ ^[0-9]+$ && $MAX_ROTATIONS =~ ^[0-9]+$ && $MAX_ROTATIONS -le 4 ]] \
    || block "controller retry/poll contract is invalid"

exec 9> "$RUN_DIR/existing_judge_controller.lock"
flock -n 9 || block "another import-only controller owns this run"
rm -f "$RUN_DIR/EXISTING_JUDGE_BLOCKED"

on_term() {
    trap - TERM
    (( ROTATION < MAX_ROTATIONS )) || block "controller walltime retry bound reached"
    log "controller walltime signal; requeueing rotation=$ROTATION/$MAX_ROTATIONS"
    scontrol requeue "$SLURM_JOB_ID" || block "controller requeue failed"
    exit 0
}
trap on_term TERM

job_state() {
    local job=$1 state
    state=$(sacct -X -j "$job" --format=JobIDRaw,State -n -P 2>/dev/null \
        | awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}') || return 1
    state=${state%%+*}; state=${state%% *}
    printf '%s\n' "$state"
}

job_exit_code() {
    local job=$1 value
    value=$(sacct -X -j "$job" --format=JobIDRaw,ExitCode -n -P 2>/dev/null \
        | awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}') || return 1
    value=${value%%:*}
    [[ $value =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$value"
}

is_terminal_state() {
    [[ $1 =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)$ ]]
}

wait_job() {
    local label=$1 job=$2 require_success=${3:-false} queue state
    [[ $job =~ ^[1-9][0-9]*$ ]] || block "$label has invalid job id: $job"
    while :; do
        queue=$(squeue -h -j "$job" -o '%A' 2>/dev/null) || queue=query_failed
        if [[ $queue != query_failed && -z $queue ]]; then
            state=$(job_state "$job") || state=
            if [[ -n $state ]] && is_terminal_state "$state"; then
                break
            fi
        fi
        log "$label job=$job state=${state:-pending}"
        sleep "$POLL_SECONDS"
    done
    sacct -X -j "$job" --format=JobIDRaw,State,ExitCode,Elapsed,Account,Partition,QOS -n -P \
        | awk -F '|' '$1 !~ /\./ && NF {print}' || true
    if [[ $require_success == true && $state != COMPLETED ]]; then
        block "$label ended in $state"
    fi
}

judge_progress_signature() {
    local output_bytes=0 journal_bytes=0
    [[ ! -f $OUTPUT ]] || output_bytes=$(wc -c < "$OUTPUT")
    [[ ! -f $JOURNAL ]] || journal_bytes=$(wc -c < "$JOURNAL")
    printf '%s:%s\n' "$output_bytes" "$journal_bytes"
}

wait_judge_job() {
    local label=$1 job=$2 queue queue_state state signature previous='' now last_progress grace_start
    last_progress=$(date +%s)
    while :; do
        queue=$(squeue -h -j "$job" -o '%A|%T' 2>/dev/null) || queue=query_failed
        if [[ $queue != query_failed && -z $queue ]]; then
            state=$(job_state "$job") || state=
            if [[ -n $state ]] && is_terminal_state "$state"; then
                break
            fi
        fi
        signature=$(judge_progress_signature)
        now=$(date +%s)
        queue_state=
        if [[ $queue != query_failed ]]; then
            queue_state=$(awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}' <<<"$queue")
        fi
        if [[ $queue_state != RUNNING ]]; then
            # Queue wait, configuration, completion, and transient scheduler
            # query failures are not evidence of a stuck provider process.
            previous=$signature
            last_progress=$now
            log "$label job=$job scheduler_state=${queue_state:-unknown} progress=$signature"
            sleep "$POLL_SECONDS"
            continue
        fi
        if [[ $signature != "$previous" ]]; then
            previous=$signature
            last_progress=$now
        elif (( now - last_progress >= JUDGE_NO_PROGRESS_SECONDS )); then
            log "$label no progress for $((now-last_progress))s; sending TERM to job=$job"
            scancel --signal=TERM "$job" 2>/dev/null || true
            grace_start=$now
            while (( $(date +%s) - grace_start < JUDGE_CANCEL_GRACE_SECONDS )); do
                state=$(job_state "$job") || state=
                [[ -n $state ]] && is_terminal_state "$state" && break
                sleep "$POLL_SECONDS"
            done
            [[ -n ${state:-} ]] && is_terminal_state "$state" || scancel "$job" 2>/dev/null || true
            wait_job "$label-cancel" "$job" false
            return
        fi
        log "$label job=$job progress=$signature"
        sleep "$POLL_SECONDS"
    done
    sacct -X -j "$job" --format=JobIDRaw,State,ExitCode,Elapsed,Account,Partition,QOS -n -P \
        | awk -F '|' '$1 !~ /\./ && NF {print}' || true
}

latest_attempt() {
    local prefix=$1 maximum=$2 path name number latest=0 expected
    for path in "$RUN_DIR"/${prefix}_[0-9]*; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || block "$prefix receipt is not regular: $path"
        name=${path##*/}; number=${name##*_}
        [[ $number =~ ^[1-9][0-9]*$ && $number -le $maximum ]] \
            || block "$prefix receipt index is invalid: $name"
        (( number > latest )) && latest=$number
    done
    for (( expected=1; expected<=latest; expected++ )); do
        [[ -f $RUN_DIR/${prefix}_${expected} && ! -L $RUN_DIR/${prefix}_${expected} ]] \
            || block "$prefix receipt sequence has a gap before $expected"
    done
    printf '%s\n' "$latest"
}

publish_immutable() {
    local source=$1 target=$2
    chmod 0400 "$source"
    if [[ -e $target || -L $target ]]; then
        [[ -f $target && ! -L $target && $(stat -c '%a' "$target") == 400 ]] \
            || block "immutable target is invalid: $target"
        cmp -s "$source" "$target" || block "immutable target drift: $target"
        rm -f "$source"
    else
        mv "$source" "$target"
    fi
}

verify_import() {
    [[ -f $RUN_DIR/existing_import_receipt.json \
        && ! -L $RUN_DIR/existing_import_receipt.json \
        && -f $RUN_DIR/existing_import_receipt.json.sha256 \
        && ! -L $RUN_DIR/existing_import_receipt.json.sha256 \
        && -d $DELIVERABLES && ! -L $DELIVERABLES ]] \
        || block "immutable external-deliverables import receipts are missing"
    [[ $(sha256sum "$RUN_DIR/existing_import_receipt.json" | awk '{print $1}') == "$IMPORT_RECEIPT_SHA256" ]] \
        || block "import receipt hash drift"
    sha256sum -c --status "$RUN_DIR/existing_import_receipt.json.sha256" \
        || block "import receipt sidecar verification failed"
    "$PYTHON_BIN" "$E2E_DIR/campaign.py" coverage \
        --dataset "$DATASET" --deliverables "$DELIVERABLES" --expected-tasks 220 >/dev/null \
        || block "imported deliverables do not have exact 220-task terminal coverage"
}

office_inventory() {
    "$PYTHON_BIN" "$E2E_DIR/preconvert_closure.py" inventory --root "$DELIVERABLES"
}

remaining_office() {
    office_inventory | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin)["produced_missing"]))'
}

transport_valid() {
    local receipt=$RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX manifest digest
    [[ -f $receipt && ! -L $receipt ]] || return 1
    [[ $(sed -n 's/^schema=//p' "$receipt") == gdpval.transport-prebuild.v1 ]] || return 1
    manifest=$(sed -n 's/^manifest=//p' "$receipt")
    digest=$(sed -n 's/^manifest_sha256=//p' "$receipt")
    [[ $manifest == "$CORRECTED_TRANSPORT_VIEW_ROOT/manifest.json" \
        && $digest =~ ^[0-9a-f]{64}$ && -f $manifest && ! -L $manifest \
        && $(sha256sum "$manifest" | awk '{print $1}') == "$digest" ]]
}

repair_judge_state() {
    [[ -e $JOURNAL || -L $JOURNAL ]] || return 0
    [[ -f $JOURNAL && ! -L $JOURNAL && -s $JOURNAL ]] \
        || block "judge journal is invalid"
    if [[ ! -e $OUTPUT && ! -L $OUTPUT ]]; then
        : > "$OUTPUT"; chmod 0600 "$OUTPUT"
    fi
    report=$("$PYTHON_BIN" "$E2E_DIR/judge_state.py" sanitize \
        --output "$OUTPUT" --journal "$JOURNAL" --deliverables "$JUDGE_CACHE_ROOT" \
        --state-root "$STATE_ROOT" --max-attempts 3) \
        || block "judge-state repair failed"
    status=$(printf '%s' "$report" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])')
    case "$status" in CLEAN|SANITIZED|RECOVERED) log "judge-state=$status" ;; *) block "invalid judge-state=$status" ;; esac
}

strict_result() {
    "$PYTHON_BIN" "$E2E_DIR/campaign.py" result --output "$OUTPUT" --journal "$JOURNAL" \
        --dataset "$DATASET" --expected-tasks 220 --json
}

validate_final() {
    [[ -f $FINAL_MARKER && ! -L $FINAL_MARKER && ! -s $FINAL_MARKER \
        && $(stat -c '%a' "$FINAL_MARKER") == 400 \
        && -f $FINAL_RECEIPT && ! -L $FINAL_RECEIPT && $(stat -c '%a' "$FINAL_RECEIPT") == 400 \
        && -f $FINAL_SHA && ! -L $FINAL_SHA && $(stat -c '%a' "$FINAL_SHA") == 400 ]] \
        || return 1
    sha256sum -c --status "$FINAL_SHA" || return 1
    [[ $(strict_result) == "$(<"$FINAL_RECEIPT")" ]] || return 1
    "$PYTHON_BIN" "$E2E_DIR/prepare_existing_campaign.py" verify-envelope \
        --run-dir "$RUN_DIR" --suffix "$JUDGE_DIR_SUFFIX" >/dev/null
}

verify_import
if [[ -e $FINAL_MARKER || -L $FINAL_MARKER ]]; then
    validate_final || block "published final receipt no longer validates"
    log "import-only campaign already complete"
    exit 0
fi

# Office closure: null finish markers remain untouched as terminal
# no-submission rollouts. The ordinary live judge runtime decides their score.
attempt=$(latest_attempt EXISTING_PRECONVERT_JOB "$MAX_PRECONVERT_ATTEMPTS")
if (( attempt > 0 )); then
    job=$(slurm_read_job_receipt "$RUN_DIR/EXISTING_PRECONVERT_JOB_${attempt}") \
        || block "preconvert receipt is invalid"
    state=$(job_state "$job") || state=
    [[ -n $state ]] && is_terminal_state "$state" || wait_job "preconvert-$attempt" "$job" false
fi
while [[ $(remaining_office) != 0 && $attempt -lt $MAX_PRECONVERT_ATTEMPTS ]]; do
    attempt=$((attempt + 1))
    receipt=$RUN_DIR/EXISTING_PRECONVERT_JOB_$attempt
    job_name="gdp-${RUN_ID:0:18}-expc${attempt}"
    job=$(slurm_submit_or_adopt "$receipt" "existing-preconvert-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
        -o "$RUN_DIR/logs/%j_existing_preconvert.out" -e "$RUN_DIR/logs/%j_existing_preconvert.err" \
        --export=RUN_DIR="$RUN_DIR",CHECKPOINT_E2E_ACTIVE_PACKAGE="$DURABLE_ACTIVE_PACKAGE",CHECKPOINT_E2E_EXECUTION_PACKAGE="$CHECKPOINT_E2E_EXECUTION_PACKAGE",PRECONVERT_ATTEMPT="$attempt",PATH="$SAFE_PATH" \
        "$E2E_DIR/preconvert_closure.sbatch") || block "could not submit preconvert attempt $attempt"
    wait_job "preconvert-$attempt" "$job" false
done
[[ $(remaining_office) == 0 ]] || block "Office closure remained incomplete after four attempts"
office_tmp=$RUN_DIR/PRECONVERT_PASS_EXISTING.tmp.$$
office_inventory > "$office_tmp"
publish_immutable "$office_tmp" "$RUN_DIR/PRECONVERT_PASS_EXISTING"
verify_import

# Provider-free transport view publication.
attempt=$(latest_attempt EXISTING_TRANSPORT_JOB "$MAX_TRANSPORT_ATTEMPTS")
if (( attempt > 0 )) && ! transport_valid; then
    job=$(slurm_read_job_receipt "$RUN_DIR/EXISTING_TRANSPORT_JOB_${attempt}") \
        || block "transport receipt is invalid"
    state=$(job_state "$job") || state=
    [[ -n $state ]] && is_terminal_state "$state" || wait_job "transport-$attempt" "$job" false
fi
while ! transport_valid && (( attempt < MAX_TRANSPORT_ATTEMPTS )); do
    attempt=$((attempt + 1))
    receipt=$RUN_DIR/EXISTING_TRANSPORT_JOB_$attempt
    job_name="gdp-${RUN_ID:0:18}-extv${attempt}"
    job=$(slurm_submit_or_adopt "$receipt" "existing-transport-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
        -t 04:00:00 --cpus-per-task=4 --mem=32G \
        -o "$RUN_DIR/logs/%j_existing_transport.out" -e "$RUN_DIR/logs/%j_existing_transport.err" \
        --export=RUN_DIR="$RUN_DIR",CHECKPOINT_E2E_ACTIVE_PACKAGE="$DURABLE_ACTIVE_PACKAGE",CHECKPOINT_E2E_EXECUTION_PACKAGE="$CHECKPOINT_E2E_EXECUTION_PACKAGE",CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT="$CORRECTED_TRANSPORT_VIEW_ROOT",TRANSPORT_RECEIPT_SUFFIX="$JUDGE_DIR_SUFFIX",PATH="$SAFE_PATH" \
        "$E2E_DIR/transport_prebuild.sbatch") || block "could not submit transport attempt $attempt"
    wait_job "transport-$attempt" "$job" false
done
transport_valid || block "transport view did not validate after two attempts"
mars_stage_file "$CORRECTED_TRANSPORT_VIEW_ROOT/reference_views.yaml" "$LOCAL_REFERENCE_VIEW_OVERLAY"

# Exact resolved-config fingerprint, still provider-free.
bash "$E2E_DIR/prepare_existing_fingerprint.sh" "$RUN_DIR" \
    || block "provider-free fingerprint gate failed"
FINGERPRINT_RECEIPT=$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json
EXPECTED_JUDGE_FINGERPRINT=$("$PYTHON_BIN" -c \
    'import json,sys,re; v=json.load(open(sys.argv[1]))["fingerprint"]; assert re.fullmatch(r"[0-9a-f]{64}",v); print(v)' \
    "$FINGERPRINT_RECEIPT") || block "fingerprint receipt is invalid"

if [[ $AUTHORIZE != true ]]; then
    temporary=$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED.tmp.$$
    : > "$temporary"; chmod 0400 "$temporary"
    publish_immutable "$temporary" "$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED"
    log "provider authorization required; resume with CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true"
    exit 0
fi
rm -f "$RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED"

mkdir -p "$JUDGE_DIR"
STAGES_FILE=$JUDGE_DIR/stages.txt
EXPECTED_STAGES='[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1, waivable_failure_classes: [timeout_exceeded, transient]}}, {num_tasks: 220, num_models: 4}]'
stages_tmp=$STAGES_FILE.tmp.$$
printf '%s' "$EXPECTED_STAGES" > "$stages_tmp"; chmod 0400 "$stages_tmp"
if [[ -e $STAGES_FILE || -L $STAGES_FILE ]]; then
    [[ -f $STAGES_FILE && ! -L $STAGES_FILE && $(stat -c '%a' "$STAGES_FILE") == 400 ]] \
        || block "stages file is invalid"
    cmp -s "$stages_tmp" "$STAGES_FILE" || block "stages file drift"
    rm -f "$stages_tmp"
else
    mv "$stages_tmp" "$STAGES_FILE"
fi

attempt=$(latest_attempt EXISTING_JUDGE_JOB "$MAX_JUDGE_ATTEMPTS")
last_judge_job=
if (( attempt > 0 )); then
    job=$(slurm_read_job_receipt "$RUN_DIR/EXISTING_JUDGE_JOB_${attempt}") \
        || block "judge receipt is invalid"
    last_judge_job=$job
    state=$(job_state "$job") || state=
    [[ -n $state ]] && is_terminal_state "$state" || wait_judge_job "judge-$attempt" "$job"
fi
repair_judge_state
result_pass=0
strict_result > "$RUN_DIR/final_receipt_existing.tmp" 2>/dev/null && result_pass=1 || true
if (( result_pass == 0 )) && [[ $PERSISTENT_JUDGE_SESSION == true && -n $last_judge_job ]] \
    && [[ $(job_exit_code "$last_judge_job" || true) == 76 ]]; then
    block "strict result failed after the persistent 16->8->4->1 ladder"
fi
while (( result_pass == 0 && attempt < MAX_JUDGE_ATTEMPTS )); do
    attempt=$((attempt + 1))
    concurrency=${JUDGE_CONCURRENCIES[attempt - 1]}
    receipt=$RUN_DIR/EXISTING_JUDGE_JOB_$attempt
    job_name="gdp-${RUN_ID:0:18}-exj${concurrency}"
    job=$(slurm_submit_or_adopt "$receipt" "existing-judge-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
        -t "$JUDGE_WALL" --cpus-per-task=32 --mem=184G \
        -o "$RUN_DIR/logs/%j_existing_judge.out" -e "$RUN_DIR/logs/%j_existing_judge.err" \
        --export=RUN_DIR="$RUN_DIR",STAGES_FILE="$STAGES_FILE",DATASET="$DATASET",CONCURRENCY="$concurrency",NUM_COMPARISON_TRIALS=4,NUM_REPEATS=1,MULTISTAGE_SEED=42,MODEL_NAME="$MODEL_NAME",JUDGE_DIR_SUFFIX="$JUDGE_DIR_SUFFIX",CHECKPOINT_E2E_ACTIVE_PACKAGE="$DURABLE_ACTIVE_PACKAGE",CHECKPOINT_E2E_EXECUTION_PACKAGE="$CHECKPOINT_E2E_EXECUTION_PACKAGE",CHECKPOINT_E2E_JUDGE_RUNTIME_OVERLAY="$DURABLE_CORRECTED_RUNTIME_OVERLAY",CHECKPOINT_E2E_JUDGE_TRANSPORT_OVERLAY="$DURABLE_ACTIVE_PACKAGE/true3_transport.yaml",CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT="$CORRECTED_TRANSPORT_VIEW_ROOT",CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION="$PERSISTENT_JUDGE_SESSION",GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS="$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS",GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS=900,PATH="$SAFE_PATH" \
        "$E2E_DIR/judge.sbatch") || block "could not submit judge attempt $attempt"
    last_judge_job=$job
    wait_judge_job "judge-$attempt-c$concurrency" "$job"
    repair_judge_state
    strict_result > "$RUN_DIR/final_receipt_existing.tmp" 2>/dev/null && result_pass=1 || true
    if (( result_pass == 0 )) && [[ $PERSISTENT_JUDGE_SESSION == true \
        && $(job_exit_code "$job" || true) == 76 ]]; then
        block "strict result failed after the persistent 16->8->4->1 ladder"
    fi
done
(( result_pass == 1 )) || block "strict 220-task/880-vote result failed after bounded judge recovery"

verify_import
bash "$E2E_DIR/prepare_existing_fingerprint.sh" "$RUN_DIR" >/dev/null \
    || block "final fingerprint revalidation failed"
publish_immutable "$RUN_DIR/final_receipt_existing.tmp" "$FINAL_RECEIPT"
sha_tmp=$FINAL_SHA.tmp.$$
sha256sum "$FINAL_RECEIPT" > "$sha_tmp"
publish_immutable "$sha_tmp" "$FINAL_SHA"
"$PYTHON_BIN" "$E2E_DIR/prepare_existing_campaign.py" publish-envelope \
    --run-dir "$RUN_DIR" --suffix "$JUDGE_DIR_SUFFIX" >/dev/null \
    || block "final import envelope publication failed"
marker_tmp=$FINAL_MARKER.tmp.$$
: > "$marker_tmp"
publish_immutable "$marker_tmp" "$FINAL_MARKER"
validate_final || block "published final receipt bundle failed validation"
rm -f "$RUN_DIR/EXISTING_JUDGE_BLOCKED"
log "EXISTING_CAMPAIGN_COMPLETE $(<"$FINAL_RECEIPT")"
