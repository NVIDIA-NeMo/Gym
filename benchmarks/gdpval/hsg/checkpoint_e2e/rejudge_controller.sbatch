#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Resume-safe controller for a scientifically corrected sibling judge run.
#SBATCH --job-name=gdpval-e2e-rejudge
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
: "${ACTIVE_PACKAGE:?set ACTIVE_PACKAGE}"
: "${CORRECTED_RUNTIME_OVERLAY:?set CORRECTED_RUNTIME_OVERLAY}"
: "${CORRECTED_TRANSPORT_VIEW_ROOT:?set CORRECTED_TRANSPORT_VIEW_ROOT}"
: "${EXPECTED_JUDGE_FINGERPRINT:?set EXPECTED_JUDGE_FINGERPRINT}"

REQUESTED_ACTIVE_PACKAGE=$ACTIVE_PACKAGE
REQUESTED_CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY
REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT
REQUESTED_EXPECTED_JUDGE_FINGERPRINT=$EXPECTED_JUDGE_FINGERPRINT
REQUESTED_JUDGE_DIR_SUFFIX=${JUDGE_DIR_SUFFIX:-nested_refs_v1}
AUTHORIZE=${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}
ROTATION=${SLURM_RESTART_COUNT:-0}
MAX_ROTATIONS=${CHECKPOINT_E2E_REJUDGE_MAX_ROTATIONS:-2}
POLL_SECONDS=${CHECKPOINT_E2E_POLL_SECONDS:-60}
JUDGE_WALL=${CHECKPOINT_E2E_REJUDGE_WALL:-04:00:00}
JUDGE_CONCURRENCIES=(16 8 4 1)

[[ $REQUESTED_JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || { echo "REJUDGE_FAIL: unsafe judge suffix: $REQUESTED_JUDGE_DIR_SUFFIX" >&2; exit 64; }
[[ $REQUESTED_EXPECTED_JUDGE_FINGERPRINT =~ ^[0-9a-f]{64}$ ]] \
    || { echo "REJUDGE_FAIL: invalid expected fingerprint" >&2; exit 64; }
[[ $AUTHORIZE == true ]] \
    || { echo "REJUDGE_FAIL: provider calls were not explicitly authorized" >&2; exit 64; }
[[ $ROTATION =~ ^[0-9]+$ && $MAX_ROTATIONS =~ ^[0-9]+$ && $POLL_SECONDS =~ ^[1-9][0-9]*$ ]] \
    || { echo "REJUDGE_FAIL: invalid controller retry settings" >&2; exit 64; }
(( MAX_ROTATIONS <= 4 && POLL_SECONDS <= 300 )) \
    || { echo "REJUDGE_FAIL: controller retry settings exceed safety bounds" >&2; exit 64; }

# Preserve the package which prepared the original campaign before loading its
# immutable settings. The corrected package owns only the sibling judging run.
# shellcheck disable=SC1090
source "$RUN_DIR/settings.env"
CAMPAIGN_E2E_SCRIPT=$E2E_SCRIPT
CAMPAIGN_E2E_DIR=$E2E_DIR
CAMPAIGN_JUDGE_RUNTIME_OVERLAY=$JUDGE_RUNTIME_OVERLAY
CAMPAIGN_TRANSPORT_VIEW_ROOT=$TRANSPORT_VIEW_ROOT
ACTIVE_PACKAGE=$REQUESTED_ACTIVE_PACKAGE
CORRECTED_RUNTIME_OVERLAY=$REQUESTED_CORRECTED_RUNTIME_OVERLAY
CORRECTED_TRANSPORT_VIEW_ROOT=$REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT
EXPECTED_JUDGE_FINGERPRINT=$REQUESTED_EXPECTED_JUDGE_FINGERPRINT
JUDGE_DIR_SUFFIX=$REQUESTED_JUDGE_DIR_SUFFIX
E2E_DIR=$ACTIVE_PACKAGE
CORRECTED_TRANSPORT_OVERLAY=$ACTIVE_PACKAGE/true3_transport.yaml
CORRECTED_CANDIDATE_VIEW=$CORRECTED_TRANSPORT_VIEW_ROOT/candidate
CORRECTED_REFERENCE_OVERLAY=$CORRECTED_TRANSPORT_VIEW_ROOT/reference_views.yaml
JUDGE_DIR=$RUN_DIR/judge_$JUDGE_DIR_SUFFIX
OUTPUT=$JUDGE_DIR/gdpval_aav2.jsonl
JOURNAL=$JUDGE_DIR/gdpval_aav2_multistage_state.jsonl
DISTRIBUTION_PATH=$JUDGE_DIR/occupation_distribution.json
STATE_ROOT=$JUDGE_DIR/.checkpoint_e2e_judge_state
FINAL_RECEIPT=$RUN_DIR/final_receipt_$JUDGE_DIR_SUFFIX.json
FINAL_SHA=$FINAL_RECEIPT.sha256
FINAL_MARKER=$RUN_DIR/CAMPAIGN_${JUDGE_DIR_SUFFIX}_COMPLETE
FINGERPRINT_RECEIPT=$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json
TRANSPORT_RECEIPT=$RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX
STAGES_FILE=$JUDGE_DIR/stages.txt
PYTHON_BIN=${CHECKPOINT_E2E_PYTHON:-python3}
FINAL_TEMP=$FINAL_RECEIPT.tmp.$$
FINAL_SHA_TEMP=$FINAL_SHA.tmp.$$
FINAL_MARKER_TEMP=$FINAL_MARKER.tmp.$$

# shellcheck disable=SC1090
source "$E2E_DIR/slurm_receipts.sh"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
fail() { log "REJUDGE_BLOCKED: $*"; printf '%s\n' "$*" > "$RUN_DIR/REJUDGE_${JUDGE_DIR_SUFFIX}_BLOCKED"; exit 1; }

cleanup_temporaries() {
    rm -f "$FINAL_TEMP" "$FINAL_SHA_TEMP" "$FINAL_MARKER_TEMP"
}
trap cleanup_temporaries EXIT

[[ ${#JUDGE_DIR_SUFFIX} -le 32 ]] || fail "judge suffix exceeds 32 characters"
[[ $ACTIVE_PACKAGE == /* && -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE \
    && $ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR" ]] \
    || fail "corrected package is missing or aliases the original campaign package"
[[ $CORRECTED_RUNTIME_OVERLAY == "$RUN_DIR"/* && -d $CORRECTED_RUNTIME_OVERLAY \
    && ! -L $CORRECTED_RUNTIME_OVERLAY \
    && $CORRECTED_RUNTIME_OVERLAY != "$CAMPAIGN_JUDGE_RUNTIME_OVERLAY" ]] \
    || fail "corrected runtime must be a distinct sibling below the campaign run"
[[ $CORRECTED_TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* && -d $CORRECTED_TRANSPORT_VIEW_ROOT \
    && ! -L $CORRECTED_TRANSPORT_VIEW_ROOT \
    && $CORRECTED_TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT" ]] \
    || fail "corrected transport view must be a distinct sibling below the campaign run"
[[ $JUDGE_DIR_SUFFIX != e2e ]] || fail "corrected rejudge cannot target the original judge_e2e output"

exec 9> "$RUN_DIR/rejudge_${JUDGE_DIR_SUFFIX}.lock"
flock -n 9 || fail "another corrected-judge controller owns this run"
rm -f "$RUN_DIR/REJUDGE_${JUDGE_DIR_SUFFIX}_BLOCKED"

on_term() {
    trap - TERM
    (( ROTATION < MAX_ROTATIONS )) || fail "controller walltime retry limit reached"
    log "controller walltime signal; requeueing rotation=$ROTATION/$MAX_ROTATIONS"
    scontrol requeue "$SLURM_JOB_ID" || fail "controller requeue failed"
    exit 0
}
trap on_term TERM

terminal_state() {
    local job=$1
    sacct -X -j "$job" --format=JobIDRaw,State -n -P 2>/dev/null \
        | awk -F '|' -v wanted="$job" '
            $1 == wanted {
                state=$2; sub(/\+.*/, "", state); sub(/ .*/, "", state); print state; exit
            }
        '
}

is_terminal() {
    case "$(terminal_state "$1")" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
            return 0 ;;
        *) return 1 ;;
    esac
}

wait_for_job() {
    local job=$1 states
    while :; do
        states=$(squeue -h -j "$job" -o '%T' 2>/dev/null) || states=
        if [[ -z $states ]] && is_terminal "$job"; then
            break
        fi
        log "judge job=$job state=${states:-accounting-pending}"
        sleep "$POLL_SECONDS"
    done
    sacct -X -j "$job" --format=JobIDRaw,State,ExitCode,Elapsed,Account,Partition,QOS -n -P \
        | awk -F '|' -v wanted="$job" '$1 == wanted {print}' || true
}

strict_result() {
    local result fingerprint
    result=$("$PYTHON_BIN" "$E2E_DIR/campaign.py" result \
        --output "$OUTPUT" --journal "$JOURNAL" \
        --dataset "$DATASET" --expected-tasks 220 --json) || return
    fingerprint=$("$PYTHON_BIN" -c 'import json,sys; values={json.loads(line)["fingerprint"] for line in open(sys.argv[1]) if line.strip()}; assert len(values)==1; print(values.pop())' "$JOURNAL") \
        || return
    [[ $fingerprint == "$EXPECTED_JUDGE_FINGERPRINT" ]] || {
        log "corrected journal fingerprint drift: $fingerprint != $EXPECTED_JUDGE_FINGERPRINT"
        return 1
    }
    printf '%s\n' "$result"
}

repair_state() {
    local report status
    [[ -e $OUTPUT || -L $OUTPUT || -e $JOURNAL || -L $JOURNAL ]] || return 0
    [[ -f $JOURNAL && ! -L $JOURNAL && -s $JOURNAL ]] \
        || fail "judge output exists without a usable journal"
    if [[ ! -e $OUTPUT && ! -L $OUTPUT ]]; then
        : > "$OUTPUT"
        chmod 0600 "$OUTPUT"
    fi
    [[ -f $OUTPUT && ! -L $OUTPUT ]] || fail "judge output is not a regular file"
    report=$("$PYTHON_BIN" "$E2E_DIR/judge_state.py" sanitize \
        --output "$OUTPUT" --journal "$JOURNAL" \
        --deliverables "$CORRECTED_CANDIDATE_VIEW" --state-root "$STATE_ROOT" \
        --max-attempts 3) || fail "judge-state repair failed"
    status=$(printf '%s\n' "$report" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["status"])') \
        || fail "invalid judge-state repair report"
    case "$status" in CLEAN|SANITIZED|RECOVERED) log "judge-state=$status" ;; *) fail "unexpected judge-state=$status" ;; esac
}

file_mode() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import stat,sys; print(format(stat.S_IMODE(Path(sys.argv[1]).stat().st_mode), "o"))' "$1"
}

journal_fingerprint() {
    "$PYTHON_BIN" -c 'import json,sys; values={json.loads(line)["fingerprint"] for line in open(sys.argv[1]) if line.strip()}; assert len(values)==1; print(values.pop())' "$1"
}

validate_original_campaign() {
    local receipt=$RUN_DIR/final_receipt.json
    local sidecar=$receipt.sha256
    local marker=$RUN_DIR/CAMPAIGN_COMPLETE
    local output=$RUN_DIR/judge_e2e/gdpval_aav2.jsonl
    local journal=$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl
    local current stored original_fingerprint
    [[ -f $marker && ! -L $marker && ! -s $marker && $(file_mode "$marker") == 400 ]] \
        || fail "original completion marker is missing or invalid"
    [[ -f $receipt && ! -L $receipt && -s $receipt && $(file_mode "$receipt") == 400 ]] \
        || fail "original final receipt is missing or invalid"
    [[ -f $sidecar && ! -L $sidecar && -s $sidecar && $(file_mode "$sidecar") == 400 ]] \
        || fail "original final receipt sidecar is missing or invalid"
    sha256sum -c --status "$sidecar" || fail "original final receipt digest mismatch"
    current=$("$PYTHON_BIN" "$CAMPAIGN_E2E_DIR/campaign.py" result \
        --output "$output" --journal "$journal" \
        --dataset "$DATASET" --expected-tasks 220 --json) \
        || fail "original strict result no longer validates"
    stored=$(<"$receipt")
    [[ $current == "$stored" ]] || fail "original final receipt no longer matches its strict result"
    original_fingerprint=$(journal_fingerprint "$journal") \
        || fail "original campaign fingerprint is invalid"
    [[ $EXPECTED_JUDGE_FINGERPRINT != "$original_fingerprint" ]] \
        || fail "corrected judge fingerprint did not enter a new semantic namespace"
}

validate_corrected_final() {
    local current stored
    [[ -f $FINAL_MARKER && ! -L $FINAL_MARKER && ! -s $FINAL_MARKER \
        && $(file_mode "$FINAL_MARKER") == 400 ]] \
        || fail "corrected completion marker is invalid"
    [[ -f $FINAL_RECEIPT && ! -L $FINAL_RECEIPT && -s $FINAL_RECEIPT \
        && $(file_mode "$FINAL_RECEIPT") == 400 ]] \
        || fail "corrected final receipt is invalid"
    [[ -f $FINAL_SHA && ! -L $FINAL_SHA && -s $FINAL_SHA \
        && $(file_mode "$FINAL_SHA") == 400 ]] \
        || fail "corrected final receipt sidecar is invalid"
    sha256sum -c --status "$FINAL_SHA" || fail "corrected final receipt digest mismatch"
    current=$(strict_result) || fail "corrected completion marker no longer validates"
    stored=$(<"$FINAL_RECEIPT")
    [[ $current == "$stored" ]] || fail "corrected final receipt no longer matches its strict result"
}

publish_immutable() {
    local source=$1 target=$2 label=$3
    chmod 0400 "$source"
    if [[ -e $target || -L $target ]]; then
        [[ -f $target && ! -L $target && $(file_mode "$target") == 400 ]] \
            || fail "$label target is not an immutable regular file"
        cmp -s "$source" "$target" || fail "$label drift after interrupted publication"
        rm -f "$source"
    else
        mv "$source" "$target"
    fi
}

validate_receipts() {
    [[ -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE ]] || fail "corrected package is invalid"
    [[ -d $CORRECTED_RUNTIME_OVERLAY && ! -L $CORRECTED_RUNTIME_OVERLAY ]] \
        || fail "corrected runtime is invalid"
    [[ -f $CORRECTED_TRANSPORT_OVERLAY && ! -L $CORRECTED_TRANSPORT_OVERLAY \
        && -f $CORRECTED_REFERENCE_OVERLAY && ! -L $CORRECTED_REFERENCE_OVERLAY \
        && -d $CORRECTED_CANDIDATE_VIEW && ! -L $CORRECTED_CANDIDATE_VIEW ]] \
        || fail "corrected config/view contract is incomplete"
    [[ -f $FINGERPRINT_RECEIPT && ! -L $FINGERPRINT_RECEIPT \
        && $(file_mode "$FINGERPRINT_RECEIPT") == 400 ]] \
        || fail "provider-free fingerprint receipt is missing"
    local preprocessed=$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl
    local preprocessed_sha256 distribution_sha256
    [[ -f $preprocessed && ! -L $preprocessed && -s $preprocessed \
        && $(file_mode "$preprocessed") == 400 ]] \
        || fail "sibling preprocessed input is missing or mutable"
    [[ -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH \
        && $(file_mode "$DISTRIBUTION_PATH") == 400 ]] \
        || fail "sibling occupation distribution is missing or mutable"
    preprocessed_sha256=$(sha256sum "$preprocessed" | awk '{print $1}')
    distribution_sha256=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
    "$PYTHON_BIN" -c 'import json,os,sys; d=json.load(open(sys.argv[1])); e=sys.argv[2]; r=sys.argv[3]; p=sys.argv[4]; s=sys.argv[5]; q=sys.argv[6]; t=sys.argv[7]; assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS" and d["fingerprint"]==e and d["expected_fingerprint"]==e and os.path.realpath(d["runtime_root"])==os.path.realpath(r) and os.path.realpath(d["preprocessed_input"])==os.path.realpath(p) and d["preprocessed_input_sha256"]==s and os.path.realpath(d["distribution_path"])==os.path.realpath(q) and d["distribution_sha256"]==t and d["materialized_row_count"]==220 and d["reference_count"]==9 and d["config_count"]==6' \
        "$FINGERPRINT_RECEIPT" "$EXPECTED_JUDGE_FINGERPRINT" "$CORRECTED_RUNTIME_OVERLAY" \
        "$preprocessed" "$preprocessed_sha256" "$DISTRIBUTION_PATH" "$distribution_sha256" \
        || fail "provider-free fingerprint receipt drift"
    [[ -f $TRANSPORT_RECEIPT && ! -L $TRANSPORT_RECEIPT \
        && $(file_mode "$TRANSPORT_RECEIPT") == 400 ]] \
        || fail "corrected transport receipt is missing"
    local schema manifest manifest_sha
    schema=$(sed -n 's/^schema=//p' "$TRANSPORT_RECEIPT")
    manifest=$(sed -n 's/^manifest=//p' "$TRANSPORT_RECEIPT")
    manifest_sha=$(sed -n 's/^manifest_sha256=//p' "$TRANSPORT_RECEIPT")
    [[ $schema == gdpval.transport-prebuild.v1 \
        && $manifest == "$CORRECTED_TRANSPORT_VIEW_ROOT/manifest.json" \
        && $manifest_sha =~ ^[0-9a-f]{64}$ \
        && $(sha256sum "$manifest" | awk '{print $1}') == "$manifest_sha" ]] \
        || fail "corrected transport receipt drift"
    "$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR" >/dev/null \
        || fail "original campaign immutable-input preflight failed"
    validate_original_campaign
    "$PYTHON_BIN" "$E2E_DIR/transport_runtime.py" validate \
        --gym-root "$GYM_ROOT" --runtime-root "$CORRECTED_RUNTIME_OVERLAY" \
        --package-root "$E2E_DIR" >/dev/null \
        || fail "corrected runtime validation failed"
}

validate_receipts
[[ ! -e $JUDGE_DIR && ! -L $JUDGE_DIR ]] || [[ -d $JUDGE_DIR && ! -L $JUDGE_DIR ]] \
    || fail "corrected judge directory is not a real directory"
mkdir -p "$JUDGE_DIR"
EXPECTED_STAGES='[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1, waivable_failure_classes: [timeout_exceeded, transient]}}, {num_tasks: 220, num_models: 4}]'
if [[ -e $STAGES_FILE || -L $STAGES_FILE ]]; then
    [[ -f $STAGES_FILE && ! -L $STAGES_FILE && $(<"$STAGES_FILE") == "$EXPECTED_STAGES" ]] \
        || fail "corrected stages file is invalid or drifted"
else
    stages_temporary=$STAGES_FILE.tmp.$$
    printf '%s' "$EXPECTED_STAGES" > "$stages_temporary"
    chmod 0400 "$stages_temporary"
    mv "$stages_temporary" "$STAGES_FILE"
fi

if [[ -e $FINAL_MARKER || -L $FINAL_MARKER ]]; then
    validate_corrected_final
    log "corrected campaign already complete"
    exit 0
fi

result_pass=0
for attempt in 1 2 3 4; do
    repair_state
    if strict_result > "$FINAL_TEMP" 2>/dev/null; then
        result_pass=1
        break
    fi
    concurrency=${JUDGE_CONCURRENCIES[attempt - 1]}
    job_receipt=$RUN_DIR/REJUDGE_${JUDGE_DIR_SUFFIX}_JOB_$attempt
    job_name="gdp-${RUN_ID:0:17}-nr${attempt}c${concurrency}"
    job=$(slurm_submit_or_adopt "$job_receipt" "rejudge-${JUDGE_DIR_SUFFIX}-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
        -t "$JUDGE_WALL" --cpus-per-task=32 --mem=184G \
        -o "$RUN_DIR/logs/%j_rejudge.out" -e "$RUN_DIR/logs/%j_rejudge.err" \
        --export=ALL,RUN_DIR="$RUN_DIR",STAGES_FILE="$STAGES_FILE",DATASET="$DATASET",CONCURRENCY="$concurrency",NUM_COMPARISON_TRIALS=4,NUM_REPEATS=1,MULTISTAGE_SEED=42,MODEL_NAME="$MODEL_NAME",JUDGE_DIR_SUFFIX="$JUDGE_DIR_SUFFIX",CHECKPOINT_E2E_ACTIVE_PACKAGE="$ACTIVE_PACKAGE",CHECKPOINT_E2E_JUDGE_RUNTIME_OVERLAY="$CORRECTED_RUNTIME_OVERLAY",CHECKPOINT_E2E_JUDGE_TRANSPORT_OVERLAY="$CORRECTED_TRANSPORT_OVERLAY",CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT="$CORRECTED_TRANSPORT_VIEW_ROOT",GDPVAL_JUDGE_REQUEST_TIMEOUT_SECONDS=900 \
        "$E2E_DIR/judge.sbatch") || fail "could not submit/adopt judge attempt=$attempt"
    wait_for_job "$job"
done

if (( result_pass == 0 )); then
    repair_state
    strict_result > "$FINAL_TEMP" \
        || fail "strict 220-task/880-vote result failed after c16->8->4->1"
fi

validate_receipts
"$PYTHON_BIN" "$E2E_DIR/campaign.py" coverage \
    --dataset "$DATASET" --deliverables "$DELIVERABLES" --expected-tasks 220 >/dev/null \
    || fail "final rollout coverage recheck failed"
publish_immutable "$FINAL_TEMP" "$FINAL_RECEIPT" "corrected final receipt"
sha256sum "$FINAL_RECEIPT" > "$FINAL_SHA_TEMP"
publish_immutable "$FINAL_SHA_TEMP" "$FINAL_SHA" "corrected final receipt sidecar"
: > "$FINAL_MARKER_TEMP"
publish_immutable "$FINAL_MARKER_TEMP" "$FINAL_MARKER" "corrected completion marker"
validate_corrected_final
rm -f "$RUN_DIR/REJUDGE_${JUDGE_DIR_SUFFIX}_BLOCKED"
log "CORRECTED_CAMPAIGN_COMPLETE $(<"$FINAL_RECEIPT")"
