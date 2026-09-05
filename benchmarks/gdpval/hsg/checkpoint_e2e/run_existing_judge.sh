#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Import and judge an existing GDPVal deliverables tree without policy rollouts.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1
SAFE_PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH=$SAFE_PATH

SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
IMPORT_PY="$SCRIPT_DIR/prepare_existing_campaign.py"
BASE_LAUNCHER="$SCRIPT_DIR/run_checkpoint_e2e.sh"
CAMPAIGN_PY="$SCRIPT_DIR/campaign.py"
PYTHON_BIN=${CHECKPOINT_E2E_PYTHON:-python3}
OWNER_ROOT=${CHECKPOINT_E2E_OWNER_ROOT:-/lustre/fsw/portfolios/llmservice/users/spanev}
AAV2_ROOT=${CHECKPOINT_E2E_AAV2_ROOT:-$OWNER_ROOT/gdpval_colo/aav2}
DATASET=${CHECKPOINT_E2E_DATASET:-$AAV2_ROOT/gdpval_benchmark.local.jsonl}
VERSION=$(<"$SCRIPT_DIR/VERSION")
EXISTING_ROOT=${CHECKPOINT_E2E_EXISTING_ROOT:-$AAV2_ROOT/checkpoint_e2e_existing_v${VERSION//./_}_runs}

fail() { echo "EXISTING_JUDGE_FAIL: $*" >&2; exit 64; }

usage() {
    cat <<'EOF'
Usage:
  run_existing_judge.sh prepare CHECKPOINT EXTERNAL_DELIVERABLES
  run_existing_judge.sh all CHECKPOINT EXTERNAL_DELIVERABLES
  run_existing_judge.sh bootstrap CHECKPOINT EXTERNAL_DELIVERABLES
  run_existing_judge.sh submit RUN_DIR
  run_existing_judge.sh resume RUN_DIR
  run_existing_judge.sh status RUN_DIR
  run_existing_judge.sh result RUN_DIR

prepare is provider-free and submits no jobs. submit/resume submit only a CPU
controller plus CPU Office/transport/judge jobs; policy rollout/GPU jobs are
never submitted. Set CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true explicitly
to let the controller pass the provider-call gate.

all prepares synchronously and then launches the controller. bootstrap submits
one credential-free CPU job that performs the same prepare/copy/launch sequence
after the calling terminal disconnects.
EOF
}

[[ -f $IMPORT_PY && ! -L $IMPORT_PY && -x $BASE_LAUNCHER \
    && -f $CAMPAIGN_PY && ! -L $CAMPAIGN_PY ]] || fail "package is incomplete"

resolve_directory() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.is_absolute(); r=p.resolve(strict=True); assert p==r and r.is_dir() and not p.is_symlink(); print(r)' "$1" \
        || fail "path must be an absolute resolved real directory: $1"
}

publish_env() {
    local target=$1
    shift
    local temporary="$target.tmp.$$" pair key value
    : > "$temporary"
    for pair in "$@"; do
        key=${pair%%=*}; value=${pair#*=}
        [[ $key =~ ^[A-Z][A-Z0-9_]*$ && $value != *$'\n'* && $value != *$'\r'* ]] \
            || fail "unsafe generated import setting: $key"
        printf '%s=%q\n' "$key" "$value" >> "$temporary"
    done
    chmod 0400 "$temporary"
    if [[ -e $target || -L $target ]]; then
        [[ -f $target && ! -L $target && $(stat -c '%a' "$target") == 400 ]] \
            || fail "import settings are not immutable: $target"
        cmp -s "$temporary" "$target" || fail "import settings drift: $target"
        rm -f "$temporary"
    else
        mv "$temporary" "$target"
    fi
}

load_run() {
    RUN_DIR=$(resolve_directory "$1")
    [[ -f $RUN_DIR/settings.env && ! -L $RUN_DIR/settings.env \
        && -f $RUN_DIR/existing_judge.env && ! -L $RUN_DIR/existing_judge.env ]] \
        || fail "run is not a prepared existing-deliverables campaign: $RUN_DIR"
    # Generated owner-only files containing validated absolute paths.
    # shellcheck disable=SC1090
    source "$RUN_DIR/settings.env"
    # shellcheck disable=SC1090
    source "$RUN_DIR/existing_judge.env"
    [[ ${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-} =~ ^[1-4]$ ]] \
        || fail "stored Gemini concurrency must be an integer from 1 through 4"
    export GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS
    [[ $ACCOUNT == nemotron_n3_post ]] || fail "campaign account is not nemotron_n3_post: $ACCOUNT"
    [[ $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]] \
        || fail "unexpected CPU routing: $CPU_PARTITION/$CPU_QOS"
    [[ $ACTIVE_PACKAGE == "$RUN_DIR/existing_judge_package" \
        && -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE ]] \
        || fail "run-owned active package is invalid"
    "$PYTHON_BIN" "$ACTIVE_PACKAGE/prepare_existing_campaign.py" verify --run-dir "$RUN_DIR" >/dev/null \
        || fail "immutable import verification failed"
    [[ $(sha256sum "$RUN_DIR/existing_import_receipt.json" | awk '{print $1}') == "$IMPORT_RECEIPT_SHA256" ]] \
        || fail "import receipt hash no longer matches existing_judge.env"
}

prepare_run() {
    [[ $# == 2 ]] || { usage >&2; exit 64; }
    local checkpoint source identity import_id campaign_root location settings run_dir runtime gemini_concurrency
    checkpoint=$(resolve_directory "$1")
    source=$(resolve_directory "$2")
    gemini_concurrency=${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-2}
    [[ $gemini_concurrency =~ ^[1-4]$ ]] \
        || fail "GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS must be an integer from 1 through 4"
    [[ -f $DATASET && ! -L $DATASET ]] || fail "dataset is unavailable: $DATASET"
    identity=$("$PYTHON_BIN" "$IMPORT_PY" identify \
        --source "$source" --dataset "$DATASET" --expected-tasks 220) \
        || fail "external deliverables did not pass the import gate"
    import_id=$(printf '%s' "$identity" | "$PYTHON_BIN" -c \
        'import json,sys,re; v=json.load(sys.stdin)["import_id"]; assert re.fullmatch(r"import-[0-9a-f]{24}",v); print(v)') \
        || fail "import identity is invalid"
    mkdir -p "$EXISTING_ROOT"
    campaign_root="$EXISTING_ROOT/$import_id"
    [[ $campaign_root == "$EXISTING_ROOT"/* && ! -L $campaign_root ]] \
        || fail "unsafe import campaign root"

    CHECKPOINT_E2E_ROOT="$campaign_root" "$BASE_LAUNCHER" prepare "$checkpoint"
    location=$("$PYTHON_BIN" "$CAMPAIGN_PY" locate \
        --checkpoint "$checkpoint" --campaign-root "$campaign_root")
    run_dir=$(printf '%s\n' "$location" | sed -n 's/^RUN_DIR=//p')
    [[ -n $run_dir ]] || fail "could not locate the prepared import run"
    RUN_DIR=$(resolve_directory "$run_dir")
    settings="$RUN_DIR/settings.env"
    [[ -f $settings && ! -L $settings ]] || fail "base campaign settings are missing"

    "$PYTHON_BIN" "$IMPORT_PY" prepare \
        --run-dir "$RUN_DIR" --source "$source" --dataset "$DATASET" \
        --package "$SCRIPT_DIR" --expected-tasks 220 --expected-import-id "$import_id"

    # shellcheck disable=SC1090
    source "$settings"
    [[ $ACCOUNT == nemotron_n3_post ]] || fail "campaign account is not nemotron_n3_post: $ACCOUNT"
    [[ $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]] \
        || fail "unexpected CPU routing: $CPU_PARTITION/$CPU_QOS"
    ACTIVE_PACKAGE="$RUN_DIR/existing_judge_package"
    runtime="$RUN_DIR/judge_runtime_overlay_existing"
    "$PYTHON_BIN" "$ACTIVE_PACKAGE/transport_runtime.py" materialize \
        --gym-root "$GYM_ROOT" --runtime-root "$runtime" --package-root "$ACTIVE_PACKAGE"
    import_receipt_sha=$(sha256sum "$RUN_DIR/existing_import_receipt.json" | awk '{print $1}')
    publish_env "$RUN_DIR/existing_judge.env" \
        "IMPORT_ID=$import_id" \
        "IMPORT_RECEIPT_SHA256=$import_receipt_sha" \
        "ACTIVE_PACKAGE=$ACTIVE_PACKAGE" \
        "CORRECTED_RUNTIME_OVERLAY=$runtime" \
        "CORRECTED_TRANSPORT_VIEW_ROOT=$RUN_DIR/judge_transport_views_existing" \
        "JUDGE_DIR_SUFFIX=existing" \
        "GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS=$gemini_concurrency"
    load_run "$RUN_DIR"
    printf 'EXISTING_PREPARED import_id=%s\nRUN_DIR=%s\nSOURCE_SNAPSHOT=%s\n' \
        "$import_id" "$RUN_DIR" "$RUN_DIR/deliverables"
}

all_run() {
    [[ $# == 2 ]] || { usage >&2; exit 64; }
    prepare_run "$1" "$2"
    launch_controller "$RUN_DIR"
}

launch_bootstrap() {
    [[ $# == 2 ]] || { usage >&2; exit 64; }
    local checkpoint source authorize account cpu_partition cpu_qos model_name state_id state_dir gemini_concurrency persistent_session
    local gym_root_override expected_gym_revision export_spec old_run_dir receipt job_name job
    local identity package_identity checkpoint_identity expected_run_id import_id package_sha
    local submissions path name number latest=0 latest_job attempt prepared_run
    checkpoint=$(resolve_directory "$1")
    source=$(resolve_directory "$2")
    authorize=${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}
    [[ $authorize == true || $authorize == false ]] \
        || fail "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS must be true or false"
    gemini_concurrency=${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-2}
    [[ $gemini_concurrency =~ ^[1-4]$ ]] \
        || fail "GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS must be an integer from 1 through 4"
    persistent_session=${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:-false}
    [[ $persistent_session == true || $persistent_session == false ]] \
        || fail "CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION must be true or false"
    model_name=${CHECKPOINT_E2E_MODEL_NAME:-}
    [[ $model_name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] \
        || fail "bootstrap requires an explicit safe CHECKPOINT_E2E_MODEL_NAME"
    account=${CHECKPOINT_E2E_ACCOUNT:-nemotron_n3_post}
    cpu_partition=${CHECKPOINT_E2E_CPU_PARTITION:-cpu}
    cpu_qos=${CHECKPOINT_E2E_CPU_QOS:-cpu-normal}
    [[ $account == nemotron_n3_post && $cpu_partition == cpu && $cpu_qos == cpu-normal ]] \
        || fail "bootstrap must use nemotron_n3_post and cpu/cpu-normal"
    for value in "$checkpoint" "$source" "$SCRIPT_DIR" "$OWNER_ROOT" "$AAV2_ROOT" \
        "$DATASET" "$EXISTING_ROOT"; do
        [[ $value == /* && $value != *,* && $value != *$'\n'* && $value != *$'\r'* ]] \
            || fail "unsafe bootstrap path: $value"
    done
    gym_root_override=${CHECKPOINT_E2E_GYM_ROOT:-}
    expected_gym_revision=${CHECKPOINT_E2E_EXPECTED_GYM_REVISION:-}
    if [[ -n $gym_root_override || -n $expected_gym_revision ]]; then
        [[ -n $gym_root_override && -n $expected_gym_revision ]] \
            || fail "Gym root and expected revision overrides must be supplied together"
        gym_root_override=$(resolve_directory "$gym_root_override")
        [[ $gym_root_override != *,* && $expected_gym_revision =~ ^[0-9a-f]{40}$ ]] \
            || fail "unsafe Gym override contract"
    fi
    identity=$("$PYTHON_BIN" "$IMPORT_PY" identify \
        --source "$source" --dataset "$DATASET" --expected-tasks 220) \
        || fail "external deliverables did not pass the bootstrap import gate"
    import_id=$(printf '%s' "$identity" | "$PYTHON_BIN" -c \
        'import json,sys,re; v=json.load(sys.stdin)["import_id"]; assert re.fullmatch(r"import-[0-9a-f]{24}",v); print(v)') \
        || fail "bootstrap import identity is invalid"
    package_identity=$("$PYTHON_BIN" "$IMPORT_PY" identify-package --package "$SCRIPT_DIR") \
        || fail "active package did not pass the bootstrap inventory gate"
    package_sha=$(printf '%s' "$package_identity" | "$PYTHON_BIN" -c \
        'import json,sys,re; v=json.load(sys.stdin)["inventory_sha256"]; assert re.fullmatch(r"[0-9a-f]{64}",v); print(v)') \
        || fail "bootstrap package identity is invalid"
    checkpoint_identity=$("$PYTHON_BIN" "$CAMPAIGN_PY" locate \
        --checkpoint "$checkpoint" --campaign-root "$EXISTING_ROOT/checkpoint_identity") \
        || fail "checkpoint identity could not be computed"
    expected_run_id=$(printf '%s\n' "$checkpoint_identity" | sed -n 's/^RUN_ID=//p')
    [[ $expected_run_id =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] \
        || fail "checkpoint identity is invalid"
    state_id=$(printf '%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n' \
        "$checkpoint_identity" "$identity" "$package_identity" "$model_name" "$authorize" \
        "$gemini_concurrency" "$persistent_session" \
        "${gym_root_override:-default}" "${expected_gym_revision:-default}" \
        | sha256sum | awk '{print substr($1,1,24)}')
    state_dir="$EXISTING_ROOT/bootstrap/$state_id"
    submissions=$state_dir/submissions
    mkdir -p "$state_dir/logs" "$submissions"
    [[ -d $state_dir && ! -L $state_dir ]] || fail "bootstrap state directory is invalid"
    if [[ -e $state_dir/BOOTSTRAP_COMPLETE || -L $state_dir/BOOTSTRAP_COMPLETE ]]; then
        [[ -f $state_dir/BOOTSTRAP_COMPLETE && ! -L $state_dir/BOOTSTRAP_COMPLETE \
            && ! -s $state_dir/BOOTSTRAP_COMPLETE \
            && $(stat -c '%a' "$state_dir/BOOTSTRAP_COMPLETE") == 400 \
            && -f $state_dir/PREPARED_RUN_DIR && ! -L $state_dir/PREPARED_RUN_DIR ]] \
            || fail "completed bootstrap receipt is invalid"
        prepared_run=$(<"$state_dir/PREPARED_RUN_DIR")
        [[ $prepared_run == /* && -d $prepared_run && ! -L $prepared_run \
            && ${prepared_run##*/} == "$expected_run_id" ]] \
            || fail "completed bootstrap points at the wrong checkpoint run"
        "$PYTHON_BIN" "$prepared_run/existing_judge_package/campaign.py" verify \
            --run-dir "$prepared_run" >/dev/null \
            || fail "completed bootstrap checkpoint provenance no longer validates"
        "$PYTHON_BIN" "$prepared_run/existing_judge_package/prepare_existing_campaign.py" verify \
            --run-dir "$prepared_run" >/dev/null \
            || fail "completed bootstrap import provenance no longer validates"
        printf 'EXISTING_BOOTSTRAP_COMPLETE=%s\nBOOTSTRAP_STATE_DIR=%s\nPROVIDER_AUTHORIZED=%s\n' \
            "$prepared_run" "$state_dir" "$authorize"
        return
    fi
    # slurm_receipts derives its crash-safe identity from RUN_DIR. This state
    # directory exists before the content-derived campaign run is prepared.
    old_run_dir=${RUN_DIR:-}
    RUN_DIR=$state_dir
    # shellcheck disable=SC1090
    source "$SCRIPT_DIR/slurm_receipts.sh"
    for path in "$submissions"/attempt_[0-9]*.jobid; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || fail "bootstrap receipt is not a regular file: $path"
        name=${path##*/}; number=${name#attempt_}; number=${number%.jobid}
        [[ $number =~ ^[1-9][0-9]*$ ]] || fail "invalid bootstrap receipt: $name"
        (( number > latest )) && latest=$number
    done
    for (( number=1; number<=latest; number++ )); do
        [[ -f $submissions/attempt_${number}.jobid && ! -L $submissions/attempt_${number}.jobid ]] \
            || fail "bootstrap receipt sequence has a gap before attempt $number"
    done
    if (( latest > 0 )); then
        latest_job=$(slurm_read_job_receipt "$submissions/attempt_${latest}.jobid") \
            || fail "latest bootstrap receipt is invalid"
        case $(job_liveness "$latest_job") in
            live)
                slurm_publish_job_receipt "$state_dir/BOOTSTRAP.jobid" "$latest_job" true >/dev/null \
                    || fail "could not refresh bootstrap receipt"
                RUN_DIR=$old_run_dir
                printf 'EXISTING_BOOTSTRAP_ALREADY_RUNNING=%s\nBOOTSTRAP_STATE_DIR=%s\nPROVIDER_AUTHORIZED=%s\n' \
                    "$latest_job" "$state_dir" "$authorize"
                return
                ;;
            terminal) ;;
            *) fail "bootstrap $latest_job has no live or terminal Slurm evidence" ;;
        esac
    fi
    attempt=$((latest + 1))
    (( attempt <= 4 )) || fail "bootstrap submission bound reached: $latest/4"
    receipt=$submissions/attempt_${attempt}.jobid
    job_name="gdp-existing-boot-${state_id:0:10}"
    export_spec="CHECKPOINT=$checkpoint,EXTERNAL_SOURCE=$source,ACTIVE_PACKAGE=$SCRIPT_DIR,BOOTSTRAP_STATE_DIR=$state_dir,AUTHORIZE_PROVIDER_CALLS=$authorize,EXPECTED_IMPORT_ID=$import_id,EXPECTED_PACKAGE_SOURCE_SHA256=$package_sha,EXPECTED_RUN_ID=$expected_run_id,CHECKPOINT_E2E_OWNER_ROOT=$OWNER_ROOT,CHECKPOINT_E2E_AAV2_ROOT=$AAV2_ROOT,CHECKPOINT_E2E_DATASET=$DATASET,CHECKPOINT_E2E_EXISTING_ROOT=$EXISTING_ROOT,CHECKPOINT_E2E_MODEL_NAME=$model_name,GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS=$gemini_concurrency,CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION=$persistent_session"
    if [[ -n $gym_root_override ]]; then
        export_spec+=",CHECKPOINT_E2E_GYM_ROOT=$gym_root_override,CHECKPOINT_E2E_EXPECTED_GYM_REVISION=$expected_gym_revision"
    fi
    job=$(slurm_submit_or_adopt "$receipt" "existing-bootstrap-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$account" -p "$cpu_partition" --qos="$cpu_qos" \
        -t 04:00:00 --cpus-per-task=4 --mem=16G \
        -o "$state_dir/logs/%j_bootstrap.out" -e "$state_dir/logs/%j_bootstrap.err" \
        --export="$export_spec" \
        "$SCRIPT_DIR/existing_judge_bootstrap.sbatch") \
        || fail "could not submit or adopt existing-judge bootstrap"
    slurm_publish_job_receipt "$state_dir/BOOTSTRAP.jobid" "$job" true >/dev/null \
        || fail "could not publish current bootstrap receipt"
    RUN_DIR=$old_run_dir
    printf 'EXISTING_BOOTSTRAP=%s\nBOOTSTRAP_STATE_DIR=%s\nPROVIDER_AUTHORIZED=%s\n' \
        "$job" "$state_dir" "$authorize"
}

job_liveness() {
    local job=$1 queue accounting state
    queue=$(squeue -h -j "$job" -o '%A' 2>/dev/null) || queue=
    if [[ -n $queue ]] && grep -qx "$job" <<<"$queue"; then
        printf 'live\n'; return
    fi
    accounting=$(sacct -X -j "$job" --format=JobIDRaw,State -n -P 2>/dev/null) \
        || { printf 'unknown\n'; return; }
    state=$(awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}' <<<"$accounting")
    state=${state%%+*}; state=${state%% *}
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
            printf 'terminal\n' ;;
        *) printf 'unknown\n' ;;
    esac
}

launch_controller() {
    [[ $# == 1 ]] || { usage >&2; exit 64; }
    load_run "$1"
    local authorize=${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}
    [[ $authorize == true || $authorize == false ]] \
        || fail "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS must be true or false"
    local persistent_session=${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:-false}
    [[ $persistent_session == true || $persistent_session == false ]] \
        || fail "CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION must be true or false"
    # shellcheck disable=SC1090
    source "$ACTIVE_PACKAGE/slurm_receipts.sh"
    local submissions="$RUN_DIR/existing_controller_submissions" latest=0 path name number
    mkdir -p "$submissions" "$RUN_DIR/logs"
    for path in "$submissions"/attempt_[0-9]*.jobid; do
        [[ -e $path || -L $path ]] || continue
        [[ -f $path && ! -L $path ]] || fail "controller receipt is not a regular file: $path"
        name=${path##*/}; number=${name#attempt_}; number=${number%.jobid}
        [[ $number =~ ^[1-9][0-9]*$ ]] || fail "invalid controller receipt: $name"
        (( number > latest )) && latest=$number
    done
    for (( number=1; number<=latest; number++ )); do
        [[ -f $submissions/attempt_${number}.jobid && ! -L $submissions/attempt_${number}.jobid ]] \
            || fail "controller receipt sequence has a gap before attempt $number"
    done
    if (( latest > 0 )); then
        latest_job=$(slurm_read_job_receipt "$submissions/attempt_${latest}.jobid") \
            || fail "latest controller receipt is invalid"
        case $(job_liveness "$latest_job") in
            live)
                slurm_publish_job_receipt "$RUN_DIR/EXISTING_CONTROLLER.jobid" "$latest_job" true >/dev/null \
                    || fail "could not refresh controller receipt"
                printf 'EXISTING_ALREADY_RUNNING=%s\nRUN_DIR=%s\n' "$latest_job" "$RUN_DIR"
                return
                ;;
            terminal) ;;
            *) fail "controller $latest_job has no live or terminal Slurm evidence" ;;
        esac
    fi
    attempt=$((latest + 1))
    (( attempt <= 4 )) || fail "controller submission bound reached: $latest/4"
    job_name="gdp-${RUN_ID:0:20}-exctl"
    receipt="$submissions/attempt_${attempt}.jobid"
    job=$(slurm_submit_or_adopt "$receipt" "existing-controller-a${attempt}" "$job_name" \
        --parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" \
        -t 12:00:00 --cpus-per-task=2 --mem=8G \
        -o "$RUN_DIR/logs/%j_existing_controller.out" \
        -e "$RUN_DIR/logs/%j_existing_controller.err" \
        --export=RUN_DIR="$RUN_DIR",CHECKPOINT_E2E_EXECUTION_PACKAGE="$ACTIVE_PACKAGE",CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS="$authorize",CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION="$persistent_session",GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS="$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS",PATH="$SAFE_PATH" \
        "$ACTIVE_PACKAGE/existing_judge_controller.sbatch") \
        || fail "could not submit or adopt import-only controller"
    slurm_publish_job_receipt "$RUN_DIR/EXISTING_CONTROLLER.jobid" "$job" true >/dev/null \
        || fail "could not publish current controller receipt"
    printf 'EXISTING_CONTROLLER=%s\nRUN_DIR=%s\nPROVIDER_AUTHORIZED=%s\n' "$job" "$RUN_DIR" "$authorize"
}

show_status() {
    [[ $# == 1 ]] || { usage >&2; exit 64; }
    load_run "$1"
    local status=PREPARED job state
    if [[ -f $RUN_DIR/CAMPAIGN_${JUDGE_DIR_SUFFIX}_COMPLETE ]]; then
        status=PASS
    elif [[ -f $RUN_DIR/EXISTING_JUDGE_BLOCKED ]]; then
        status=BLOCKED
    elif [[ -f $RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED ]]; then
        status=AWAITING_JUDGE_AUTHORIZATION
    elif [[ -f $RUN_DIR/EXISTING_CONTROLLER.jobid ]]; then
        # shellcheck disable=SC1090
        source "$ACTIVE_PACKAGE/slurm_receipts.sh"
        job=$(slurm_read_job_receipt "$RUN_DIR/EXISTING_CONTROLLER.jobid") \
            || fail "current controller receipt is invalid"
        state=$(job_liveness "$job")
        [[ $state != live ]] || status=RUNNING
        [[ $state != terminal ]] || status=RETRYABLE
    fi
    printf 'status=%s run_dir=%s import_id=%s\n' "$status" "$RUN_DIR" "$IMPORT_ID"
}

show_result() {
    [[ $# == 1 ]] || { usage >&2; exit 64; }
    load_run "$1"
    local judge_dir="$RUN_DIR/judge_$JUDGE_DIR_SUFFIX"
    local receipt="$RUN_DIR/final_receipt_$JUDGE_DIR_SUFFIX.json"
    local sidecar="$receipt.sha256" marker="$RUN_DIR/CAMPAIGN_${JUDGE_DIR_SUFFIX}_COMPLETE"
    [[ -f $marker && ! -L $marker && ! -s $marker && $(stat -c '%a' "$marker") == 400 \
        && -f $receipt && ! -L $receipt && $(stat -c '%a' "$receipt") == 400 \
        && -f $sidecar && ! -L $sidecar && $(stat -c '%a' "$sidecar") == 400 ]] \
        || fail "final import-only receipt bundle is incomplete"
    sha256sum -c --status "$sidecar" || fail "final receipt digest mismatch"
    "$PYTHON_BIN" "$ACTIVE_PACKAGE/campaign.py" verify --run-dir "$RUN_DIR" >/dev/null \
        || fail "checkpoint campaign provenance no longer validates"
    "$PYTHON_BIN" "$ACTIVE_PACKAGE/prepare_existing_campaign.py" verify-envelope \
        --run-dir "$RUN_DIR" --suffix "$JUDGE_DIR_SUFFIX" >/dev/null \
        || fail "final import envelope no longer validates"
    current=$("$PYTHON_BIN" "$ACTIVE_PACKAGE/campaign.py" result \
        --output "$judge_dir/gdpval_aav2.jsonl" \
        --journal "$judge_dir/gdpval_aav2_multistage_state.jsonl" \
        --dataset "$DATASET" --expected-tasks 220 --json) \
        || fail "strict final result no longer validates"
    [[ $current == "$(<"$receipt")" ]] || fail "final receipt content drift"
    printf '%s\n' "$current"
}

[[ $# -ge 1 ]] || { usage >&2; exit 64; }
action=$1; shift
case "$action" in
    prepare) prepare_run "$@" ;;
    all) all_run "$@" ;;
    bootstrap) launch_bootstrap "$@" ;;
    submit|resume) launch_controller "$@" ;;
    status) show_status "$@" ;;
    result) show_result "$@" ;;
    *) usage >&2; exit 64 ;;
esac
