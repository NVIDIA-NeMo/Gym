#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Non-provider phase: build immutable candidate/reference transport views once.
#SBATCH --job-name=gdpval-e2e-transport
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1
export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${RUN_DIR:?set RUN_DIR}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"

REQUESTED_ACTIVE_PACKAGE=${CHECKPOINT_E2E_ACTIVE_PACKAGE:-}
REQUESTED_TRANSPORT_VIEW_ROOT=${CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT:-}
TRANSPORT_RECEIPT_SUFFIX=${TRANSPORT_RECEIPT_SUFFIX:-}
REQUESTED_DERIVATIVE_CACHE_ROOT=${CHECKPOINT_E2E_TRANSPORT_DERIVATIVE_CACHE_ROOT:-}

RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] || {
    echo "CHECKPOINT_E2E_TRANSPORT_FAIL: invalid run id" >&2; exit 64;
}
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "CHECKPOINT_E2E_TRANSPORT_FAIL: invalid Slurm user" >&2; exit 64;
}
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-transport
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]] || {
    echo "CHECKPOINT_E2E_TRANSPORT_FAIL: bootstrap root resolves to Lustre" >&2; exit 64;
}
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" transport "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
CAMPAIGN_E2E_SCRIPT=$E2E_SCRIPT
CAMPAIGN_E2E_DIR=$E2E_DIR
CAMPAIGN_TRANSPORT_VIEW_ROOT=$TRANSPORT_VIEW_ROOT
override_count=0
for requested in "$REQUESTED_ACTIVE_PACKAGE" "$REQUESTED_TRANSPORT_VIEW_ROOT" \
    "$TRANSPORT_RECEIPT_SUFFIX"; do
    [[ -z $requested ]] || override_count=$((override_count + 1))
done
(( override_count == 0 || override_count == 3 )) || {
    echo "CHECKPOINT_E2E_TRANSPORT_FAIL: corrected transport package, root, and receipt suffix must be supplied together" >&2
    exit 64
}
CORRECTED_REJUDGE=false
if (( override_count == 3 )); then
    CORRECTED_REJUDGE=true
    TRANSPORT_VIEW_ROOT=$REQUESTED_TRANSPORT_VIEW_ROOT
fi
if [[ $CORRECTED_REJUDGE == true ]]; then
    [[ $REQUESTED_ACTIVE_PACKAGE == "$RUN_DIR"/* \
        && -d $REQUESTED_ACTIVE_PACKAGE && ! -L $REQUESTED_ACTIVE_PACKAGE ]] || {
        echo "CHECKPOINT_E2E_TRANSPORT_FAIL: durable corrected package is invalid" >&2
        exit 64
    }
fi
mars_stage_gym "$GYM_ROOT" "$GYM_REVISION"
mars_stage_runtime
mars_stage_container "$GDPVAL_SIF" "$GDPVAL_SIF_SIGNATURE"
mars_stage_apptainer "$APPTAINER_BIN" "$APPTAINER_SIGNATURE"
mars_stage_file "$REFERENCE_OVERLAY" "$MARS_RUN_ROOT/config/reference_paths.yaml"
E2E_DIR=$MARS_PACKAGE
TRANSPORT_VIEWS_PY=$E2E_DIR/transport_views.py
GYM_ROOT=$MARS_GYM
JUDGE_RUNTIME_OVERLAY=$MARS_RUNTIME
GDPVAL_SIF=$MARS_GDPVAL_SIF
APPTAINER_BIN=$MARS_APPTAINER_BIN
REFERENCE_OVERLAY=$MARS_RUN_ROOT/config/reference_paths.yaml
CAMPAIGN_E2E_SCRIPT=$E2E_DIR/run_checkpoint_e2e.sh
export CHECKPOINT_E2E_LOCAL_PACKAGE=$E2E_DIR
export CHECKPOINT_E2E_LOCAL_GYM=$GYM_ROOT
export CHECKPOINT_E2E_LOCAL_RUNTIME=$JUDGE_RUNTIME_OVERLAY
export CHECKPOINT_E2E_LOCAL_ENV_FILE=$MARS_RUN_ROOT/config/aav2.env
mars_stage_file "$ENV_FILE" "$CHECKPOINT_E2E_LOCAL_ENV_FILE"
# shellcheck disable=SC1090
source "$E2E_DIR/judge_process_group.sh"

fail() { echo "CHECKPOINT_E2E_TRANSPORT_FAIL: $*" >&2; exit 64; }

on_term() {
    trap - TERM
    echo "transport prebuild received TERM; incomplete atomic staging will not be published"
    stop_judge_process_group || true
    exit 75
}
trap on_term TERM

DERIVATIVE_CACHE_ROOT=${REQUESTED_DERIVATIVE_CACHE_ROOT:-$MARS_RUN_ROOT/cache/transport_derivatives}
[[ $DERIVATIVE_CACHE_ROOT == /raid/scratch/* && $DERIVATIVE_CACHE_ROOT != *:* \
    && $DERIVATIVE_CACHE_ROOT != *$'\n'* && ! -L $DERIVATIVE_CACHE_ROOT ]] \
    || fail "shared derivative cache root is unsafe: $DERIVATIVE_CACHE_ROOT"
mkdir -p "$DERIVATIVE_CACHE_ROOT"
[[ -d $DERIVATIVE_CACHE_ROOT && ! -L $DERIVATIVE_CACHE_ROOT ]] \
    || fail "shared derivative cache root is not a directory: $DERIVATIVE_CACHE_ROOT"
chmod 0700 "$DERIVATIVE_CACHE_ROOT"

# The GDPVal image is large.  Hash it once per exact path/size/mtime identity,
# under a shared lock, then reuse the immutable digest receipt.  The first
# hasher rechecks the nanosecond stat identity after reading all bytes.
container_identity_dir="$DERIVATIVE_CACHE_ROOT/container_identities"
mkdir -p "$container_identity_dir"
chmod 0700 "$container_identity_dir"
container_stat_json() {
    python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]).resolve(strict=True); s=p.stat(); print(json.dumps({"path":str(p),"bytes":s.st_size,"mtime_ns":s.st_mtime_ns},sort_keys=True,separators=(",",":")))' "$1"
}
container_stat_before=$(container_stat_json "$GDPVAL_SIF") \
    || fail "could not stat GDPVal container"
container_identity_key=$(printf '%s' "$container_stat_before" | sha256sum | awk '{print $1}')
[[ $container_identity_key =~ ^[0-9a-f]{64}$ ]] \
    || fail "could not key GDPVal container identity"
container_identity_lock="$container_identity_dir/$container_identity_key.lock"
container_identity_receipt="$container_identity_dir/$container_identity_key.json"
exec {container_identity_fd}>"$container_identity_lock"
chmod 0600 "$container_identity_lock"
flock -x "$container_identity_fd" || fail "could not lock GDPVal container identity"
if [[ -e $container_identity_receipt || -L $container_identity_receipt ]]; then
    [[ -f $container_identity_receipt && ! -L $container_identity_receipt \
        && $(stat -c '%a' "$container_identity_receipt") == 400 ]] \
        || fail "GDPVal container identity receipt is invalid"
else
    container_sha256=$(sha256sum "$GDPVAL_SIF" | awk '{print $1}')
    [[ $container_sha256 =~ ^[0-9a-f]{64}$ ]] || fail "could not fingerprint GDPVal container"
    container_stat_after=$(container_stat_json "$GDPVAL_SIF") \
        || fail "could not restat GDPVal container"
    [[ $container_stat_after == "$container_stat_before" ]] \
        || fail "GDPVal container changed while it was hashed"
    container_identity_temporary="$container_identity_receipt.tmp.$$"
    python3 -c 'import json,sys; print(json.dumps({"schema":"gdpval.container-identity.v1","identity":json.loads(sys.argv[1]),"sha256":sys.argv[2]},indent=2,sort_keys=True))' \
        "$container_stat_before" "$container_sha256" > "$container_identity_temporary"
    chmod 0400 "$container_identity_temporary"
    mv "$container_identity_temporary" "$container_identity_receipt"
fi
container_sha256=$(python3 -c 'import json,re,sys; expected=json.loads(sys.argv[1]); actual=json.load(open(sys.argv[2])); valid=isinstance(actual,dict) and set(actual)=={"schema","identity","sha256"} and actual["schema"]=="gdpval.container-identity.v1" and actual["identity"]==expected and isinstance(actual["sha256"],str) and re.fullmatch("[0-9a-f]{64}",actual["sha256"]); sys.stdout.write(actual["sha256"] if valid else ""); raise SystemExit(0 if valid else 64)' \
    "$container_stat_before" "$container_identity_receipt") \
    || fail "GDPVal container identity receipt drift"
flock -u "$container_identity_fd" || fail "could not unlock GDPVal container identity"
exec {container_identity_fd}>&-

if [[ $CORRECTED_REJUDGE == true ]]; then
    [[ $TRANSPORT_RECEIPT_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
        && ${#TRANSPORT_RECEIPT_SUFFIX} -le 32 ]] \
        || fail "invalid corrected transport receipt suffix: $TRANSPORT_RECEIPT_SUFFIX"
    [[ $E2E_DIR == /raid/scratch/* && -d $E2E_DIR && ! -L $E2E_DIR ]] \
        || fail "node-local corrected package is missing"
    [[ -f $TRANSPORT_VIEWS_PY && ! -L $TRANSPORT_VIEWS_PY ]] \
        || fail "corrected transport-view builder is missing"
    [[ $TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* && $TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT" \
        && ! -L $TRANSPORT_VIEW_ROOT ]] \
        || fail "corrected transport view must be a distinct sibling below the campaign run"
fi

"$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR"

configure_gdpval_container_python \
    "$APPTAINER_BIN/apptainer" "$GDPVAL_SIF" "$E2E_DIR" "$RUN_DIR" "$REFERENCE_OVERLAY" \
    || fail "could not build the exact container bind contract"
container_tail_start=$((${#GDPVAL_CONTAINER_PYTHON[@]} - 2))
(( container_tail_start >= 2 )) || fail "container command is too short to extend safely"
CONTAINER_CACHE_ROOT=/gdpval-transport-derivative-cache
CONTAINER_PYTHON=(
    "${GDPVAL_CONTAINER_PYTHON[@]:0:$container_tail_start}"
    --bind "$DERIVATIVE_CACHE_ROOT:$CONTAINER_CACHE_ROOT"
    --bind "$GYM_ROOT:$GYM_ROOT:ro"
    --bind "$JUDGE_RUNTIME_OVERLAY:$JUDGE_RUNTIME_OVERLAY:ro"
    "${GDPVAL_CONTAINER_PYTHON[@]:$container_tail_start}"
)
run_container_python() {
    local log=$1
    shift
    start_judge_process_group "$log" "${CONTAINER_PYTHON[@]}" "$@" \
        || fail "could not start containerized transport command"
    wait_judge_process_group || fail "could not wait for containerized transport command"
    (( GYM_RC == 0 )) || fail "containerized transport command failed with rc=$GYM_RC"
}

mkdir -p "$RUN_DIR/logs"
log_file="$MARS_JOB_ROOT/logs/transport_prebuild.log"
durable_log="$RUN_DIR/logs/transport_prebuild_${SLURM_JOB_ID:-manual}.log"
copy_transport_log() {
    if [[ -f $log_file ]]; then
        cp -- "$log_file" "$durable_log.tmp.$$"
        chmod 0600 "$durable_log.tmp.$$"
        mv -f "$durable_log.tmp.$$" "$durable_log"
    fi
}
trap copy_transport_log EXIT
run_container_python "$log_file" "$TRANSPORT_VIEWS_PY" build \
    --candidate-root "$DELIVERABLES" \
    --reference-overlay "$REFERENCE_OVERLAY" \
    --output-root "$TRANSPORT_VIEW_ROOT" \
    --min-video-bytes 8388608 \
    --derivative-profile reference-pdf-v1+video-h264-720p-crf26-aac128-min8m+ref-video-bundle8-v1 \
    --preconvert-py "$GYM_ROOT/resources_servers/gdpval/preconvert.py" \
    --derivative-cache-root "$CONTAINER_CACHE_ROOT" \
    --derivative-cache-label "$DERIVATIVE_CACHE_ROOT" \
    --container-sha256 "$container_sha256"

manifest="$TRANSPORT_VIEW_ROOT/manifest.json"
[[ -f $manifest && ! -L $manifest ]] || fail "transport manifest was not published"
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
receipt="$RUN_DIR/TRANSPORT_PREBUILD_PASS"
if [[ $CORRECTED_REJUDGE == true ]]; then
    receipt="$RUN_DIR/TRANSPORT_PREBUILD_PASS_$TRANSPORT_RECEIPT_SUFFIX"
fi
temporary="${receipt}.tmp.$$"
printf 'schema=gdpval.transport-prebuild.v1\nmanifest=%s\nmanifest_sha256=%s\n' \
    "$manifest" "$manifest_sha" > "$temporary"
chmod 0400 "$temporary"
if [[ -e $receipt || -L $receipt ]]; then
    [[ -f $receipt && ! -L $receipt ]] || fail "transport pass receipt is not a regular file"
    cmp -s "$temporary" "$receipt" || fail "transport pass receipt drift"
    rm -f "$temporary"
else
    mv "$temporary" "$receipt"
fi
echo "TRANSPORT_PREBUILD_PASS root=$TRANSPORT_VIEW_ROOT manifest_sha256=$manifest_sha"
