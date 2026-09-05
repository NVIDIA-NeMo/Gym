#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provider-free fingerprint gate for an import-only GDPVal judging campaign.

set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1

[[ $# == 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 64; }
RUN_DIR=$(cd -P -- "$1" && pwd -P)
SETTINGS_FILE=${CHECKPOINT_E2E_LOCAL_SETTINGS:-$RUN_DIR/settings.env}
EXISTING_SETTINGS_FILE=${CHECKPOINT_E2E_LOCAL_EXISTING_SETTINGS:-$RUN_DIR/existing_judge.env}
if [[ -n ${CHECKPOINT_E2E_LOCAL_SETTINGS:-}${CHECKPOINT_E2E_LOCAL_EXISTING_SETTINGS:-} ]]; then
    [[ ${CHECKPOINT_E2E_LOCAL_SETTINGS:-} == /raid/scratch/* \
        && ${CHECKPOINT_E2E_LOCAL_EXISTING_SETTINGS:-} == /raid/scratch/* ]] || {
        echo "EXISTING_FINGERPRINT_FAIL: local settings overrides are incomplete or unsafe" >&2
        exit 64
    }
fi
[[ -f $SETTINGS_FILE && ! -L $SETTINGS_FILE \
    && -f $EXISTING_SETTINGS_FILE && ! -L $EXISTING_SETTINGS_FILE ]] || {
    echo "EXISTING_FINGERPRINT_FAIL: prepared settings are missing" >&2
    exit 64
}

# Both files are generated owner-only from validated absolute paths.
# shellcheck disable=SC1090
source "$SETTINGS_FILE"
# shellcheck disable=SC1090
source "$EXISTING_SETTINGS_FILE"
E2E_EXECUTION_DIR=$ACTIVE_PACKAGE
DURABLE_CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY
PYTHON_BIN=${CHECKPOINT_E2E_PYTHON:-python3}
if [[ -n ${CHECKPOINT_E2E_LOCAL_PACKAGE:-} ]]; then
    [[ $CHECKPOINT_E2E_LOCAL_PACKAGE == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_GYM == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_RUNTIME == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_REFERENCE_OVERLAY == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_ENV_FILE == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_COMPONENT_VENVS == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_DATASET_OVERLAY == /raid/scratch/* \
        && $CHECKPOINT_E2E_LOCAL_REFERENCE_VIEW_OVERLAY == /raid/scratch/* ]] || {
        echo "EXISTING_FINGERPRINT_FAIL: unsafe node-local execution overrides" >&2
        exit 64
    }
    E2E_EXECUTION_DIR=$CHECKPOINT_E2E_LOCAL_PACKAGE
    GYM_ROOT=$CHECKPOINT_E2E_LOCAL_GYM
    CORRECTED_RUNTIME_OVERLAY=$CHECKPOINT_E2E_LOCAL_RUNTIME
    REFERENCE_OVERLAY=$CHECKPOINT_E2E_LOCAL_REFERENCE_OVERLAY
    ENV_FILE=$CHECKPOINT_E2E_LOCAL_ENV_FILE
    COMPONENT_VENVS=$CHECKPOINT_E2E_LOCAL_COMPONENT_VENVS
    DATASET_OVERLAY=$CHECKPOINT_E2E_LOCAL_DATASET_OVERLAY
    REFERENCE_VIEW_OVERLAY=$CHECKPOINT_E2E_LOCAL_REFERENCE_VIEW_OVERLAY
    PYTHON_BIN=$GYM_ROOT/.venv/bin/python
else
    COMPONENT_VENVS=$RUN_DIR/judge_component_venvs
fi
[[ ${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-} =~ ^[1-4]$ ]] || {
    echo "EXISTING_FINGERPRINT_FAIL: Gemini concurrency must be an integer from 1 through 4" >&2
    exit 64
}
export GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS

fail() { echo "EXISTING_FINGERPRINT_FAIL: $*" >&2; exit 64; }

[[ $ACCOUNT == nemotron_n3_post ]] || fail "campaign account is not nemotron_n3_post: $ACCOUNT"
[[ $CPU_PARTITION == cpu && $CPU_QOS == cpu-normal ]] \
    || fail "unexpected CPU routing: $CPU_PARTITION/$CPU_QOS"
[[ $ACTIVE_PACKAGE == "$RUN_DIR/existing_judge_package" \
    && -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE ]] \
    || fail "run-owned active package is invalid"
[[ $DURABLE_CORRECTED_RUNTIME_OVERLAY == "$RUN_DIR"/* \
    && -d $DURABLE_CORRECTED_RUNTIME_OVERLAY && ! -L $DURABLE_CORRECTED_RUNTIME_OVERLAY ]] \
    || fail "run-owned judge runtime is invalid"
[[ $CORRECTED_TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* \
    && -d $CORRECTED_TRANSPORT_VIEW_ROOT && ! -L $CORRECTED_TRANSPORT_VIEW_ROOT ]] \
    || fail "run-owned transport view is invalid"
[[ $JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && ${#JUDGE_DIR_SUFFIX} -le 32 && $JUDGE_DIR_SUFFIX != e2e ]] \
    || fail "invalid import judge suffix"

[[ -f $RUN_DIR/existing_import_receipt.json && ! -L $RUN_DIR/existing_import_receipt.json \
    && $(sha256sum "$RUN_DIR/existing_import_receipt.json" | awk '{print $1}') == "$IMPORT_RECEIPT_SHA256" ]] \
    || fail "immutable import receipt verification failed"
sha256sum -c --status "$RUN_DIR/existing_import_receipt.json.sha256" \
    || fail "immutable import receipt sidecar verification failed"
"$PYTHON_BIN" "$E2E_EXECUTION_DIR/transport_runtime.py" validate \
    --gym-root "$GYM_ROOT" --runtime-root "$CORRECTED_RUNTIME_OVERLAY" \
    --package-root "$E2E_EXECUTION_DIR" >/dev/null \
    || fail "judge runtime verification failed"

receipt="$RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX"
[[ -f $receipt && ! -L $receipt ]] || fail "transport prebuild receipt is missing"
manifest=$(sed -n 's/^manifest=//p' "$receipt")
manifest_sha=$(sed -n 's/^manifest_sha256=//p' "$receipt")
[[ $(sed -n 's/^schema=//p' "$receipt") == gdpval.transport-prebuild.v1 \
    && $manifest == "$CORRECTED_TRANSPORT_VIEW_ROOT/manifest.json" \
    && $manifest_sha =~ ^[0-9a-f]{64}$ \
    && $(sha256sum "$manifest" | awk '{print $1}') == "$manifest_sha" ]] \
    || fail "transport prebuild receipt drift"

JUDGE_DIR="$RUN_DIR/judge_$JUDGE_DIR_SUFFIX"
OUTPUT="$JUDGE_DIR/gdpval_aav2.jsonl"
PREPROCESSED="$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl"
DISTRIBUTION_PATH="$JUDGE_DIR/occupation_distribution.json"
DATASET_OVERLAY=${DATASET_OVERLAY:-$JUDGE_DIR/dataset_overlay.yaml}
CANDIDATE_VIEW="$CORRECTED_TRANSPORT_VIEW_ROOT/candidate"
REFERENCE_VIEW_OVERLAY=${REFERENCE_VIEW_OVERLAY:-$CORRECTED_TRANSPORT_VIEW_ROOT/reference_views.yaml}
FINGERPRINT_RECEIPT="$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json"
[[ -d $CANDIDATE_VIEW && ! -L $CANDIDATE_VIEW \
    && -f $REFERENCE_VIEW_OVERLAY && ! -L $REFERENCE_VIEW_OVERLAY ]] \
    || fail "transport views are incomplete"

mkdir -p "$JUDGE_DIR/preprocessed_datasets"
"$PYTHON_BIN" "$E2E_EXECUTION_DIR/prepare_existing_campaign.py" prepare-input \
    --run-dir "$RUN_DIR" --output "$PREPROCESSED" >/dev/null \
    || fail "could not freeze provider-free benchmark input"

dataset_overlay_tmp="$DATASET_OVERLAY.tmp.$$"
cat > "$dataset_overlay_tmp" <<EOF
gdpval_stirrup_agent:
  responses_api_agents:
    stirrup_agent:
      datasets:
        - name: gdpval
          type: benchmark
          jsonl_fpath: $DATASET
          prompt_config: null
          prepare_script: benchmarks/gdpval/prepare.py
          num_repeats: 1
EOF
chmod 0400 "$dataset_overlay_tmp"
if [[ -e $DATASET_OVERLAY || -L $DATASET_OVERLAY ]]; then
    [[ -f $DATASET_OVERLAY && ! -L $DATASET_OVERLAY \
        && $(stat -c '%a' "$DATASET_OVERLAY") == 400 ]] \
        || fail "dataset overlay is not an immutable regular file"
    cmp -s "$dataset_overlay_tmp" "$DATASET_OVERLAY" || fail "dataset overlay drift"
    rm -f "$dataset_overlay_tmp"
else
    mv "$dataset_overlay_tmp" "$DATASET_OVERLAY"
fi

# Resolve the same endpoint/model interpolation as the live judge without
# contacting it. Never print secret values.
for try in 1 2 3 4 5 6; do
    if [[ -r $ENV_FILE ]]; then
        while IFS= read -r line; do
            [[ $line == export\ * ]] || continue
            name=${line#export }; name=${name%%=*}
            [[ -n ${!name:-} ]] || eval "$line"
        done < "$ENV_FILE" 2>/dev/null
    fi
    [[ -n ${JUDGE_API_KEY:-} ]] && break
    echo "fingerprint environment read attempt $try failed; retrying in 10s" >&2
    sleep 10
done
[[ -n ${JUDGE_API_KEY:-} && $JUDGE_API_KEY == sk-* ]] \
    || fail "JUDGE_API_KEY must be the sk- LiteLLM key"
: "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"

export PERSIST_DELIVERABLES_DIR="$CANDIDATE_VIEW"
export EXECUTE_ONLY=false
export JUDGE_ONLY=true
export RERUN_INCOMPLETE=true
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3
export STIRRUP_PER_TASK_TIMEOUT_S=1500
export JUDGE_SAMPLING_SEED=9
export GDPVAL_MAX_FILE_BYTES_FOR_JUDGE=335544320
export GDPVAL_MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE=368000000
export GDPVAL_MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE=490000000
export GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE=330301440
export GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE=436207616
export GDPVAL_MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE=495000000
export NEMO_GYM_EXTRA_ROOTS="$CORRECTED_RUNTIME_OVERLAY"
export PYTHONPATH="$CORRECTED_RUNTIME_OVERLAY:$GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

probe() {
    local expected=$1 destination=$2
    "$PYTHON_BIN" "$E2E_EXECUTION_DIR/fingerprint_probe.py" \
        --gym-root "$GYM_ROOT" --runtime-root "$CORRECTED_RUNTIME_OVERLAY" \
        --dataset "$DATASET" --preprocessed-input "$PREPROCESSED" \
        --distribution-path "$DISTRIBUTION_PATH" \
        --output "$OUTPUT" --venv-dir "$COMPONENT_VENVS" \
        --candidate-view "$CANDIDATE_VIEW" --model-name "$MODEL_NAME" \
        --concurrency 8 --expected "$expected" \
        --config "$GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model.yaml" \
        --config "$GYM_ROOT/benchmarks/gdpval/config.yaml" \
        --config "$REFERENCE_OVERLAY" \
        --config "$E2E_EXECUTION_DIR/true3_transport.yaml" \
        --config "$REFERENCE_VIEW_OVERLAY" \
        --config "$DATASET_OVERLAY" > "$destination"
}

exec 9> "$RUN_DIR/fingerprint_${JUDGE_DIR_SUFFIX}.lock"
flock 9
discovery="$FINGERPRINT_RECEIPT.discover.$$"
temporary="$FINGERPRINT_RECEIPT.tmp.$$"
trap 'rm -f "$discovery" "$temporary"' EXIT

if [[ -e $FINGERPRINT_RECEIPT || -L $FINGERPRINT_RECEIPT ]]; then
    [[ -f $FINGERPRINT_RECEIPT && ! -L $FINGERPRINT_RECEIPT \
        && $(stat -c '%a' "$FINGERPRINT_RECEIPT") == 400 ]] \
        || fail "fingerprint receipt is not immutable"
    [[ -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH \
        && $(stat -c '%a' "$DISTRIBUTION_PATH") == 400 ]] \
        || fail "occupation distribution is not immutable"
    distribution_sha=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
    "$PYTHON_BIN" -c 'import json,os,sys; d=json.load(open(sys.argv[1])); p=sys.argv[2]; s=sys.argv[3]; assert os.path.realpath(d["distribution_path"])==os.path.realpath(p) and d["distribution_sha256"]==s' \
        "$FINGERPRINT_RECEIPT" "$DISTRIBUTION_PATH" "$distribution_sha" \
        || fail "occupation distribution receipt drift"
    fingerprint=$("$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS"; print(d["fingerprint"])' "$FINGERPRINT_RECEIPT")
    probe "$fingerprint" "$temporary"
    cmp -s "$temporary" "$FINGERPRINT_RECEIPT" \
        || fail "existing fingerprint receipt no longer matches runtime/profile inputs"
    printf 'FINGERPRINT_PASS fingerprint=%s receipt=%s\n' "$fingerprint" "$FINGERPRINT_RECEIPT"
    exit 0
fi

set +e
probe 0000000000000000000000000000000000000000000000000000000000000000 "$discovery"
discovery_rc=$?
set -e
[[ $discovery_rc == 1 ]] || fail "fingerprint discovery failed with rc=$discovery_rc"
fingerprint=$("$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="MISMATCH"; print(d["fingerprint"])' "$discovery")
[[ $fingerprint =~ ^[0-9a-f]{64}$ ]] || fail "fingerprint discovery emitted an invalid digest"
[[ -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH ]] \
    || fail "fingerprint discovery did not publish a regular occupation distribution"
chmod 0400 "$DISTRIBUTION_PATH"
probe "$fingerprint" "$temporary"
preprocessed_sha=$(sha256sum "$PREPROCESSED" | awk '{print $1}')
distribution_sha=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
"$PYTHON_BIN" -c 'import json,os,sys; d=json.load(open(sys.argv[1])); f,r,p,s,q,t=sys.argv[2:]; assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS" and d["fingerprint"]==f and d["expected_fingerprint"]==f and os.path.realpath(d["runtime_root"])==os.path.realpath(r) and os.path.realpath(d["preprocessed_input"])==os.path.realpath(p) and d["preprocessed_input_sha256"]==s and os.path.realpath(d["distribution_path"])==os.path.realpath(q) and d["distribution_sha256"]==t and d["materialized_row_count"]==220 and d["reference_count"]==9 and d["config_count"]==6' \
    "$temporary" "$fingerprint" "$CORRECTED_RUNTIME_OVERLAY" "$PREPROCESSED" "$preprocessed_sha" "$DISTRIBUTION_PATH" "$distribution_sha" \
    || fail "fingerprint receipt validation failed"
chmod 0400 "$temporary"
mv "$temporary" "$FINGERPRINT_RECEIPT"
rm -f "$discovery"
trap - EXIT
printf 'FINGERPRINT_PASS fingerprint=%s receipt=%s\n' "$fingerprint" "$FINGERPRINT_RECEIPT"
