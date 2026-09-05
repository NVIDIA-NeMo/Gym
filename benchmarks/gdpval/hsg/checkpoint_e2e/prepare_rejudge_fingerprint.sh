#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Publish a provider-free fingerprint receipt for a corrected sibling judge run.

set -euo pipefail
umask 077

[[ $# == 1 ]] || { echo "usage: $0 RUN_DIR" >&2; exit 64; }
RUN_DIR=$(cd -P -- "$1" && pwd -P)
: "${ACTIVE_PACKAGE:?set ACTIVE_PACKAGE}"
: "${CORRECTED_RUNTIME_OVERLAY:?set CORRECTED_RUNTIME_OVERLAY}"
: "${CORRECTED_TRANSPORT_VIEW_ROOT:?set CORRECTED_TRANSPORT_VIEW_ROOT}"
REQUESTED_ACTIVE_PACKAGE=$ACTIVE_PACKAGE
REQUESTED_CORRECTED_RUNTIME_OVERLAY=$CORRECTED_RUNTIME_OVERLAY
REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT=$CORRECTED_TRANSPORT_VIEW_ROOT
REQUESTED_JUDGE_DIR_SUFFIX=${JUDGE_DIR_SUFFIX:-nested_refs_v1}
[[ $REQUESTED_JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ \
    && ${#REQUESTED_JUDGE_DIR_SUFFIX} -le 32 ]] \
    || { echo "invalid JUDGE_DIR_SUFFIX" >&2; exit 64; }

# shellcheck disable=SC1090
source "$RUN_DIR/settings.env"
CAMPAIGN_E2E_SCRIPT=$E2E_SCRIPT
CAMPAIGN_E2E_DIR=$E2E_DIR
CAMPAIGN_JUDGE_RUNTIME_OVERLAY=$JUDGE_RUNTIME_OVERLAY
CAMPAIGN_TRANSPORT_VIEW_ROOT=$TRANSPORT_VIEW_ROOT
ACTIVE_PACKAGE=$REQUESTED_ACTIVE_PACKAGE
CORRECTED_RUNTIME_OVERLAY=$REQUESTED_CORRECTED_RUNTIME_OVERLAY
CORRECTED_TRANSPORT_VIEW_ROOT=$REQUESTED_CORRECTED_TRANSPORT_VIEW_ROOT
JUDGE_DIR_SUFFIX=$REQUESTED_JUDGE_DIR_SUFFIX
JUDGE_DIR=$RUN_DIR/judge_$JUDGE_DIR_SUFFIX
OUTPUT=$JUDGE_DIR/gdpval_aav2.jsonl
RECEIPT=$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json
SOURCE_PREPROCESSED=$RUN_DIR/judge_e2e/preprocessed_datasets/benchmark.jsonl
PREPROCESSED=$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl
DISTRIBUTION_PATH=$JUDGE_DIR/occupation_distribution.json
DATASET_OVERLAY=$RUN_DIR/judge_e2e/dataset_overlay.yaml
CANDIDATE_VIEW=$CORRECTED_TRANSPORT_VIEW_ROOT/candidate
REFERENCE_VIEW_OVERLAY=$CORRECTED_TRANSPORT_VIEW_ROOT/reference_views.yaml

[[ $ACTIVE_PACKAGE == /* && -d $ACTIVE_PACKAGE && ! -L $ACTIVE_PACKAGE \
    && $ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR" \
    && $CORRECTED_RUNTIME_OVERLAY == "$RUN_DIR"/* \
    && -d $CORRECTED_RUNTIME_OVERLAY && ! -L $CORRECTED_RUNTIME_OVERLAY \
    && $CORRECTED_RUNTIME_OVERLAY != "$CAMPAIGN_JUDGE_RUNTIME_OVERLAY" \
    && $CORRECTED_TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* \
    && $CORRECTED_TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT" \
    && -d $CORRECTED_TRANSPORT_VIEW_ROOT && ! -L $CORRECTED_TRANSPORT_VIEW_ROOT \
    && -d $CANDIDATE_VIEW && ! -L $CANDIDATE_VIEW \
    && -f $REFERENCE_VIEW_OVERLAY && ! -L $REFERENCE_VIEW_OVERLAY \
    && -f $SOURCE_PREPROCESSED && ! -L $SOURCE_PREPROCESSED \
    && -f $DATASET_OVERLAY && ! -L $DATASET_OVERLAY \
    && $JUDGE_DIR_SUFFIX != e2e ]] \
    || { echo "corrected fingerprint inputs are incomplete" >&2; exit 64; }
"$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR" >/dev/null
python3 "$ACTIVE_PACKAGE/transport_runtime.py" validate \
    --gym-root "$GYM_ROOT" --runtime-root "$CORRECTED_RUNTIME_OVERLAY" \
    --package-root "$ACTIVE_PACKAGE" >/dev/null

# Resolve the judge endpoint/model environment exactly as judge.sbatch does.
# Configuration parsing is provider-free, but these values affect the resolved
# runtime config (and therefore its fingerprint). Never print their values.
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
[[ -n ${JUDGE_API_KEY:-} && ${JUDGE_API_KEY} == sk-* ]] \
    || { echo "JUDGE_API_KEY must be the sk- LiteLLM key" >&2; exit 64; }
: "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"

# Serialize both sibling-input publication and receipt publication. The prepared
# input path itself is fingerprinted, so it belongs to the same transaction.
exec 9> "$RUN_DIR/fingerprint_${JUDGE_DIR_SUFFIX}.lock"
flock 9
mkdir -p "$JUDGE_DIR/preprocessed_datasets"
preprocessed_temporary=$PREPROCESSED.tmp.$$
trap 'rm -f "$preprocessed_temporary"' EXIT
if [[ -e $PREPROCESSED || -L $PREPROCESSED ]]; then
    preprocessed_mode=$(python3 -c 'from pathlib import Path; import stat,sys; print(format(stat.S_IMODE(Path(sys.argv[1]).stat().st_mode), "o"))' "$PREPROCESSED" 2>/dev/null || true)
    [[ -f $PREPROCESSED && ! -L $PREPROCESSED && $preprocessed_mode == 400 \
        && -s $PREPROCESSED ]] \
        || { echo "sibling preprocessed input is not an immutable regular file" >&2; exit 64; }
    cmp -s "$SOURCE_PREPROCESSED" "$PREPROCESSED" \
        || { echo "sibling preprocessed input differs from the frozen campaign input" >&2; exit 64; }
else
    cp --reflink=auto -- "$SOURCE_PREPROCESSED" "$preprocessed_temporary"
    cmp -s "$SOURCE_PREPROCESSED" "$preprocessed_temporary" \
        || { echo "sibling preprocessed input copy verification failed" >&2; exit 64; }
    chmod 0400 "$preprocessed_temporary"
    mv "$preprocessed_temporary" "$PREPROCESSED"
fi
trap - EXIT
export PERSIST_DELIVERABLES_DIR=$CANDIDATE_VIEW
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CORRECTED_RUNTIME_OVERLAY:$GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

probe() {
    local expected=$1 destination=$2
    "$GYM_ROOT/.venv/bin/python" "$ACTIVE_PACKAGE/fingerprint_probe.py" \
        --gym-root "$GYM_ROOT" --runtime-root "$CORRECTED_RUNTIME_OVERLAY" \
        --dataset "$DATASET" --preprocessed-input "$PREPROCESSED" \
        --distribution-path "$DISTRIBUTION_PATH" \
        --output "$OUTPUT" --venv-dir "$RUN_DIR/judge_component_venvs" \
        --candidate-view "$CANDIDATE_VIEW" --model-name "$MODEL_NAME" \
        --concurrency 8 --expected "$expected" \
        --config "$GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model.yaml" \
        --config "$GYM_ROOT/benchmarks/gdpval/config.yaml" \
        --config "$REFERENCE_OVERLAY" \
        --config "$ACTIVE_PACKAGE/true3_transport.yaml" \
        --config "$REFERENCE_VIEW_OVERLAY" \
        --config "$DATASET_OVERLAY" > "$destination"
}

preprocessed_sha256=$(sha256sum "$PREPROCESSED" | awk '{print $1}')
discovery=$RECEIPT.discover.$$
temporary=$RECEIPT.tmp.$$
trap 'rm -f "$discovery" "$temporary"' EXIT

if [[ -e $RECEIPT || -L $RECEIPT ]]; then
    receipt_mode=$(python3 -c 'from pathlib import Path; import stat,sys; print(format(stat.S_IMODE(Path(sys.argv[1]).stat().st_mode), "o"))' "$RECEIPT" 2>/dev/null || true)
    [[ -f $RECEIPT && ! -L $RECEIPT && $receipt_mode == 400 ]] \
        || { echo "fingerprint receipt is not a regular file" >&2; exit 64; }
    distribution_mode=$(python3 -c 'from pathlib import Path; import stat,sys; print(format(stat.S_IMODE(Path(sys.argv[1]).stat().st_mode), "o"))' "$DISTRIBUTION_PATH" 2>/dev/null || true)
    [[ -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH && $distribution_mode == 400 ]] \
        || { echo "occupation distribution is not immutable" >&2; exit 64; }
    distribution_sha256=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
    python3 -c 'import json,os,sys; d=json.load(open(sys.argv[1])); p=sys.argv[2]; s=sys.argv[3]; assert os.path.realpath(d["distribution_path"])==os.path.realpath(p) and d["distribution_sha256"]==s' \
        "$RECEIPT" "$DISTRIBUTION_PATH" "$distribution_sha256" \
        || { echo "occupation distribution receipt drift" >&2; exit 64; }
    fingerprint=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS"; print(d["fingerprint"])' "$RECEIPT")
    probe "$fingerprint" "$temporary"
    cmp -s "$temporary" "$RECEIPT" \
        || { echo "existing fingerprint receipt no longer matches current runtime/profile inputs" >&2; exit 64; }
    printf 'FINGERPRINT_PASS fingerprint=%s receipt=%s\n' "$fingerprint" "$RECEIPT"
    exit 0
fi

set +e
probe 0000000000000000000000000000000000000000000000000000000000000000 "$discovery"
discovery_rc=$?
set -e
[[ $discovery_rc == 1 ]] \
    || { echo "fingerprint discovery failed with unexpected rc=$discovery_rc" >&2; exit 64; }
fingerprint=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="MISMATCH"; print(d["fingerprint"])' "$discovery")
original_fingerprint=$(python3 -c 'import json,sys; values={json.loads(line)["fingerprint"] for line in open(sys.argv[1]) if line.strip()}; assert len(values)==1; print(values.pop())' \
    "$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl")
[[ $fingerprint =~ ^[0-9a-f]{64}$ && $fingerprint != "$original_fingerprint" ]] \
    || { echo "corrected fingerprint did not enter a new semantic namespace" >&2; exit 64; }
[[ -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH ]] \
    || { echo "fingerprint discovery did not publish a regular occupation distribution" >&2; exit 64; }
chmod 0400 "$DISTRIBUTION_PATH"
probe "$fingerprint" "$temporary"
distribution_sha256=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
python3 -c 'import json,os,sys; d=json.load(open(sys.argv[1])); e=sys.argv[2]; r=sys.argv[3]; p=sys.argv[4]; s=sys.argv[5]; q=sys.argv[6]; t=sys.argv[7]; assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS" and d["fingerprint"]==e and d["expected_fingerprint"]==e and d["runtime_root"]==r and d["preprocessed_input"]==p and d["preprocessed_input_sha256"]==s and os.path.realpath(d["distribution_path"])==os.path.realpath(q) and d["distribution_sha256"]==t and d["materialized_row_count"]==220 and d["reference_count"]==9 and d["config_count"]==6' \
    "$temporary" "$fingerprint" "$CORRECTED_RUNTIME_OVERLAY" "$PREPROCESSED" "$preprocessed_sha256" "$DISTRIBUTION_PATH" "$distribution_sha256"
chmod 0400 "$temporary"
mv "$temporary" "$RECEIPT"
trap - EXIT
rm -f "$discovery"
printf 'FINGERPRINT_PASS fingerprint=%s receipt=%s\n' "$fingerprint" "$RECEIPT"
