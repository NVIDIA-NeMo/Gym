#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Strict cached-deliverable AA-v2 judge. No policy GPUs are used.
#SBATCH --job-name=gdpval-e2e-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=184G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${RUN_DIR:?set RUN_DIR}"
: "${STAGES_FILE:?set STAGES_FILE}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"

# A completed campaign can be re-judged under a corrected, immutable transport
# runtime without mutating its frozen rollout state or its original judge
# output.  Preserve these opt-in overrides before settings.env restores the
# package that prepared the campaign.
REQUESTED_ACTIVE_PACKAGE=${CHECKPOINT_E2E_ACTIVE_PACKAGE:-}
REQUESTED_JUDGE_RUNTIME_OVERLAY=${CHECKPOINT_E2E_JUDGE_RUNTIME_OVERLAY:-}
REQUESTED_JUDGE_TRANSPORT_OVERLAY=${CHECKPOINT_E2E_JUDGE_TRANSPORT_OVERLAY:-}
REQUESTED_TRANSPORT_VIEW_ROOT=${CHECKPOINT_E2E_TRANSPORT_VIEW_ROOT:-}
REQUESTED_JUDGE_DIR_SUFFIX=${JUDGE_DIR_SUFFIX:-}

RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] || {
    echo "CHECKPOINT_E2E_JUDGE_FAIL: invalid run id" >&2; exit 64;
}
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "CHECKPOINT_E2E_JUDGE_FAIL: invalid Slurm user" >&2; exit 64;
}
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-judge
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]] || {
    echo "CHECKPOINT_E2E_JUDGE_FAIL: bootstrap root resolves to Lustre" >&2; exit 64;
}
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" judge "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_uv
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
mars_stage_file "$STAGES_FILE" "$MARS_RUN_ROOT/config/stages.txt"
STAGES_FILE=$MARS_RUN_ROOT/config/stages.txt
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
# Keep the original launcher as the authority for the campaign's immutable
# checkpoint, dataset, rollout, and transport-view preflight.
CAMPAIGN_E2E_SCRIPT=$E2E_SCRIPT
CAMPAIGN_E2E_DIR=$E2E_DIR
CAMPAIGN_JUDGE_RUNTIME_OVERLAY=$JUDGE_RUNTIME_OVERLAY
CAMPAIGN_JUDGE_TRANSPORT_OVERLAY=$JUDGE_TRANSPORT_OVERLAY
CAMPAIGN_TRANSPORT_VIEW_ROOT=$TRANSPORT_VIEW_ROOT
override_count=0
for requested in "$REQUESTED_ACTIVE_PACKAGE" "$REQUESTED_JUDGE_RUNTIME_OVERLAY" \
    "$REQUESTED_JUDGE_TRANSPORT_OVERLAY" "$REQUESTED_TRANSPORT_VIEW_ROOT"; do
    [[ -z $requested ]] || override_count=$((override_count + 1))
done
(( override_count == 0 || override_count == 4 )) || {
    echo "CHECKPOINT_E2E_JUDGE_FAIL: corrected rejudge overrides must be supplied as one complete bundle" >&2
    exit 64
}
CORRECTED_REJUDGE=false
if (( override_count == 4 )); then
    CORRECTED_REJUDGE=true
    TRANSPORT_VIEW_ROOT=$REQUESTED_TRANSPORT_VIEW_ROOT
    JUDGE_DELIVERABLES=$TRANSPORT_VIEW_ROOT/candidate
fi
if [[ $CORRECTED_REJUDGE == true ]]; then
    [[ $REQUESTED_ACTIVE_PACKAGE == "$RUN_DIR"/* \
        && -d $REQUESTED_ACTIVE_PACKAGE && ! -L $REQUESTED_ACTIVE_PACKAGE \
        && $REQUESTED_JUDGE_RUNTIME_OVERLAY == "$RUN_DIR"/* \
        && -d $REQUESTED_JUDGE_RUNTIME_OVERLAY && ! -L $REQUESTED_JUDGE_RUNTIME_OVERLAY \
        && -f $REQUESTED_JUDGE_TRANSPORT_OVERLAY \
        && ! -L $REQUESTED_JUDGE_TRANSPORT_OVERLAY ]] || {
        echo "CHECKPOINT_E2E_JUDGE_FAIL: durable corrected runtime contract is invalid" >&2
        exit 64
    }
fi
mars_stage_gym "$GYM_ROOT" "$GYM_REVISION"
mars_stage_runtime
mars_stage_container "$GDPVAL_SIF" "$GDPVAL_SIF_SIGNATURE"
mars_stage_apptainer "$APPTAINER_BIN" "$APPTAINER_SIGNATURE"
mars_stage_file "$REFERENCE_OVERLAY" "$MARS_RUN_ROOT/config/reference_paths.yaml"
mars_stage_file "$ENV_FILE" "$MARS_RUN_ROOT/config/aav2.env"
mars_stage_file "$TRANSPORT_VIEW_ROOT/reference_views.yaml" "$MARS_RUN_ROOT/config/reference_views.yaml"
E2E_DIR=$MARS_PACKAGE
GYM_ROOT=$MARS_GYM
JUDGE_RUNTIME_OVERLAY=$MARS_RUNTIME
JUDGE_TRANSPORT_OVERLAY=$E2E_DIR/true3_transport.yaml
TRANSPORT_VIEWS_PY=$E2E_DIR/transport_views.py
GDPVAL_SIF=$MARS_GDPVAL_SIF
APPTAINER_BIN=$MARS_APPTAINER_BIN
REFERENCE_OVERLAY=$MARS_RUN_ROOT/config/reference_paths.yaml
REFERENCE_VIEW_OVERLAY=$MARS_RUN_ROOT/config/reference_views.yaml
ENV_FILE=$MARS_RUN_ROOT/config/aav2.env
LOCAL_COMPONENT_VENVS=$MARS_RUN_ROOT/component_venvs
LOCAL_DATASET_OVERLAY=$MARS_RUN_ROOT/config/dataset_overlay.yaml
CAMPAIGN_E2E_SCRIPT=$E2E_DIR/run_checkpoint_e2e.sh
export CHECKPOINT_E2E_PYTHON=$MARS_PYTHON
export CHECKPOINT_E2E_LOCAL_PACKAGE=$E2E_DIR
export CHECKPOINT_E2E_LOCAL_GYM=$GYM_ROOT
export CHECKPOINT_E2E_LOCAL_RUNTIME=$JUDGE_RUNTIME_OVERLAY
export CHECKPOINT_E2E_LOCAL_ENV_FILE=$ENV_FILE
export CHECKPOINT_E2E_LOCAL_SETTINGS=$MARS_JOB_ROOT/settings.env
# shellcheck disable=SC1090
source "$E2E_DIR/judge_process_group.sh"

fail() { echo "CHECKPOINT_E2E_JUDGE_FAIL: $*" >&2; exit 64; }

CONCURRENCY=${CONCURRENCY:-16}
CONCURRENCY_LADDER=${CONCURRENCY_LADDER:-16,8,4,1}
PERSISTENT_JUDGE_SESSION=${CHECKPOINT_E2E_PERSISTENT_JUDGE_SESSION:-false}
JUDGE_SERVICE_STARTUP_TIMEOUT_SECONDS=${CHECKPOINT_E2E_JUDGE_SERVICE_STARTUP_TIMEOUT_SECONDS:-2100}
NUM_COMPARISON_TRIALS=${NUM_COMPARISON_TRIALS:-4}
NUM_REPEATS=${NUM_REPEATS:-1}
MULTISTAGE_SEED=${MULTISTAGE_SEED:-42}
ROTATION=${SLURM_RESTART_COUNT:-0}
JUDGE_DIR_SUFFIX=${REQUESTED_JUDGE_DIR_SUFFIX:-e2e}
[[ $JUDGE_DIR_SUFFIX =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || { echo "CHECKPOINT_E2E_JUDGE_FAIL: invalid judge directory suffix: $JUDGE_DIR_SUFFIX" >&2; exit 64; }
(( ${#JUDGE_DIR_SUFFIX} <= 32 )) || fail "judge directory suffix exceeds 32 characters"
JUDGE_DIR="$RUN_DIR/judge_${JUDGE_DIR_SUFFIX}"
OUTPUT_JSONL="$JUDGE_DIR/gdpval_aav2.jsonl"
JOURNAL="$JUDGE_DIR/gdpval_aav2_multistage_state.jsonl"
DISTRIBUTION_PATH="$JUDGE_DIR/occupation_distribution.json"

if [[ $CORRECTED_REJUDGE == true ]]; then
    [[ $JUDGE_DIR_SUFFIX != e2e ]] || fail "corrected rejudge cannot target the original judge_e2e output"
    [[ $E2E_DIR == /raid/scratch/* && -d $E2E_DIR && ! -L $E2E_DIR ]] \
        || fail "node-local corrected package is missing"
    [[ $JUDGE_RUNTIME_OVERLAY == /raid/scratch/* \
        && -d $JUDGE_RUNTIME_OVERLAY && ! -L $JUDGE_RUNTIME_OVERLAY ]] \
        || fail "node-local corrected runtime is missing"
    [[ $JUDGE_TRANSPORT_OVERLAY == "$E2E_DIR/true3_transport.yaml" \
        && -f $JUDGE_TRANSPORT_OVERLAY && ! -L $JUDGE_TRANSPORT_OVERLAY ]] \
        || fail "node-local corrected transport overlay is missing"
    [[ $TRANSPORT_VIEW_ROOT == "$RUN_DIR"/* && -d $TRANSPORT_VIEW_ROOT \
        && ! -L $TRANSPORT_VIEW_ROOT && $TRANSPORT_VIEW_ROOT != "$CAMPAIGN_TRANSPORT_VIEW_ROOT" ]] \
        || fail "corrected transport view must be a distinct sibling below the campaign run"
    [[ -f $TRANSPORT_VIEWS_PY && ! -L $TRANSPORT_VIEWS_PY ]] \
        || fail "corrected transport-view validator is missing"
fi

[[ -r $STAGES_FILE ]] || fail "stages file is unreadable: $STAGES_FILE"
STAGES="$(<"$STAGES_FILE")"
[[ $STAGES == '[{num_tasks: 45, partial_completion: {min_success_fraction: 0.9, min_per_reference_success_fraction: 0.5, min_successful_rows_per_reference: 1, waivable_failure_classes: [timeout_exceeded, transient]}}, {num_tasks: 220, num_models: 4}]' ]] \
    || fail "unexpected multistage schedule: $STAGES"
[[ $NUM_COMPARISON_TRIALS == 4 && $NUM_REPEATS == 1 && $MULTISTAGE_SEED == 42 ]] \
    || fail "judge trial/repeat/seed contract drift"
GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS=${GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS:-2}
[[ $GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS =~ ^[1-4]$ ]] \
    || fail "Gemini concurrency must be an integer from 1 through 4"
export GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS
echo "judge provider cap: gemini_max_concurrent_requests=$GDPVAL_GEMINI_MAX_CONCURRENT_REQUESTS"
[[ $PERSISTENT_JUDGE_SESSION == true || $PERSISTENT_JUDGE_SESSION == false ]] \
    || fail "persistent judge session opt-in must be true or false"
[[ $CONCURRENCY =~ ^[1-9][0-9]*$ ]] || fail "judge concurrency must be a positive integer"
[[ $CONCURRENCY_LADDER == 16,8,4,1 ]] || fail "judge concurrency ladder must be exactly 16,8,4,1"
[[ $JUDGE_SERVICE_STARTUP_TIMEOUT_SECONDS =~ ^[1-9][0-9]*$ \
    && $JUDGE_SERVICE_STARTUP_TIMEOUT_SECONDS -le 3600 ]] \
    || fail "judge service startup timeout must be a positive value at most 3600 seconds"

# Recheck all immutable inputs on the compute node immediately before provider
# calls. This is cheap and avoids trusting flaky login-node Lustre reads.
"$CAMPAIGN_E2E_SCRIPT" _compute-preflight "$RUN_DIR"
if [[ $CORRECTED_REJUDGE == true ]]; then
    "${CHECKPOINT_E2E_PYTHON:-python3}" "$E2E_DIR/transport_runtime.py" validate \
        --gym-root "$GYM_ROOT" \
        --runtime-root "$JUDGE_RUNTIME_OVERLAY" \
        --package-root "$E2E_DIR"
fi
"${CHECKPOINT_E2E_PYTHON:-python3}" "$E2E_DIR/campaign.py" coverage \
    --dataset "$DATASET" --deliverables "$DELIVERABLES" --expected-tasks 220

# The multistage fingerprint binds the prepared-input path, not only its bytes.
# Verify the provider-free receipt against the exact immutable sibling path
# before loading credentials or contacting the judge endpoint.
if [[ $CORRECTED_REJUDGE == true ]]; then
    PREPROCESSED_INPUT=$JUDGE_DIR/preprocessed_datasets/benchmark.jsonl
    FINGERPRINT_RECEIPT=$RUN_DIR/fingerprint_$JUDGE_DIR_SUFFIX.json
    [[ -f $PREPROCESSED_INPUT && ! -L $PREPROCESSED_INPUT && -s $PREPROCESSED_INPUT \
        && -f $DISTRIBUTION_PATH && ! -L $DISTRIBUTION_PATH \
        && $(stat -c '%a' "$DISTRIBUTION_PATH") == 400 \
        && -f $FINGERPRINT_RECEIPT && ! -L $FINGERPRINT_RECEIPT ]] \
        || fail "provider-free fingerprint input/distribution/receipt is missing"
    preprocessed_sha256=$(sha256sum "$PREPROCESSED_INPUT" | awk '{print $1}')
    distribution_sha256=$(sha256sum "$DISTRIBUTION_PATH" | awk '{print $1}')
    "${CHECKPOINT_E2E_PYTHON:-python3}" -c \
        'import json,os,sys; d=json.load(open(sys.argv[1])); p=sys.argv[2]; s=sys.argv[3]; q=sys.argv[4]; t=sys.argv[5]; assert d["schema"]=="gdpval.multistage-fingerprint-probe.v1" and d["status"]=="PASS" and os.path.realpath(d["preprocessed_input"])==os.path.realpath(p) and d["preprocessed_input_sha256"]==s and os.path.realpath(d["distribution_path"])==os.path.realpath(q) and d["distribution_sha256"]==t' \
        "$FINGERPRINT_RECEIPT" "$PREPROCESSED_INPUT" "$preprocessed_sha256" \
        "$DISTRIBUTION_PATH" "$distribution_sha256" \
        || fail "provider-free fingerprint input/distribution path/hash drift"
fi

# Load secrets inside the job; never print their values. A short retry handles
# transient per-inode Lustre failures seen on both login and compute nodes.
for try in 1 2 3 4 5 6; do
    if [[ -r $ENV_FILE ]]; then
        while IFS= read -r line; do
            [[ $line == export\ * ]] || continue
            name=${line#export }; name=${name%%=*}
            [[ -n ${!name:-} ]] || eval "$line"
        done < "$ENV_FILE" 2>/dev/null
    fi
    [[ -n ${JUDGE_API_KEY:-} ]] && break
    echo "judge environment read attempt $try failed; retrying in 10s"
    sleep 10
done
[[ -n ${JUDGE_API_KEY:-} && ${JUDGE_API_KEY} == sk-* ]] \
    || fail "JUDGE_API_KEY must be the sk- LiteLLM key"
: "${JUDGE_BASE_URL:?JUDGE_BASE_URL is required}"
models_json=$(curl -sf -m 20 -H "Authorization: Bearer $JUDGE_API_KEY" \
    "$JUDGE_BASE_URL/models") || fail "judge endpoint rejected the key"
printf '%s' "$models_json" | "${CHECKPOINT_E2E_PYTHON:-python3}" -c '
import json, sys
required = {
    "openai/openai/gpt-5.5",
    "gcp/google/gemini-3.1-pro-preview",
    "aws/anthropic/bedrock-claude-opus-4-8",
}
document = json.load(sys.stdin)
available = {row.get("id") for row in document.get("data", []) if isinstance(row, dict)}
missing = sorted(required - available)
if missing:
    raise SystemExit(f"judge model catalog is missing: {missing}")
' || fail "judge endpoint does not advertise the complete three-model panel"
unset models_json

anchors=$(grep -c 'deliverables_dir:' "$REFERENCE_OVERLAY")
[[ $anchors == 9 ]] || fail "reference overlay has $anchors anchors, expected 9"
while IFS= read -r path; do
    [[ -d $path ]] || fail "reference anchor is unreadable: $path"
done < <(sed -n 's|^[[:space:]]*deliverables_dir:[[:space:]]*\(/lustre/.*\)$|\1|p' "$REFERENCE_OVERLAY")

# The non-provider transport phase builds these views once. Judge retries only
# validate and reuse them, so transport work is never mistaken for a stalled
# provider request by the judge watchdog.
configure_gdpval_container_python \
    "$APPTAINER_BIN/apptainer" "$GDPVAL_SIF" "$E2E_DIR" "$RUN_DIR" "$REFERENCE_OVERLAY" \
    || fail "could not build the exact container bind contract"
container_tail_start=$((${#GDPVAL_CONTAINER_PYTHON[@]} - 2))
(( container_tail_start >= 2 )) || fail "container command is too short to extend safely"
CONTAINER_PYTHON=(
    "${GDPVAL_CONTAINER_PYTHON[@]:0:$container_tail_start}"
    --bind "$GYM_ROOT:$GYM_ROOT:ro"
    --bind "$JUDGE_RUNTIME_OVERLAY:$JUDGE_RUNTIME_OVERLAY:ro"
    "${GDPVAL_CONTAINER_PYTHON[@]:$container_tail_start}"
)
"${CONTAINER_PYTHON[@]}" "$TRANSPORT_VIEWS_PY" check-tools
TRANSPORT_PREBUILD_RECEIPT=$RUN_DIR/TRANSPORT_PREBUILD_PASS
if [[ $CORRECTED_REJUDGE == true ]]; then
    TRANSPORT_PREBUILD_RECEIPT=$RUN_DIR/TRANSPORT_PREBUILD_PASS_$JUDGE_DIR_SUFFIX
fi
[[ -f $TRANSPORT_PREBUILD_RECEIPT && ! -L $TRANSPORT_PREBUILD_RECEIPT ]] \
    || fail "transport prebuild pass receipt is missing"
receipt_schema=$(sed -n 's/^schema=//p' "$TRANSPORT_PREBUILD_RECEIPT")
receipt_manifest=$(sed -n 's/^manifest=//p' "$TRANSPORT_PREBUILD_RECEIPT")
receipt_manifest_sha=$(sed -n 's/^manifest_sha256=//p' "$TRANSPORT_PREBUILD_RECEIPT")
[[ $receipt_schema == gdpval.transport-prebuild.v1 \
    && $receipt_manifest == "$TRANSPORT_VIEW_ROOT/manifest.json" \
    && $receipt_manifest_sha =~ ^[0-9a-f]{64}$ ]] \
    || fail "transport prebuild pass receipt is malformed"
[[ $(sha256sum "$receipt_manifest" | awk '{print $1}') == "$receipt_manifest_sha" ]] \
    || fail "transport prebuild manifest digest mismatch"
manifest_profile=$("${CHECKPOINT_E2E_PYTHON:-python3}" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("derivative_profile", ""))' \
    "$receipt_manifest") || fail "transport manifest profile is unreadable"
overlay_profile=$(sed -n 's/^[[:space:]]*judge_transport_derivative_profile:[[:space:]]*//p' \
    "$JUDGE_TRANSPORT_OVERLAY")
[[ $manifest_profile == reference-pdf-v1+video-h264-720p-crf26-aac128-min8m+ref-video-bundle8-v1 \
    && $overlay_profile == "$manifest_profile" ]] \
    || fail "transport derivative profile does not match the fingerprinted overlay"
"${CONTAINER_PYTHON[@]}" "$TRANSPORT_VIEWS_PY" validate \
    --output-root "$TRANSPORT_VIEW_ROOT"
CANDIDATE_VIEW="$TRANSPORT_VIEW_ROOT/candidate"
[[ -r $REFERENCE_VIEW_OVERLAY && -d $CANDIDATE_VIEW ]] \
    || fail "transport views did not publish their complete contract"

mkdir -p "$JUDGE_DIR/logs"
export PATH="$MARS_UV_DIR:$PATH"
mkdir -p "$UV_CACHE_DIR"
[[ $UV_CACHE_DIR == /raid/scratch/* ]] || fail "UV cache is not node-local"
[[ $(command -v uv) == "$MARS_UV" ]] || fail "uv is not staged node-locally"

export PERSIST_DELIVERABLES_DIR="$CANDIDATE_VIEW"
mkdir -p "$RAY_TMPDIR"
[[ $TMPDIR == /raid/scratch/* && $RAY_TMPDIR == /raid/scratch/* ]] \
    || fail "judge temporary directories are not node-local"
export EXECUTE_ONLY=false
export JUDGE_ONLY=true
export RERUN_INCOMPLETE=true
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3
export STIRRUP_PER_TASK_TIMEOUT_S="${JUDGE_TASK_TIMEOUT_SECONDS:-1500}"
[[ $STIRRUP_PER_TASK_TIMEOUT_S =~ ^[1-9][0-9]*$ && $STIRRUP_PER_TASK_TIMEOUT_S -le 1500 ]] \
    || fail "judge task timeout must be a positive value at most 1500 seconds"
export JUDGE_SAMPLING_SEED="${CHECKPOINT_E2E_JUDGE_SAMPLING_SEED:-9}"
export NEMO_GYM_EXTRA_ROOTS="$JUDGE_RUNTIME_OVERLAY"
export PYTHONPATH="$JUDGE_RUNTIME_OVERLAY:$GYM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Preserve exact large FLOAT-WAV submissions while retaining a 5 MB margin
# below Gemini's observed 500 MB serialized-request hard limit.
export GDPVAL_MAX_FILE_BYTES_FOR_JUDGE=335544320
export GDPVAL_MAX_TOTAL_RAW_ATTACHMENT_BYTES_FOR_JUDGE=368000000
export GDPVAL_MAX_TOTAL_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE=490000000
export GDPVAL_MAX_SECTION_RAW_ATTACHMENT_BYTES_FOR_JUDGE=330301440
export GDPVAL_MAX_SECTION_ENCODED_ATTACHMENT_CHARS_FOR_JUDGE=436207616
export GDPVAL_MAX_TOTAL_SERIALIZED_REQUEST_BYTES_FOR_JUDGE=495000000

# Gym's default head-server port is process-global. CPU nodes are shared, so
# two unrelated eval jobs can otherwise collide before any provider request.
# Give each Slurm job a deterministic slot below the Linux ephemeral range.
# A node-local advisory lock closes the probe-to-bind race between these jobs;
# checking every port in the slot also avoids unrelated listening services.
command -v flock >/dev/null || fail "flock is unavailable on the judge node"
JUDGE_PORT_BASE=12000
JUDGE_PORT_SLOT_WIDTH=20
JUDGE_PORT_SLOT_COUNT=1000
JUDGE_PORT_SEED=${SLURM_JOB_ID:-$$}
[[ $JUDGE_PORT_SEED =~ ^[1-9][0-9]*$ ]] || fail "judge port seed must be a positive integer"
JUDGE_PORT_LOCK_ROOT=/tmp/gdpval_e2e_port_locks_${UID}
mkdir "$JUDGE_PORT_LOCK_ROOT" 2>/dev/null || true
[[ -d $JUDGE_PORT_LOCK_ROOT && ! -L $JUDGE_PORT_LOCK_ROOT && -O $JUDGE_PORT_LOCK_ROOT ]] \
    || fail "unsafe judge port lock directory: $JUDGE_PORT_LOCK_ROOT"
chmod 0700 "$JUDGE_PORT_LOCK_ROOT"
JUDGE_PORT_SLOT=
for ((probe = 0; probe < JUDGE_PORT_SLOT_COUNT; probe++)); do
    candidate_slot=$(((JUDGE_PORT_SEED + probe) % JUDGE_PORT_SLOT_COUNT))
    candidate_head=$((JUDGE_PORT_BASE + candidate_slot * JUDGE_PORT_SLOT_WIDTH))
    candidate_lock=$JUDGE_PORT_LOCK_ROOT/slot_${candidate_slot}.lock
    exec 9>"$candidate_lock"
    if ! flock -n 9; then
        exec 9>&-
        continue
    fi
    candidate_available=true
    for ((candidate_offset = 0; candidate_offset < JUDGE_PORT_SLOT_WIDTH; candidate_offset++)); do
        candidate_port=$((candidate_head + candidate_offset))
        if (exec 7<>"/dev/tcp/127.0.0.1/$candidate_port") >/dev/null 2>&1; then
            candidate_available=false
            break
        fi
    done
    if [[ $candidate_available == true ]]; then
        JUDGE_PORT_SLOT=$candidate_slot
        JUDGE_HEAD_PORT=$candidate_head
        # Keep FD 9 open so the advisory lock survives until this job exits.
        break
    fi
    flock -u 9
    exec 9>&-
done
[[ -n $JUDGE_PORT_SLOT ]] || fail "no free job-scoped Gym port slot is available"
JUDGE_PORT_RANGE_LOW=$((JUDGE_HEAD_PORT + 1))
JUDGE_PORT_RANGE_HIGH=$((JUDGE_HEAD_PORT + JUDGE_PORT_SLOT_WIDTH - 1))
echo "judge Gym ports head=$JUDGE_HEAD_PORT components=$JUDGE_PORT_RANGE_LOW-$JUDGE_PORT_RANGE_HIGH slot=$JUDGE_PORT_SLOT"

DATASET_OVERLAY=$LOCAL_DATASET_OVERLAY
DATASET_OVERLAY_TMP="$DATASET_OVERLAY.tmp.$$"
cat > "$DATASET_OVERLAY_TMP" <<EOF
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
chmod 0400 "$DATASET_OVERLAY_TMP"
if [[ -e $DATASET_OVERLAY || -L $DATASET_OVERLAY ]]; then
    [[ -f $DATASET_OVERLAY && ! -L $DATASET_OVERLAY ]] \
        || fail "dataset overlay is not a regular file: $DATASET_OVERLAY"
    cmp -s "$DATASET_OVERLAY_TMP" "$DATASET_OVERLAY" \
        || fail "dataset overlay drift: $DATASET_OVERLAY"
    rm -f "$DATASET_OVERLAY_TMP"
else
    mv "$DATASET_OVERLAY_TMP" "$DATASET_OVERLAY"
fi

on_term() {
    trap - TERM
    echo "judge stop signal at rotation=$ROTATION; preserving partial state for the next concurrency"
    stop_judge_process_group || true
    exit 75
}
trap on_term TERM

cleanup_session() {
    stop_judge_process_group || true
    if [[ -n ${JUDGE_LOG:-} && -f $JUDGE_LOG && -n ${DURABLE_JUDGE_LOG:-} ]]; then
        cp -- "$JUDGE_LOG" "$DURABLE_JUDGE_LOG.tmp.$$" || true
        if [[ -f $DURABLE_JUDGE_LOG.tmp.$$ ]]; then
            chmod 0600 "$DURABLE_JUDGE_LOG.tmp.$$"
            mv -f "$DURABLE_JUDGE_LOG.tmp.$$" "$DURABLE_JUDGE_LOG"
        fi
    fi
}
trap cleanup_session EXIT

strict_result() {
    "${CHECKPOINT_E2E_PYTHON:-python3}" "$E2E_DIR/campaign.py" result \
        --output "$OUTPUT_JSONL" --journal "$JOURNAL" \
        --dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS"
}

repair_judge_state() {
    local report status post
    [[ -e $OUTPUT_JSONL || -e $JOURNAL ]] || return 0
    [[ -s $JOURNAL && -f $JOURNAL && ! -L $JOURNAL ]] \
        || fail "judge output exists without a usable multistage journal: $JOURNAL"
    if [[ ! -e $OUTPUT_JSONL && ! -L $OUTPUT_JSONL ]]; then
        : > "$OUTPUT_JSONL"
        chmod 0600 "$OUTPUT_JSONL"
    fi
    [[ -f $OUTPUT_JSONL && ! -L $OUTPUT_JSONL ]] \
        || fail "judge output is not a regular file: $OUTPUT_JSONL"
    report=$("${CHECKPOINT_E2E_PYTHON:-python3}" "$E2E_DIR/judge_state.py" sanitize \
        --output "$OUTPUT_JSONL" --journal "$JOURNAL" --deliverables "$CANDIDATE_VIEW" \
        --state-root "$JUDGE_DIR/.checkpoint_e2e_judge_state" --max-attempts 3) \
        || fail "judge-state repair failed between persistent passes"
    status=$(printf '%s\n' "$report" \
        | "${CHECKPOINT_E2E_PYTHON:-python3}" -c 'import json,sys; print(json.load(sys.stdin)["status"])') \
        || fail "judge-state repair emitted an invalid report"
    case "$status" in
        CLEAN)
            ;;
        SANITIZED|RECOVERED)
            post=$("${CHECKPOINT_E2E_PYTHON:-python3}" "$E2E_DIR/judge_state.py" audit \
                --output "$OUTPUT_JSONL" --journal "$JOURNAL" --deliverables "$CANDIDATE_VIEW" \
                --max-attempts 3) || fail "post-repair judge-state audit failed"
            [[ $(printf '%s\n' "$post" \
                | "${CHECKPOINT_E2E_PYTHON:-python3}" -c 'import json,sys; print(json.load(sys.stdin)["status"])') == CLEAN ]] \
                || fail "judge-state repair did not leave a clean active cache"
            ;;
        *)
            fail "unexpected judge-state repair status: $status"
            ;;
    esac
    echo "judge-state=$status"
}

cd "$GYM_ROOT"
GYM_PYTHON="$GYM_ROOT/.venv/bin/python"
GYM_ENTRYPOINT="$E2E_DIR/gym_entrypoint.py"
[[ -x $GYM_PYTHON && -f $GYM_ENTRYPOINT && ! -L $GYM_ENTRYPOINT ]] \
    || fail "node-local Gym entrypoint is unavailable"
POLICY_URL=http://127.0.0.1:9/v1
REUSE_PREPROCESSED_ARGS=()
if [[ $CORRECTED_REJUDGE == true ]]; then
    REUSE_PREPROCESSED_ARGS+=(++reuse_existing_data_preparation=true)
fi

# Recovery concurrency belongs only to Gym's top-level dispatcher. Keep the
# pinned PR #2588 stirrup-agent config untouched so its scientific fingerprint stays
# identical across resume passes.
JUDGE_LOG="$MARS_JOB_ROOT/logs/judge.log"
DURABLE_JUDGE_LOG="$JUDGE_DIR/logs/judge_${SLURM_JOB_ID}_${ROTATION}.log"
echo "judge runtime started; detailed log: $JUDGE_LOG"
if [[ $PERSISTENT_JUDGE_SESSION == false ]]; then
    # Proven legacy path: one service lifecycle and one dispatcher pass. The
    # controller may submit a smaller-concurrency follow-up job.
    start_judge_process_group "$JUDGE_LOG" "$GYM_PYTHON" "$GYM_ENTRYPOINT" eval run \
        --config responses_api_models/vllm_model/configs/vllm_model.yaml \
        --config benchmarks/gdpval/config.yaml \
        --config "$REFERENCE_OVERLAY" \
        --config "$JUDGE_TRANSPORT_OVERLAY" \
        --config "$REFERENCE_VIEW_OVERLAY" \
        --config "$DATASET_OVERLAY" \
        --split benchmark --output "$OUTPUT_JSONL" \
        --model-url "$POLICY_URL" --model-api-key dummy --model "$MODEL_NAME" \
        --concurrency "$CONCURRENCY" --num-repeats 1 \
        --max-output-tokens 262144 --resume \
        "${REUSE_PREPROCESSED_ARGS[@]}" \
        ++head_server.host=127.0.0.1 \
        ++head_server.port="$JUDGE_HEAD_PORT" \
        ++port_range_low="$JUDGE_PORT_RANGE_LOW" \
        ++port_range_high="$JUDGE_PORT_RANGE_HIGH" \
        ++overwrite_metrics_conflicts=true \
        ++skip_venv_if_present=true \
        ++uv_venv_dir="$LOCAL_COMPONENT_VENVS" \
        "++policy_model.responses_api_models.vllm_model.base_url=[\"$POLICY_URL\"]" \
        ++policy_model.responses_api_models.vllm_model.model="$MODEL_NAME" \
        ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=true \
        ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.rerun_incomplete=true \
        ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir="$CANDIDATE_VIEW" \
        ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
        ++gdpval_resources_server.resources_servers.gdpval.num_comparison_trials=4 \
        ++gdpval_resources_server.resources_servers.gdpval.preconvert_office_to_pdf=false \
        ++multistage.enabled=true \
        "++multistage.stages=$STAGES" \
        ++multistage.dataset_path="$DATASET" \
        ++multistage.distribution_path="$DISTRIBUTION_PATH" \
        ++multistage.seed=42
    wait_judge_process_group
    gym_rc=$GYM_RC
    echo "gym_rc=$gym_rc; applying exact scientific acceptance gate"
    strict_result && exit 0
    (( gym_rc != 0 )) && exit "$gym_rc"
    exit 75
fi

# Opt-in persistent path: the service owner has no rollout-collection driver.
# It retains the same Ray cluster, ports, component processes, and component
# environments while the helper invokes only the pinned driver for each
# bounded resume pass.
HEAD_URL="http://127.0.0.1:$JUDGE_HEAD_PORT"
SESSION_PY="$E2E_DIR/judge_session.py"
[[ -f $SESSION_PY && ! -L $SESSION_PY && -x $GYM_PYTHON ]] \
    || fail "persistent judge-session helper or Gym Python is unavailable"
start_judge_process_group "$JUDGE_LOG" "$GYM_PYTHON" "$GYM_ENTRYPOINT" env start \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config benchmarks/gdpval/config.yaml \
    --config "$REFERENCE_OVERLAY" \
    --config "$JUDGE_TRANSPORT_OVERLAY" \
    --config "$REFERENCE_VIEW_OVERLAY" \
    --config "$DATASET_OVERLAY" \
    --model-url "$POLICY_URL" --model-api-key dummy --model "$MODEL_NAME" \
    "${REUSE_PREPROCESSED_ARGS[@]}" \
    ++split=benchmark \
    ++output_jsonl_fpath="$OUTPUT_JSONL" \
    ++num_samples_in_parallel=16 \
    ++num_repeats=1 \
    ++responses_create_params.max_output_tokens=262144 \
    ++resume_from_cache=true \
    ++head_server.host=127.0.0.1 \
    ++head_server.port="$JUDGE_HEAD_PORT" \
    ++port_range_low="$JUDGE_PORT_RANGE_LOW" \
    ++port_range_high="$JUDGE_PORT_RANGE_HIGH" \
    ++overwrite_metrics_conflicts=true \
    ++skip_venv_if_present=true \
    ++uv_venv_dir="$LOCAL_COMPONENT_VENVS" \
    "++policy_model.responses_api_models.vllm_model.base_url=[\"$POLICY_URL\"]" \
    ++policy_model.responses_api_models.vllm_model.model="$MODEL_NAME" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=true \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.rerun_incomplete=true \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir="$CANDIDATE_VIEW" \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.num_comparison_trials=4 \
    ++gdpval_resources_server.resources_servers.gdpval.preconvert_office_to_pdf=false \
    ++multistage.enabled=true \
    "++multistage.stages=$STAGES" \
    ++multistage.dataset_path="$DATASET" \
    ++multistage.distribution_path="$DISTRIBUTION_PATH" \
    ++multistage.seed=42

if ! "$GYM_PYTHON" "$SESSION_PY" wait \
    --head-url "$HEAD_URL" --owner-pid "$GYM_PID" \
    --timeout-seconds "$JUDGE_SERVICE_STARTUP_TIMEOUT_SECONDS" \
    --expected-service policy_model \
    --expected-service gdpval_judge_model \
    --expected-service gdpval_resources_server \
    --expected-service gdpval_stirrup_agent \
    --expected-service gdpval_gpt55_judge_model \
    --expected-service gdpval_gemini31_judge_model \
    --expected-service gdpval_claude48_judge_model; then
    echo "judge service startup failed; node-local log tail follows" >&2
    tail -n 240 "$JUDGE_LOG" >&2 || true
    exit 64
fi

IFS=, read -r -a judge_concurrencies <<< "$CONCURRENCY_LADDER"
for pass_index in "${!judge_concurrencies[@]}"; do
    concurrency=${judge_concurrencies[$pass_index]}
    if (( pass_index > 0 )); then
        repair_judge_state
    fi
    kill -0 "$GYM_PID" 2>/dev/null || {
        wait_judge_process_group
        (( GYM_RC != 0 )) && exit "$GYM_RC"
        exit 75
    }
    echo "persistent judge pass=$((pass_index + 1))/${#judge_concurrencies[@]} concurrency=$concurrency"
    pass_rc=0
    "$GYM_PYTHON" "$SESSION_PY" run-pass \
        --head-url "$HEAD_URL" --concurrency "$concurrency" >> "$JUDGE_LOG" 2>&1 \
        || pass_rc=$?
    echo "persistent judge pass concurrency=$concurrency rc=$pass_rc; applying exact scientific acceptance gate"
    strict_result && exit 0
    if ! kill -0 "$GYM_PID" 2>/dev/null; then
        wait_judge_process_group
        (( GYM_RC != 0 )) && exit "$GYM_RC"
        exit 75
    fi
done

repair_judge_state
strict_result && exit 0
echo "strict final result remained incomplete after persistent concurrency ladder $CONCURRENCY_LADDER" >&2
exit 76
