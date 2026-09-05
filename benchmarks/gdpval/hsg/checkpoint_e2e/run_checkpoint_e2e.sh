#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One-checkpoint GDPVal rollout + AA-v2 multistage ELO launcher for HSG.
set -euo pipefail

umask 077

SCRIPT_DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CAMPAIGN_PY="$SCRIPT_DIR/campaign.py"
TRANSPORT_RUNTIME_PY="$SCRIPT_DIR/transport_runtime.py"
TRANSPORT_VIEWS_PY="$SCRIPT_DIR/transport_views.py"
FINGERPRINT_PROBE_PY="$SCRIPT_DIR/fingerprint_probe.py"
TRANSPORT_OVERLAY="$SCRIPT_DIR/true3_transport.yaml"
TRANSPORT_PATCH="$SCRIPT_DIR/runtime_sources/pr2588_true3_transport.patch"
PROVIDER_IMAGE_CAPS_PATCH="$SCRIPT_DIR/runtime_sources/provider_image_caps.patch"
PROVIDER_AGGREGATE_MEDIA_CAPS_PATCH="$SCRIPT_DIR/runtime_sources/provider_aggregate_media_caps.patch"
RECURSIVE_REFERENCE_ASSETS_PATCH="$SCRIPT_DIR/runtime_sources/recursive_reference_assets.patch"
STRICT_COMPARISON_TRIALS_PATCH="$SCRIPT_DIR/runtime_sources/strict_comparison_trials.patch"
PROVIDER_CONTEXT_FALLBACK_PATCH="$SCRIPT_DIR/runtime_sources/provider_context_fallback.patch"
PROVIDER_RATE_LIMIT_BACKOFF_PATCH="$SCRIPT_DIR/runtime_sources/provider_rate_limit_backoff.patch"
PARTIAL_PDF_OVERFLOW_PATCH="$SCRIPT_DIR/runtime_sources/partial_pdf_overflow.patch"
GEMINI_PDF_PART_CAP_PATCH="$SCRIPT_DIR/runtime_sources/gemini_pdf_part_cap.patch"
TRANSPORT_ASSIGNMENT="$SCRIPT_DIR/runtime_sources/transport_assignment.py"
SLURM_RECEIPTS_SH="$SCRIPT_DIR/slurm_receipts.sh"
ROLLOUT_LIFECYCLE_SH="$SCRIPT_DIR/rollout_lifecycle.sh"
ROLLOUT_SHARD_COVERAGE_PY="$SCRIPT_DIR/rollout_shard_coverage.py"
PYTHON_BIN="${CHECKPOINT_E2E_PYTHON:-python3}"

[[ -r $SLURM_RECEIPTS_SH ]] || {
    echo "CHECKPOINT_E2E_FAIL: Slurm receipt helper is missing: $SLURM_RECEIPTS_SH" >&2
    exit 64
}
# shellcheck disable=SC1090
source "$SLURM_RECEIPTS_SH"

OWNER_ROOT="${CHECKPOINT_E2E_OWNER_ROOT:-/lustre/fsw/portfolios/llmservice/users/spanev}"
AAV2_ROOT="${CHECKPOINT_E2E_AAV2_ROOT:-$OWNER_ROOT/gdpval_colo/aav2}"
UNIFIED_ROOT="${CHECKPOINT_E2E_UNIFIED_ROOT:-$OWNER_ROOT/gdpval_colo/unified}"
CAMPAIGN_ROOT="${CHECKPOINT_E2E_ROOT:-$AAV2_ROOT/checkpoint_e2e_true3_v1_4_13_runs}"
# The maintained rollout harness was intentionally removed from the newer
# judging branch. Keep the proven rollout tree separate from the PR runtime so
# each phase runs the source that actually owns its contract.
ROLLOUT_GYM_ROOT="${CHECKPOINT_E2E_ROLLOUT_GYM_ROOT:-$OWNER_ROOT/gdpval_integ}"
EXPECTED_ROLLOUT_GYM_REVISION="${CHECKPOINT_E2E_EXPECTED_ROLLOUT_GYM_REVISION:-626d2c2654912ec2f0c62d2d440888751a3a5b96}"
GYM_ROOT="${CHECKPOINT_E2E_GYM_ROOT:-$OWNER_ROOT/gdpval_colo/runtime/gym-pr2588-d3f146d}"
EXPECTED_GYM_REVISION="${CHECKPOINT_E2E_EXPECTED_GYM_REVISION:-d3f146d386c7dfe07d4fabce32c4c8b14c7917d2}"
DATASET="${CHECKPOINT_E2E_DATASET:-$AAV2_ROOT/gdpval_benchmark.local.jsonl}"
REFERENCE_OVERLAY="${CHECKPOINT_E2E_REFERENCE_OVERLAY:-$AAV2_ROOT/aa_v2_reference_paths.mirrored.yaml}"
ENV_FILE="${CHECKPOINT_E2E_ENV_FILE:-$AAV2_ROOT/aav2.env}"
ROLLOUT_SBATCH="${CHECKPOINT_E2E_ROLLOUT_SBATCH:-$SCRIPT_DIR/gdpval_rollout.sbatch}"
SERVE_SCRIPT="${CHECKPOINT_E2E_SERVE_SCRIPT:-$UNIFIED_ROOT/serve/serve_vllm_replica.sh}"
JUDGE_SBATCH="${CHECKPOINT_E2E_JUDGE_SBATCH:-$SCRIPT_DIR/judge.sbatch}"
PARSER_ROOT="${CHECKPOINT_E2E_PARSER_ROOT:-$AAV2_ROOT/parsers}"
PARSER_PLUGIN="${CHECKPOINT_E2E_PARSER_PLUGIN:-$PARSER_ROOT/ultra_v3_reasoning_parser.py}"
VLLM_CONTAINER="${CHECKPOINT_E2E_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/danz/images/vllm-hsg-03-16.sqsh}"
GDPVAL_SIF="${CHECKPOINT_E2E_GDPVAL_SIF:-/lustre/fsw/portfolios/llmservice/users/vadams/containers/python-3.12.gdpval.sqsh}"
AGENT_SIF="${CHECKPOINT_E2E_AGENT_SIF:-/lustre/fsw/portfolios/llmservice/users/agronskiy/images/apptainer/python-3.13.gdpval.gym-32083a.sif}"
APPTAINER_BIN="${CHECKPOINT_E2E_APPTAINER_BIN:-$OWNER_ROOT/gdpval_colo/apptainer-1.5.1-bigsession/bin}"

ACCOUNT="${CHECKPOINT_E2E_ACCOUNT:-nemotron_n3_post}"
GPU_PARTITION="${CHECKPOINT_E2E_GPU_PARTITION:-batch}"
GPU_QOS="${CHECKPOINT_E2E_GPU_QOS:-normal}"
CPU_PARTITION="${CHECKPOINT_E2E_CPU_PARTITION:-cpu}"
CPU_QOS="${CHECKPOINT_E2E_CPU_QOS:-cpu-normal}"
ROLLOUT_CONCURRENCY="${CHECKPOINT_E2E_ROLLOUT_CONCURRENCY:-20}"
RECOVERY_CONCURRENCY="${CHECKPOINT_E2E_RECOVERY_CONCURRENCY:-8}"
ROLLOUT_WALL="${CHECKPOINT_E2E_ROLLOUT_WALL:-03:15:00}"
JUDGE_WALL="${CHECKPOINT_E2E_JUDGE_WALL:-04:00:00}"
EXPECTED_TASKS=220
SHARD_COUNT=6

fail() {
    echo "CHECKPOINT_E2E_FAIL: $*" >&2
    exit 64
}

usage() {
    cat <<'EOF'
Usage: run_checkpoint_e2e.sh prepare|submit|all|resume|status|result CHECKPOINT

  prepare  Build and verify the deterministic campaign; no Slurm/provider work.
  submit   Submit six rollout shards and the autonomous CPU controller.
  all      Prepare, then submit.
  resume   Restart only the controller/recovery path when no controller is live.
  status   Read-only progress summary.
  result   Require accepted Stage 0 calibration plus exact 220/880 Stage 1, then show ELO.

Set CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS=true for submit/all/resume to allow
the controller to start the three-provider judging phase. Secrets stay in the
protected CHECKPOINT_E2E_ENV_FILE and are never written into campaign state.
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command is unavailable: $1"
}

normalize_job_id() {
    slurm_normalize_job_id "$1" || fail "invalid Slurm job id"
}

read_job_id() {
    slurm_read_job_receipt "$1" || fail "invalid job receipt: $1"
}

validate_dependency_policy() {
    local config
    config=$(scontrol show config 2>/dev/null) \
        || fail "could not read Slurm dependency policy"
    awk -F '=' '
        /^[[:space:]]*DependencyParameters[[:space:]]*=/ {
            count = split($2, values, ",")
            for (i = 1; i <= count; i++) {
                gsub(/[[:space:]]/, "", values[i])
                if (values[i] == "kill_invalid_depend") found = 1
            }
        }
        END {exit !found}
    ' <<<"$config" || fail "Slurm must enable DependencyParameters=kill_invalid_depend"
}

resolve_checkpoint() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=True))' "$1"
}

location_lines() {
    "$PYTHON_BIN" "$CAMPAIGN_PY" locate --checkpoint "$1" --campaign-root "$CAMPAIGN_ROOT"
}

locate_run() {
    local checkpoint=$1 location
    location="$(location_lines "$checkpoint")"
    RUN_ID="$(printf '%s\n' "$location" | sed -n 's/^RUN_ID=//p')"
    RUN_DIR="$(printf '%s\n' "$location" | sed -n 's/^RUN_DIR=//p')"
    [[ -n $RUN_ID && -n $RUN_DIR ]] || fail "could not derive campaign location"
    SETTINGS="$RUN_DIR/settings.env"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

stat_signature() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import sys; s=Path(sys.argv[1]).stat(); print(f"{s.st_size}:{s.st_mtime_ns}")' "$1"
}

file_mode() {
    "$PYTHON_BIN" -c 'from pathlib import Path; import stat,sys; print(format(stat.S_IMODE(Path(sys.argv[1]).stat().st_mode), "o"))' "$1"
}

safe_model_name() {
    if [[ -n ${CHECKPOINT_E2E_MODEL_NAME:-} ]]; then
        printf '%s\n' "$CHECKPOINT_E2E_MODEL_NAME"
        return
    fi
    local checkpoint=$1 leaf parent model
    leaf="$(basename "$checkpoint")"
    parent="$(basename "$(dirname "$checkpoint")")"
    if [[ $leaf == hf ]]; then
        model="$(basename "$(dirname "$(dirname "$checkpoint")")")-$parent"
    else
        model="$leaf"
    fi
    printf '%s\n' "$model" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-//; s/-$//'
}

publish_env_file() {
    local target=$1
    shift
    local temporary="${target}.tmp.$$" pair key value
    mkdir -p "$(dirname "$target")"
    : > "$temporary"
    for pair in "$@"; do
        key=${pair%%=*}
        value=${pair#*=}
        [[ $key =~ ^[A-Z][A-Z0-9_]*$ ]] || fail "unsafe generated setting name: $key"
        printf '%s=%q\n' "$key" "$value" >> "$temporary"
    done
    chmod 0400 "$temporary"
    if [[ -e $target ]]; then
        cmp -s "$temporary" "$target" || {
            rm -f "$temporary"
            fail "prepared file drift: $target"
        }
        rm -f "$temporary"
    else
        mv "$temporary" "$target"
    fi
}

publish_runtime_pins() {
    local target=$1
    local temporary="${target}.tmp.$$" path
    shift
    : > "$temporary"
    for path in "$@"; do
        [[ -r $path ]] || fail "runtime source is unreadable: $path"
        sha256sum "$path" >> "$temporary"
    done
    chmod 0400 "$temporary"
    if [[ -e $target ]]; then
        cmp -s "$temporary" "$target" || {
            rm -f "$temporary"
            fail "runtime source inventory drift: $target"
        }
        rm -f "$temporary"
    else
        mv "$temporary" "$target"
    fi
}

source_settings() {
    [[ -f $SETTINGS && ! -L $SETTINGS ]] || fail "campaign is not prepared: $SETTINGS"
    [[ $(file_mode "$SETTINGS") == 400 ]] || fail "settings permissions drift: $SETTINGS"
    # settings.env is generated locally from validated absolute paths and kept 0400.
    # shellcheck disable=SC1090
    source "$SETTINGS"
}

acquire_launcher_lock() {
    local lock="$RUN_DIR/launcher.lock"
    require_command flock
    if [[ -e $lock || -L $lock ]]; then
        [[ -f $lock && ! -L $lock ]] || fail "launcher lock is not a regular file: $lock"
    fi
    exec 8> "$lock"
    chmod 0600 "$lock"
    flock -n 8 || fail "another submit/resume process owns this campaign: $RUN_DIR"
}

prepare_campaign() {
    local checkpoint=$1 model_name runner_sha rollout_sha serve_sha judge_sha overlay_sha parser_sha
    local gym_revision rollout_gym_revision
    local family="${CHECKPOINT_E2E_MODEL_FAMILY:-super35-nemotron}"
    [[ $family == super35-nemotron ]] \
        || fail "unsupported default model family '$family'; provide a compatible launcher profile"
    for path in "$CAMPAIGN_PY" "$TRANSPORT_RUNTIME_PY" "$TRANSPORT_VIEWS_PY" "$FINGERPRINT_PROBE_PY" \
        "$TRANSPORT_OVERLAY" "$TRANSPORT_PATCH" "$PROVIDER_IMAGE_CAPS_PATCH" \
        "$PROVIDER_AGGREGATE_MEDIA_CAPS_PATCH" \
        "$RECURSIVE_REFERENCE_ASSETS_PATCH" "$STRICT_COMPARISON_TRIALS_PATCH" \
        "$PROVIDER_CONTEXT_FALLBACK_PATCH" \
        "$PROVIDER_RATE_LIMIT_BACKOFF_PATCH" \
        "$PARTIAL_PDF_OVERFLOW_PATCH" \
        "$GEMINI_PDF_PART_CAP_PATCH" \
        "$TRANSPORT_ASSIGNMENT" "$SLURM_RECEIPTS_SH" \
        "$ROLLOUT_LIFECYCLE_SH" "$ROLLOUT_SHARD_COVERAGE_PY" \
        "$DATASET" "$ROLLOUT_SBATCH" "$SERVE_SCRIPT" \
        "$JUDGE_SBATCH" "$REFERENCE_OVERLAY" "$PARSER_PLUGIN" "$VLLM_CONTAINER" "$GDPVAL_SIF" \
        "$AGENT_SIF" "$APPTAINER_BIN/apptainer"; do
        [[ -r $path ]] || fail "required path is unreadable: $path"
    done
    [[ -x $GYM_ROOT/.venv/bin/gym ]] || fail "Gym CLI is missing: $GYM_ROOT/.venv/bin/gym"
    [[ -x $ROLLOUT_GYM_ROOT/.venv/bin/gym ]] \
        || fail "rollout Gym CLI is missing: $ROLLOUT_GYM_ROOT/.venv/bin/gym"
    if [[ $EXPECTED_GYM_REVISION == unversioned ]]; then
        gym_revision=unversioned
    else
        require_command git
        gym_revision=$(git -C "$GYM_ROOT" rev-parse HEAD 2>/dev/null) \
            || fail "Gym root is not a readable Git checkout: $GYM_ROOT"
        [[ $gym_revision == "$EXPECTED_GYM_REVISION" ]] \
            || fail "Gym revision drift: $gym_revision != $EXPECTED_GYM_REVISION"
    fi
    if [[ $EXPECTED_ROLLOUT_GYM_REVISION == unversioned ]]; then
        rollout_gym_revision=unversioned
    else
        rollout_gym_revision=$(git -C "$ROLLOUT_GYM_ROOT" rev-parse HEAD 2>/dev/null) \
            || fail "rollout Gym root is not a readable Git checkout: $ROLLOUT_GYM_ROOT"
        [[ $rollout_gym_revision == "$EXPECTED_ROLLOUT_GYM_REVISION" ]] \
            || fail "rollout Gym revision drift: $rollout_gym_revision != $EXPECTED_ROLLOUT_GYM_REVISION"
    fi

    mkdir -p "$CAMPAIGN_ROOT"
    "$PYTHON_BIN" "$CAMPAIGN_PY" prepare \
        --checkpoint "$checkpoint" \
        --dataset "$DATASET" \
        --campaign-root "$CAMPAIGN_ROOT" \
        --shards "$SHARD_COUNT" \
        --expected-tasks "$EXPECTED_TASKS"
    locate_run "$checkpoint"

    JUDGE_RUNTIME_OVERLAY="$RUN_DIR/judge_runtime_overlay"
    if [[ $gym_revision == unversioned ]]; then
        # Unit fixtures may use a minimal unversioned tree. Production defaults
        # never take this branch; exact PR materialization is mandatory there.
        JUDGE_RUNTIME_OVERLAY="$GYM_ROOT"
    else
        "$PYTHON_BIN" "$TRANSPORT_RUNTIME_PY" materialize \
            --gym-root "$GYM_ROOT" \
            --runtime-root "$JUDGE_RUNTIME_OVERLAY" \
            --package-root "$SCRIPT_DIR"
    fi

    model_name="$(safe_model_name "$checkpoint")"
    runner_sha="$(sha256_file "$ROLLOUT_GYM_ROOT/benchmarks/gdpval/run_gdpval_rollouts.sh")"
    rollout_sha="$(sha256_file "$ROLLOUT_SBATCH")"
    serve_sha="$(sha256_file "$SERVE_SCRIPT")"
    judge_sha="$(sha256_file "$JUDGE_SBATCH")"
    overlay_sha="$(sha256_file "$REFERENCE_OVERLAY")"
    parser_sha="$(sha256_file "$PARSER_PLUGIN")"

    publish_env_file "$RUN_DIR/model_profile.env" \
        "POLICY_SERVE_SCRIPT=$SERVE_SCRIPT" \
        "MODEL_NAME=$model_name" \
        "MODEL_PATH=$checkpoint" \
        "CONTAINER_IMAGE=$VLLM_CONTAINER" \
        "NODES_PER_REPLICA=1" \
        "GPUS_PER_NODE=4" \
        "TENSOR_PARALLEL_SIZE=4" \
        "BASE_API_PORT=5000" \
        "GPU_MEMORY_UTILIZATION=0.85" \
        "MAX_MODEL_LEN=262144" \
        "MAX_NUM_SEQS=32" \
        "MAX_NUM_BATCHED_TOKENS=8192" \
        "KV_CACHE_DTYPE=fp8" \
        "DTYPE=bfloat16" \
        "EXTRA_MOUNTS=$PARSER_ROOT:/parsers:ro" \
        "VLLM_EXTRA_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser-plugin /parsers/ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3" \
        "USES_REASONING_PARSER=true" \
        "USES_INTERLEAVED_REASONING=true" \
        "NEMO_GYM_REASONING_FIELD=reasoning" \
        "EXPECTED_RUNNER_SHA=${runner_sha:0:16}"

    publish_runtime_pins "$RUN_DIR/runtime_sources.sha256" \
        "$SCRIPT_DIR/VERSION" \
        "$SCRIPT_DIR/run_checkpoint_e2e.sh" \
        "$SLURM_RECEIPTS_SH" \
        "$CAMPAIGN_PY" \
        "$SCRIPT_DIR/controller.sbatch" \
        "$SCRIPT_DIR/rejudge_controller.sbatch" \
        "$SCRIPT_DIR/rejudge_bootstrap.sbatch" \
        "$SCRIPT_DIR/launch_rejudge.sh" \
        "$SCRIPT_DIR/prepare_rejudge_fingerprint.sh" \
        "$SCRIPT_DIR/judge_state.py" \
        "$SCRIPT_DIR/judge_process_group.sh" \
        "$SCRIPT_DIR/preflight.sbatch" \
        "$SCRIPT_DIR/preconvert_closure.sbatch" \
        "$SCRIPT_DIR/preconvert_closure.py" \
        "$SCRIPT_DIR/transport_prebuild.sbatch" \
        "$TRANSPORT_RUNTIME_PY" \
        "$TRANSPORT_VIEWS_PY" \
        "$FINGERPRINT_PROBE_PY" \
        "$TRANSPORT_OVERLAY" \
        "$TRANSPORT_PATCH" \
        "$PROVIDER_IMAGE_CAPS_PATCH" \
        "$PROVIDER_AGGREGATE_MEDIA_CAPS_PATCH" \
        "$RECURSIVE_REFERENCE_ASSETS_PATCH" \
        "$STRICT_COMPARISON_TRIALS_PATCH" \
        "$PROVIDER_CONTEXT_FALLBACK_PATCH" \
        "$PROVIDER_RATE_LIMIT_BACKOFF_PATCH" \
        "$PARTIAL_PDF_OVERFLOW_PATCH" \
        "$GEMINI_PDF_PART_CAP_PATCH" \
        "$TRANSPORT_ASSIGNMENT" \
        "$ROLLOUT_LIFECYCLE_SH" \
        "$ROLLOUT_SHARD_COVERAGE_PY" \
        "$JUDGE_SBATCH" \
        "$ROLLOUT_GYM_ROOT/.venv/bin/gym" \
        "$ROLLOUT_GYM_ROOT/benchmarks/gdpval/config.yaml" \
        "$ROLLOUT_GYM_ROOT/benchmarks/gdpval/prepare.py" \
        "$ROLLOUT_GYM_ROOT/benchmarks/gdpval/run_gdpval_rollouts.sh" \
        "$ROLLOUT_GYM_ROOT/nemo_gym/rollout_collection.py" \
        "$ROLLOUT_GYM_ROOT/nemo_gym/rollout_reverification.py" \
        "$ROLLOUT_GYM_ROOT/resources_servers/gdpval/app.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_agents/stirrup_agent/app.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_agents/stirrup_agent/file_reader.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_agents/stirrup_agent/stirrup_utils.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_models/openai_model/client.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_models/openai_model/app.py" \
        "$ROLLOUT_GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model.yaml" \
        "$GYM_ROOT/.venv/bin/gym" \
        "$GYM_ROOT/benchmarks/gdpval/config.yaml" \
        "$GYM_ROOT/benchmarks/gdpval/prepare.py" \
        "$GYM_ROOT/nemo_gym/deliverables.py" \
        "$GYM_ROOT/nemo_gym/rollout_collection.py" \
        "$GYM_ROOT/nemo_gym/rollout_reverification.py" \
        "$GYM_ROOT/resources_servers/gdpval/app.py" \
        "$GYM_ROOT/resources_servers/gdpval/comparison.py" \
        "$GYM_ROOT/resources_servers/gdpval/judge_panel.py" \
        "$GYM_ROOT/resources_servers/gdpval/multistage_elo.py" \
        "$GYM_ROOT/resources_servers/gdpval/multistage_orchestrator.py" \
        "$GYM_ROOT/resources_servers/gdpval/preconvert.py" \
        "$GYM_ROOT/resources_servers/gdpval/scoring.py" \
        "$GYM_ROOT/responses_api_agents/stirrup_agent/app.py" \
        "$GYM_ROOT/responses_api_agents/stirrup_agent/file_reader.py" \
        "$GYM_ROOT/responses_api_agents/stirrup_agent/stirrup_utils.py" \
        "$GYM_ROOT/responses_api_models/openai_model/client.py" \
        "$GYM_ROOT/responses_api_models/openai_model/app.py" \
        "$GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model.yaml"

    publish_env_file "$SETTINGS" \
        "RUN_ID=$RUN_ID" \
        "RUN_DIR=$RUN_DIR" \
        "CHECKPOINT=$checkpoint" \
        "MODEL_NAME=$model_name" \
        "DATASET=$DATASET" \
        "REFERENCE_OVERLAY=$REFERENCE_OVERLAY" \
        "PROFILE=$RUN_DIR/model_profile.env" \
        "PROFILE_SHA256=$(sha256_file "$RUN_DIR/model_profile.env")" \
        "DELIVERABLES=$RUN_DIR/deliverables" \
        "ROLLOUT_GYM_ROOT=$ROLLOUT_GYM_ROOT" \
        "ROLLOUT_GYM_REVISION=$rollout_gym_revision" \
        "GYM_ROOT=$GYM_ROOT" \
        "GYM_REVISION=$gym_revision" \
        "UNIFIED_ROOT=$UNIFIED_ROOT" \
        "AAV2_ROOT=$AAV2_ROOT" \
        "ENV_FILE=$ENV_FILE" \
        "ROLLOUT_SBATCH=$ROLLOUT_SBATCH" \
        "SERVE_SCRIPT=$SERVE_SCRIPT" \
        "JUDGE_SBATCH=$JUDGE_SBATCH" \
        "JUDGE_RUNTIME_OVERLAY=$JUDGE_RUNTIME_OVERLAY" \
        "JUDGE_TRANSPORT_OVERLAY=$TRANSPORT_OVERLAY" \
        "TRANSPORT_VIEWS_PY=$TRANSPORT_VIEWS_PY" \
        "FINGERPRINT_PROBE_PY=$FINGERPRINT_PROBE_PY" \
        "SLURM_RECEIPTS_SH=$SLURM_RECEIPTS_SH" \
        "TRANSPORT_VIEW_ROOT=$RUN_DIR/judge_transport_views" \
        "JUDGE_DELIVERABLES=$RUN_DIR/judge_transport_views/candidate" \
        "GDPVAL_SIF=$GDPVAL_SIF" \
        "AGENT_SIF=$AGENT_SIF" \
        "APPTAINER_BIN=$APPTAINER_BIN" \
        "VLLM_CONTAINER=$VLLM_CONTAINER" \
        "PARSER_PLUGIN=$PARSER_PLUGIN" \
        "RUNTIME_PINS=$RUN_DIR/runtime_sources.sha256" \
        "ACCOUNT=$ACCOUNT" \
        "GPU_PARTITION=$GPU_PARTITION" \
        "GPU_QOS=$GPU_QOS" \
        "CPU_PARTITION=$CPU_PARTITION" \
        "CPU_QOS=$CPU_QOS" \
        "ROLLOUT_CONCURRENCY=$ROLLOUT_CONCURRENCY" \
        "RECOVERY_CONCURRENCY=$RECOVERY_CONCURRENCY" \
        "JUDGE_TASK_TIMEOUT_SECONDS=1500" \
        "ROLLOUT_WALL=$ROLLOUT_WALL" \
        "JUDGE_WALL=$JUDGE_WALL" \
        "EXPECTED_TASKS=$EXPECTED_TASKS" \
        "SHARD_COUNT=$SHARD_COUNT" \
        "RUNNER_SHA256=$runner_sha" \
        "ROLLOUT_SBATCH_SHA256=$rollout_sha" \
        "SERVE_SCRIPT_SHA256=$serve_sha" \
        "JUDGE_SBATCH_SHA256=$judge_sha" \
        "REFERENCE_OVERLAY_SHA256=$overlay_sha" \
        "PARSER_PLUGIN_SHA256=$parser_sha" \
        "VLLM_CONTAINER_SIGNATURE=$(stat_signature "$VLLM_CONTAINER")" \
        "GDPVAL_SIF_SIGNATURE=$(stat_signature "$GDPVAL_SIF")" \
        "AGENT_SIF_SIGNATURE=$(stat_signature "$AGENT_SIF")" \
        "APPTAINER_SIGNATURE=$(stat_signature "$APPTAINER_BIN/apptainer")" \
        "E2E_SCRIPT=$SCRIPT_DIR/run_checkpoint_e2e.sh" \
        "E2E_DIR=$SCRIPT_DIR"

    mkdir -p "$RUN_DIR/logs" "$RUN_DIR/deliverables"
    "$PYTHON_BIN" "$CAMPAIGN_PY" verify --run-dir "$RUN_DIR"
    echo "PREPARED run_id=$RUN_ID"
    echo "RUN_DIR=$RUN_DIR"
    echo "MODEL_NAME=$model_name"
}

assert_sha() {
    local path=$1 expected=$2 label=$3 actual
    actual="$(sha256_file "$path")"
    [[ $actual == "$expected" ]] || fail "$label source drift: $actual != $expected ($path)"
}

compute_preflight() {
    local run_dir=$1
    RUN_DIR=$run_dir
    SETTINGS=${CHECKPOINT_E2E_LOCAL_SETTINGS:-$RUN_DIR/settings.env}
    if [[ -n ${CHECKPOINT_E2E_LOCAL_SETTINGS:-} ]]; then
        [[ $SETTINGS == /raid/scratch/* ]] || fail "node-local settings override is unsafe"
    fi
    source_settings
    if [[ -n ${CHECKPOINT_E2E_LOCAL_PACKAGE:-} ]]; then
        [[ $CHECKPOINT_E2E_LOCAL_PACKAGE == /raid/scratch/* \
            && $CHECKPOINT_E2E_LOCAL_GYM == /raid/scratch/* \
            && $CHECKPOINT_E2E_LOCAL_RUNTIME == /raid/scratch/* \
            && $CHECKPOINT_E2E_LOCAL_ENV_FILE == /raid/scratch/* ]] \
            || fail "node-local preflight overrides are unsafe"
        E2E_DIR=$CHECKPOINT_E2E_LOCAL_PACKAGE
        CAMPAIGN_PY=$E2E_DIR/campaign.py
        TRANSPORT_RUNTIME_PY=$E2E_DIR/transport_runtime.py
        PYTHON_BIN=$CHECKPOINT_E2E_LOCAL_GYM/.venv/bin/python
        GYM_ROOT=$CHECKPOINT_E2E_LOCAL_GYM
        JUDGE_RUNTIME_OVERLAY=$CHECKPOINT_E2E_LOCAL_RUNTIME
        ENV_FILE=$CHECKPOINT_E2E_LOCAL_ENV_FILE
    fi
    local rehash=${CHECKPOINT_E2E_REHASH_REFERENCE_ASSETS:-false}
    local -a verify_args=(verify --run-dir "$RUN_DIR")
    [[ $rehash == true || $rehash == false ]] \
        || fail "CHECKPOINT_E2E_REHASH_REFERENCE_ASSETS must be true or false"
    [[ $rehash != true ]] || verify_args+=(--rehash-reference-assets)
    "$PYTHON_BIN" "$CAMPAIGN_PY" "${verify_args[@]}"
    if [[ $GYM_REVISION != unversioned ]]; then
        local observed_gym_revision
        if [[ -f $GYM_ROOT/.checkpoint_e2e_revision && ! -L $GYM_ROOT/.checkpoint_e2e_revision ]]; then
            observed_gym_revision=$(<"$GYM_ROOT/.checkpoint_e2e_revision")
        else
            observed_gym_revision=$(git -C "$GYM_ROOT" rev-parse HEAD 2>/dev/null)
        fi
        [[ $observed_gym_revision == "$GYM_REVISION" ]] \
            || fail "Gym revision changed after prepare"
    fi
    if [[ $ROLLOUT_GYM_REVISION != unversioned ]]; then
        [[ $(git -C "$ROLLOUT_GYM_ROOT" rev-parse HEAD 2>/dev/null) == "$ROLLOUT_GYM_REVISION" ]] \
            || fail "rollout Gym revision changed after prepare"
    fi
    assert_sha "$ROLLOUT_GYM_ROOT/benchmarks/gdpval/run_gdpval_rollouts.sh" "$RUNNER_SHA256" runner
    assert_sha "$ROLLOUT_SBATCH" "$ROLLOUT_SBATCH_SHA256" rollout
    assert_sha "$SERVE_SCRIPT" "$SERVE_SCRIPT_SHA256" serve
    assert_sha "$JUDGE_SBATCH" "$JUDGE_SBATCH_SHA256" judge
    assert_sha "$REFERENCE_OVERLAY" "$REFERENCE_OVERLAY_SHA256" reference-overlay
    assert_sha "$PARSER_PLUGIN" "$PARSER_PLUGIN_SHA256" reasoning-parser
    assert_sha "$PROFILE" "$PROFILE_SHA256" model-profile
    sha256sum -c --status "$RUNTIME_PINS" || fail "pinned Gym runtime sources changed"
    if [[ $GYM_REVISION != unversioned ]]; then
        "$PYTHON_BIN" "$TRANSPORT_RUNTIME_PY" validate \
            --gym-root "$GYM_ROOT" \
            --runtime-root "$JUDGE_RUNTIME_OVERLAY" \
            --package-root "$E2E_DIR"
    fi
    local vllm_container_signature
    vllm_container_signature=$(stat_signature "$VLLM_CONTAINER")
    if [[ $vllm_container_signature != "$VLLM_CONTAINER_SIGNATURE" ]]; then
        local rollout_receipt="$RUN_DIR/ROLLOUT_COVERAGE_PASS"
        local drift_receipt="$RUN_DIR/VLLM_CONTAINER_DRIFT_AFTER_ROLLOUTS"
        local drift_temporary="${drift_receipt}.tmp.$$"
        [[ -f $rollout_receipt && ! -L $rollout_receipt ]] \
            || fail "vLLM container changed before immutable rollout coverage was recorded"
        "$PYTHON_BIN" "$CAMPAIGN_PY" coverage \
            --dataset "$DATASET" --deliverables "$DELIVERABLES" \
            --expected-tasks "$EXPECTED_TASKS" >/dev/null \
            || fail "vLLM container changed while rollout coverage is incomplete"
        printf '%s\n' \
            'schema=gdpval.vllm-container-post-rollout-drift.v1' \
            "path=$VLLM_CONTAINER" \
            "expected_signature=$VLLM_CONTAINER_SIGNATURE" \
            "observed_signature=$vllm_container_signature" \
            "rollout_coverage_receipt_sha256=$(sha256_file "$rollout_receipt")" \
            > "$drift_temporary"
        chmod 0400 "$drift_temporary"
        if [[ -e $drift_receipt || -L $drift_receipt ]]; then
            [[ -f $drift_receipt && ! -L $drift_receipt ]] \
                || fail "vLLM post-rollout drift receipt is not a regular file"
            cmp -s "$drift_temporary" "$drift_receipt" \
                || fail "vLLM container changed again after post-rollout drift was recorded"
            rm -f "$drift_temporary"
        else
            mv "$drift_temporary" "$drift_receipt"
        fi
        echo "VLLM_CONTAINER_POST_ROLLOUT_DRIFT expected=$VLLM_CONTAINER_SIGNATURE observed=$vllm_container_signature"
    fi
    [[ $(stat_signature "$GDPVAL_SIF") == "$GDPVAL_SIF_SIGNATURE" ]] \
        || fail "GDPVal container changed"
    [[ $(stat_signature "$AGENT_SIF") == "$AGENT_SIF_SIGNATURE" ]] \
        || fail "agent sandbox changed"
    [[ $(stat_signature "$APPTAINER_BIN/apptainer") == "$APPTAINER_SIGNATURE" ]] \
        || fail "Apptainer runtime changed"
    [[ -r $ENV_FILE ]] || fail "protected environment file is unreadable on compute node"
    local receipt="$RUN_DIR/PREFLIGHT_PASS"
    local temporary="${receipt}.tmp.$$"
    printf 'campaign=%s\ncheckpoint=%s\nrunner_sha256=%s\n' \
        "$RUN_ID" "$CHECKPOINT" "$RUNNER_SHA256" > "$temporary"
    chmod 0400 "$temporary"
    mv -f "$temporary" "$receipt"
    echo "PREFLIGHT_PASS run_id=$RUN_ID"
}

job_liveness() {
    local job=$1 queue accounting state
    [[ $job =~ ^[1-9][0-9]*$ ]] || fail "invalid recorded job id: $job"
    # Slurm may return a nonzero "invalid job id" after a terminal job leaves
    # squeue even though sacct already has authoritative terminal evidence.
    # Fall through to accounting instead of making a safe resume impossible.
    queue=$(squeue -h -j "$job" -o '%A' 2>/dev/null) || queue=
    if [[ -n $queue ]] && grep -qx "$job" <<<"$queue"; then
        printf 'live\n'
        return
    fi
    if ! accounting=$(sacct -X -j "$job" --format=JobIDRaw,State -n -P 2>/dev/null); then
        printf 'unknown\n'
        return
    fi
    state=$(awk -F '|' -v wanted="$job" '$1 == wanted {print $2; exit}' <<<"$accounting")
    state=${state%%+*}
    state=${state%% *}
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED)
            printf 'terminal\n'
            ;;
        *)
            printf 'unknown\n'
            ;;
    esac
}

submit_rollout_shard() {
    local shard=$1 shard_path=$2 run_dir=$3 concurrency=$4 job_name=$5 dependency=${6:-} role=$7
    local -a args=(--parsable -J "$job_name" -N 1 -A "$ACCOUNT" -p "$GPU_PARTITION" --qos="$GPU_QOS"
        -t "$ROLLOUT_WALL" -o "$run_dir/logs/%j_rollout.out" -e "$run_dir/logs/%j_rollout.err")
    [[ -n $dependency ]] && args+=(--dependency="$dependency")
    mkdir -p "$run_dir/logs"
    slurm_submit_or_adopt "$run_dir/JOBID" "$role" "$job_name" "${args[@]}" \
        --export=ALL,RUN_DIR="$run_dir",DATASET="$shard_path",PROFILE="$PROFILE",PERSIST_DELIVERABLES_DIR="$DELIVERABLES",CONCURRENCY="$concurrency",AGENT_MAX_TURNS=250,STIRRUP_PER_TASK_TIMEOUT_S=10200,MAX_ROTATIONS=3,TREE="$ROLLOUT_GYM_ROOT",ENV_FILE="$ENV_FILE",EXPECTED_RUNNER_SHA="${RUNNER_SHA256:0:16}",AGENT_SIF="$AGENT_SIF",APPTAINER_BIN="$APPTAINER_BIN",ROLLOUT_PACKAGE_DIR="$SCRIPT_DIR" \
        "$ROLLOUT_SBATCH" || fail "could not submit or adopt rollout role=$role"
}

submit_controller() {
    local dependency=${1:-} generation=${2:-initial}
    local authorize=${CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS:-false}
    [[ $generation =~ ^(initial|after-[1-9][0-9]*)$ ]] \
        || fail "invalid controller submission generation: $generation"
    [[ $authorize == true || $authorize == false ]] \
        || fail "CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS must be true or false"
    local job_name="gdp-e2e-${RUN_ID:0:28}-ctl"
    local receipt="$RUN_DIR/controller_submissions/${generation}.jobid"
    local -a args=(--parsable -J "$job_name" -A "$ACCOUNT" -p "$CPU_PARTITION"
        --qos="$CPU_QOS" -t 12:00:00 --cpus-per-task=2 --mem=8G
        -o "$RUN_DIR/logs/%j_controller.out" -e "$RUN_DIR/logs/%j_controller.err")
    [[ -n $dependency ]] && args+=(--dependency="$dependency")
    slurm_submit_or_adopt "$receipt" "controller-${generation}" "$job_name" "${args[@]}" \
        --export=ALL,RUN_DIR="$RUN_DIR",E2E_DIR="$SCRIPT_DIR",CHECKPOINT_E2E_AUTHORIZE_PROVIDER_CALLS="$authorize" \
        "$SCRIPT_DIR/controller.sbatch" || fail "could not submit or adopt controller generation=$generation"
}

submit_campaign() {
    local checkpoint=$1
    locate_run "$checkpoint"
    [[ -d $RUN_DIR ]] || fail "campaign is not prepared: $RUN_DIR"
    acquire_launcher_lock
    source_settings
    [[ ! -e $RUN_DIR/CAMPAIGN_COMPLETE ]] || {
        echo "ALREADY_COMPLETE run_id=$RUN_ID"
        return
    }
    if [[ -s $RUN_DIR/CONTROLLER_JOBID ]]; then
        local recorded_controller recorded_state
        recorded_controller=$(read_job_id "$RUN_DIR/CONTROLLER_JOBID")
        recorded_state=$(job_liveness "$recorded_controller")
        if [[ $recorded_state == live ]]; then
            echo "ALREADY_RUNNING controller=$recorded_controller run_id=$RUN_ID"
            return
        fi
        [[ $recorded_state == terminal ]] \
            || fail "controller $recorded_controller has no live or terminal Slurm evidence; retry later"
    fi
    [[ ! -e $RUN_DIR/FLEET_SUBMITTED ]] \
        || fail "campaign was already submitted; use resume instead of submit"
    validate_dependency_policy

    local preflight_job preflight_state
    if [[ -s $RUN_DIR/PREFLIGHT_JOBID ]]; then
        preflight_job="$(read_job_id "$RUN_DIR/PREFLIGHT_JOBID")"
        preflight_state="$(job_liveness "$preflight_job")"
        case "$preflight_state" in
            live) ;;
            terminal)
                [[ -e $RUN_DIR/PREFLIGHT_PASS ]] \
                    || fail "preflight $preflight_job terminated without a pass receipt"
                preflight_job=""
                ;;
            unknown)
                fail "preflight $preflight_job has no live or terminal Slurm evidence; retry later"
                ;;
        esac
    elif [[ -e $RUN_DIR/PREFLIGHT_PASS ]]; then
        preflight_job=""
    else
        local preflight_name="gdp-e2e-${RUN_ID:0:28}-pre"
        preflight_job="$(slurm_submit_or_adopt "$RUN_DIR/PREFLIGHT_JOBID" preflight "$preflight_name" \
            --parsable -J "$preflight_name" \
            -A "$ACCOUNT" -p "$CPU_PARTITION" --qos="$CPU_QOS" -t 00:15:00 \
            --cpus-per-task=2 --mem=8G \
            -o "$RUN_DIR/logs/%j_preflight.out" -e "$RUN_DIR/logs/%j_preflight.err" \
            --export=ALL,RUN_DIR="$RUN_DIR",E2E_DIR="$SCRIPT_DIR" \
            "$SCRIPT_DIR/preflight.sbatch")" || fail "could not submit or adopt preflight"
    fi

    local jobs_file="$RUN_DIR/FLEET_JOBS.tsv" shard run_dir shard_path job rows dependency
    : > "$jobs_file.tmp.$$"
    for (( shard=0; shard<SHARD_COUNT; shard++ )); do
        printf -v shard_path '%s/shards/shard_%02d_of_%02d.jsonl' "$RUN_DIR" "$shard" "$SHARD_COUNT"
        printf -v run_dir '%s/rollout_s%02d' "$RUN_DIR" "$shard"
        rows=$(awk 'NF {n++} END {print n+0}' "$shard_path")
        if [[ -s $run_dir/JOBID ]]; then
            job="$(read_job_id "$run_dir/JOBID")"
        else
            dependency=""
            [[ -n $preflight_job ]] && dependency="afterok:$preflight_job"
            job="$(submit_rollout_shard "$shard" "$shard_path" "$run_dir" "$ROLLOUT_CONCURRENCY" \
                "gdp-${RUN_ID:0:20}-s${shard}" "$dependency" "rollout-s${shard}")"
        fi
        printf '%s\t%s\t%s\t%s\n' "$shard" "$job" "$rows" "$run_dir" >> "$jobs_file.tmp.$$"
    done
    chmod 0400 "$jobs_file.tmp.$$"
    mv -f "$jobs_file.tmp.$$" "$jobs_file"
    : > "$RUN_DIR/FLEET_SUBMITTED"
    chmod 0400 "$RUN_DIR/FLEET_SUBMITTED"

    local fleet_ids controller dependency
    fleet_ids="$(cut -f2 "$jobs_file" | paste -sd: -)"
    dependency="afterany:$fleet_ids"
    controller="$(submit_controller "$dependency" initial)"
    slurm_publish_job_receipt "$RUN_DIR/CONTROLLER_JOBID" "$controller" true >/dev/null \
        || fail "could not publish current controller receipt"
    echo "SUBMITTED rollout_jobs=$(tr ':' ',' <<<"$fleet_ids") controller=$controller"
    echo "RUN_DIR=$RUN_DIR"
}

submit_recovery() {
    local run_dir=$1 round=$2
    [[ $round =~ ^[1-9][0-9]*$ ]] || fail "recovery round must be positive"
    RUN_DIR=$run_dir
    SETTINGS=$RUN_DIR/settings.env
    source_settings
    local recovery="$RUN_DIR/recovery_r${round}" residue="$RUN_DIR/recovery_r${round}/residue.jsonl"
    local shards="$RUN_DIR/recovery_r${round}/shards" jobs_file="$RUN_DIR/recovery_r${round}/JOBS.tsv"
    mkdir -p "$recovery"
    "$PYTHON_BIN" "$CAMPAIGN_PY" residue --dataset "$DATASET" --deliverables "$DELIVERABLES" \
        --output "$residue" --shards-dir "$shards" --max-shards "$SHARD_COUNT" \
        --expected-tasks "$EXPECTED_TASKS"
    local manifest="$shards/manifest.json" shard_count
    shard_count="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["shard_count"])' "$manifest")"
    if (( shard_count == 0 )); then
        : > "$recovery/NOT_NEEDED"
        echo "RECOVERY_NOT_NEEDED round=$round"
        return
    fi
    local temporary="$jobs_file.tmp.$$" shard shard_path shard_run rows conc job
    : > "$temporary"
    for (( shard=0; shard<shard_count; shard++ )); do
        printf -v shard_path '%s/shard_%02d_of_%02d.jsonl' "$shards" "$shard" "$shard_count"
        printf -v shard_run '%s/rollout_s%02d' "$recovery" "$shard"
        rows=$(awk 'NF {n++} END {print n+0}' "$shard_path")
        conc=$rows
        (( conc > RECOVERY_CONCURRENCY )) && conc=$RECOVERY_CONCURRENCY
        if [[ -s $shard_run/JOBID ]]; then
            job="$(read_job_id "$shard_run/JOBID")"
        else
            job="$(submit_rollout_shard "$shard" "$shard_path" "$shard_run" "$conc" \
                "gdp-${RUN_ID:0:18}-r${round}s${shard}" "" "recovery-r${round}-s${shard}")"
        fi
        printf '%s\t%s\t%s\t%s\n' "$shard" "$job" "$rows" "$shard_run" >> "$temporary"
    done
    chmod 0400 "$temporary"
    mv -f "$temporary" "$jobs_file"
    : > "$recovery/SUBMITTED"
    chmod 0400 "$recovery/SUBMITTED"
    echo "SUBMITTED_RECOVERY round=$round jobs=$(cut -f2 "$jobs_file" | paste -sd, -)"
}

resume_campaign() {
    local checkpoint=$1
    locate_run "$checkpoint"
    [[ -d $RUN_DIR ]] || fail "campaign is not prepared: $RUN_DIR"
    acquire_launcher_lock
    source_settings
    if [[ -e $RUN_DIR/CAMPAIGN_COMPLETE ]]; then
        result_campaign "$checkpoint"
        return
    fi
    [[ -e $RUN_DIR/FLEET_SUBMITTED && -s $RUN_DIR/FLEET_JOBS.tsv ]] \
        || fail "initial rollout fleet was never submitted; use submit"
    if [[ -s $RUN_DIR/CONTROLLER_JOBID ]]; then
        local recorded_controller recorded_state
        recorded_controller=$(read_job_id "$RUN_DIR/CONTROLLER_JOBID")
        recorded_state=$(job_liveness "$recorded_controller")
        if [[ $recorded_state == live ]]; then
            echo "ALREADY_RUNNING controller=$recorded_controller"
            return
        fi
        [[ $recorded_state == terminal ]] \
            || fail "controller $recorded_controller has no live or terminal Slurm evidence; refusing resume"
    fi
    local controller generation=initial
    [[ -z ${recorded_controller:-} ]] || generation="after-${recorded_controller}"
    controller="$(submit_controller "" "$generation")"
    slurm_publish_job_receipt "$RUN_DIR/CONTROLLER_JOBID" "$controller" true >/dev/null \
        || fail "could not publish current controller receipt"
    echo "RESUMED controller=$controller"
    echo "RUN_DIR=$RUN_DIR"
}

coverage_count() {
    local report
    if [[ ! -d $DELIVERABLES ]]; then
        echo 0
        return
    fi
    report=$("$PYTHON_BIN" "$CAMPAIGN_PY" coverage --dataset "$DATASET" --deliverables "$DELIVERABLES" \
        --expected-tasks "$EXPECTED_TASKS" --json 2>/dev/null) || true
    if [[ -n $report ]]; then
        printf '%s\n' "$report" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["completed"])'
    else
        echo 0
    fi
}

judge_progress() {
    local output="$RUN_DIR/judge_e2e/gdpval_aav2.jsonl"
    if [[ ! -s $output ]]; then
        echo 'stage0=0/45(min41) stage1=0/220 trials=0/1044-1060 invalid=0 errors=0'
        return
    fi
    "$PYTHON_BIN" -c 'import json,sys
s=[0,0]; trials=invalid=errors=0
for line in open(sys.argv[1]):
 r=json.loads(line); i=r.get("stage_index");
 if i in (0,1): s[i]+=1
 j=r.get("judge_response") or {}; trials+=int(j.get("total_judged",0) or 0); invalid+=int(j.get("total_invalid",0) or 0)
 errors+=int(bool(j.get("error") or j.get("ref_errors") or r.get("invalid_judge_response")))
print(f"stage0={s[0]}/45(min41) stage1={s[1]}/220 trials={trials}/1044-1060 invalid={invalid} errors={errors}")' "$output"
}

status_campaign() {
    local checkpoint=$1
    locate_run "$checkpoint"
    echo "RUN_ID=$RUN_ID"
    echo "RUN_DIR=$RUN_DIR"
    if [[ ! -f $SETTINGS ]]; then
        echo "STATE=UNPREPARED"
        echo "NEXT=prepare"
        return
    fi
    source_settings
    local state=PREPARED controller=none live=0 next=submit
    if [[ -s $RUN_DIR/CONTROLLER_JOBID ]]; then
        controller="$(read_job_id "$RUN_DIR/CONTROLLER_JOBID")"
        case "$(job_liveness "$controller")" in
            live) live=1 ;;
            terminal) live=0 ;;
            unknown) live=-1 ;;
        esac
    fi
    if [[ -e $RUN_DIR/CAMPAIGN_COMPLETE ]]; then
        state=PASS; next=result
    elif (( live == 1 )); then
        state=RUNNING; next=status
    elif (( live == -1 )); then
        state=UNKNOWN_SLURM_STATE; next=status
    elif [[ -e $RUN_DIR/JUDGE_AUTHORIZATION_REQUIRED ]]; then
        state=AWAITING_JUDGE_AUTHORIZATION; next=resume
    elif [[ -e $RUN_DIR/CONTROLLER_BLOCKED ]]; then
        state=BLOCKED; next=inspect_then_resume
    elif [[ -e $RUN_DIR/FLEET_SUBMITTED ]]; then
        state=RETRYABLE; next=resume
    fi
    echo "STATE=$state"
    echo "CONTROLLER=$controller live=$live"
    echo "ROLLOUT=$(coverage_count)/$EXPECTED_TASKS"
    echo "JUDGE=$(judge_progress)"
    [[ ! -e $RUN_DIR/PREFLIGHT_PASS ]] || echo "PREFLIGHT=PASS"
    [[ ! -e $RUN_DIR/PRECONVERT_PASS ]] || echo "PRECONVERT=PASS"
    if [[ -e $RUN_DIR/CONTROLLER_BLOCKED ]]; then
        echo "BLOCKER=$(tr '\n' ' ' < "$RUN_DIR/CONTROLLER_BLOCKED")"
    fi
    echo "NEXT=$next"
}

result_campaign() {
    local checkpoint=$1
    locate_run "$checkpoint"
    source_settings
    local receipt="$RUN_DIR/final_receipt.json"
    local sidecar="$RUN_DIR/final_receipt.json.sha256"
    local marker="$RUN_DIR/CAMPAIGN_COMPLETE"
    [[ -f $marker && ! -L $marker && ! -s $marker && $(file_mode "$marker") == 400 ]] \
        || fail "authoritative completion marker is missing or invalid: $marker"
    [[ -f $receipt && ! -L $receipt && -s $receipt && $(file_mode "$receipt") == 400 ]] \
        || fail "authoritative final receipt is missing or invalid: $receipt"
    [[ -f $sidecar && ! -L $sidecar && -s $sidecar && $(file_mode "$sidecar") == 400 ]] \
        || fail "final receipt sidecar is missing or invalid: $sidecar"
    sha256sum -c --status "$sidecar" || fail "final receipt digest mismatch"
    "$PYTHON_BIN" "$CAMPAIGN_PY" verify --run-dir "$RUN_DIR" \
        --rehash-reference-assets >/dev/null
    sha256sum -c --status "$RUNTIME_PINS" || fail "pinned runtime sources changed"
    "$PYTHON_BIN" "$CAMPAIGN_PY" coverage --dataset "$DATASET" --deliverables "$DELIVERABLES" \
        --expected-tasks "$EXPECTED_TASKS"
    local current_result stored_result
    current_result=$("$PYTHON_BIN" "$CAMPAIGN_PY" result \
        --output "$RUN_DIR/judge_e2e/gdpval_aav2.jsonl" \
        --journal "$RUN_DIR/judge_e2e/gdpval_aav2_multistage_state.jsonl" \
        --dataset "$DATASET" --expected-tasks "$EXPECTED_TASKS" --json)
    stored_result=$(<"$receipt")
    [[ $current_result == "$stored_result" ]] \
        || fail "final receipt no longer matches the strict result"
    printf '%s\n' "$current_result"
    echo "FINAL_RECEIPT=$receipt"
    echo "COMPLETION_MARKER=$marker"
}

main() {
    require_command "$PYTHON_BIN"
    [[ -r $CAMPAIGN_PY ]] || fail "campaign helper is missing: $CAMPAIGN_PY"
    local action=${1:-} raw_checkpoint=${2:-} checkpoint
    [[ -n $action && -n $raw_checkpoint && $# == 2 ]] || { usage >&2; exit 64; }
    checkpoint="$(resolve_checkpoint "$raw_checkpoint")"
    case "$action" in
        prepare) prepare_campaign "$checkpoint" ;;
        submit) submit_campaign "$checkpoint" ;;
        all) prepare_campaign "$checkpoint"; submit_campaign "$checkpoint" ;;
        resume) resume_campaign "$checkpoint" ;;
        status) status_campaign "$checkpoint" ;;
        result) result_campaign "$checkpoint" ;;
        _compute-preflight) compute_preflight "$raw_checkpoint" ;;
        _submit-recovery) fail "internal recovery requires RUN_DIR and round" ;;
        *) usage >&2; exit 64 ;;
    esac
}

# Internal actions use a different shape and never resolve their first argument
# as a checkpoint.
if [[ ${1:-} == _compute-preflight ]]; then
    [[ $# == 2 ]] || fail "usage: $0 _compute-preflight RUN_DIR"
    compute_preflight "$2"
elif [[ ${1:-} == _submit-recovery ]]; then
    [[ $# == 3 ]] || fail "usage: $0 _submit-recovery RUN_DIR ROUND"
    submit_recovery "$2" "$3"
else
    main "$@"
fi
