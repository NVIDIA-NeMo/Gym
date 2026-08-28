#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Unified Apex Agents rollout collection modeled on Serge Panev's GDPVal
# collocated launcher:
#   - N_REPLICAS = N_NODES / NODES_PER_REPLICA
#   - replicas are assembled rack-locally whenever the allocation permits
#   - each serving process publishes one ready endpoint
#   - Serge's least-connections proxy balances all rollout traffic globally
#   - one node-local Gym worker runs on every active allocation node
#   - round-robin input shards and per-worker outputs are merged resume-safely
#   - one resumable output is retained across explicit walltime requeues

#SBATCH --job-name=apex-k3-unified
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
# Use a dedicated pre-walltime signal so a normal `scancel` SIGTERM does not
# enter the requeue path.
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"480","reason":"benchmarking","description":"Apex Agents Kimi-K3 rollouts"}}'
#SBATCH --output=results/slurm-logs/%x-%j.log

set -Eeuo pipefail
export PYTHONUNBUFFERED=1

readonly DEFAULT_APEX_GYM_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/artij/ApexGym"
readonly DEFAULT_GYM_CONTAINER="/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/"\
"nemo_rl/images/high_stripe/rl.48761755.sqsh"
readonly DEFAULT_PROFILE="${DEFAULT_APEX_GYM_DIR}/scripts/profiles/apex-kimi-k3.env"
readonly DEFAULT_DATASET="${DEFAULT_APEX_GYM_DIR}/benchmarks/apex_agents/data/apex_agents_benchmark.jsonl"

export APEX_GYM_DIR=${APEX_GYM_DIR:-${DEFAULT_APEX_GYM_DIR}}
export PROFILE=${PROFILE:-${DEFAULT_PROFILE}}
export DATASET=${DATASET:-${DEFAULT_DATASET}}
export GYM_CONFIG=${GYM_CONFIG:-${APEX_GYM_DIR}/env.yaml}
export GYM_CONTAINER=${GYM_CONTAINER:-${DEFAULT_GYM_CONTAINER}}
export GYM_BIN=${GYM_BIN:-${APEX_GYM_DIR}/.venv/bin/gym}
export APEX_IMAGE=${APEX_IMAGE:-${APEX_GYM_DIR}/responses_api_agents/apex_agent/images/archipelago-fixed.sif}
export WORLD_CACHE_DIR=${WORLD_CACHE_DIR:-${APEX_GYM_DIR}/benchmarks/apex_agents/data/world_cache}
export SHARED_RUNTIME_DIR=${SHARED_RUNTIME_DIR:-${APEX_GYM_DIR}/results/apex_agents_runtime}
export CONCURRENCY=${CONCURRENCY:-120}
export NUM_REPEATS=${NUM_REPEATS:-1}
export LIMIT=${LIMIT:-}
export SPLIT=${SPLIT:-benchmark}
export MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-32768}
export APEX_AGENT_TIMEOUT=${APEX_AGENT_TIMEOUT:-12600}
export APEX_AGENT_SUPPORTS_VISION=${APEX_AGENT_SUPPORTS_VISION:-true}
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-3}
export MAX_ROTATIONS=${MAX_ROTATIONS:-8}
export SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-7200}
export POLICY_LABEL=${POLICY_LABEL:-Kimi-K3}
export K3_CPUS_PER_NODE=${K3_CPUS_PER_NODE:-40}
export GYM_CPUS_PER_NODE=${GYM_CPUS_PER_NODE:-96}
export GYM_HEAD_PORT=${GYM_HEAD_PORT:-20000}
export USER_NAME=${USER_NAME:-${USER:-$(whoami)}}
export RUN_DIR=${RUN_DIR:-${APEX_GYM_DIR}/results/apex_k3_unified/manual}
export OUTPUT_JSONL=${OUTPUT_JSONL:-${RUN_DIR}/rollouts.jsonl}
export SHARD_ROOT=${SHARD_ROOT:-${RUN_DIR}/shards}
export SHARD_INPUT_DIR=${SHARD_INPUT_DIR:-${SHARD_ROOT}/inputs}
export SHARD_OUTPUT_DIR=${SHARD_OUTPUT_DIR:-${SHARD_ROOT}/outputs}
export EFFECTIVE_DATASET=${EFFECTIVE_DATASET:-${SHARD_ROOT}/source.jsonl}
export ROTATION=${SLURM_RESTART_COUNT:-0}

log() {
    printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 64
}

load_profile() {
    [[ -r "${PROFILE}" ]] || die "profile is not readable: ${PROFILE}"
    set -a
    # shellcheck disable=SC1090
    . "${PROFILE}"
    set +a
    : "${POLICY_SERVE_SCRIPT:?profile must set POLICY_SERVE_SCRIPT}"
    : "${MODEL_NAME:?profile must set MODEL_NAME}"
    : "${MODEL_PATH:?profile must set MODEL_PATH}"
    : "${CONTAINER_IMAGE:?profile must set CONTAINER_IMAGE}"
    : "${LAUNCHER_PATH:?profile must set LAUNCHER_PATH}"
    : "${POLICY_LB_SCRIPT:?profile must set POLICY_LB_SCRIPT}"
    : "${POLICY_LB_PYTHON:?profile must set POLICY_LB_PYTHON}"
    export NODES_PER_REPLICA=${NODES_PER_REPLICA:-4}
    export BASE_API_PORT=${BASE_API_PORT:-12500}
}

validate_common() {
    [[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || die "CONCURRENCY must be a positive integer"
    [[ "${NUM_REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_REPEATS must be a positive integer"
    [[ -z "${LIMIT}" || "${LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "LIMIT must be empty or a positive integer"
    [[ "${SPLIT}" =~ ^[A-Za-z0-9._-]+$ ]] || die "SPLIT contains unsupported characters"
    [[ "${MAX_OUTPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "MAX_OUTPUT_TOKENS must be a positive integer"
    [[ "${APEX_AGENT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "APEX_AGENT_TIMEOUT must be a positive integer"
    [[ "${APEX_AGENT_SUPPORTS_VISION}" == "true" || "${APEX_AGENT_SUPPORTS_VISION}" == "false" ]] ||
        die "APEX_AGENT_SUPPORTS_VISION must be true or false"
    [[ "${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] ||
        die "NEMO_GYM_MAX_ROLLOUT_ATTEMPTS must be a positive integer"
    [[ "${NODES_PER_REPLICA}" =~ ^[1-9][0-9]*$ ]] || die "NODES_PER_REPLICA must be positive"
    [[ "${BASE_API_PORT}" =~ ^[0-9]+$ ]] || die "BASE_API_PORT must be an integer"
    [[ "${K3_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "K3_CPUS_PER_NODE must be positive"
    [[ "${GYM_CPUS_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "GYM_CPUS_PER_NODE must be positive"
    (( K3_CPUS_PER_NODE + GYM_CPUS_PER_NODE <= 144 )) ||
        die "K3_CPUS_PER_NODE + GYM_CPUS_PER_NODE must not exceed the 144 allocated CPUs per node"
    [[ "${GYM_HEAD_PORT}" =~ ^[0-9]+$ ]] && (( GYM_HEAD_PORT >= 1 && GYM_HEAD_PORT <= 65535 )) ||
        die "GYM_HEAD_PORT must be between 1 and 65535"
    [[ "${NEMO_GYM_REASONING_FIELD:-}" == "reasoning" || \
        "${NEMO_GYM_REASONING_FIELD:-}" == "reasoning_content" ]] ||
        die "${POLICY_LABEL} requires exactly one reasoning field"

    local required
    for required in \
        "${POLICY_SERVE_SCRIPT}" \
        "${MODEL_PATH}/config.json" \
        "${CONTAINER_IMAGE}" \
        "${LAUNCHER_PATH}" \
        "${POLICY_LB_SCRIPT}" \
        "${POLICY_LB_PYTHON}" \
        "${GYM_CONTAINER}" \
        "${GYM_CONFIG}" \
        "${GYM_BIN}" \
        "${DATASET}" \
        "${APEX_IMAGE}" \
        "${APEX_GYM_DIR}/scripts/prepare_apex_retry_rows.py" \
        "${APEX_GYM_DIR}/scripts/merge_apex_k3_shards.py"; do
        [[ -r "${required}" ]] || die "required path is not readable: ${required}"
    done
    [[ -x "${GYM_BIN}" ]] || die "Gym executable is not executable: ${GYM_BIN}"
    [[ -x "${POLICY_LB_PYTHON}" ]] || die "LB Python is not executable: ${POLICY_LB_PYTHON}"
    "${POLICY_LB_PYTHON}" -c 'import aiohttp' ||
        die "LB Python cannot import aiohttp: ${POLICY_LB_PYTHON}"
    if [[ ! -d "${WORLD_CACHE_DIR}" ]] ||
        ! find "${WORLD_CACHE_DIR}" -type f ! -path '*/.locks/*' -print -quit | grep -q .; then
        die "Apex world cache is missing or empty: ${WORLD_CACHE_DIR}"
    fi
}

validate_local() {
    load_profile
    validate_common
    local urls='["http://127.0.0.1:12500/v1"]'
    "${GYM_BIN}" env resolve \
        --config "${GYM_CONFIG}" \
        --config "${APEX_GYM_DIR}/benchmarks/apex_agents/config.yaml" \
        --config "${APEX_GYM_DIR}/responses_api_models/vllm_model/configs/vllm_model.yaml" \
        "++head_server.host=127.0.0.1" \
        "++head_server.port=${GYM_HEAD_PORT}" \
        "++apex_agent_concurrency=1" \
        "++apex_agent_timeout=${APEX_AGENT_TIMEOUT}" \
        "++apex_agent_max_output_tokens=${MAX_OUTPUT_TOKENS}" \
        "++apex_agent_supports_vision=${APEX_AGENT_SUPPORTS_VISION}" \
        "++apex_agent_sandbox_provider=apex_agent_apptainer_sandbox" \
        "++apex_agent_apptainer_direct_exec=true" \
        "++apex_agent_apptainer_fakeroot_for_root=false" \
        "++apex_agents_image=${APEX_IMAGE}" \
        "++apex_agents_auto_build=false" \
        "++apex_world_cache_dir=${WORLD_CACHE_DIR}" \
        "++policy_model.responses_api_models.vllm_model.base_url=${urls}" \
        "++policy_model.responses_api_models.vllm_model.model=${MODEL_NAME}" \
        >/dev/null
    log "Validation passed: paths, ${POLICY_LABEL} profile, world cache, and node-local Gym config are valid."
}

run_gym_worker() {
    load_profile
    validate_common
    : "${SHARD_INDEX:?SHARD_INDEX must be set}"
    : "${WORKER_CONCURRENCY:?WORKER_CONCURRENCY must be set}"
    : "${POLICY_URL:?POLICY_URL must be set}"
    : "${INPUT_JSONL:?INPUT_JSONL must be set}"
    : "${SHARD_OUTPUT_JSONL:?SHARD_OUTPUT_JSONL must be set}"
    : "${APEX_ARTIFACT_DIR:?APEX_ARTIFACT_DIR must be set}"

    unset NEMO_GYM_CONFIG_DICT NEMO_GYM_CONFIG_PATH
    ulimit -n 65535
    export HOME="/home/${USER_NAME}"
    export USER="${USER_NAME}"
    export PYTHONUNBUFFERED=1
    export UV_PROJECT_ENVIRONMENT="${APEX_GYM_DIR}/.venv"
    local local_root="/tmp/${USER_NAME}/apex-k3-${SLURM_JOB_ID}-${ROTATION}/shard-${SHARD_INDEX}"
    export UV_CACHE_DIR="${local_root}/uv-cache"
    export UV_PYTHON_INSTALL_DIR="${SHARED_RUNTIME_DIR}/python"
    # Ray appends /ray/session_<timestamp>_<pid>/sockets/... to this value.
    # Keep the prefix short enough for Linux's 107-byte AF_UNIX path limit.
    export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID}-${ROTATION}-${SHARD_INDEX}"
    export TMPDIR="${local_root}/tmp"
    export APPTAINER_CACHEDIR="${SHARED_RUNTIME_DIR}/apptainer-cache"
    export APPTAINER_TMPDIR="${local_root}/apptainer-tmp"
    export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
    export SINGULARITY_TMPDIR="${APPTAINER_TMPDIR}"
    mkdir -p \
        "${RAY_TMPDIR}" \
        "${TMPDIR}" \
        "${UV_CACHE_DIR}" \
        "${APPTAINER_CACHEDIR}" \
        "${APPTAINER_TMPDIR}" \
        "${APEX_ARTIFACT_DIR}" \
        "${RUN_DIR}/logs"

    cd "${APEX_GYM_DIR}"
    "${GYM_BIN}" --help >/dev/null

    local retry_preparation
    retry_preparation=$(python3 "${APEX_GYM_DIR}/scripts/prepare_apex_retry_rows.py" \
        --output "${SHARD_OUTPUT_JSONL}")
    log "Worker ${SHARD_INDEX}: retry preparation before resume: ${retry_preparation}"

    local policy_urls_omega="[\"${POLICY_URL}\"]"
    # `gym eval run` preprocesses the dataset declared on the selected agent
    # and then replaces input_jsonl_fpath with that preprocessed output. Thus,
    # --input alone does not shard a benchmark run; replace the agent's one
    # benchmark dataset so every worker materializes only its assigned shard.
    local dataset_omega
    dataset_omega="[{name:apex_agents,type:benchmark,jsonl_fpath:\"${INPUT_JSONL}\",prompt_config:null,prepare_script:\"benchmarks/apex_agents/prepare.py\",num_repeats:1,license:CC-BY-4.0}]"

    local common_args=(
        --config "${GYM_CONFIG}"
        --benchmark apex_agents
        --model-type vllm_model
        --model "${MODEL_NAME}"
        --model-url "${POLICY_URL}"
        --model-api-key dummy
        "++head_server.host=127.0.0.1"
        "++head_server.port=${GYM_HEAD_PORT}"
        "++apex_agent_concurrency=${WORKER_CONCURRENCY}"
        "++apex_agent_timeout=${APEX_AGENT_TIMEOUT}"
        "++apex_agent_max_output_tokens=${MAX_OUTPUT_TOKENS}"
        "++apex_agent_supports_vision=${APEX_AGENT_SUPPORTS_VISION}"
        "++apex_agent_sandbox_provider=apex_agent_apptainer_sandbox"
        "++apex_agent_apptainer_direct_exec=true"
        "++apex_agent_apptainer_fakeroot_for_root=false"
        "++apex_agent_apptainer_extra_start_args=[--writable-tmpfs,--cleanenv,--pid,--no-mount,\"home,tmp,bind-paths\",--home,/root]"
        "++apex_agents_image=${APEX_IMAGE}"
        "++apex_agents_auto_build=false"
        "++apex_world_cache_dir=${WORLD_CACHE_DIR}"
        "++apex_artifact_output_dir=${APEX_ARTIFACT_DIR}"
        "++skip_venv_if_present=true"
        "++apex_agents_benchmark_agent.responses_api_agents.apex_agent.datasets=${dataset_omega}"
        "++policy_model.responses_api_models.vllm_model.base_url=${policy_urls_omega}"
        "++policy_model.responses_api_models.vllm_model.model=${MODEL_NAME}"
        "++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=${USES_REASONING_PARSER:-true}"
        "++policy_model.responses_api_models.vllm_model.uses_interleaved_reasoning=${USES_INTERLEAVED_REASONING:-true}"
    )

    local eval_args=(
        --agent apex_agents_benchmark_agent
        --input "${INPUT_JSONL}"
        --output "${SHARD_OUTPUT_JSONL}"
        --num-repeats "${NUM_REPEATS}"
        --concurrency "${WORKER_CONCURRENCY}"
        --split "${SPLIT}"
        --resume
    )

    log "Worker ${SHARD_INDEX}: node=$(hostname), concurrency=${WORKER_CONCURRENCY}, policy=${POLICY_URL}"
    log "Worker ${SHARD_INDEX}: input=${INPUT_JSONL}, output=${SHARD_OUTPUT_JSONL}, tmp=${local_root}"
    log "Worker ${SHARD_INDEX}: timeout=${APEX_AGENT_TIMEOUT}s, max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
    setsid "${GYM_BIN}" eval run \
        "${common_args[@]}" \
        "${eval_args[@]}" &
    local eval_pid=$!

    terminate_eval() {
        kill -TERM -- "-${eval_pid}" 2>/dev/null || kill -TERM "${eval_pid}" 2>/dev/null || true
    }
    trap 'terminate_eval; wait "${eval_pid}" 2>/dev/null || true; exit 143' TERM INT
    local eval_rc=0
    wait "${eval_pid}" || eval_rc=$?

    retry_preparation=$(python3 "${APEX_GYM_DIR}/scripts/prepare_apex_retry_rows.py" \
        --output "${SHARD_OUTPUT_JSONL}")
    log "Worker ${SHARD_INDEX}: retry preparation after run: ${retry_preparation}"

    local completed=0
    [[ ! -f "${SHARD_OUTPUT_JSONL}" ]] || completed=$(awk 'NF {n++} END {print n+0}' "${SHARD_OUTPUT_JSONL}")
    local expected
    expected=$(awk 'NF {n++} END {print n+0}' "${INPUT_JSONL}")
    expected=$((expected * NUM_REPEATS))
    log "Worker ${SHARD_INDEX}: eval rc=${eval_rc}; output=${completed}/${expected}"
    (( eval_rc == 0 )) || return "${eval_rc}"
    (( completed == expected )) || die "worker ${SHARD_INDEX} output is incomplete"
}

gym_step_pids=()
serve_pids=()
merge_loop_pid=""
lb_pid=""

cleanup() {
    trap - EXIT TERM INT USR1
    local pid
    for pid in "${gym_step_pids[@]:-}"; do
        [[ -z "${pid}" ]] || kill -TERM "${pid}" 2>/dev/null || true
    done
    [[ -z "${merge_loop_pid:-}" ]] || kill -TERM "${merge_loop_pid}" 2>/dev/null || true
    [[ -z "${lb_pid:-}" ]] || kill -TERM "${lb_pid}" 2>/dev/null || true
    for pid in "${serve_pids[@]:-}"; do
        [[ -z "${pid}" ]] || kill -TERM "${pid}" 2>/dev/null || true
    done
    for pid in "${gym_step_pids[@]:-}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
    [[ -z "${merge_loop_pid:-}" ]] || wait "${merge_loop_pid}" 2>/dev/null || true
    [[ -z "${lb_pid:-}" ]] || wait "${lb_pid}" 2>/dev/null || true
    for pid in "${serve_pids[@]:-}"; do
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
    if [[ -n "${SHARD_COUNT:-}" && -d "${SHARD_OUTPUT_DIR:-/nonexistent}" ]]; then
        merge_outputs >/dev/null 2>&1 || true
    fi
}

prepare_shards() {
    python3 - \
        "${DATASET}" \
        "${EFFECTIVE_DATASET}" \
        "${SHARD_INPUT_DIR}" \
        "${SHARD_OUTPUT_DIR}" \
        "${SHARD_COUNT}" \
        "${LIMIT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
effective = Path(sys.argv[2])
input_dir = Path(sys.argv[3])
output_dir = Path(sys.argv[4])
shard_count = int(sys.argv[5])
limit = int(sys.argv[6]) if sys.argv[6] else None

raw_source = source.read_bytes()
raw_lines = [line + b"\n" for line in raw_source.splitlines() if line.strip()]
if limit is not None:
    raw_lines = raw_lines[:limit]
if not raw_lines:
    raise SystemExit("selected Apex dataset is empty")

task_ids = []
seen_task_ids = set()
for line_number, line in enumerate(raw_lines, 1):
    task_id = json.loads(line).get("task_id")
    if not task_id or task_id in seen_task_ids:
        raise SystemExit(f"missing or duplicate task_id at selected line {line_number}: {task_id!r}")
    task_ids.append(task_id)
    seen_task_ids.add(task_id)

manifest = {
    "source": str(source),
    "source_sha256": hashlib.sha256(raw_source).hexdigest(),
    "limit": limit,
    "selected_rows": len(raw_lines),
    "shard_count": shard_count,
    "task_ids_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
}
manifest_path = effective.with_name("manifest.json")
existing_outputs = list(output_dir.glob("shard-*.jsonl"))
if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    if any(path.stat().st_size for path in existing_outputs):
        raise SystemExit("dataset, limit, or shard topology changed while resumable outputs exist; use a new RUN_DIR")

input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
effective.parent.mkdir(parents=True, exist_ok=True)
shards = [[] for _ in range(shard_count)]
for index, line in enumerate(raw_lines):
    shards[index % shard_count].append(line)

def replace_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)

replace_bytes(effective, b"".join(raw_lines))
for index, shard in enumerate(shards):
    replace_bytes(input_dir / f"shard-{index}.jsonl", b"".join(shard))

temporary_manifest = manifest_path.with_suffix(".json.tmp")
temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
os.replace(temporary_manifest, manifest_path)
print(
    f"Prepared {len(raw_lines)} tasks across {shard_count} node-local Gym workers: "
    + ", ".join(str(len(shard)) for shard in shards)
)
PY
}

merge_outputs() {
    python3 "${APEX_GYM_DIR}/scripts/merge_apex_k3_shards.py" \
        --source "${EFFECTIVE_DATASET}" \
        --output-dir "${SHARD_OUTPUT_DIR}" \
        --shard-count "${SHARD_COUNT}" \
        --combined "${OUTPUT_JSONL}" \
        --num-repeats "${NUM_REPEATS}" \
        --max-attempts "${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}"
}

count_retryable_rows() {
    python3 "${APEX_GYM_DIR}/scripts/merge_apex_k3_shards.py" \
        --source "${EFFECTIVE_DATASET}" \
        --output-dir "${SHARD_OUTPUT_DIR}" \
        --shard-count "${SHARD_COUNT}" \
        --combined "${OUTPUT_JSONL}" \
        --num-repeats "${NUM_REPEATS}" \
        --max-attempts "${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}" \
        --print-retryable
}

on_walltime() {
    trap - USR1
    log "Walltime signal at rotation ${ROTATION}; preserving ${OUTPUT_JSONL}."
    cleanup
    if (( ROTATION + 1 >= MAX_ROTATIONS )); then
        log "Reached MAX_ROTATIONS=${MAX_ROTATIONS}; not requeueing."
        exit 0
    fi
    log "Requeueing job ${SLURM_JOB_ID} for rotation $((ROTATION + 1))."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}

run_main() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "submit this script with sbatch"
    load_profile
    validate_common

    mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    local n_nodes=${#nodes[@]}
    (( n_nodes % NODES_PER_REPLICA == 0 )) ||
        die "${n_nodes} nodes is not a multiple of NODES_PER_REPLICA=${NODES_PER_REPLICA}"
    export N_REPLICAS=$((n_nodes / NODES_PER_REPLICA))

    local selected_rows
    selected_rows=$(awk 'NF {n++} END {print n+0}' "${DATASET}")
    if [[ -n "${LIMIT}" && "${LIMIT}" -lt "${selected_rows}" ]]; then
        selected_rows=${LIMIT}
    fi
    (( selected_rows > 0 )) || die "selected dataset is empty"

    export SHARD_COUNT=${n_nodes}
    (( CONCURRENCY < SHARD_COUNT )) && SHARD_COUNT=${CONCURRENCY}
    (( selected_rows < SHARD_COUNT )) && SHARD_COUNT=${selected_rows}

    mkdir -p \
        "${RUN_DIR}/logs" \
        "${RUN_DIR}/server_info" \
        "${RUN_DIR}/artifacts" \
        "${SHARD_INPUT_DIR}" \
        "${SHARD_OUTPUT_DIR}" \
        "${SHARED_RUNTIME_DIR}"
    log "Topology: ${n_nodes} nodes = ${N_REPLICAS} replica(s) x ${NODES_PER_REPLICA}"
    log "Collectors: ${SHARD_COUNT} node-local Gym worker(s), global concurrency=${CONCURRENCY}, "\
"selected tasks=${selected_rows}"
    log "Per-node CPU budget: policy=${K3_CPUS_PER_NODE}, Gym=${GYM_CPUS_PER_NODE}, "\
"unallocated=$((144 - K3_CPUS_PER_NODE - GYM_CPUS_PER_NODE))"
    prepare_shards

    # Distributed policy replicas are generally faster when their nodes are
    # rack-local. Group by hostname prefix before pooling cross-rack leftovers.
    local -A rack_of=()
    local -a racks=() replica_placements=() placement_pool=()
    local node rack rack_count
    for node in "${nodes[@]}"; do
        rack=${node%%-T*}
        if [[ -z "${rack_of[${rack}]+x}" ]]; then
            racks+=("${rack}")
            rack_of["${rack}"]=""
        fi
        rack_of["${rack}"]+=" ${node}"
    done
    log "Allocation spans ${#racks[@]} rack(s):"
    local -a rack_nodes=()
    for rack in "${racks[@]}"; do
        read -ra rack_nodes <<< "${rack_of[${rack}]}"
        rack_count=${#rack_nodes[@]}
        log "  ${rack}: ${rack_count} node(s)"
        while (( ${#rack_nodes[@]} >= NODES_PER_REPLICA )); do
            replica_placements+=("$(IFS=,; echo "${rack_nodes[*]:0:NODES_PER_REPLICA}")")
            rack_nodes=("${rack_nodes[@]:NODES_PER_REPLICA}")
        done
        placement_pool+=("${rack_nodes[@]}")
    done
    while (( ${#placement_pool[@]} >= NODES_PER_REPLICA )); do
        replica_placements+=("$(IFS=,; echo "${placement_pool[*]:0:NODES_PER_REPLICA}")")
        placement_pool=("${placement_pool[@]:NODES_PER_REPLICA}")
        log "WARNING: replica $((${#replica_placements[@]} - 1)) straddles racks; expect lower serving throughput"
    done
    (( ${#replica_placements[@]} >= N_REPLICAS )) ||
        die "could place only ${#replica_placements[@]}/${N_REPLICAS} ${POLICY_LABEL} replicas"

    local replica_index replica_nodes first_rack same_rack placement_node
    for ((replica_index = 0; replica_index < N_REPLICAS; replica_index++)); do
        replica_nodes=${replica_placements[replica_index]}
        first_rack=${replica_nodes%%-T*}
        same_rack=yes
        for placement_node in ${replica_nodes//,/ }; do
            [[ "${placement_node%%-T*}" == "${first_rack}" ]] || same_rack=no
        done
        rm -f "${RUN_DIR}/replica_${replica_index}/server_info/"*.env 2>/dev/null || true
        (
            # The batch allocation exports both SLURM_CPUS_PER_TASK=144 and
            # SLURM_TRES_PER_TASK=cpu=144. Serge's wrapper does not pass -c to
            # its nested srun, so provide one consistent configured CPU default
            # instead of leaving Slurm two conflicting environment-derived values.
            unset SLURM_TRES_PER_TASK
            export SLURM_CPUS_PER_TASK="${K3_CPUS_PER_NODE}"
            REPLICA_NODES="${replica_nodes}" \
            OUTPUT_DIR="${RUN_DIR}/replica_${replica_index}" \
            API_PORT=$((BASE_API_PORT + replica_index)) \
            LOG_DIR="${RUN_DIR}/logs" \
                exec bash "${POLICY_SERVE_SCRIPT}"
        ) &
        serve_pids+=("$!")
        log "Replica ${replica_index}: nodes=${replica_nodes}, rack-local=${same_rack}, port=$((BASE_API_PORT + replica_index))"
    done

    local serve_wait=${SERVE_WAIT_SECONDS:-3600}
    local deadline=$((SECONDS + serve_wait))
    local -a policy_urls=()
    log "Waiting up to ${serve_wait}s for ${N_REPLICAS} ${POLICY_LABEL} replica(s)."
    while ((SECONDS < deadline)); do
        policy_urls=()
        for ((replica_index = 0; replica_index < N_REPLICAS; replica_index++)); do
            local env_file=""
            env_file=$(find "${RUN_DIR}/replica_${replica_index}/server_info" -maxdepth 1 -name '*.env' -print -quit 2>/dev/null || true)
            [[ -n "${env_file}" ]] || continue
            local url=""
            url=$(sed -n 's/^SERVER_URL=//p' "${env_file}" | head -n 1)
            [[ -n "${url}" ]] && policy_urls+=("${url}")
        done
        (( ${#policy_urls[@]} == N_REPLICAS )) && break
        local alive=0
        local pid
        for pid in "${serve_pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive=$((alive + 1))
        done
        (( alive > 0 )) || die "all ${POLICY_LABEL} serving processes exited before readiness"
        sleep 10
    done
    (( ${#policy_urls[@]} == N_REPLICAS )) ||
        die "${POLICY_LABEL} readiness timed out (${#policy_urls[@]}/${N_REPLICAS} replicas ready)"

    local url served
    for url in "${policy_urls[@]}"; do
        served=$(curl -sf "${url}/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
        [[ "${served}" == "${MODEL_NAME}" ]] || die "${url} serves ${served}, expected ${MODEL_NAME}"
    done
    log "${POLICY_LABEL} ready: ${policy_urls[*]}"

    # Present one endpoint to every Gym worker. Least-connections keeps a
    # global queue across replicas, avoiding static-shard tail imbalance.
    local -a policy_origins=()
    for url in "${policy_urls[@]}"; do
        policy_origins+=("${url%/v1}")
    done
    local policy_backends
    policy_backends=$(IFS=,; echo "${policy_origins[*]}")
    local head_node=${nodes[0]}
    local head_ip
    head_ip=$(getent hosts "${head_node}" | awk '{print $1; exit}')
    [[ -n "${head_ip}" ]] || die "could not resolve batch head node ${head_node}"
    local lb_port_file="${RUN_DIR}/server_info/lb-${SLURM_JOB_ID}-${ROTATION}.port"
    local lb_log="${RUN_DIR}/logs/lb-${SLURM_JOB_ID}-${ROTATION}.log"
    rm -f "${lb_port_file}"
    "${POLICY_LB_PYTHON}" "${POLICY_LB_SCRIPT}" \
        --port 0 \
        --port-file "${lb_port_file}" \
        --backends "${policy_backends}" \
        --wait-for-backends \
        >"${lb_log}" 2>&1 &
    lb_pid=$!

    local lb_deadline=$((SECONDS + 180))
    while [[ ! -s "${lb_port_file}" && ${SECONDS} -lt ${lb_deadline} ]]; do
        kill -0 "${lb_pid}" 2>/dev/null || {
            tail -n 50 "${lb_log}" >&2 || true
            die "${POLICY_LABEL} load balancer exited before readiness"
        }
        sleep 2
    done
    [[ -s "${lb_port_file}" ]] || die "${POLICY_LABEL} load balancer did not publish a port within 180s"
    local lb_port
    # Serge's proxy intentionally writes the port without a trailing newline.
    # `read` still populates the value in that case but returns nonzero at EOF,
    # which aborts this set -e script before Gym starts.
    lb_port=$(<"${lb_port_file}")
    [[ "${lb_port}" =~ ^[1-9][0-9]*$ ]] && (( lb_port <= 65535 )) ||
        die "${POLICY_LABEL} load balancer published an invalid port: ${lb_port}"
    local balanced_policy_url="http://${head_ip}:${lb_port}/v1"
    local lb_models
    if ! lb_models=$(curl -sf --max-time 30 "http://127.0.0.1:${lb_port}/v1/models"); then
        tail -n 50 "${lb_log}" >&2 || true
        die "${POLICY_LABEL} load balancer localhost readiness probe failed"
    fi
    if ! served=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<<"${lb_models}"); then
        die "${POLICY_LABEL} load balancer returned an invalid /v1/models response"
    fi
    [[ "${served}" == "${MODEL_NAME}" ]] ||
        die "${balanced_policy_url} serves ${served}, expected ${MODEL_NAME}"
    log "${POLICY_LABEL} least-connections endpoint: ${balanced_policy_url}; backends=${policy_backends}"

    local container_env="APEX_GYM_DIR,PROFILE,DATASET,GYM_CONFIG,GYM_BIN,APEX_IMAGE,WORLD_CACHE_DIR"
    container_env+=",SHARED_RUNTIME_DIR,RUN_DIR,OUTPUT_JSONL,CONCURRENCY,NUM_REPEATS,LIMIT,MAX_OUTPUT_TOKENS"
    container_env+=",SHARD_ROOT,SHARD_INPUT_DIR,SHARD_OUTPUT_DIR,EFFECTIVE_DATASET,SHARD_COUNT"
    container_env+=",APEX_AGENT_TIMEOUT,NEMO_GYM_MAX_ROLLOUT_ATTEMPTS,MAX_ROTATIONS,ROTATION"
    container_env+=",SERVER_WAIT_SECONDS,USER_NAME,N_REPLICAS,GYM_HEAD_PORT"
    container_env+=",SHARD_INDEX,WORKER_CONCURRENCY,POLICY_URL,INPUT_JSONL,SHARD_OUTPUT_JSONL,APEX_ARTIFACT_DIR"
    container_env+=",NEMO_GYM_REASONING_FIELD"
    container_env+=",MODEL_NAME,MODEL_PATH,CONTAINER_IMAGE,LAUNCHER_PATH,NODES_PER_REPLICA"
    container_env+=",USES_REASONING_PARSER,USES_INTERLEAVED_REASONING"

    local base_concurrency=$((CONCURRENCY / SHARD_COUNT))
    local extra_concurrency=$((CONCURRENCY % SHARD_COUNT))
    local worker_index worker_concurrency replica_slot
    local -a worker_replica_nodes=()
    for ((worker_index = 0; worker_index < SHARD_COUNT; worker_index++)); do
        worker_concurrency=${base_concurrency}
        (( worker_index < extra_concurrency )) && worker_concurrency=$((worker_concurrency + 1))

        # Interleave collectors across replicas so low-concurrency canaries still
        # exercise every model replica before using a second node in a replica.
        replica_index=$((worker_index % N_REPLICAS))
        replica_slot=$((worker_index / N_REPLICAS))
        IFS=, read -ra worker_replica_nodes <<< "${replica_placements[replica_index]}"
        node=${worker_replica_nodes[replica_slot]}

        log "Launching Gym worker ${worker_index}: node=${node}, placement-replica=${replica_index}, "\
"concurrency=${worker_concurrency}, policy=least-connections"
        SHARD_INDEX="${worker_index}" \
        WORKER_CONCURRENCY="${worker_concurrency}" \
        POLICY_URL="${balanced_policy_url}" \
        INPUT_JSONL="${SHARD_INPUT_DIR}/shard-${worker_index}.jsonl" \
        SHARD_OUTPUT_JSONL="${SHARD_OUTPUT_DIR}/shard-${worker_index}.jsonl" \
        APEX_ARTIFACT_DIR="${RUN_DIR}/artifacts/shard-${worker_index}" \
            env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
            srun --overlap --exact \
                --nodes=1 \
                --ntasks=1 \
                --nodelist="${node}" \
                --cpus-per-task="${GYM_CPUS_PER_NODE}" \
                --gres=none \
                --cpu-bind=none \
                --kill-on-bad-exit=1 \
                --export=ALL \
                --output="${RUN_DIR}/logs/gym-${SLURM_JOB_ID}-${ROTATION}-shard-${worker_index}-%N.log" \
                --no-container-mount-home \
                --container-image="${GYM_CONTAINER}" \
                --container-mounts="/lustre:/lustre,/home:/home" \
                --container-workdir="${APEX_GYM_DIR}" \
                --container-env="${container_env}" \
                bash "${APEX_GYM_DIR}/scripts/run_apex_agents_k3_unified.sbatch" --gym-worker &
        gym_step_pids+=("$!")
    done

    (
        while true; do
            merge_outputs >/dev/null 2>&1 || true
            sleep 60
        done
    ) &
    merge_loop_pid=$!

    local gym_rc=0
    local pid
    for pid in "${gym_step_pids[@]}"; do
        wait "${pid}" || gym_rc=1
    done
    gym_step_pids=()

    kill "${merge_loop_pid}" 2>/dev/null || true
    wait "${merge_loop_pid}" 2>/dev/null || true
    merge_loop_pid=""

    local completed
    completed=$(merge_outputs)
    local expected
    expected=$(awk 'NF {n++} END {print n+0}' "${EFFECTIVE_DATASET}")
    expected=$((expected * NUM_REPEATS))
    log "Gym workers rc=${gym_rc}; combined output=${completed}/${expected} rows at ${OUTPUT_JSONL}"
    if (( completed != expected )); then
        local retryable
        retryable=$(count_retryable_rows)
        log "Incomplete output has ${retryable} retryable row(s); max attempts=${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS}."
        if (( retryable > 0 )); then
            if (( ROTATION + 1 >= MAX_ROTATIONS )); then
                die "rollout output is incomplete with retryable rows, but MAX_ROTATIONS=${MAX_ROTATIONS} was reached"
            fi
            log "Auto-requeueing job ${SLURM_JOB_ID} for retry rotation $((ROTATION + 1))."
            cleanup
            scontrol requeue "${SLURM_JOB_ID}"
            exit 0
        fi
        log "No retryable rows remain; all missing rows are terminal or reached the attempt limit."
    elif (( gym_rc != 0 )); then
        log "WARNING: a Gym worker exited nonzero, but all expected output rows were recovered."
    fi
}

case "${1:-}" in
    --validate)
        validate_local
        ;;
    --gym-worker)
        run_gym_worker
        ;;
    "")
        trap cleanup EXIT
        trap on_walltime USR1
        trap 'exit 143' TERM
        trap 'exit 130' INT
        run_main
        ;;
    *)
        die "unknown argument: $1"
        ;;
esac
