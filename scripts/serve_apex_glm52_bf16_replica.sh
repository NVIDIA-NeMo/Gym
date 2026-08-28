#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start one GLM-5.2 BF16 vLLM endpoint on an explicit 4-, 8-, or 16-node subset of
# an existing Slurm allocation. The parent rollout harness provides
# REPLICA_NODES, OUTPUT_DIR, API_PORT, and the model settings through a profile.

set -Eeuo pipefail

log() {
    printf '[glm52-bf16 %s] %s\n' "$(date +%H:%M:%S)" "$*"
}

die() {
    log "FATAL: $*" >&2
    exit 1
}

require_positive_int() {
    local name=$1
    local value=$2
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got ${value}"
}

: "${REPLICA_NODES:?REPLICA_NODES must list the nodes for this endpoint}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
: "${MODEL_NAME:?MODEL_NAME must be set}"
: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${CONTAINER_IMAGE:?CONTAINER_IMAGE must be set}"

API_PORT=${API_PORT:-10240}
BASE_RAY_PORT=${BASE_RAY_PORT:-6379}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-16}
DATA_PARALLEL_SIZE_LOCAL=${DATA_PARALLEL_SIZE_LOCAL:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
API_SERVER_COUNT=${API_SERVER_COUNT:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
K3_CPUS_PER_NODE=${K3_CPUS_PER_NODE:-40}
CONTAINER_MOUNTS=${CONTAINER_MOUNTS:-/lustre:/lustre}
LOG_DIR=${LOG_DIR:-${OUTPUT_DIR}}
CACHE_ROOT=${CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${USER}/cache}
HF_HOME=${HF_HOME:-${CACHE_ROOT}/huggingface}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-300}
GLM52_WARMUP_MAX_TOKENS=${GLM52_WARMUP_MAX_TOKENS:-256}
GLM52_ENFORCE_EAGER=${GLM52_ENFORCE_EAGER:-false}

require_positive_int API_PORT "${API_PORT}"
require_positive_int BASE_RAY_PORT "${BASE_RAY_PORT}"
require_positive_int GPUS_PER_NODE "${GPUS_PER_NODE}"
require_positive_int TENSOR_PARALLEL_SIZE "${TENSOR_PARALLEL_SIZE}"
require_positive_int PIPELINE_PARALLEL_SIZE "${PIPELINE_PARALLEL_SIZE}"
require_positive_int DATA_PARALLEL_SIZE "${DATA_PARALLEL_SIZE}"
require_positive_int DATA_PARALLEL_SIZE_LOCAL "${DATA_PARALLEL_SIZE_LOCAL}"
require_positive_int API_SERVER_COUNT "${API_SERVER_COUNT}"
require_positive_int MAX_MODEL_LEN "${MAX_MODEL_LEN}"
require_positive_int K3_CPUS_PER_NODE "${K3_CPUS_PER_NODE}"
require_positive_int VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS "${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}"
require_positive_int GLM52_WARMUP_MAX_TOKENS "${GLM52_WARMUP_MAX_TOKENS}"
[[ "${GLM52_ENFORCE_EAGER}" == "true" || "${GLM52_ENFORCE_EAGER}" == "false" ]] ||
    die "GLM52_ENFORCE_EAGER must be true or false; got ${GLM52_ENFORCE_EAGER}"

(( DATA_PARALLEL_SIZE % DATA_PARALLEL_SIZE_LOCAL == 0 )) ||
    die "DATA_PARALLEL_SIZE must be divisible by DATA_PARALLEL_SIZE_LOCAL"

IFS=',' read -ra nodes <<< "${REPLICA_NODES}"
node_count=${#nodes[@]}
expected_nodes=$((DATA_PARALLEL_SIZE / DATA_PARALLEL_SIZE_LOCAL))
(( node_count == expected_nodes )) ||
    die "received ${node_count} nodes, but DP=${DATA_PARALLEL_SIZE} and DPL=${DATA_PARALLEL_SIZE_LOCAL} require ${expected_nodes}"
(( TENSOR_PARALLEL_SIZE <= GPUS_PER_NODE )) ||
    die "TP=${TENSOR_PARALLEL_SIZE} exceeds the ${GPUS_PER_NODE} GPUs available per node"

[[ -r "${MODEL_PATH}/config.json" ]] || die "model config is not readable: ${MODEL_PATH}/config.json"
[[ -r "${CONTAINER_IMAGE}" ]] || die "container is not readable: ${CONTAINER_IMAGE}"

mkdir -p "${OUTPUT_DIR}/server_info" "${LOG_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}"

head_host=${nodes[0]}
head_ip=$(getent hosts "${head_host}" | awk '{print $1; exit}')
[[ -n "${head_ip}" ]] || die "could not resolve head node ${head_host}"

ray_port=${RAY_PORT:-${BASE_RAY_PORT}}
server_url="http://${head_ip}:${API_PORT}/v1"
safe_model_name=${MODEL_NAME//\//_}
server_info_file="${OUTPUT_DIR}/server_info/${safe_model_name}.env"
head_ip_file="${OUTPUT_DIR}/.ray_head_ip_${SLURM_JOB_ID}_${API_PORT}"
vllm_log="${LOG_DIR}/glm52-bf16-${SLURM_JOB_ID}-${API_PORT}.log"
rm -f "${server_info_file}" "${server_info_file}.tmp" "${head_ip_file}"

export API_PORT API_SERVER_COUNT BASE_RAY_PORT CACHE_ROOT CONTAINER_IMAGE DATA_PARALLEL_SIZE
export DATA_PARALLEL_SIZE_LOCAL GPU_MEMORY_UTILIZATION HF_HOME MAX_MODEL_LEN MODEL_NAME MODEL_PATH
export OUTPUT_DIR PIPELINE_PARALLEL_SIZE RAY_PORT="${ray_port}" TENSOR_PARALLEL_SIZE VLLM_CACHE_ROOT
export GLM52_ENFORCE_EAGER GLM52_WARMUP_MAX_TOKENS VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
export VLLM_LOG="${vllm_log}"

step_pid=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${step_pid}" ]] && kill -0 "${step_pid}" 2>/dev/null; then
        kill "${step_pid}" 2>/dev/null || true
        wait "${step_pid}" 2>/dev/null || true
    fi
    rm -f "${head_ip_file}"
    exit "${status}"
}
trap cleanup EXIT INT TERM

log "nodes=${REPLICA_NODES}; head=${head_host} (${head_ip}); API=${server_url}"
log "TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DATA_PARALLEL_SIZE}, DPL=${DATA_PARALLEL_SIZE_LOCAL}"
log "eager=${GLM52_ENFORCE_EAGER}; execute-model timeout=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}s; warm-up tokens=${GLM52_WARMUP_MAX_TOKENS}"

srun --overlap --exact \
    --nodes="${node_count}" \
    --ntasks="${node_count}" \
    --ntasks-per-node=1 \
    --nodelist="${REPLICA_NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${K3_CPUS_PER_NODE}" \
    --cpu-bind=none \
    --kill-on-bad-exit=1 \
    --label \
    --mpi=pmix \
    --export=ALL \
    --no-container-mount-home \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    bash -lc '
        set -Eeuo pipefail

        # Do not depend on the submit shell PATH. The GLM image installs its
        # Ray and vLLM entrypoints in /usr/local/bin.
        export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
        for required_command in ray vllm curl; do
            command -v "${required_command}" >/dev/null 2>&1 || {
                echo "ERROR: ${required_command} is unavailable inside ${CONTAINER_IMAGE}; PATH=${PATH}" >&2
                exit 64
            }
        done

        export HF_HOME VLLM_CACHE_ROOT
        export HF_TOKEN="${HF_TOKEN:-}"
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        export VLLM_ENGINE_READY_TIMEOUT_S=3600
        # execute_model is one distributed model step, not the whole response.
        # A 300-second expiry therefore indicates a deadlocked rank and should
        # fail fast instead of being hidden behind a longer request timeout.
        export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS
        export RAY_CGRAPH_get_timeout=600
        export RAY_raylet_start_wait_time_s=120
        export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
        export VLLM_ALLREDUCE_USE_SYMM_MEM=0
        export VLLM_USE_DEEP_GEMM=0
        export FLASHINFER_WORKSPACE_BASE=/tmp

        HEAD_IP_FILE="${OUTPUT_DIR}/.ray_head_ip_${SLURM_JOB_ID}_${API_PORT}"
        RAY_FIXED_PORTS="--node-manager-port=8266 --object-manager-port=8267 --metrics-export-port=8269 --dashboard-agent-grpc-port=8270 --dashboard-agent-listen-port=8271 --runtime-env-agent-port=8272"

        if [[ "${SLURM_PROCID}" -eq 0 ]]; then
            vllm_pid=""
            tail_pid=""
            fatal_monitor_pid=""
            rank0_cleanup() {
                status=$?
                trap - EXIT INT TERM
                [[ -z "${fatal_monitor_pid}" ]] || kill "${fatal_monitor_pid}" 2>/dev/null || true
                [[ -z "${tail_pid}" ]] || kill "${tail_pid}" 2>/dev/null || true
                [[ -z "${vllm_pid}" ]] || kill "${vllm_pid}" 2>/dev/null || true
                ray stop >/dev/null 2>&1 || true
                rm -f "${HEAD_IP_FILE}"
                exit "${status}"
            }
            trap rank0_cleanup EXIT INT TERM

            rm -f "${HEAD_IP_FILE}"
            HEAD_IP=$(hostname -I | awk "{print \$1}")
            if [[ -z "${HEAD_IP}" ]]; then
                HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
            fi
            [[ -n "${HEAD_IP}" ]] || { echo "ERROR: could not determine Ray head IP" >&2; exit 1; }
            printf "%s\n" "${HEAD_IP}" > "${HEAD_IP_FILE}"

            echo "Starting Ray head on ${HEAD_IP}:${RAY_PORT}"
            ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" ${RAY_FIXED_PORTS} --disable-usage-stats
            export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

            for _ in $(seq 1 360); do
                active_nodes=$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_" || true)
                (( active_nodes >= DATA_PARALLEL_SIZE / DATA_PARALLEL_SIZE_LOCAL )) && break
                sleep 5
            done
            active_nodes=$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_" || true)
            expected_nodes=$((DATA_PARALLEL_SIZE / DATA_PARALLEL_SIZE_LOCAL))
            if (( active_nodes < expected_nodes )); then
                echo "ERROR: only ${active_nodes}/${expected_nodes} Ray nodes joined" >&2
                ray status --address "${RAY_ADDRESS}" || true
                ray stop || true
                exit 1
            fi
            ray status --address "${RAY_ADDRESS}" || true

            serve_port="${API_PORT}"
            unset VLLM_PORT
            echo "Starting GLM-5.2 BF16 vLLM server on port ${serve_port}"
            # Blackwell startup has independent FlashInfer and TorchInductor
            # autotuners. Disable both problematic benchmark paths while
            # retaining ordinary Inductor compilation and CUDA graphs.
            compilation_config="{\"inductor_compile_config\": {\"combo_kernels\": false, "
            compilation_config+="\"benchmark_combo_kernel\": false}, "
            compilation_config+="\"pass_config\": {\"fuse_allreduce_rms\": false}}"
            execution_args=()
            if [[ "${GLM52_ENFORCE_EAGER}" == "true" ]]; then
                echo "DP=${DATA_PARALLEL_SIZE}: forcing eager execution"
                execution_args+=(--enforce-eager)
            fi
            vllm serve "${MODEL_PATH}" \
                --enable-log-requests \
                --trust-remote-code \
                --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
                --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
                --data-parallel-size "${DATA_PARALLEL_SIZE}" \
                --data-parallel-backend ray \
                --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}" \
                --distributed-executor-backend ray \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
                --port "${serve_port}" \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm45 \
                --enable-prefix-caching \
                --enable-chunked-prefill \
                --enable-expert-parallel \
                --no-enable-flashinfer-autotune \
                --api-server-count "${API_SERVER_COUNT}" \
                --chat-template-content-format string \
                --served-model-name "${MODEL_NAME}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --compilation-config "${compilation_config}" \
                --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" \
                "${execution_args[@]}" \
                > "${VLLM_LOG}" 2>&1 &
            vllm_pid=$!

            # The API process can remain alive after a Ray/NCCL engine thread
            # dies. Convert those terminal log signatures into a nonzero rank0
            # exit so srun --kill-on-bad-exit tears down every worker promptly.
            (
                while kill -0 "${vllm_pid}" 2>/dev/null; do
                    if grep -Eq \
                        "Watchdog caught collective operation timeout|EngineCore encountered a fatal error|RayWorkerProc rank=.*died unexpectedly" \
                        "${VLLM_LOG}"; then
                        echo "ERROR: fatal distributed inference failure detected; terminating vLLM" >&2
                        kill "${vllm_pid}" 2>/dev/null || true
                        exit 1
                    fi
                    sleep 2
                done
            ) &
            fatal_monitor_pid=$!

            tail -n +1 -F "${VLLM_LOG}" &
            tail_pid=$!
            while ! curl -sf --max-time 5 "http://127.0.0.1:${serve_port}/v1/models" >/dev/null 2>&1; do
                if ! kill -0 "${vllm_pid}" 2>/dev/null; then
                    echo "ERROR: vLLM exited before readiness" >&2
                    exit 1
                fi
                sleep 2
            done

            # /v1/models only proves that the API process is listening. Exercise
            # every DP engine with a representative prefill and forced decode
            # before publishing the endpoint to the rollout harness. This keeps
            # runtime compilation out of the first real Apex rollouts and must
            # cross the 71-token point where the DP4 canary previously hung.
            echo "Warming ${DATA_PARALLEL_SIZE} GLM data-parallel engine(s)"
            warmup_text=""
            for _ in $(seq 1 1024); do
                warmup_text+=" warmup"
            done
            printf -v warmup_payload \
                "{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"%s Continue writing WARMUP tokens.\"}],\"temperature\":0,\"ignore_eos\":true,\"max_tokens\":%s}" \
                "${MODEL_NAME}" "${warmup_text}" "${GLM52_WARMUP_MAX_TOKENS}"

            warmup_pids=()
            warmup_outputs=()
            warmup_max_time=$((VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS + 60))
            for warmup_index in $(seq 0 $((DATA_PARALLEL_SIZE - 1))); do
                warmup_output="/tmp/glm52-warmup-${SLURM_JOB_ID}-${API_PORT}-${warmup_index}.json"
                rm -f "${warmup_output}"
                warmup_outputs+=("${warmup_output}")
                curl --fail --silent --show-error \
                    --max-time "${warmup_max_time}" \
                    -H "Content-Type: application/json" \
                    -d "${warmup_payload}" \
                    "http://127.0.0.1:${serve_port}/v1/chat/completions" \
                    > "${warmup_output}" &
                warmup_pids+=("$!")
            done

            warmup_failed=0
            for warmup_pid in "${warmup_pids[@]}"; do
                if ! wait "${warmup_pid}"; then
                    warmup_failed=1
                fi
            done
            for warmup_output in "${warmup_outputs[@]}"; do
                if [[ ! -s "${warmup_output}" ]] || ! grep -q "\"choices\"" "${warmup_output}"; then
                    echo "ERROR: invalid GLM warm-up response in ${warmup_output}" >&2
                    warmup_failed=1
                fi
            done
            rm -f "${warmup_outputs[@]}"
            (( warmup_failed == 0 )) || { echo "ERROR: GLM inference warm-up failed" >&2; exit 1; }
            echo "GLM inference warm-up completed across ${DATA_PARALLEL_SIZE} engine(s)"

            SERVER_INFO_FILE="${OUTPUT_DIR}/server_info/${MODEL_NAME//\//_}.env"
            {
                echo "MODEL_NAME=${MODEL_NAME}"
                echo "HEAD_IP=${HEAD_IP}"
                echo "VLLM_PORT=${serve_port}"
                echo "SERVER_URL=http://${HEAD_IP}:${serve_port}/v1"
            } > "${SERVER_INFO_FILE}.tmp"
            mv "${SERVER_INFO_FILE}.tmp" "${SERVER_INFO_FILE}"
            echo "[glm52-bf16] READY at http://${HEAD_IP}:${serve_port}/v1"

            wait "${vllm_pid}"
        fi

        for _ in $(seq 1 180); do
            [[ -s "${HEAD_IP_FILE}" ]] && break
            sleep 1
        done
        [[ -s "${HEAD_IP_FILE}" ]] || { echo "ERROR: timed out waiting for ${HEAD_IP_FILE}" >&2; exit 1; }
        HEAD_IP=$(<"${HEAD_IP_FILE}")
        export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
        echo "Starting Ray worker for ${RAY_ADDRESS}"
        until ray start --address="${RAY_ADDRESS}" ${RAY_FIXED_PORTS} --disable-usage-stats; do
            echo "Retrying Ray worker connection to ${RAY_ADDRESS}"
            sleep 5
        done
        tail -f /dev/null
    ' &
step_pid=$!
wait "${step_pid}"
