#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Standalone GLM-5.2 BF16 vLLM launcher derived from the original server-only
# recipe. This control canary deliberately does not start Gym or Apex. The only
# topology change from the original recipe is 8 nodes / Ray DP=8 instead of
# 16 nodes / Ray DP=16; TP remains 4 GPUs per node.

#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --switches=1
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -Eeuo pipefail

OUTPUT_DIR=""
readonly LUSTRE_DIR=/lustre

VLLM_IMAGE="${VLLM_IMAGE:-/lustre/fsw/portfolios/llmservice/users/mengxiwu/containers/vllm-glm52-arm64-cu130-ray.sqsh}"
MODEL_PATH="${MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/venkats/models/GLM-5.2}"
MODEL_NAME="${MODEL_NAME:-GLM-52-bf16-parallel}"
NUM_INSTANCES=""
INSTANCE_IDX=""

NUM_NODES=8
ACCOUNT=nemotron_n4_post
PARTITION=batch
QOS="${QOS:-normal}"
RESERVATION="${RESERVATION:-}"
TIME_LIMIT=04:00:00

readonly BASE_RAY_PORT=6379
readonly BASE_VLLM_PORT=10240

TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=8
DATA_PARALLEL_SIZE_LOCAL=1
GPU_MEMORY_UTILIZATION=0.85
API_SERVER_COUNT=1
MAX_MODEL_LEN=262144

CACHE_ROOT="${CACHE_ROOT:-/lustre/fsw/portfolios/llmservice/users/${USER:-$(id -un 2>/dev/null || echo unknown)}/cache}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"

usage() {
    echo "Usage: $0 --num-instances <n> --output-dir <path> [OPTIONS]"
    echo ""
    echo "Required outside Slurm:"
    echo "  --num-instances <n>        Number of standalone endpoints to submit"
    echo "  --output-dir <path>        Output directory for logs and server_info"
    echo ""
    echo "Optional overrides:"
    echo "  --vllm-image <path>        vLLM Ray container image"
    echo "  --model-path <path>        Model checkpoint path"
    echo "  --model-name <name>        Base served model name"
    echo "  --account <name>           Slurm account (default: ${ACCOUNT})"
    echo "  --partition <name>         Slurm partition (default: ${PARTITION})"
    echo "  --qos <name>               Slurm QoS (default: ${QOS})"
    echo "  --reservation <name>       Optional Slurm reservation"
    echo "  --time-limit <time>        Slurm walltime (default: ${TIME_LIMIT})"
    echo "  --gpu-memory-utilization <fraction>"
    echo "  --max-model-len <n>        Served prompt + output context"
    echo "  --cache-root <path>        Host-visible cache root"
    exit 1
}

require_positive_int() {
    local name=$1
    local value=$2
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer; got '${value}'" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-image)
            VLLM_IMAGE=$2
            shift 2
            ;;
        --model-path)
            MODEL_PATH=$2
            shift 2
            ;;
        --model-name)
            MODEL_NAME=$2
            shift 2
            ;;
        --num-instances)
            NUM_INSTANCES=$2
            shift 2
            ;;
        --instance-idx)
            INSTANCE_IDX=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --account)
            ACCOUNT=$2
            shift 2
            ;;
        --partition)
            PARTITION=$2
            shift 2
            ;;
        --qos)
            QOS=$2
            shift 2
            ;;
        --reservation)
            RESERVATION=$2
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT=$2
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION=$2
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN=$2
            shift 2
            ;;
        --cache-root)
            CACHE_ROOT=$2
            HF_HOME="${CACHE_ROOT}/huggingface"
            VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

require_positive_int "NUM_NODES" "${NUM_NODES}"
require_positive_int "TENSOR_PARALLEL_SIZE" "${TENSOR_PARALLEL_SIZE}"
require_positive_int "DATA_PARALLEL_SIZE" "${DATA_PARALLEL_SIZE}"
require_positive_int "MAX_MODEL_LEN" "${MAX_MODEL_LEN}"

if [[ "${DATA_PARALLEL_SIZE}" -ne "${NUM_NODES}" ]]; then
    echo "ERROR: standalone DP=${DATA_PARALLEL_SIZE} must match NUM_NODES=${NUM_NODES}" >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${NUM_INSTANCES}" || -z "${OUTPUT_DIR}" ]]; then
        echo "ERROR: --num-instances and --output-dir are required" >&2
        usage
    fi
    require_positive_int "--num-instances" "${NUM_INSTANCES}"

    OUTPUT_DIR=$(readlink -m "${OUTPUT_DIR}")
    mkdir -p "${OUTPUT_DIR}/slurm_logs" "${OUTPUT_DIR}/server_info"

    for ((i = 0; i < NUM_INSTANCES; i++)); do
        instance_model_name="${MODEL_NAME}-${i}"
        sbatch_args=(
            --account "${ACCOUNT}"
            --partition "${PARTITION}"
            --time "${TIME_LIMIT}"
            --nodes "${NUM_NODES}"
            --ntasks "${NUM_NODES}"
            --ntasks-per-node 1
            --gres gpu:4
            --switches 1
            --job-name "${instance_model_name}"
            --output "${OUTPUT_DIR}/slurm_logs/%j_%x.out"
            --error "${OUTPUT_DIR}/slurm_logs/%j_%x.err"
        )
        [[ -z "${QOS}" ]] || sbatch_args+=(--qos "${QOS}")
        [[ -z "${RESERVATION}" ]] || sbatch_args+=(--reservation "${RESERVATION}")

        sbatch "${sbatch_args[@]}" "$0" \
            --vllm-image "${VLLM_IMAGE}" \
            --model-path "${MODEL_PATH}" \
            --model-name "${MODEL_NAME}" \
            --num-instances "${NUM_INSTANCES}" \
            --instance-idx "${i}" \
            --output-dir "${OUTPUT_DIR}" \
            --time-limit "${TIME_LIMIT}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --cache-root "${CACHE_ROOT}"
    done
    exit 0
fi

if [[ -z "${NUM_INSTANCES}" || -z "${INSTANCE_IDX}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: missing standalone job arguments inside Slurm" >&2
    exit 1
fi
if ! [[ "${INSTANCE_IDX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --instance-idx must be a non-negative integer" >&2
    exit 1
fi

INSTANCE_MODEL_NAME="${MODEL_NAME}-${INSTANCE_IDX}"
RAY_PORT=$((BASE_RAY_PORT + INSTANCE_IDX))
VLLM_PORT=$((BASE_VLLM_PORT + INSTANCE_IDX))
LOG_FILE="${OUTPUT_DIR}/${INSTANCE_MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}"
touch "${LOG_FILE}"

export OUTPUT_DIR LUSTRE_DIR VLLM_IMAGE MODEL_PATH INSTANCE_MODEL_NAME LOG_FILE
export RAY_PORT VLLM_PORT NUM_NODES INSTANCE_IDX
export TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE DATA_PARALLEL_SIZE DATA_PARALLEL_SIZE_LOCAL
export GPU_MEMORY_UTILIZATION API_SERVER_COUNT MAX_MODEL_LEN
export CACHE_ROOT HF_HOME VLLM_CACHE_ROOT

srun \
    --no-container-mount-home \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${OUTPUT_DIR}:/outputs,${LUSTRE_DIR}:/lustre" \
    --export=ALL \
    --mpi=pmix \
    bash -lc '
        set -Eeuo pipefail
        cd /outputs

        export HF_HOME VLLM_CACHE_ROOT
        export HF_TOKEN="${HF_TOKEN:-}"
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
        export VLLM_ENGINE_READY_TIMEOUT_S=3600
        export RAY_CGRAPH_get_timeout=600
        export RAY_raylet_start_wait_time_s=120
        export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
        export VLLM_ALLREDUCE_USE_SYMM_MEM=0
        export VLLM_USE_DEEP_GEMM=0
        export FLASHINFER_WORKSPACE_BASE=/tmp

        HEAD_IP_FILE="/outputs/.ray_head_ip_${SLURM_JOB_ID}_${INSTANCE_IDX}"
        RAY_FIXED_PORTS="--node-manager-port=8266 --object-manager-port=8267 --metrics-export-port=8269 --dashboard-agent-grpc-port=8270 --dashboard-agent-listen-port=8271 --runtime-env-agent-port=8272"

        if [[ "${SLURM_PROCID}" -eq 0 ]]; then
            cleanup_rank0() {
                status=$?
                trap - EXIT INT TERM
                [[ -z "${TAIL_PID:-}" ]] || kill "${TAIL_PID}" 2>/dev/null || true
                [[ -z "${VLLM_PID:-}" ]] || kill "${VLLM_PID}" 2>/dev/null || true
                ray stop >/dev/null 2>&1 || true
                rm -f "${HEAD_IP_FILE}"
                exit "${status}"
            }
            trap cleanup_rank0 EXIT INT TERM

            rm -f "${HEAD_IP_FILE}"
            HEAD_IP=$(hostname -I | awk "{print \$1}")
            if [[ -z "${HEAD_IP}" ]]; then
                HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
            fi
            [[ -n "${HEAD_IP}" ]] || { echo "ERROR: could not determine head node IP" >&2; exit 1; }
            echo "${HEAD_IP}" > "${HEAD_IP_FILE}"

            echo "Starting Ray head on ${HEAD_IP}:${RAY_PORT}"
            ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" ${RAY_FIXED_PORTS} --disable-usage-stats
            export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

            echo "Waiting for ${NUM_NODES} Ray nodes"
            for _ in $(seq 1 360); do
                active_nodes=$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_" || true)
                [[ "${active_nodes}" -ge "${NUM_NODES}" ]] && break
                sleep 5
            done
            active_nodes=$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_" || true)
            if [[ "${active_nodes}" -lt "${NUM_NODES}" ]]; then
                echo "ERROR: only ${active_nodes}/${NUM_NODES} Ray nodes joined" >&2
                ray status --address "${RAY_ADDRESS}" || true
                exit 1
            fi
            ray status --address "${RAY_ADDRESS}" || true

            SERVE_PORT="${VLLM_PORT}"
            unset VLLM_PORT
            echo "Starting original GLM-5.2 BF16 server: TP=${TENSOR_PARALLEL_SIZE}, DP=${DATA_PARALLEL_SIZE}"
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
                --port "${SERVE_PORT}" \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm45 \
                --enable-prefix-caching \
                --enable-chunked-prefill \
                --enable-expert-parallel \
                --no-enable-flashinfer-autotune \
                --api-server-count "${API_SERVER_COUNT}" \
                --chat-template-content-format string \
                --served-model-name "${INSTANCE_MODEL_NAME}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
                --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" \
                > "${LOG_FILE}" 2>&1 &
            VLLM_PID=$!

            tail -n +1 -F "${LOG_FILE}" &
            TAIL_PID=$!
            while ! grep -Eq "Uvicorn running on|Application startup complete\\." "${LOG_FILE}" 2>/dev/null; do
                if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                    echo "ERROR: vLLM server exited before readiness" >&2
                    exit 1
                fi
                sleep 2
            done

            SERVER_INFO_FILE="/outputs/server_info/${INSTANCE_MODEL_NAME}.env"
            {
                echo "MODEL_NAME=${INSTANCE_MODEL_NAME}"
                echo "INSTANCE_IDX=${INSTANCE_IDX}"
                echo "HEAD_IP=${HEAD_IP}"
                echo "VLLM_PORT=${SERVE_PORT}"
                echo "SERVER_URL=http://${HEAD_IP}:${SERVE_PORT}/v1"
            } > "${SERVER_INFO_FILE}"
            echo "READY: http://${HEAD_IP}:${SERVE_PORT}/v1"

            wait "${VLLM_PID}"
        else
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
        fi
    '
