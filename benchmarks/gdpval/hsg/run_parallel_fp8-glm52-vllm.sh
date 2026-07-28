#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --switches=1
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

OUTPUT_DIR=""
LUSTRE_DIR=/lustre

# Keep the upstream flag names for compatibility with the existing launcher stack.
VLLM_IMAGE="${VLLM_IMAGE:-}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${MODEL_NAME:-GLM-52-fp8-parallel}"
NUM_INSTANCES=""
INSTANCE_IDX=""

# FP8 GLM 5.2 uses the serving shape measured fastest on this hardware:
# a single TP=8 replica spanning 2 nodes. FP8
# halves weight-memory traffic vs bf16, so decode is materially faster and
# the freed HBM goes to KV cache.
NUM_NODES=2
# Slurm settings match the bf16 launcher.
# Set GDPVAL_SLURM_ACCOUNT (or --account) to your Slurm account.
# export QOS="${QOS-nemotron-priority}"
# export RESERVATION="${RESERVATION-sla_res_nemotron}"
ACCOUNT="${GDPVAL_SLURM_ACCOUNT:-}"
PARTITION=batch
export QOS="${QOS-normal}"
export RESERVATION="${RESERVATION-}"
TIME_LIMIT=04:00:00

BASE_RAY_PORT=6379
BASE_VLLM_PORT=10240

TENSOR_PARALLEL_SIZE=8
PIPELINE_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE_LOCAL=1
GPU_MEMORY_UTILIZATION=0.90
API_SERVER_COUNT=1
# Bumped from 202752 -> 262144 so the served context covers the rollout
# token budget (MAX_TOKENS=262144 in test_parallel_rollout_run.sh); the
# old 202752 was below that budget and could truncate/reject long requests.
# Not pushed to GLM-5.2's native 1M window: KV cache here is bf16 (no
# --kv-cache-dtype fp8), so 1M would collapse per-replica concurrency.
MAX_MODEL_LEN=262144

CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-$HOME}/gdpval-cache}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"

usage() {
    echo "Usage: $0 --num-instances <n> --output-dir <path> [OPTIONS]"
    echo ""
    echo "Required outside Slurm:"
    echo "  --num-instances <n>        Number of endpoint instances to launch"
    echo "  --output-dir <path>        Output directory for logs and server_info"
    echo ""
    echo "Optional overrides:"
    echo "  --vllm-image <path>        vLLM Ray container image (default: ${VLLM_IMAGE})"
    echo "  --model-path <path>        Model checkpoint path (default: ${MODEL_PATH})"
    echo "  --model-name <name>        Base served model name (default: ${MODEL_NAME})"
    echo "  --num-nodes <n>            Nodes per GLM 5.2 bf16 server job (default: ${NUM_NODES})"
    echo "  --account <name>           Slurm account for self-submitted jobs (default: ${ACCOUNT})"
    echo "  --partition <name>         Slurm partition for self-submitted jobs (default: ${PARTITION})"
    echo "  --qos <name>               Slurm QoS for self-submitted jobs (default: ${QOS:-<none>})"
    echo "  --reservation <name>       Slurm reservation for self-submitted jobs (default: ${RESERVATION:-<none>})"
    echo "  --time-limit <time>        Slurm walltime for self-submitted jobs (default: ${TIME_LIMIT})"
    echo "  --tp-size <n>              Tensor parallel size (default: ${TENSOR_PARALLEL_SIZE})"
    echo "  --pp-size <n>              Pipeline parallel size (default: ${PIPELINE_PARALLEL_SIZE})"
    echo "  --dp-size <n>              Data parallel size (default: ${DATA_PARALLEL_SIZE})"
    echo "  --dp-size-local <n>        Local data parallel size per node (default: ${DATA_PARALLEL_SIZE_LOCAL})"
    echo "  --gpu-memory-utilization <fraction>"
    echo "                             GPU memory utilization (default: ${GPU_MEMORY_UTILIZATION})"
    echo "  --api-server-count <n>     vLLM API server count (default: ${API_SERVER_COUNT})"
    echo "  --max-model-len <n>        Served context length, prompt + output (default: ${MAX_MODEL_LEN})"
    echo "  --cache-root <path>        Host-visible cache root (default: ${CACHE_ROOT})"
    echo ""
    echo "Notes:"
    echo "  - The launcher writes one server_info/<model>-<idx>.env file per endpoint."
    exit 1
}

require_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        echo "ERROR: ${name} must be a positive integer; got '${value}'" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-image)
            VLLM_IMAGE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-instances)
            NUM_INSTANCES="$2"
            shift 2
            ;;
        --instance-idx)
            INSTANCE_IDX="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --qos)
            QOS="$2"
            export QOS
            shift 2
            ;;
        --reservation)
            RESERVATION="$2"
            export RESERVATION
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --tp-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --pp-size)
            PIPELINE_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --dp-size)
            DATA_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --dp-size-local)
            DATA_PARALLEL_SIZE_LOCAL="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --api-server-count)
            API_SERVER_COUNT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --cache-root)
            CACHE_ROOT="$2"
            HF_HOME="${CACHE_ROOT}/huggingface"
            VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

require_positive_int "--num-nodes" "${NUM_NODES}"
require_positive_int "--tp-size" "${TENSOR_PARALLEL_SIZE}"
require_positive_int "--pp-size" "${PIPELINE_PARALLEL_SIZE}"
require_positive_int "--dp-size" "${DATA_PARALLEL_SIZE}"
require_positive_int "--dp-size-local" "${DATA_PARALLEL_SIZE_LOCAL}"
require_positive_int "--api-server-count" "${API_SERVER_COUNT}"
require_positive_int "--max-model-len" "${MAX_MODEL_LEN}"

if [[ $((DATA_PARALLEL_SIZE % DATA_PARALLEL_SIZE_LOCAL)) -ne 0 ]]; then
    echo "ERROR: --dp-size must be divisible by --dp-size-local" >&2
    exit 1
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${VLLM_IMAGE}" || -z "${MODEL_PATH}" || -z "${MODEL_NAME}" || -z "${NUM_INSTANCES}" || -z "${OUTPUT_DIR}" ]]; then
        echo "ERROR: missing required arguments"
        usage
    fi

    require_positive_int "--num-instances" "${NUM_INSTANCES}"

    mkdir -p "${OUTPUT_DIR}/slurm_logs"
    mkdir -p "${OUTPUT_DIR}/server_info"

    for ((i=0; i<NUM_INSTANCES; i++)); do
        INSTANCE_MODEL_NAME="${MODEL_NAME}-${i}"
        SERVER_STDOUT_PATH="${OUTPUT_DIR}/slurm_logs/%j_%x.out"
        SERVER_STDERR_PATH="${OUTPUT_DIR}/slurm_logs/%j_%x.err"
        SBATCH_ARGS=(
            -A "${ACCOUNT}"
            -p "${PARTITION}"
            --time "${TIME_LIMIT}"
            --nodes "${NUM_NODES}"
            --ntasks "${NUM_NODES}"
            --ntasks-per-node 1
            --gres gpu:4
            --switches 1
            -J "${INSTANCE_MODEL_NAME}"
            --output "${SERVER_STDOUT_PATH}"
            --error "${SERVER_STDERR_PATH}"
        )

        if [[ -n "${QOS}" ]]; then
            SBATCH_ARGS+=(--qos "${QOS}")
        fi

        if [[ -n "${RESERVATION}" ]]; then
            SBATCH_ARGS+=(--reservation "${RESERVATION}")
        fi

        sbatch \
            "${SBATCH_ARGS[@]}" \
            "$0" \
            --vllm-image "${VLLM_IMAGE}" \
            --model-path "${MODEL_PATH}" \
            --model-name "${MODEL_NAME}" \
            --num-instances "${NUM_INSTANCES}" \
            --instance-idx "${i}" \
            --output-dir "${OUTPUT_DIR}" \
            --num-nodes "${NUM_NODES}" \
            --time-limit "${TIME_LIMIT}" \
            --tp-size "${TENSOR_PARALLEL_SIZE}" \
            --pp-size "${PIPELINE_PARALLEL_SIZE}" \
            --dp-size "${DATA_PARALLEL_SIZE}" \
            --dp-size-local "${DATA_PARALLEL_SIZE_LOCAL}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --api-server-count "${API_SERVER_COUNT}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --cache-root "${CACHE_ROOT}"
    done
    exit 0
fi

if [[ -z "${VLLM_IMAGE}" || -z "${MODEL_PATH}" || -z "${MODEL_NAME}" || -z "${NUM_INSTANCES}" || -z "${INSTANCE_IDX}" || -z "${OUTPUT_DIR}" ]]; then
    echo "ERROR: missing required arguments inside Slurm job"
    exit 1
fi

if ! [[ "${INSTANCE_IDX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --instance-idx must be a non-negative integer"
    exit 1
fi

INSTANCE_MODEL_NAME="${MODEL_NAME}-${INSTANCE_IDX}"
RAY_PORT=$((BASE_RAY_PORT + INSTANCE_IDX))
VLLM_PORT=$((BASE_VLLM_PORT + INSTANCE_IDX))
LOG_FILE="${OUTPUT_DIR}/${INSTANCE_MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${VLLM_CACHE_ROOT}"

export WORK_DIR="${OUTPUT_DIR}"
export OUTPUT_DIR LUSTRE_DIR VLLM_IMAGE MODEL_PATH INSTANCE_MODEL_NAME LOG_FILE
export RAY_PORT VLLM_PORT NUM_NODES INSTANCE_IDX
export TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE DATA_PARALLEL_SIZE DATA_PARALLEL_SIZE_LOCAL
export GPU_MEMORY_UTILIZATION API_SERVER_COUNT
export MAX_MODEL_LEN
export CACHE_ROOT HF_HOME VLLM_CACHE_ROOT

srun \
    --kill-on-bad-exit=1 \
    --no-container-mount-home \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${OUTPUT_DIR}:/outputs,${LUSTRE_DIR}:/lustre" \
    --export=ALL \
    --mpi=pmix \
    bash -lc '
        set -euo pipefail
        cd /outputs

        export HF_HOME="${HF_HOME}"
        export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}"

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

        if [ "${SLURM_PROCID}" -eq 0 ]; then
            rm -f "${HEAD_IP_FILE}"

            HEAD_IP=$(hostname -I | awk "{print \$1}")
            if [ -z "${HEAD_IP}" ]; then
                HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
            fi

            if [ -z "${HEAD_IP}" ]; then
                echo "ERROR: could not determine head node IP"
                exit 1
            fi

            echo "${HEAD_IP}" > "${HEAD_IP_FILE}"
            echo "=== [rank0] Starting Ray head on ${HEAD_IP}:${RAY_PORT} ==="
            ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" ${RAY_FIXED_PORTS} --disable-usage-stats

            export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
            echo "=== [rank0] Waiting for ${NUM_NODES} Ray node(s) ==="
            for _ in $(seq 1 360); do
                if [ "$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_")" -ge "${NUM_NODES}" ]; then
                    break
                fi
                sleep 5
            done

            if [ "$(ray status --address "${RAY_ADDRESS}" 2>/dev/null | grep -c "node_")" -lt "${NUM_NODES}" ]; then
                echo "ERROR: timed out waiting for ${NUM_NODES} Ray nodes"
                ray status --address "${RAY_ADDRESS}" || true
                ray stop || true
                rm -f "${HEAD_IP_FILE}"
                exit 1
            fi
            ray status --address "${RAY_ADDRESS}" || true

            echo "=== [rank0] Starting GLM 5.2 fp8 vLLM server (TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DATA_PARALLEL_SIZE}, DPL=${DATA_PARALLEL_SIZE_LOCAL}) ==="
            SERVE_PORT="${VLLM_PORT}"
            unset VLLM_PORT

            vllm serve "${MODEL_PATH}" \
                --trust-remote-code \
                --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
                --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
                --distributed-executor-backend ray \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
                --port "${SERVE_PORT}" \
                --enable-auto-tool-choice \
                --tool-call-parser glm47 \
                --reasoning-parser glm45 \
                --enable-prefix-caching \
                --enable-chunked-prefill \
                --enable-expert-parallel \
                --api-server-count "${API_SERVER_COUNT}" \
                --chat-template-content-format string \
                --served-model-name "${INSTANCE_MODEL_NAME}" \
                --max-model-len "${MAX_MODEL_LEN}" \
                --kv-cache-dtype fp8 \
                --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
                --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" > "${LOG_FILE}" 2>&1 &

            VLLM_PID=$!

            echo "=== [rank0] Waiting for server readiness (tailing ${LOG_FILE}) ==="
            tail -f "${LOG_FILE}" &
            TAIL_PID=$!

            while ! grep -Eq "Uvicorn running on|Application startup complete\\." "${LOG_FILE}" 2>/dev/null; do
                if grep -Eq "Failed to run autotuning code block|Engine core initialization failed|EngineCore failed to start" "${LOG_FILE}" 2>/dev/null; then
                    echo "ERROR: vLLM reported a fatal startup failure; inspect ${LOG_FILE}"
                    kill "${VLLM_PID}" 2>/dev/null || true
                    wait "${VLLM_PID}" 2>/dev/null || true
                    kill "${TAIL_PID}" 2>/dev/null || true
                    ray stop || true
                    rm -f "${HEAD_IP_FILE}"
                    exit 1
                fi
                if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
                    echo "ERROR: vLLM server process died"
                    kill "${TAIL_PID}" 2>/dev/null || true
                    ray stop || true
                    rm -f "${HEAD_IP_FILE}"
                    exit 1
                fi
                sleep 2
            done

            echo ""
            echo "=== Server is ready on http://${HEAD_IP}:${SERVE_PORT}/v1 ==="
            echo ""

            RUNTIME_SERVER_INFO_DIR="/outputs/server_info"
            RUNTIME_SERVER_INFO_FILE="${RUNTIME_SERVER_INFO_DIR}/${INSTANCE_MODEL_NAME}.env"
            mkdir -p "${RUNTIME_SERVER_INFO_DIR}"

            SERVER_URL="http://${HEAD_IP}:${SERVE_PORT}/v1"
            {
                echo "MODEL_NAME=${INSTANCE_MODEL_NAME}"
                echo "INSTANCE_IDX=${INSTANCE_IDX}"
                echo "HEAD_IP=${HEAD_IP}"
                echo "VLLM_PORT=${SERVE_PORT}"
                echo "SERVER_URL=${SERVER_URL}"
            } > "${RUNTIME_SERVER_INFO_FILE}"

            sync || true
            echo "Wrote server info to ${RUNTIME_SERVER_INFO_FILE}"

            wait "${VLLM_PID}"
            kill "${TAIL_PID}" 2>/dev/null || true
            ray stop || true
            rm -f "${HEAD_IP_FILE}"
        else
            for _ in $(seq 1 180); do
                [ -s "${HEAD_IP_FILE}" ] && break
                sleep 1
            done

            if [ ! -s "${HEAD_IP_FILE}" ]; then
                echo "ERROR: timed out waiting for Ray head IP file: ${HEAD_IP_FILE}"
                exit 1
            fi

            HEAD_IP=$(cat "${HEAD_IP_FILE}")
            export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

            echo "=== [rank${SLURM_PROCID}] Starting Ray worker for ${RAY_ADDRESS} ==="
            until ray start --address="${RAY_ADDRESS}" ${RAY_FIXED_PORTS} --disable-usage-stats; do
                echo "Retrying Ray worker connection to ${RAY_ADDRESS}"
                sleep 5
            done

            tail -f /dev/null
        fi
    '
