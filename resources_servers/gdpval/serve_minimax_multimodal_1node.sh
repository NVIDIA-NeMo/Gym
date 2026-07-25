#!/bin/bash
# Serve MiniMax-M3 as a MULTIMODAL (vision-enabled) OpenAI endpoint from the
# vendor vLLM container on ONE 4-GPU GB200 node at TP=4 — a single node-local
# replica, one HTTP endpoint. No Ray, no cross-node reductions.
#
# WHY single-node: TP=8 spanning two GB200 nodes loaded fine but produced GARBAGE
# logits (the cross-node MXFP8 MoE / fused allreduce path on this container is
# numerically unreliable — 'trtllm' allreduce crashes multi-node, 'mnnvl' corrupts,
# PyNCCL fallback still garbage). Keeping the whole replica on one node eliminates
# every cross-node reduction, so we inherit the numerics the coworker's single-node
# recipe proved. MiniMax-M3 MXFP8 is ~52 GiB/rank at TP=8, so ~104 GiB/GPU at TP=4,
# which fits a GB200's ~186 GB.
#
# Same vendor-container plumbing as serve_minimax_multimodal_2node.sh, minus Ray:
#   - REMOVED --language-model-only        -> loads the vision encoder (see images)
#   - ADDED   --mm-encoder-attn-backend TORCH_SDPA  (SM100 encoder-attn fallback)
#
# Run EITHER way (only 1 node / 4 GPUs is used even if you allocated more):
#   A) batch:        sbatch serve_minimax_multimodal_1node.sh
#   B) interactive:  salloc -A <acct> -p <part> -N 1 --ntasks-per-node=1 \
#                           --gres=gpu:4 -t 04:00:00
#                    bash serve_minimax_multimodal_1node.sh
#
# The endpoint's /v1 URL is echoed AND written to
#   ${OUTPUT_DIR}/server_info/${MODEL_NAME}.env  (SERVER_URL=...).
# Point gym at it:
#   export MINIMAX_BASE_URL=$(. ${OUTPUT_DIR}/server_info/minimax-m3.env; echo "$SERVER_URL")

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 04:00:00
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

# --- Knobs (override via env) ------------------------------------------------
# Local MXFP8 M3 checkpoint. /lustre is bind-mounted into the container below, so
# any /lustre path — including the snapshot's ../../blobs symlinks — resolves.
MODEL_PATH="${MINIMAX_MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_compeval/checkpoints/hf_cache/MiniMax-M3-MXFP8}"
# Vendor vLLM image with M3 support (+ Ray baked in), pre-staged as a .sqsh.
VLLM_IMAGE="${MINIMAX_IMAGE:-/lustre/fsw/portfolios/llmservice/users/vadams/containers/vllm-openai_minimax-m3-ray2.56.0.sqsh}"
MODEL_NAME="${MODEL_NAME:-minimax-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/minimax_judge}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre}"
TP="${TP:-4}"                          # GPUs on this node == tensor-parallel size
# API server port. NOT named VLLM_PORT — vLLM treats that reserved env var as the
# base for internal distributed ports and can pin/collide them.
API_PORT="${API_PORT:-5000}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a Slurm allocation. Use 'sbatch $0' or run under 'salloc -N 1 ...'." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/server_info"
LOG_FILE="${OUTPUT_DIR}/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}"

# Single-node: no cross-node allreduce backend is set (custom/PyNCCL intra-node is
# correct and fast). VLLM_ALLREDUCE_USE_SYMM_MEM=0 kept from the reference env.
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export OUTPUT_DIR LUSTRE_DIR VLLM_IMAGE MODEL_PATH MODEL_NAME LOG_FILE TP API_PORT

echo "Serving '${MODEL_NAME}' (multimodal) at TP=${TP} on 1 node"
echo "Model:  ${MODEL_PATH}"
echo "Image:  ${VLLM_IMAGE}"

srun \
    --nodes=1 --ntasks=1 --gpus-per-node="${TP}" \
    --no-container-mount-home \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${OUTPUT_DIR}:/outputs,${LUSTRE_DIR}:/lustre" \
    --export=ALL \
    --mpi=pmix \
    bash -c '
    set -euo pipefail
    cd /outputs

    export FLASHINFER_WORKSPACE_BASE=/tmp

    HEAD_IP=$(hostname -I | awk "{print \$1}")
    if [ -z "$HEAD_IP" ]; then
        HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
    fi
    if [ -z "$HEAD_IP" ]; then
        echo "ERROR: could not determine node IP"
        exit 1
    fi

    echo "=== Starting vLLM server (MiniMax-M3, multimodal, TP=${TP}, single node) ==="

    # Coworker-proven MiniMax-M3 serve args, adapted for multimodal + single-node TP.
    vllm serve "${MODEL_PATH}" \
        --served-model-name "${MODEL_NAME}" \
        --trust-remote-code \
        --block-size 128 \
        --tensor-parallel-size "${TP}" \
        --enable-expert-parallel \
        --enable-auto-tool-choice \
        --tool-call-parser minimax_m3 \
        --reasoning-parser minimax_m3 \
        --mm-encoder-attn-backend TORCH_SDPA \
        --enable-prefix-caching \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 128 \
        --max-model-len 1048576 \
        --host 0.0.0.0 \
        --port "${API_PORT}" \
        --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
        --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" > "${LOG_FILE}" 2>&1 &

    VLLM_PID=$!

    echo "=== Waiting for server readiness (tailing ${LOG_FILE}) ==="
    tail -f "${LOG_FILE}" &
    TAIL_PID=$!

    while ! grep -q "Application startup complete." "${LOG_FILE}" 2>/dev/null; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server process died"
            kill "$TAIL_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 2
    done

    SERVER_URL="http://${HEAD_IP}:${API_PORT}/v1"
    echo ""
    echo "=== Server is ready on ${SERVER_URL} ==="
    echo ""

    RUNTIME_SERVER_INFO_FILE="/outputs/server_info/${MODEL_NAME}.env"
    {
        echo "MODEL_NAME=${MODEL_NAME}"
        echo "HEAD_IP=${HEAD_IP}"
        echo "API_PORT=${API_PORT}"
        echo "SERVER_URL=${SERVER_URL}"
    } > "${RUNTIME_SERVER_INFO_FILE}"
    sync || true
    echo "Wrote server info to ${RUNTIME_SERVER_INFO_FILE}"
    echo "Wire gym in with:  export MINIMAX_BASE_URL=${SERVER_URL}"

    wait "$VLLM_PID"
    kill "$TAIL_PID" 2>/dev/null || true
    '
