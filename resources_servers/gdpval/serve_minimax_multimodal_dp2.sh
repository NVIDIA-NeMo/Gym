#!/bin/bash
# Serve MiniMax-M3 as a MULTIMODAL (vision-enabled) OpenAI endpoint across TWO
# 4-GPU GB200 nodes as TP=4 x DP=2 — TWO independent, node-local replicas behind
# ONE HTTP endpoint on node 0 (vLLM native multi-node data-parallel, internal
# load balancing). No Ray.
#
# WHY this shape (and NOT TP=8 across nodes): TP=8 spanning two nodes produced
# GARBAGE logits — the container's cross-node MXFP8 MoE / allreduce path is
# numerically broken. Here each replica is TP=4 on a SINGLE node, so it reproduces
# the (validated) single-node numerics exactly; the only cross-node traffic is the
# DP request router (control plane), never tensor math.
#
# CRITICAL: --enable-expert-parallel is DELIBERATELY OMITTED. With DP, enabling EP
# makes vLLM build ONE "wide-EP" group across all DP*TP=8 ranks -> cross-node
# expert all-to-all -> the same garbage. Without it, each replica shards its experts
# by TP=4 intra-node (same per-GPU memory as the single-node run), fully isolated.
#
# Layout:
#   node 0 (rank 0): API server + DP rank 0 (TP=4)   -> serves :$API_PORT
#   node 1 (rank 1): --headless   DP rank 1 (TP=4)   -> joins DP coordinator
# One endpoint on node 0 load-balances requests across both replicas.
#
# Run EITHER way (2 nodes, 4 GPUs each):
#   A) batch:        sbatch serve_minimax_multimodal_dp2.sh
#   B) interactive:  salloc -A <acct> -p <part> -N 2 --ntasks-per-node=1 \
#                           --gres=gpu:4 -t 04:00:00
#                    bash serve_minimax_multimodal_dp2.sh
#
# Endpoint /v1 URL is echoed by node 0 AND written to
#   ${OUTPUT_DIR}/server_info/${MODEL_NAME}.env  (SERVER_URL=...).
# Point gym at it:
#   export MINIMAX_BASE_URL=$(. ${OUTPUT_DIR}/server_info/minimax-m3.env; echo "$SERVER_URL")

#SBATCH -A nemotron_n4_post
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -t 04:00:00
#SBATCH --switches=1
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

set -euo pipefail

# --- Knobs (override via env) ------------------------------------------------
# Local MXFP8 M3 checkpoint. /lustre is bind-mounted into the container below, so
# any /lustre path — including the snapshot's ../../blobs symlinks — resolves.
MODEL_PATH="${MINIMAX_MODEL_PATH:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_compeval/checkpoints/hf_cache/MiniMax-M3-MXFP8}"
# Vendor vLLM image with M3 support (+ Ray baked in, though DP doesn't need it).
VLLM_IMAGE="${MINIMAX_IMAGE:-/lustre/fsw/portfolios/llmservice/users/vadams/containers/vllm-openai_minimax-m3-ray2.56.0.sqsh}"
MODEL_NAME="${MODEL_NAME:-minimax-m3}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fsw/portfolios/llmservice/users/${USER}/minimax_judge}"
LUSTRE_DIR="${LUSTRE_DIR:-/lustre}"
TP="${TP:-4}"                          # GPUs per node == tensor-parallel size per replica
DP="${DP:-2}"                          # data-parallel replicas == number of nodes
DP_RPC_PORT="${DP_RPC_PORT:-13345}"    # cross-node DP coordination RPC port
# API server port. NOT named VLLM_PORT — vLLM treats that reserved env var as the
# base for internal distributed ports and can pin/collide them.
API_PORT="${API_PORT:-5000}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: not inside a Slurm allocation. Use 'sbatch $0' or run under 'salloc -N 2 ...'." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/server_info"

# Single-replica-per-node: no cross-node allreduce backend forced (intra-node
# custom/PyNCCL is correct). Kept from the reference env.
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export OUTPUT_DIR LUSTRE_DIR VLLM_IMAGE MODEL_PATH MODEL_NAME TP DP DP_RPC_PORT API_PORT

echo "Serving '${MODEL_NAME}' (multimodal) at TP=${TP} x DP=${DP} across ${DP} nodes (node-local replicas)"
echo "Model:  ${MODEL_PATH}"
echo "Image:  ${VLLM_IMAGE}"

srun \
    --nodes="${DP}" --ntasks-per-node=1 --gpus-per-node="${TP}" \
    --no-container-mount-home \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${OUTPUT_DIR}:/outputs,${LUSTRE_DIR}:/lustre" \
    --export=ALL \
    --mpi=pmix \
    bash -c '
    set -euo pipefail
    cd /outputs

    export FLASHINFER_WORKSPACE_BASE=/tmp

    RANK="${SLURM_PROCID:-0}"
    HEAD_IP_FILE="/outputs/.dp_head_ip_${SLURM_JOB_ID}"
    LOG_FILE="/outputs/${MODEL_NAME}_${SLURM_JOB_ID}_rank${RANK}.log"

    # Shared serve args. NOTE: NO --enable-expert-parallel (keeps experts node-local
    # per replica; enabling it would span EP across nodes and corrupt output).
    COMMON=(
      vllm serve "${MODEL_PATH}"
      --served-model-name "${MODEL_NAME}"
      --trust-remote-code
      --block-size 128
      --tensor-parallel-size "${TP}"
      --data-parallel-size "${DP}"
      --data-parallel-size-local 1
      --data-parallel-rpc-port "${DP_RPC_PORT}"
      --enable-auto-tool-choice
      --tool-call-parser minimax_m3
      --reasoning-parser minimax_m3
      --mm-encoder-attn-backend TORCH_SDPA
      --enable-prefix-caching
      --gpu-memory-utilization 0.90
      --max-num-seqs 128
      --max-model-len 1048576
      --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}"
      --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}"
    )

    if [ "$RANK" -eq 0 ]; then
        rm -f "$HEAD_IP_FILE"
        HEAD_IP=$(hostname -I | awk "{print \$1}")
        [ -z "$HEAD_IP" ] && HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
        [ -z "$HEAD_IP" ] && { echo "ERROR: could not determine head node IP"; exit 1; }
        echo "$HEAD_IP" > "$HEAD_IP_FILE"

        echo "=== [rank0 $(hostname)] API server + DP rank0 (TP=${TP}) -> http://${HEAD_IP}:${API_PORT}/v1 ==="
        "${COMMON[@]}" \
            --data-parallel-address "$HEAD_IP" \
            --host 0.0.0.0 --port "$API_PORT" \
            > "$LOG_FILE" 2>&1 &
        VLLM_PID=$!

        echo "=== [rank0] Waiting for server readiness (tailing ${LOG_FILE}) ==="
        tail -f "$LOG_FILE" &
        TAIL_PID=$!
        while ! grep -q "Application startup complete." "$LOG_FILE" 2>/dev/null; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "ERROR: vLLM server process died"
                kill "$TAIL_PID" 2>/dev/null || true
                rm -f "$HEAD_IP_FILE"
                exit 1
            fi
            sleep 2
        done

        SERVER_URL="http://${HEAD_IP}:${API_PORT}/v1"
        echo ""; echo "=== Server is ready on ${SERVER_URL} ==="; echo ""
        INFO="/outputs/server_info/${MODEL_NAME}.env"
        {
            echo "MODEL_NAME=${MODEL_NAME}"
            echo "HEAD_IP=${HEAD_IP}"
            echo "API_PORT=${API_PORT}"
            echo "SERVER_URL=${SERVER_URL}"
        } > "$INFO"
        sync || true
        echo "Wrote server info to ${INFO}"
        echo "Wire gym in with:  export MINIMAX_BASE_URL=${SERVER_URL}"

        wait "$VLLM_PID"
        kill "$TAIL_PID" 2>/dev/null || true
        rm -f "$HEAD_IP_FILE"
    else
        for _ in $(seq 1 180); do [ -s "$HEAD_IP_FILE" ] && break; sleep 1; done
        [ -s "$HEAD_IP_FILE" ] || { echo "ERROR: timed out waiting for ${HEAD_IP_FILE}"; exit 1; }
        HEAD_IP=$(cat "$HEAD_IP_FILE")

        echo "=== [rank${RANK} $(hostname)] HEADLESS DP rank${RANK} (TP=${TP}) -> coordinator ${HEAD_IP}:${DP_RPC_PORT} ==="
        exec "${COMMON[@]}" \
            --headless \
            --data-parallel-start-rank "$RANK" \
            --data-parallel-address "$HEAD_IP" \
            > "$LOG_FILE" 2>&1
    fi
    '
