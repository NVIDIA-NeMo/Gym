#!/bin/bash

set -euo pipefail

# One four-GPU GB200 node is dedicated to prefill and one to decode.  The
# client-facing vllm-router runs with prefill on the first allocated node.
NUM_NODES=${NUM_NODES:?Set NUM_NODES=2}
MODEL=${MODEL:?Set MODEL to the model path or ID}
CONTAINER=${CONTAINER:?Set CONTAINER to the vLLM container image}
MOUNTS=${MOUNTS:?Set MOUNTS to the required container mounts}

if [[ "$NUM_NODES" != 2 ]]; then
    echo "This prefill/decode-disaggregated configuration requires NUM_NODES=2." >&2
    exit 2
fi

command=$(cat <<EOF
set -euo pipefail

prefill_host=\${PD_PREFILL_HOST:?Missing prefill node address}
decode_host=\${PD_DECODE_HOST:?Missing decode node address}

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout
# when KV transfer is enabled.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# NIXL uses UCX for cross-node KV transfer. Explicitly enable UCX's CUDA
# transports and the GB200 InfiniBand interface; otherwise UCX treats VRAM as
# host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND.
export UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_ADDR_TYPE=eth
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0

common_args=(
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --tensor-parallel-size 4
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --api-server-count 1
    --kv-cache-dtype fp8
    -cc.pass_config.fuse_allreduce_rms=False
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --max-num-batched-tokens 8192
)

wait_for_server() {
    local pid=\$1
    local url=\$2

    while ! curl -fsS "\$url/health" >/dev/null; do
        if ! kill -0 "\$pid" 2>/dev/null; then
            echo "vLLM at \$url exited before becoming healthy." >&2
            wait "\$pid"
        fi
        sleep 5
    done
}

if [[ "\$SLURM_PROCID" == 0 ]]; then
    # Prefill is a node-local TP=4 vLLM server on the first Slurm node.
    pip install vllm-router
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$prefill_host \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --host \$prefill_host \
        --port 8001 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        &
    prefill_pid=\$!
    trap 'kill "\$prefill_pid" 2>/dev/null || true' EXIT

    wait_for_server "\$prefill_pid" "http://\$prefill_host:8001"

    vllm-router \
        --policy round_robin \
        --vllm-pd-disaggregation \
        --prefill http://\$prefill_host:8001 \
        --decode http://\$decode_host:8002 \
        --host \$prefill_host \
        --port 8000 \
        --intra-node-data-parallel-size 1
elif [[ "\$SLURM_PROCID" == 1 ]]; then
    # Decode is a separate node-local TP=4 vLLM server on the second node.
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$decode_host \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --host \$decode_host \
        --port 8002 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
else
    echo "Unexpected Slurm task rank: \$SLURM_PROCID" >&2
    exit 2
fi
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
CONTAINER=$CONTAINER \
MOUNTS=$MOUNTS \
sbatch \
    --nodes=$NUM_NODES \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=vllm-pd-disagg-$USER \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_pd_disagg_base.sh bash -lc "$command"
