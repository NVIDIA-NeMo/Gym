#!/bin/bash

set -euo pipefail

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

# TODO: bake vllm-router into the evaluation container.
pip install vllm-router

host=\$(hostname -I | awk '{print \$1}')
common_args=(
    --gpu-memory-utilization 0.9
    --distributed-executor-backend ray
    --data-parallel-backend ray
    --data-parallel-size 1
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
    --host \$host
)

# The prefill API and engine are packed on the Ray head.
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
VLLM_RAY_DP_PACK_STRATEGY=fill \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve "$MODEL" "\${common_args[@]}" \
    --port 8001 \
    --data-parallel-size-local 1 \
    --data-parallel-address \$host \
    --master-addr \$host \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
    &
prefill_pid=\$!

wait_for_server() {
    local pid=\$1
    local port=\$2

    while ! curl -fsS "http://\$host:\$port/health" >/dev/null; do
        if ! kill -0 "\$pid" 2>/dev/null; then
            echo "vLLM on port \$port exited before becoming healthy." >&2
            wait "\$pid"
        fi
        sleep 5
    done
}

wait_for_server "\$prefill_pid" 8001

# Decode has no local engine. With fill packing, Ray skips the fully occupied
# head node and places this TP=4 engine on the other allocated node.
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
VLLM_RAY_DP_PACK_STRATEGY=fill \
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
vllm serve "$MODEL" "\${common_args[@]}" \
    --port 8002 \
    --data-parallel-size-local 0 \
    --data-parallel-address \$host \
    --master-addr \$host \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    &
decode_pid=\$!

trap 'kill "\$prefill_pid" "\$decode_pid" 2>/dev/null || true' EXIT

wait_for_server "\$decode_pid" 8002

vllm-router \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --prefill http://\$host:8001 \
    --decode http://\$host:8002 \
    --host \$host \
    --port 8000 \
    --intra-node-data-parallel-size 1
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
    scripts/sbatch_base.sh bash -lc "$command"
