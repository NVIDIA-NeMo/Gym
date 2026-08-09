#!/bin/bash

set -euo pipefail

NUM_NODES=${NUM_NODES:?Set NUM_NODES=4}
MODEL=${MODEL:?Set MODEL to the model path or ID}
CONTAINER=${CONTAINER:?Set CONTAINER to the vLLM container image}
MOUNTS=${MOUNTS:?Set MOUNTS to the required container mounts}

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

# Prefill pool: NIXL KV producer on the first two data-parallel replicas.
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve "$MODEL" "\${common_args[@]}" \
    --port 8001 \
    --data-parallel-size-local 1 \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
    &
prefill_pid=\$!

# Decode pool: NIXL KV consumer on the remaining two data-parallel replicas.
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
vllm serve "$MODEL" "\${common_args[@]}" \
    --port 8002 \
    --data-parallel-size-local 0 \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    &
decode_pid=\$!

trap 'kill "\$prefill_pid" "\$decode_pid" 2>/dev/null || true' EXIT

until curl -fsS http://\$host:8001/health >/dev/null; do
    sleep 5
done
until curl -fsS http://\$host:8002/health >/dev/null; do
    sleep 5
done

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
