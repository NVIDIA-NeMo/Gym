#!/bin/bash
# 
# Example run:
# MODEL=/path/to/model \
# NUM_NODES=4 \
# SBATCH_ACCOUNT=my-slurm-account \
# SBATCH_PARTITION=batch \
# CONTAINER=/path/to/vllm/container \
# MOUNTS=/shared/fs:/shared/fs \
# bash benchmarks/nemotron_3_ultra/sbatch_external_vllm.sh
# 
# This script assumes:
# - GB200s which are 4 GPUs per node. If you want to use 8 GPUs per node, update the --tensor-parallel-size and --gres=gpu arguments to 8.
# - Nemotron 3 Ultra configs e.g. with the parser configs.

set -euo pipefail

# Input arguments and validation
NUM_NODES=$NUM_NODES
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS

command=$(cat <<EOF
set -euo pipefail

# TODO @bxyu-nvidia: Move this to the container prep script
pip install vllm-router

host="\$(hostname -I | awk '{print $1}')"
common_args=(
    --served-model-name nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
    --gpu-memory-utilization 0.9
    --distributed-executor-backend ray
    --data-parallel-backend ray
    --data-parallel-size 2
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
    --max-num-batched-tokens 16384
    --host \$host
)

# 1. Prefill
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
vllm serve $MODEL "\${common_args[@]}" \
    --port 8001 \
    --data-parallel-size-local 1 \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
    &
prefill_pid=\$!

# 2. Decode
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
vllm serve $MODEL "\${common_args[@]}" \
    --port 8002 \
    --data-parallel-size-local 0 \
    --kv-transfer-config \
        '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    &
decode_pid=\$!

trap 'kill "\$prefill_pid" "\$decode_pid" 2>/dev/null || true' EXIT

until curl -s http://\$host:8001/health >/dev/null; do
    sleep 5
done
until curl -s http://\$host:8002/health >/dev/null; do
    sleep 5
done

# 3. Router
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
    --job-name=vllm-$USER \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_base.sh bash -lc "$command"
