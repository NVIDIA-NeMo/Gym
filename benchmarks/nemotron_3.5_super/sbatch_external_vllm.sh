#!/bin/bash

set -euo pipefail

# Input arguments and validation
NUM_PREFILL_NODES=$NUM_PREFILL_NODES
NUM_DECODE_NODES=$NUM_DECODE_NODES
MODEL=$MODEL
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS

command=$(cat <<EOF
set -euo pipefail

prefill_hosts=( \${PD_PREFILL_HOSTS:?Missing prefill node addresses} )
decode_hosts=( \${PD_DECODE_HOSTS:?Missing decode node addresses} )

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
    --data-parallel-backend mp
    --tensor-parallel-size 4
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
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
    # P is one DP=2 deployment: this rank is its API-server and rank 0.
    prefill_host=\${prefill_hosts[0]}
    pip install vllm-router
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$prefill_host \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --host \$prefill_host \
        --port 8001 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-address \$prefill_host \
        --data-parallel-rpc-port 13345 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        &
    prefill_pid=\$!
    trap 'kill "\$prefill_pid" 2>/dev/null || true' EXIT

    wait_for_server "\$prefill_pid" "http://\$prefill_host:8001"

    until curl -fsS "http://\${decode_hosts[0]}:8002/health" >/dev/null; do
        sleep 5
    done

    vllm-router \
        --policy consistent_hash \
        --vllm-pd-disaggregation \
        --prefill http://\$prefill_host:8001 \
        --decode http://\${decode_hosts[0]}:8002 \
        --host \$prefill_host \
        --port 8000 \
        --intra-node-data-parallel-size 1
elif [[ "\$SLURM_PROCID" == 1 ]]; then
    # Prefill DP rank 1 has no API server.
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${prefill_hosts[1]} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --headless \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 1 \
        --data-parallel-address \${prefill_hosts[0]} \
        --data-parallel-rpc-port 13345 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
elif [[ "\$SLURM_PROCID" == 2 ]]; then
    # D is one DP=2 deployment: this rank is its API-server and rank 0.
    decode_host=\${decode_hosts[0]}
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$decode_host \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --host \$decode_host \
        --port 8002 \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-address \$decode_host \
        --data-parallel-rpc-port 13346 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
elif [[ "\$SLURM_PROCID" == 3 ]]; then
    # Decode DP rank 1 has no API server.
    VLLM_NIXL_SIDE_CHANNEL_HOST=\${decode_hosts[1]} \
    VLLM_NIXL_SIDE_CHANNEL_PORT=5700 \
    vllm serve "$MODEL" "\${common_args[@]}" \
        --headless \
        --data-parallel-size 2 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 1 \
        --data-parallel-address \${decode_hosts[0]} \
        --data-parallel-rpc-port 13346 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
else
    echo "Unexpected Slurm task rank: \$SLURM_PROCID" >&2
    exit 2
fi
EOF
)

batch_command=$(cat <<'EOF'
set -euo pipefail

if [[ "${SLURM_JOB_NUM_NODES:-}" != 4 ]]; then
    echo "This DP=2 prefill / DP=2 decode configuration requires exactly four nodes." >&2
    exit 2
fi

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
prefill_host_0=$(getent hosts "${nodes[0]}" | awk 'NR == 1 {print $1}')
prefill_host_1=$(getent hosts "${nodes[1]}" | awk 'NR == 1 {print $1}')
decode_host_0=$(getent hosts "${nodes[2]}" | awk 'NR == 1 {print $1}')
decode_host_1=$(getent hosts "${nodes[3]}" | awk 'NR == 1 {print $1}')
if [[ -z "$prefill_host_0" || -z "$prefill_host_1" ||
      -z "$decode_host_0" || -z "$decode_host_1" ]]; then
    echo "Could not resolve allocated prefill/decode node addresses." >&2
    exit 1
fi

# Do not start Ray: each Slurm task directly runs a vLLM DP rank.
PD_PREFILL_HOSTS="$prefill_host_0 $prefill_host_1" \
PD_DECODE_HOSTS="$decode_host_0 $decode_host_1" \
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 \
    --container-image="$CONTAINER" \
    --container-name=container-on-node \
    --container-mounts="$MOUNTS" \
    --container-workdir="$SLURM_SUBMIT_DIR" \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd "$SLURM_SUBMIT_DIR"
        exec "$@"
    ' bash "$@"
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
    --output=slurm-logs/%j-%x.log \
    --ntasks-per-node=1 \
    --exclusive \
    --segment=$NUM_NODES \
    bash -lc "$batch_command" bash bash -lc "$command"
