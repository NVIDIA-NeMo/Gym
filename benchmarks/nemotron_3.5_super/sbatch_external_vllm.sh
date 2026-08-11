#!/bin/bash

set -euo pipefail

# Input arguments and validation
NUM_PREFILL_NODES=$NUM_PREFILL_NODES
NUM_DECODE_NODES=$NUM_DECODE_NODES
MODEL=$MODEL
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_CONFIG=$VLLM_CONFIG

should_run_eval=$(( $# > 0 ))
if (( should_run_eval )); then
    EXPERIMENT_NAME=$EXPERIMENT_NAME

    # 
    EXPORT_TO_CSV=${EXPORT_TO_CSV:-0}
    EXPORT_CSV_TO_MODEL_DIR=${EXPORT_CSV_TO_MODEL_DIR:-0}
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-vllm_only}"

    EXPORT_TO_CSV=0
    EXPORT_CSV_TO_MODEL_DIR=0
fi

# Fixed vLLM Port configurations
PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT=5600
DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT=5700

ROUTER_SERVER_PORT=8000
PREFILL_SERVER_PORT=8001
DECODE_SERVER_PORT=8002

PREFILL_DP_RPC_PORT=13345
DECODE_DP_RPC_PORT=13346

EVAL_COMMAND=$(cat <<EOF
# TODO @bxyu-nvidia: Remove this once it is baked in the Gym deps
source /opt/uv_venvs/responses_api_agents/opencode_sandboxed_agent/.venv/bin/activate
uv pip install httpx-aiohttp
deactivate
source /opt/uv_venvs/resources_servers/swebench/.venv/bin/activate
uv pip install httpx-aiohttp
deactivate

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

gym eval prepare $@ +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/\$SLURM_JOB_ID/\$(date +%Y%m%d_%H%M%S)
# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
gym eval run \
    $@ \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +nemo_gym_log_dir=results/\$experiment_name/logs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=results/\$experiment_name.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++use_absolute_ip=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=http://\$(getent hosts "\$PREFILL_HEAD" | awk 'NR == 1 {print \$1}'):$ROUTER_SERVER_PORT/v1 \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL \
    ++upload_rollouts_to_wandb=false \
    ++global_aiohttp_connector_limit_per_host=16384 \
    ++port_range_low=63000 \
    ++port_range_high=64000


if (( $EXPORT_TO_CSV )); then
    python benchmarks/nemotron_3.5_super/export_to_csv.py \
        --model-path $MODEL \
        --jsonl-fpath-base \$(realpath results/\$experiment_name)

    if (( $EXPORT_CSV_TO_MODEL_DIR )); then
        cp results/\$experiment_name_export.csv $MODEL/export.csv
    fi
fi

EOF
)

command=$(cat <<EOF
#!/bin/bash

set -euo pipefail

# Input arguments and validation
PREFILL_HEAD=\$PREFILL_HEAD
DECODE_HEAD=\$DECODE_HEAD

# Nemotron's three-read Mamba SSM state must use the dimension-sequence layout when KV transfer is enabled.
# Not used when the model has no Mamba layers.
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# Generic vLLM environment variables.
export VLLM_USE_FASTOKENS=1

# NIXL uses UCX for cross-node KV transfer. Explicitly enable UCX's CUDA
# transports and the GB200 InfiniBand interface; otherwise UCX treats VRAM as
# host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND.
export UCX_TLS=rc_x,rc,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_ADDR_TYPE=eth
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0

source "$VLLM_CONFIG"

this_node_hostname=\$(hostname)
# Split nodes here by index
if (( SLURM_PROCID == 0 )); then
    # Prefill head

    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $PREFILL_SERVER_PORT \
        --data-parallel-size $NUM_PREFILL_NODES \
        --data-parallel-address \$this_node_hostname \
        --data-parallel-rpc-port $PREFILL_DP_RPC_PORT \
        --api-server-count 1 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
        &
    prefill_pid=\$!
    trap 'kill "\$prefill_pid" 2>/dev/null || true' EXIT

    until curl -fs "http://\$PREFILL_HEAD:$PREFILL_SERVER_PORT/health" >/dev/null; do
        sleep 5
    done
    until curl -fs "http://\$DECODE_HEAD:$DECODE_SERVER_PORT/health" >/dev/null; do
        sleep 5
    done

    # --intra-node-data-parallel-size must match the data-parallel-size-local above.
    # Set a super long request timeout since some reasoning requests may take a long time to generate.
    vllm-router \
        --policy consistent_hash \
        --vllm-pd-disaggregation \
        --prefill http://\$PREFILL_HEAD:$PREFILL_SERVER_PORT \
        --decode http://\$DECODE_HEAD:$DECODE_SERVER_PORT \
        --host \$PREFILL_HEAD \
        --port $ROUTER_SERVER_PORT \
        --intra-node-data-parallel-size 1 \
        --request-timeout-secs 86400 \
        --prometheus-host 0.0.0.0 \
        --prometheus-port 9000 \
        --log-level info
elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
    # Prefill worker
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" \
        --headless \
        --data-parallel-size $NUM_PREFILL_NODES \
        --data-parallel-start-rank \$SLURM_PROCID \
        --data-parallel-address \$PREFILL_HEAD \
        --data-parallel-rpc-port $PREFILL_DP_RPC_PORT \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
elif (( SLURM_PROCID == NUM_PREFILL_NODES )); then
    # Decode head

    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $DECODE_SERVER_PORT \
        --data-parallel-size $NUM_DECODE_NODES \
        --data-parallel-address \$DECODE_HEAD \
        --data-parallel-rpc-port $DECODE_DP_RPC_PORT \
        --api-server-count 1 \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
else
    # Decode worker

    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" \
        --headless \
        --data-parallel-size $NUM_DECODE_NODES \
        --data-parallel-start-rank \$(( SLURM_PROCID - $NUM_PREFILL_NODES )) \
        --data-parallel-address \$DECODE_HEAD \
        --data-parallel-rpc-port $DECODE_DP_RPC_PORT \
        --kv-transfer-config \
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi
EOF
)

NUM_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))
batch_command=$(cat <<EOF
set -euo pipefail

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
PREFILL_HEAD="\${nodes[0]}"
DECODE_HEAD="\${nodes[$NUM_PREFILL_NODES]}"

PREFILL_HEAD="\$PREFILL_HEAD" \
DECODE_HEAD="\$DECODE_HEAD" \
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 \
    --container-image=$CONTAINER \
    --container-name=container-on-node \
    --container-mounts=$MOUNTS \
    --container-workdir=\$SLURM_SUBMIT_DIR \
    --no-container-mount-home \
    bash -lc '
        set -euo pipefail
        cd "\$SLURM_SUBMIT_DIR"
        exec "\$@"
    ' bash bash -lc "\$VLLM_PD_WORKLOAD" &
server_step=\$!

cleanup_server() {
    kill "\$server_step" 2>/dev/null || true
    wait "\$server_step" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM

if (( $should_run_eval )); then
    # No need to wait for endpoint since Gym will wait for model endpoints to spin up before proceeding.

    eval_status=0
    PREFILL_HEAD="\$PREFILL_HEAD" \
    srun --overlap --exact --nodes=1 --ntasks=1 --nodelist="\$PREFILL_HEAD" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name=eval-container-on-node \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -lc '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            exec bash -lc "\$EVAL_COMMAND"
        ' || eval_status=\$?

    cleanup_server
    trap - EXIT INT TERM
    exit "\$eval_status"
fi

wait "\$server_step"
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
VLLM_PD_WORKLOAD="$command" \
EVAL_COMMAND="$EVAL_COMMAND" \
VLLM_PD_BATCH_COMMAND="$batch_command" \
sbatch \
    --nodes=$NUM_NODES \
    --time=04:00:00 \
    --job-name=gym-$EXPERIMENT_NAME-$USER \
    --output=slurm-logs/%j-%x.log \
    --ntasks-per-node=1 \
    --exclusive \
    --segment=$NUM_NODES \
    --wrap 'exec bash -lc "$VLLM_PD_BATCH_COMMAND"'
