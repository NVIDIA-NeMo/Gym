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
set -euo pipefail

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

gym eval prepare $@ +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_\$SLURM_JOB_ID/date_\$(date +%Y%m%d_%H%M%S)
# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
# ++use_absolute_ip=true: Necessary for communication between harness in sandbox and Gym model servers
# ++upload_rollouts_to_wandb=false: Rollouts file is massive. We leave on the cluster.
# global_aiohttp_connector_limit_per_host: 16k concurrent requests should be enough. We can raise further if our inference is efficient enough to support.
# port_range_low, port_range_high: Move into ephemeral ports
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
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $PREFILL_SERVER_PORT \
        --data-parallel-size $NUM_PREFILL_NODES \
        --data-parallel-address \$this_node_hostname \
        --data-parallel-rpc-port $PREFILL_DP_RPC_PORT \
        --api-server-count 1 \
        &
    prefill_pid=\$!
    trap 'kill "\$prefill_pid" 2>/dev/null || true' EXIT

    # @bxyu-nvidia: for --intra-node-data-parallel-size: Not sure what to set this to other than 1. I can't tell from the docs what is appropriate and 1 seems to work fine.
    # Set a super long request timeout since some reasoning requests may take a long time to generate.
    # Don't manually wait as vllm-router will wait for the URLs to come up
    vllm-router \
        --policy consistent_hash \
        --vllm-pd-disaggregation \
        --prefill http://\$PREFILL_HEAD:$PREFILL_SERVER_PORT \
        --decode http://\$DECODE_HEAD:$DECODE_SERVER_PORT \
        --host \$PREFILL_HEAD \
        --port $ROUTER_SERVER_PORT \
        --intra-node-data-parallel-size 1 \
        --request-timeout-secs 86400 \
        --log-level error
elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
    # Prefill worker
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
        --headless \
        --data-parallel-size $NUM_PREFILL_NODES \
        --data-parallel-start-rank \$SLURM_PROCID \
        --data-parallel-address \$PREFILL_HEAD \
        --data-parallel-rpc-port $PREFILL_DP_RPC_PORT
elif (( SLURM_PROCID == NUM_PREFILL_NODES )); then
    # Decode head

    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $DECODE_SERVER_PORT \
        --data-parallel-size $NUM_DECODE_NODES \
        --data-parallel-address \$DECODE_HEAD \
        --data-parallel-rpc-port $DECODE_DP_RPC_PORT \
        --api-server-count 1
else
    # Decode worker

    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
        --headless \
        --data-parallel-size $NUM_DECODE_NODES \
        --data-parallel-start-rank \$(( SLURM_PROCID - $NUM_PREFILL_NODES )) \
        --data-parallel-address \$DECODE_HEAD \
        --data-parallel-rpc-port $DECODE_DP_RPC_PORT
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

    # @bxyu-nvidia: Put the Gym servers on a separate node than the PREFILL_HEAD which is also running the vllm-router
    # This helps relieve so much network traffic on one node.
    if [[ -v 'nodes[1]' ]]; then
        EVAL_NODE=\${nodes[1]}
    else
        EVAL_NODE=\${nodes[0]}
    fi

    # @bxyu-nvidia: We need --cpus-per-task=SLURM_CPUS_ON_NODE, otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer errors from Gym servers and vLLM. Not sure what the correlation is
    eval_status=0
    PREFILL_HEAD="\$PREFILL_HEAD" \
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE --nodelist="\$EVAL_NODE" --gpus=0 \
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
