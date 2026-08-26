#!/bin/bash
# Reward profiling in one job: a vLLM prefill/decode endpoint plus the Gym sweep driver.
#
# Forked from ../sbatch_external_vllm.sh. The vLLM half is unchanged; the eval half runs the
# sweep instead of a benchmark split. Run prepare_sweep.sh first to produce SWEEP_DIR.

set -euo pipefail

# Input arguments and validation
NUM_PREFILL_NODES=$NUM_PREFILL_NODES
NUM_DECODE_NODES=$NUM_DECODE_NODES
MODEL=$MODEL
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_CONFIG=$VLLM_CONFIG
# Written by prepare_sweep.sh: sweep_config.yaml + rollouts_materialized_inputs.jsonl.
SWEEP_DIR=$SWEEP_DIR
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL:-512}
SLURM_COMMENT="${SLURM_COMMENT:-}"

# Unlike the benchmark launcher there is no serve-only mode: a reward-profiling job always runs
# the sweep. EXPERIMENT_NAME only names the job and its slurm log.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reward-profiling}"

# Fixed vLLM Port configurations
PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT=5600
DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT=5700

ROUTER_SERVER_PORT=8000
WORKER_SERVER_PORT=8001

eval_command=$(cat <<EOF
# Activate the container's Gym venv. SWEEP_DIR holds the artifacts prepare_sweep.sh wrote.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

# Serve every environment in the sweep from one deployment; rollout collection routes each row
# to the agent named in its own agent_ref.
gym env start --config $SWEEP_DIR/sweep_config.yaml \\
    +uv_venv_dir=/opt/uv_venvs \\
    +skip_venv_if_present=true \\
    ++use_absolute_ip=true \\
    ++policy_base_url=http://\\\$(getent hosts "\\\$ROUTER_NODE" | awk 'NR == 1 {print \\\$1}'):$ROUTER_SERVER_PORT/v1 \\
    ++policy_api_key=dummy_api_key \\
    ++policy_model_name=$MODEL &
gym_servers_pid=\\\$!
trap 'kill \\\$gym_servers_pid 2>/dev/null || true' EXIT

# --resume is load-bearing: Gym reads the pre-expanded inputs instead of re-expanding them
# (~100 min single-threaded for a full sweep), and a walltime kill continues where it stopped.
# num_repeats=1 because prepare_sweep.sh already expanded the repeats.
gym eval run --no-serve --resume \\
    --input $SWEEP_DIR/rollouts_materialized_inputs.jsonl \\
    --output $SWEEP_DIR/rollouts.jsonl \\
    ++num_repeats=1 \\
    ++num_samples_in_parallel=$NUM_SAMPLES_IN_PARALLEL \\
    +nemo_gym_log_dir=$SWEEP_DIR/logs \\
    +uv_venv_dir=/opt/uv_venvs \\
    +skip_venv_if_present=true \\
    ++global_aiohttp_connector_limit_per_host=16384 \\
    ++port_range_low=63000 \\
    ++port_range_high=64000

# Per-task reward summary. allow_partial_rollouts so a walltime kill still yields a profile.
gym eval profile \\
    --inputs $SWEEP_DIR/rollouts_materialized_inputs.jsonl \\
    --rollouts $SWEEP_DIR/rollouts.jsonl \\
    ++allow_partial_rollouts=True
EOF
)

pd_command=$(cat <<EOF
#!/bin/bash

set -euo pipefail

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

# Increase the number of file descriptors to 65k
if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

this_node_hostname=\$(hostname)
if (( SLURM_PROCID == 0 )); then
    read -r -a nodes <<< "\$ALL_NODES"

    # @bxyu-nvidia: for --intra-node-data-parallel-size: Not sure what to set this to other than 1. I can't tell from the docs what is appropriate and 1 seems to work fine.
    # Set a super long request timeout since some reasoning requests may take a long time to generate.
    # Don't manually wait as vllm-router will wait for the URLs to come up
    router_args=( \
        --prefill-policy cache_aware \
        --decode-policy cache_aware \
        --vllm-pd-disaggregation \
        --host \$this_node_hostname \
        --port $ROUTER_SERVER_PORT \
        --intra-node-data-parallel-size 1 \
        --request-timeout-secs 86400 \
        --log-level error
    )

    for (( i = 0; i < $NUM_PREFILL_NODES; i++ )); do
        router_args+=(--prefill "http://\${nodes[i]}:$WORKER_SERVER_PORT")
    done
    for (( i = 0; i < $NUM_DECODE_NODES; i++ )); do
        node_idx=\$(( $NUM_PREFILL_NODES + i ))
        router_args+=(--decode "http://\${nodes[node_idx]}:$WORKER_SERVER_PORT")
    done

    vllm-router "\${router_args[@]}" &

    router_pid=\$!
    trap 'kill "\$router_pid" 2>/dev/null || true' EXIT
fi

# Split nodes here by index
if (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
    # Prefill
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $WORKER_SERVER_PORT
else
    # Decode
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $WORKER_SERVER_PORT
fi
EOF
)

NUM_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))
batch_command=$(cat <<EOF
set -euo pipefail

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))

ALL_NODES="\${nodes[*]}" \
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
    ' bash bash -lc "\$vllm_command" &
server_step=\$!

cleanup_server() {
    kill "\$server_step" 2>/dev/null || true
    wait "\$server_step" 2>/dev/null || true
}
trap cleanup_server EXIT INT TERM

# No need to wait for endpoint since Gym will wait for model endpoints to spin up before proceeding.

    # @bxyu-nvidia: Put the Gym servers on a separate node than the PREFILL_HEAD which is also running the vllm-router
    # This helps relieve so much network traffic on one node.
    if [[ -v 'nodes[1]' ]]; then
        EVAL_NODE=\${nodes[1]}
    else
        EVAL_NODE=\${nodes[0]}
    fi

    # @bxyu-nvidia: We need --cpus-per-task=SLURM_CPUS_ON_NODE, otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer errors from Gym servers and vLLM. Not sure what the correlation is
    ROUTER_NODE="\${nodes[0]}" \
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE --nodelist="\$EVAL_NODE" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name=eval-container-on-node \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -lc '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            # `set -e` is not inherited through a fresh `bash -lc`, so a failing gym command
            # inside eval_command would otherwise exit 0 and look like a successful run.
            exec bash -lc "set -euo pipefail; \$eval_command"
        ' &
    eval_step=\$!

    completed_pid=""
    completed_status=0
    wait -n -p completed_pid "\$server_step" "\$eval_step" || completed_status=\$?

    if [[ "\$completed_pid" == "\$server_step" ]]; then
        if (( completed_status == 0 )); then
            completed_status=1
        fi
        echo "vLLM server step exited unexpectedly with status \$completed_status" >&2
        kill "\$eval_step" 2>/dev/null || true
        wait "\$eval_step" 2>/dev/null || true
        trap - EXIT INT TERM
        exit "\$completed_status"
    fi

    cleanup_server
    trap - EXIT INT TERM
    exit "\$completed_status"
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
vllm_command="$pd_command" \
eval_command="$eval_command" \
batch_command="$batch_command" \
sbatch \
    --nodes=$NUM_NODES \
    --time=04:00:00 \
    --job-name=gym-$EXPERIMENT_NAME-$USER \
    --output=slurm-logs/%j-%x.log \
    --ntasks-per-node=1 \
    --comment="$SLURM_COMMENT" \
    --exclusive \
    --segment=$NUM_NODES \
    --wrap 'exec bash -lc "$batch_command"'
