#!/bin/bash

set -euo pipefail

# Input arguments and validation
NUM_PREFILL_NODES=$NUM_PREFILL_NODES
NUM_DECODE_NODES=$NUM_DECODE_NODES
MODEL=$MODEL
MODEL_NAME="${MODEL_NAME:-$MODEL}"
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_CONFIG=$VLLM_CONFIG
SLURM_COMMENT="${SLURM_COMMENT:-}"
OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-}"
OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"
OPENSANDBOX_PROTOCOL="${OPENSANDBOX_PROTOCOL:-http}"

# The checkout this script ships in; a caller running a copy of it names its own.
gym_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [[ "${1:-}" == "--gym-root" ]]; then
    if [[ -z "${2:-}" ]]; then
        echo "--gym-root needs a path" >&2
        exit 2
    fi
    gym_root=$2
    shift 2
fi

should_run_eval=$(( $# > 0 ))
if (( should_run_eval )); then
    EXPERIMENT_NAME=$EXPERIMENT_NAME

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
WORKER_SERVER_PORT=8001

eval_command=$(cat <<EOF
set -euo pipefail

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

export NEMO_GYM_RUN_ID="\$SLURM_JOB_ID"
export NEMO_GYM_USER="\${NEMO_GYM_USER:-\$SLURM_JOB_USER}"

gym eval prepare $@ +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_\$SLURM_JOB_ID/date_\$(date +%Y%m%d_%H%M%S)
# export_to_csv.py derives <base>_aggregate_metrics.json from this, so the
# default timestamped name makes the aggregate unfindable to anything that
# did not watch the job run. Override it when results/ is already per-run.
rollouts_fpath=\${ROLLOUTS_FPATH:-results/\$experiment_name.jsonl}
# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
# ++use_absolute_ip=true: Necessary for communication between harness in sandbox and Gym model servers
# ++upload_rollouts=false: Rollouts file is massive. We leave on the cluster.
# global_aiohttp_connector_limit_per_host: 16k concurrent requests should be enough. We can raise further if our inference is efficient enough to support.
# port_range_low, port_range_high: Move into ephemeral ports
gym eval run \
    $@ \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +nemo_gym_log_dir=results/\$experiment_name/logs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=\$rollouts_fpath \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++use_absolute_ip=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=http://\$(getent hosts "\$ROUTER_NODE" | awk 'NR == 1 {print \$1}'):$ROUTER_SERVER_PORT/v1 \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL_NAME \
    ++upload_rollouts=false \
    ++global_aiohttp_connector_limit_per_host=16384 \
    ++port_range_low=63000 \
    ++port_range_high=64000


if (( $EXPORT_TO_CSV )); then
    python benchmarks/nemotron_3.5_super/export_to_csv.py \
        --model-path $MODEL \
        --jsonl-fpath-base \$(realpath "\${rollouts_fpath%.jsonl}")

    if (( $EXPORT_CSV_TO_MODEL_DIR )); then
        cp "\${rollouts_fpath%.jsonl}_export.csv" $MODEL/export.csv
    fi
fi

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
    vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
        --host \$this_node_hostname \
        --port $WORKER_SERVER_PORT
else
    # Decode
    VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
    vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
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
    job_status=\$?
    trap - EXIT INT TERM
    set +e
    kill "\$server_step" 2>/dev/null || true
    wait "\$server_step" 2>/dev/null || true
    exit "\$job_status"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
            exec bash -lc "\$eval_command"
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
        exit "\$completed_status"
    fi

    exit "\$completed_status"
fi

wait "\$server_step"
EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
submit_dir=$(pwd -P)
# An exported connection is sent as arguments; otherwise env.yaml is read.
if [[ -n "$OPENSANDBOX_DOMAIN" ]]; then
    cleanup_connection=(--domain "$OPENSANDBOX_DOMAIN" --api-key "$OPENSANDBOX_API_KEY" --protocol "$OPENSANDBOX_PROTOCOL")
else
    cleanup_connection=(--connection-config "$gym_root/env.yaml")
fi
cleanup_user=${NEMO_GYM_USER:-$USER}
main_job_id=$(
    NEMO_GYM_USER="$cleanup_user" \
    vllm_command="$pd_command" \
    eval_command="$eval_command" \
    batch_command="$batch_command" \
    sbatch \
        --parsable \
        --nodes=$NUM_NODES \
        --time=04:00:00 \
        --job-name=gym-$EXPERIMENT_NAME-$USER \
        --output=slurm-logs/%j-%x.log \
        --ntasks-per-node=1 \
        --comment="$SLURM_COMMENT" \
        --exclusive \
        --segment=$NUM_NODES \
        --wrap 'exec bash -lc "$batch_command"'
)
main_job_id=${main_job_id%%;*}

if (( should_run_eval )); then
    if ! cleanup_job_id=$(
        sbatch \
            --parsable \
            --dependency=afterany:"$main_job_id" \
            --partition=cpu \
            --qos=cpu-short \
            --gres=none \
            --gpus-per-node=0 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=256M \
            --time=00:30:00 \
            --job-name="gym-cleanup-$main_job_id" \
            --output="$submit_dir/slurm-logs/%j-gym-cleanup-$main_job_id.log" \
            "$gym_root/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py" \
            "${cleanup_connection[@]}" \
            --run-id "$main_job_id" \
            --user "$cleanup_user" \
            --reap
    ); then
        echo "Submitted batch job $main_job_id"
        echo "Failed to submit the sandbox-cleanup job for batch job $main_job_id;" \
            "it is running and its sandboxes will need reaping by hand" >&2
        exit 0
    fi
    cleanup_job_id=${cleanup_job_id%%;*}
fi

echo "Submitted batch job $main_job_id"
if (( should_run_eval )); then
    echo "Submitted cleanup job $cleanup_job_id for batch job $main_job_id"
fi
