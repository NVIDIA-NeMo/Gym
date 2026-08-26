#!/bin/bash
# Reward profiling in one job: a vLLM prefill/decode endpoint plus the Gym sweep driver.
#
# Forked from ../sbatch_external_vllm.sh. The vLLM half is unchanged; the eval half runs the
# sweep instead of a benchmark split. Run prepare_sweep.sh first to produce SWEEP_DIR.

set -euo pipefail

# Input arguments and validation
# Required:
#   MODEL        served checkpoint path (also used as policy_model_name)
#   CONTAINER    reward-profiling sqsh (built by ../../build_eval_container.sh with SKIP_PREPARE=1)
#   SWEEP_DIR    <out-dir>/<nickname> written by prepare_sweep.sh
#
# Optional, with defaults below. SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_QOS / SBATCH_GRES are
# read by sbatch itself from the environment, so they are exported rather than passed as flags.
#
# Sandbox lane: set SANDBOX_CONTAINER to a nemo-skills sandbox sqsh to run one alongside. Required
# by ns_tools and math_formal_lean, which read NEMO_SKILLS_SANDBOX_HOST/PORT and otherwise fall
# back to 127.0.0.1:6000, where nothing listens -- the lane then fails with a bare 500 per rollout.
#
# The image runs nginx over N uWSGI workers on ONE node, hashing X-Session-ID so a stateful IPython
# session stays pinned to one worker. It does not span nodes, and NEMO_SKILLS_SANDBOX_HOST takes a
# single host, so sandbox capacity is currently one node's worth. Scaling past that needs a
# balancer in front of several sandbox nodes; measured at 483 rollouts/hr on 32 workers, and the
# sandbox lane is ~42% of total sweep cost, so this is the first thing to grow.
for _required in MODEL CONTAINER SWEEP_DIR; do
    if [[ -z "${!_required:-}" ]]; then
        echo "ERROR: $_required is required. See the header of $0 for the full argument list." >&2
        exit 2
    fi
done

MODEL=$MODEL
CONTAINER=$CONTAINER
SWEEP_DIR=$SWEEP_DIR

# One prefill node feeds several decode nodes; decode is the throughput-limiting side. P4D12 is the
# largest configuration tested, and 16 nodes still fits inside one 18-node NVL72 rack, which
# --segment requires.
NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-1}
NUM_DECODE_NODES=${NUM_DECODE_NODES:-2}

VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh}
MOUNTS=${MOUNTS:-/lustre:/lustre}

# Sandbox sidecar. Empty disables it, which is correct for the no-judge and judge lanes.
# All job output lands under reward_profiling/outputs/, alongside the sweep dirs, so a run's
# artifacts and its logs are in one gitignored place rather than three.
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR=${LOG_DIR:-$RP_DIR/outputs/slurm-logs}
mkdir -p "$LOG_DIR"

SANDBOX_CONTAINER=${SANDBOX_CONTAINER:-}
SANDBOX_PORT=${SANDBOX_PORT:-6000}
# One worker per core starves the driver, which shares the node. 32 was measured; raise it when the
# sandbox gets a node to itself.
SANDBOX_WORKERS=${SANDBOX_WORKERS:-32}
if [[ -n "$SANDBOX_CONTAINER" && ! -f "$SANDBOX_CONTAINER" ]]; then
    echo "ERROR: SANDBOX_CONTAINER=$SANDBOX_CONTAINER does not exist." >&2
    exit 2
fi

# Secrets reach Gym through env.yaml, which it auto-loads from its working directory. The judge
# lane needs it: the judge config_overlay interpolates ${nv_inference_api_key}, which env.yaml resolves
# from the shell. Without the mount the config fails to parse rather than failing at judge time.
ENV_YAML=${ENV_YAML:-$PWD/env.yaml}
if [[ -f "$ENV_YAML" ]]; then
    MOUNTS="$MOUNTS,$ENV_YAML:/opt/Gym/env.yaml"
else
    echo "WARNING: no env.yaml at $ENV_YAML; judge environments will fail to resolve their API key." >&2
fi

# Scale concurrency with decode capacity: each decode engine schedules up to max_num_seqs
# sequences (512 in vllm_configs/nemotron_3.5_super.sh), so this saturates the engines instead of
# leaving them idle. A measured run at 128 against D2 sat at 12.5% of capacity, with the client
# semaphore rather than the GPUs as the limit. Stays under the 16k per-host aiohttp connector cap
# set below until roughly D32. Lower it if the driver process becomes the bottleneck.
MAX_NUM_SEQS_PER_DECODE_ENGINE=${MAX_NUM_SEQS_PER_DECODE_ENGINE:-512}
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL:-$((MAX_NUM_SEQS_PER_DECODE_ENGINE * NUM_DECODE_NODES))}

# Names the job and its slurm log only; there is no serve-only mode here.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reward-profiling}"

WALLTIME=${WALLTIME:-04:00:00}

# sbatch picks these up from the environment. Without an account it refuses the job outright, and
# without a GRES request a non-CPU partition rejects it -- both are hard errors at submit, so they
# get defaults rather than being discovered one failed submission at a time.
export SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-nemotron_n4_post}
export SBATCH_GRES=${SBATCH_GRES:-gpu:4}
# interactive schedules far sooner than normal on this cluster, and a profiling sweep is
# restartable -- it resumes from the materialized inputs -- so latency to first rollout matters
# more than protection from preemption.
export SBATCH_QOS=${SBATCH_QOS:-interactive}

# nemo_gym.server_utils.DEFAULT_HEAD_SERVER_PORT. 36 environments starting from baked venvs took
# well under a minute in testing; the ceiling is for a cold container that has to install.
HEAD_SERVER_PORT=${HEAD_SERVER_PORT:-11000}
HEAD_SERVER_TIMEOUT_S=${HEAD_SERVER_TIMEOUT_S:-900}
SLURM_COMMENT="${SLURM_COMMENT:-}"

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
    ++policy_base_url=http://\$(getent hosts "\$ROUTER_NODE" | awk 'NR == 1 {print \$1}'):$ROUTER_SERVER_PORT/v1 \
    ++policy_api_key=dummy_api_key \\
    ++policy_model_name=$MODEL &
gym_servers_pid=\$!
trap 'kill \$gym_servers_pid 2>/dev/null || true' EXIT

# Wait for the head server before starting collection. gym env start backgrounds itself and 36
# environments take a while to come up, so without this gym eval run --no-serve races it and dies
# with "Could not connect to the head server at http://127.0.0.1:11000". Any HTTP response means
# it is listening; the body does not matter.
echo "waiting for the Gym head server on :$HEAD_SERVER_PORT ..."
for _i in \$(seq 1 $((HEAD_SERVER_TIMEOUT_S / 10))); do
    if curl -s -o /dev/null -m 3 "http://127.0.0.1:$HEAD_SERVER_PORT/"; then
        echo "head server up after \$(( _i * 10 ))s"
        break
    fi
    if ! kill -0 "\$gym_servers_pid" 2>/dev/null; then
        echo "ERROR: gym env start exited before the head server came up" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -s -o /dev/null -m 3 "http://127.0.0.1:$HEAD_SERVER_PORT/"; then
    echo "ERROR: head server never came up within ${HEAD_SERVER_TIMEOUT_S}s" >&2
    exit 1
fi

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

    # Sandbox sidecar on the eval node, so ns_tools reaches it over loopback. Started before the
    # driver and waited on: if the driver starts first it burns rollouts against a dead port.
    if [[ -n "$SANDBOX_CONTAINER" ]]; then
        echo "starting sandbox on \$EVAL_NODE (${SANDBOX_WORKERS} workers, port ${SANDBOX_PORT})"
        srun --overlap --nodes=1 --ntasks=1 --nodelist="\$EVAL_NODE" --gpus=0 \
            --cpus-per-task=${SANDBOX_WORKERS} \
            --container-image=$SANDBOX_CONTAINER \
            --container-mounts=$MOUNTS \
            --no-container-mount-home \
            bash -lc 'export UWSGI_PROCESSES=${SANDBOX_WORKERS} NUM_WORKERS=${SANDBOX_WORKERS} \
                      LISTEN_PORT=${SANDBOX_PORT} NGINX_PORT=${SANDBOX_PORT}; exec /start-with-nginx.sh' \
            > "$LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" 2>&1 &
        sandbox_step=\$!

        SANDBOX_IP=\$(getent hosts "\$EVAL_NODE" | awk 'NR==1 {print \$1}')
        echo "waiting for sandbox at \$SANDBOX_IP:${SANDBOX_PORT}/health ..."
        for _i in \$(seq 1 60); do
            if curl -sf -m 3 "http://\$SANDBOX_IP:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
                echo "sandbox healthy after \${_i}0s"; break
            fi
            if ! kill -0 "\$sandbox_step" 2>/dev/null; then
                echo "ERROR: sandbox exited during startup; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" >&2
                exit 1
            fi
            sleep 10
        done
        if ! curl -sf -m 3 "http://\$SANDBOX_IP:${SANDBOX_PORT}/health" >/dev/null 2>&1; then
            echo "ERROR: sandbox never became healthy; see $LOG_DIR/\${SLURM_JOB_ID}-sandbox.log" >&2
            exit 1
        fi
        # Exported, not prefixed onto srun: bash decides which words are assignments at parse
        # time, so an expanded "VAR=value" would become the command name instead. srun propagates
        # the environment into the container, which is how ROUTER_NODE already reaches it.
        export NEMO_SKILLS_SANDBOX_HOST="\$SANDBOX_IP"
        export NEMO_SKILLS_SANDBOX_PORT="${SANDBOX_PORT}"
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
            # set -e is not inherited through a fresh bash -lc, so a failing gym command
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
    --time=$WALLTIME \
    --job-name=gym-$EXPERIMENT_NAME-$USER \
    --output="$LOG_DIR/%j-%x.log" \
    --ntasks-per-node=1 \
    --comment="$SLURM_COMMENT" \
    --exclusive \
    --segment=$NUM_NODES \
    --wrap 'exec bash -lc "$batch_command"'
