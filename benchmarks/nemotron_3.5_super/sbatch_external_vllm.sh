#!/bin/bash

set -euo pipefail

# Input arguments and validation
# PD (default): NUM_PREFILL_NODES=<P> NUM_DECODE_NODES=<D>
# Aggregated: VLLM_MODE=aggregated NUM_NODES=<replicas> (defaults to one)
VLLM_MODE="${VLLM_MODE:-pd}"
case "$VLLM_MODE" in
    pd)
        NUM_PREFILL_NODES=${NUM_PREFILL_NODES:?Required in PD mode}
        NUM_DECODE_NODES=${NUM_DECODE_NODES:?Required in PD mode}
        NUM_NODES=$((NUM_PREFILL_NODES + NUM_DECODE_NODES))
        ;;
    aggregated)
        NUM_NODES=${NUM_NODES:-1}
        NUM_PREFILL_NODES=0
        NUM_DECODE_NODES=0
        ;;
    *)
        echo "VLLM_MODE must be pd or aggregated" >&2
        exit 1
        ;;
esac
MODEL=$MODEL
MODEL_NAME="${MODEL_NAME:-$MODEL}"
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_CONFIG=$VLLM_CONFIG
ENABLE_MOONCAKE=${ENABLE_MOONCAKE:-0}
if [[ "$VLLM_MODE" == aggregated && "$ENABLE_MOONCAKE" != 0 ]]; then
    echo "ENABLE_MOONCAKE requires VLLM_MODE=pd" >&2
    exit 1
fi
# Independent mode starts one complete TP model replica per node. Coupled mode
# forms one multi-node DP/EP engine per tier for models that cannot fit per node.
VLLM_PD_DEPLOYMENT_MODE="${VLLM_PD_DEPLOYMENT_MODE:-independent}"
SLURM_COMMENT="${SLURM_COMMENT:-}"
OPENSANDBOX_DOMAIN="${OPENSANDBOX_DOMAIN:-}"
OPENSANDBOX_API_KEY="${OPENSANDBOX_API_KEY:-}"
OPENSANDBOX_PROTOCOL="${OPENSANDBOX_PROTOCOL:-http}"

case "$VLLM_PD_DEPLOYMENT_MODE" in
    independent | coupled)
        ;;
    *)
        echo "ERROR: VLLM_PD_DEPLOYMENT_MODE must be independent or coupled; got '$VLLM_PD_DEPLOYMENT_MODE'." >&2
        exit 1
        ;;
esac

if [[ "$VLLM_MODE" == aggregated && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]]; then
    echo "ERROR: VLLM_MODE=aggregated does not support VLLM_PD_DEPLOYMENT_MODE=coupled." >&2
    exit 1
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
ROUTER_METRICS_PORT=29000
WORKER_SERVER_PORT=8001

PREFILL_DP_RPC_PORT=13345
DECODE_DP_RPC_PORT=13346

ROUTER_PREFILL_POLICY="${ROUTER_PREFILL_POLICY:-cache_aware}"
ROUTER_DECODE_POLICY="${ROUTER_DECODE_POLICY:-cache_aware}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE="${ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE:-1}"
# Optional whitespace-separated flags, e.g. ROUTER_ARGS="--balance-abs-threshold 32 --balance-rel-threshold 1.1".

eval_command=$(cat <<EOF
set -euo pipefail

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

export NEMO_GYM_RUN_ID="\$SLURM_JOB_ID"
export NEMO_GYM_USER="\${NEMO_GYM_USER:-\$SLURM_JOB_USER}"

source "$VLLM_CONFIG"

gym eval prepare $@ +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_\$SLURM_JOB_ID/date_\$(date +%Y%m%d_%H%M%S)
# export_to_csv.py derives <base>_aggregate_metrics.json from this, so the
# default timestamped name makes the aggregate unfindable to anything that
# did not watch the job run. Override it when results/ is already per-run.
rollouts_fpath=\${ROLLOUTS_FPATH:-results/\$experiment_name.jsonl}

# Scrape each API server directly; the router only exposes its own metrics.
gym_config_args=(
    --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml
    --config benchmarks/nemotron_3.5_super/policy_model_override.yaml
)
inference_metrics_config="results/\$experiment_name/inference-metrics.yaml"
mkdir -p "\$(dirname "\$inference_metrics_config")"
read -r -a nodes <<< "\$ALL_NODES"
{
    printf 'inference_metrics:\n  enabled: true\n  endpoints:\n'
    for node_index in "\${!nodes[@]}"; do
        # Coupled tiers expose one API server each; other ranks are headless.
        if [[ "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]] && \
            (( node_index != 0 && node_index != $NUM_PREFILL_NODES )); then
            continue
        fi
        printf '    node%s: "http://%s:$WORKER_SERVER_PORT/metrics"\n' \
            "\$node_index" "\${nodes[node_index]}"
    done
    printf '  router_endpoints:\n    main: "http://%s:$ROUTER_METRICS_PORT/metrics"\n' "\$ROUTER_NODE"
} > "\$inference_metrics_config"
gym_config_args+=(--config "\$inference_metrics_config")

# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
# ++use_absolute_ip=true: Necessary for communication between harness in sandbox and Gym model servers
# ++upload_rollouts=false: Rollouts file is massive. We leave on the cluster.
# global_aiohttp_connector_limit_per_host: 16k concurrent requests should be enough. We can raise further if our inference is efficient enough to support.
# port_range_low, port_range_high: Move into ephemeral ports
# We add the sandbox_utils and policy_model_override yamls so users don't need to add them on every invocation
gym eval run \
    $@ \
    "\${gym_config_args[@]}" \
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
    ++port_range_high=64000 \
    "\${GYM_MODEL_PARAMS[@]}"


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

vllm_command=$(cat <<EOF
#!/bin/bash

set -euo pipefail

# Generic vLLM environment variables.
export VLLM_USE_FASTOKENS=1

# @bxyu-nvidia: This timeout keep_alive helps reduce connection reset errors between vllm-router and the prefill/decode instances.
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180

# TODO @bxyu-nvidia: Unfortunately there's an accuracy issue with the rust frontend in vLLM 0.29.0, around 1-2% delta on SWE Verified.
# export VLLM_USE_RUST_FRONTEND=1

# NIXL uses UCX for cross-node KV transfer. Explicitly enable UCX's CUDA
# transports and the GB200 InfiniBand interface; otherwise UCX treats VRAM as
# host memory and NIXL KV-cache registration fails with NIXL_ERR_BACKEND.
export UCX_TLS=rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc
export UCX_RNDV_SCHEME=get_zcopy
export UCX_RNDV_THRESH=0
export UCX_NET_DEVICES=all

# Helpful NCCL env vars to set on modern clusters.
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=1
export NCCL_NVLS_ENABLE=1

export ENABLE_MOONCAKE=$ENABLE_MOONCAKE
source "$VLLM_CONFIG"

if (( ENABLE_MOONCAKE )); then
    kv_load_failure_policy=fail
    mooncake_kv_role=kv_consumer
    if (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
        kv_load_failure_policy=recompute
        mooncake_kv_role=kv_both
    fi
    # Preserve each model's connector settings while adding the shared KV store.
    add_mooncake_to_args() {
        mooncake_args=()
        local arg config
        while (( \$# )); do
            arg=\$1
            shift
            if [[ "\$arg" == --kv-transfer-config ]]; then
                config=\${1:?Missing value for --kv-transfer-config}
                shift
            elif [[ "\$arg" == --kv-transfer-config=* ]]; then
                config=\${arg#*=}
            else
                mooncake_args+=("\$arg")
                continue
            fi
            config=\$(python3 - "\$config" "\$kv_load_failure_policy" "\$mooncake_kv_role" <<'MOONCAKE_CONNECTOR'
import json
import sys

config = json.loads(sys.argv[1])
if config["kv_connector"] != "MultiConnector":
    config = {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"connectors": [config]},
    }
config["kv_load_failure_policy"] = sys.argv[2]
connectors = config["kv_connector_extra_config"]["connectors"]
if not any(connector["kv_connector"] == "MooncakeStoreConnector" for connector in connectors):
    connectors.append({
        "kv_connector": "MooncakeStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"load_async": True, "lookup_async": True},
    })
for connector in connectors:
    if connector["kv_connector"] == "MooncakeStoreConnector":
        connector["kv_role"] = sys.argv[3]
        if sys.argv[3] == "kv_consumer":
            connector.setdefault("kv_connector_extra_config", {})["save_decode_cache"] = False
print(json.dumps(config))
MOONCAKE_CONNECTOR
)
            mooncake_args+=(--kv-transfer-config "\$config")
        done
    }
    add_mooncake_to_args "\${VLLM_COMMON_ARGS[@]}"
    VLLM_COMMON_ARGS=("\${mooncake_args[@]}")
    add_mooncake_to_args "\${VLLM_PREFILL_ARGS[@]}"
    VLLM_PREFILL_ARGS=("\${mooncake_args[@]}")
    add_mooncake_to_args "\${VLLM_DECODE_ARGS[@]}"
    VLLM_DECODE_ARGS=("\${mooncake_args[@]}")
fi

# Increase the number of file descriptors to 65k
if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

this_node_hostname=\$(hostname)
read -r -a nodes <<< "\$ALL_NODES"

mooncake_pid=""
cleanup_mooncake() {
    if [[ -n "\$mooncake_pid" ]]; then
        kill "\$mooncake_pid" 2>/dev/null || true
        wait "\$mooncake_pid" 2>/dev/null || true
    fi
}

if (( ENABLE_MOONCAKE )); then
    # Cover setup failures before the serving-mode cleanup traps take over.
    trap cleanup_mooncake EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    export MOONCAKE_CONFIG_PATH="\${MOONCAKE_CONFIG_PATH:-/tmp/mooncake-\$SLURM_JOB_ID/config.json}"
    mkdir -p "\$(dirname "\$MOONCAKE_CONFIG_PATH")"
    cat > "\$MOONCAKE_CONFIG_PATH" <<MOONCAKE_CONFIG
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "\${nodes[0]}:50051",
  "global_segment_size": "100GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0,mlx5_1,mlx5_3,mlx5_4",
  "enable_offload": false
}
MOONCAKE_CONFIG

    # @bxyu-nvidia: Specific to OCI-HSG, as with the device_names above. Otherwise we get errors like:
    # W0924 18:25:48.906106 1830563 worker_pool.cpp:477] Worker: Cannot make connection for endpoint: 10.109.24.46:15741@mlx5_2, pausing peer rail and retrying through an alternate peer RNIC
    export MC_TE_FILTERS=mlx5_0,mlx5_1,mlx5_3,mlx5_4

    uv pip install --system mooncake-transfer-engine-cuda13

    # @bxyu-nvidia: Need these for mooncake connector on GB200 https://github.com/vllm-project/vllm/blob/main/docs/features/mooncake_connector_usage.md#environment-variables
    export WITH_NVIDIA_PEERMEM=0

    if (( SLURM_PROCID == 0 )); then
        echo "Starting mooncake_master on \${nodes[0]}"
        mooncake_master \
            -rpc_port=50051 \
            -rpc_thread_num=4 \
            -metrics_port=9003 \
            -eviction_high_watermark_ratio=0.95 \
            -eviction_ratio=0.1 \
            -minloglevel=1 \
            -enable_metric_reporting=false \
            -default_kv_lease_ttl=120000 \
            -logtostderr &
        mooncake_pid=\$!
    fi

    mooncake_health_url="http://\${nodes[0]}:9003/health"
    mooncake_deadline=\$(( SECONDS + 120 ))
    echo "Waiting for Mooncake master at \$mooncake_health_url"
    until curl --fail --silent --noproxy '*' --connect-timeout 2 --max-time 5 \
        --output /dev/null "\$mooncake_health_url"; do
        if [[ -n "\$mooncake_pid" ]] && ! kill -0 "\$mooncake_pid" 2>/dev/null; then
            echo "mooncake_master exited during startup" >&2
            exit 1
        fi
        if (( SECONDS >= mooncake_deadline )); then
            echo "Timed out waiting for Mooncake master at \$mooncake_health_url" >&2
            exit 1
        fi
        sleep 1
    done
    echo "Mooncake master is ready"
fi

router_common_args=(--log-level error --prometheus-host 0.0.0.0 --prometheus-port $ROUTER_METRICS_PORT)
read -r -a router_extra_args <<< $(printf '%q' "${ROUTER_ARGS:-}")
if (( \${#router_extra_args[@]} )); then
    router_common_args+=("\${router_extra_args[@]}")
fi

if [[ "$VLLM_MODE" == pd && "$VLLM_PD_DEPLOYMENT_MODE" == coupled ]]; then
    PREFILL_HEAD=\${nodes[0]}
    DECODE_HEAD=\${nodes[$NUM_PREFILL_NODES]}

    set_headless_args() {
        # Model configs tune API processes for serving ranks. Headless ranks
        # reject this option, whether supplied as two arguments or with '='.
        headless_args=()
        while (( \$# )); do
            if [[ "\$1" == --api-server-count ]]; then
                shift 2
            elif [[ "\$1" == --api-server-count=* ]]; then
                shift
            else
                headless_args+=("\$1")
                shift
            fi
        done
    }

    if (( SLURM_PROCID == 0 )); then
        # The first prefill rank owns its tier's API server. The remaining
        # prefill ranks run headless so expert parallelism spans the tier.
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT \
            --data-parallel-size $NUM_PREFILL_NODES \
            --data-parallel-address \$PREFILL_HEAD \
            --data-parallel-rpc-port $PREFILL_DP_RPC_PORT \
            --api-server-count 16 \
            &
        prefill_pid=\$!
        coupled_pids=("\$prefill_pid")
        cleanup_coupled_head() {
            local status=\$?
            trap - EXIT INT TERM
            # Signal both local services without delaying failure propagation;
            # the enclosing srun tears down the remaining distributed workers.
            kill "\${coupled_pids[@]}" 2>/dev/null || true
            cleanup_mooncake
            exit "\$status"
        }
        trap cleanup_coupled_head EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM

        vllm-router \
            --prefill-policy $ROUTER_PREFILL_POLICY \
            --decode-policy $ROUTER_DECODE_POLICY \
            --vllm-pd-disaggregation \
            --prefill "http://\$PREFILL_HEAD:$WORKER_SERVER_PORT" \
            --decode "http://\$DECODE_HEAD:$WORKER_SERVER_PORT" \
            --host \$PREFILL_HEAD \
            --port $ROUTER_SERVER_PORT \
            --intra-node-data-parallel-size $ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE \
            --request-timeout-secs 86400 \
            --worker-startup-timeout-secs 1200 \
            "\${router_common_args[@]}" &
        router_pid=\$!
        coupled_pids+=("\$router_pid")

        # Monitor both services after readiness. Polling also catches children that
        # exited before monitoring started, which wait -n can otherwise miss.
        while kill -0 "\$prefill_pid" 2>/dev/null && kill -0 "\$router_pid" 2>/dev/null; do
            sleep 1
        done
        failed_role=prefill
        failed_pid=\$prefill_pid
        if kill -0 "\$prefill_pid" 2>/dev/null; then
            failed_role=router
            failed_pid=\$router_pid
        fi
        failed_status=0
        wait "\$failed_pid" || failed_status=\$?
        # Neither service should exit by itself, even with a zero exit status.
        (( failed_status != 0 )) || failed_status=1
        echo "ERROR: \$failed_role process exited after startup (status=\$failed_status)." >&2
        exit "\$failed_status"
    elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
        set_headless_args "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}"
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${headless_args[@]}" \
            --headless \
            --data-parallel-size $NUM_PREFILL_NODES \
            --data-parallel-start-rank \$SLURM_PROCID \
            --data-parallel-address \$PREFILL_HEAD \
            --data-parallel-rpc-port $PREFILL_DP_RPC_PORT
    elif (( SLURM_PROCID == $NUM_PREFILL_NODES )); then
        # Decode mirrors prefill with one API rank and headless ranks across
        # the other decode nodes.
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT \
            --data-parallel-size $NUM_DECODE_NODES \
            --data-parallel-address \$DECODE_HEAD \
            --data-parallel-rpc-port $DECODE_DP_RPC_PORT \
            --api-server-count 16
    else
        set_headless_args "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_DECODE_ARGS[@]}"
        VLLM_NIXL_SIDE_CHANNEL_HOST=\$this_node_hostname \
        VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_VLLM_NIXL_SIDE_CHANNEL_PORT \
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${headless_args[@]}" \
            --headless \
            --data-parallel-size $NUM_DECODE_NODES \
            --data-parallel-start-rank \$(( SLURM_PROCID - $NUM_PREFILL_NODES )) \
            --data-parallel-address \$DECODE_HEAD \
            --data-parallel-rpc-port $DECODE_DP_RPC_PORT
    fi
else
    router_pid=""

    cleanup_vllm() {
        if [[ -n "\$router_pid" ]]; then
            kill "\$router_pid" 2>/dev/null || true
            wait "\$router_pid" 2>/dev/null || true
        fi
        cleanup_mooncake
    }
    trap cleanup_vllm EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if (( SLURM_PROCID == 0 )); then

        # Set a super long request timeout since some reasoning requests may take a long time to generate.
        # Don't manually wait as vllm-router will wait for the URLs to come up
        # Set a longer worker startup timeout since some models e.g. DSv4 take > 10 mins to load.
        router_args=( \
            --host \$this_node_hostname \
            --port $ROUTER_SERVER_PORT \
            --intra-node-data-parallel-size $ROUTER_INTRA_NODE_DATA_PARALLEL_SIZE \
            --request-timeout-secs 86400 \
            --worker-startup-timeout-secs 1200 \
            "\${router_common_args[@]}"
        )

        if [[ "$VLLM_MODE" == pd ]]; then
            router_args+=( \
                --prefill-policy $ROUTER_PREFILL_POLICY \
                --decode-policy $ROUTER_DECODE_POLICY \
                --vllm-pd-disaggregation
            )
            for (( i = 0; i < $NUM_PREFILL_NODES; i++ )); do
                router_args+=(--prefill "http://\${nodes[i]}:$WORKER_SERVER_PORT")
            done
            for (( i = 0; i < $NUM_DECODE_NODES; i++ )); do
                node_idx=\$(( $NUM_PREFILL_NODES + i ))
                router_args+=(--decode "http://\${nodes[node_idx]}:$WORKER_SERVER_PORT")
            done
        else
            router_args+=(--policy $ROUTER_POLICY --worker-urls)
            for node in "\${nodes[@]}"; do
                router_args+=("http://\$node:$WORKER_SERVER_PORT")
            done
        fi

        vllm-router "\${router_args[@]}" &

        router_pid=\$!

        sleep 5
        if ! kill -0 "\$router_pid" 2>/dev/null; then
            echo "vllm-router exited during startup" >&2
            exit 1
        fi
    fi

    if [[ "$VLLM_MODE" == aggregated ]]; then
        # Reuse prefill tuning by default. Configs can supply VLLM_AGGREGATED_ARGS
        # instead (including an empty array) for independent tuning.
        if declare -p VLLM_AGGREGATED_ARGS &>/dev/null; then
            worker_args=("\${VLLM_COMMON_ARGS[@]}" "\${VLLM_AGGREGATED_ARGS[@]}")
        else
            worker_args=("\${VLLM_COMMON_ARGS[@]}" "\${VLLM_PREFILL_ARGS[@]}")
        fi
        # A complete replica must not wait for KV transfers from a prefill worker.
        # Strip both CLI forms, including connectors set in VLLM_COMMON_ARGS.
        aggregated_args=()
        skip_value=0
        for arg in "\${worker_args[@]}"; do
            if (( skip_value )); then
                skip_value=0
                continue
            fi
            if [[ "\$arg" == --kv-transfer-config ]]; then
                skip_value=1
            elif [[ "\$arg" != --kv-transfer-config=* ]]; then
                aggregated_args+=("\$arg")
            fi
        done
        vllm serve "$MODEL" --served-model-name "$MODEL_NAME" "\${aggregated_args[@]}" \
            --host \$this_node_hostname \
            --port $WORKER_SERVER_PORT
    elif (( SLURM_PROCID < $NUM_PREFILL_NODES )); then
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
fi
EOF
)

batch_command=$(cat <<EOF
set -euo pipefail

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))

ALL_NODES="\${nodes[*]}" \
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image=$CONTAINER \
    --container-name=container-on-node \
    --container-mounts=$MOUNTS \
    --container-workdir=\$SLURM_SUBMIT_DIR \
    --no-container-mount-home \
    bash -c '
        set -euo pipefail
        cd "\$SLURM_SUBMIT_DIR"
        exec "\$@"
    ' bash bash -c "\$vllm_command" &
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

    # @bxyu-nvidia: Put the Gym servers on a separate node from the one running vllm-router.
    # This helps relieve so much network traffic on one node.
    if [[ -v 'nodes[1]' ]]; then
        EVAL_NODE=\${nodes[1]}
    else
        EVAL_NODE=\${nodes[0]}
    fi

    # @bxyu-nvidia: We need --cpus-per-task=SLURM_CPUS_ON_NODE, otherwise we run into a lot of ServerDisconnectedError and ConnectionResetByPeer errors from Gym servers and vLLM. Not sure what the correlation is
    ROUTER_NODE="\${nodes[0]}" \
    ALL_NODES="\${nodes[*]}" \
    srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE --nodelist="\$EVAL_NODE" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name=eval-container-on-node \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -c '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            exec bash -c "\$eval_command"
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
SEGMENT=${SEGMENT:-$NUM_NODES}

submit_dir=$(pwd -P)
# An exported connection is sent as arguments; otherwise env.yaml is read.
if [[ -n "$OPENSANDBOX_DOMAIN" ]]; then
    cleanup_connection=(--domain "$OPENSANDBOX_DOMAIN" --api-key "$OPENSANDBOX_API_KEY" --protocol "$OPENSANDBOX_PROTOCOL")
else
    cleanup_connection=(--connection-config "$submit_dir/env.yaml")
fi
cleanup_user=${NEMO_GYM_USER:-$USER}
main_job_id=$(
    NEMO_GYM_USER="$cleanup_user" \
    vllm_command="$vllm_command" \
    eval_command="$eval_command" \
    batch_command="$batch_command" \
    sbatch \
        --parsable \
        --nodes=$NUM_NODES \
        --time="${SBATCH_TIMELIMIT:-04:00:00}" \
        --job-name=gym-$EXPERIMENT_NAME-$USER \
        --output=slurm-logs/%j-%x.log \
        --ntasks-per-node=1 \
        --comment="$SLURM_COMMENT" \
        --exclusive \
        --segment=$SEGMENT \
        ${NODELIST:+--nodelist="$NODELIST"} \
        --wrap 'exec bash -c "$batch_command"'
)
main_job_id=${main_job_id%%;*}

if (( should_run_eval )); then
    # @bxyu-nvidia: Don't run cleanup job in reservation
    unset SBATCH_RESERVATION
    if ! cleanup_job_id=$(
        sbatch \
            --parsable \
            --dependency=afterany:"$main_job_id" \
            --partition=cpu \
            --qos=cpu-normal \
            --gres=none \
            --gpus-per-node=0 \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=256M \
            --time=00:30:00 \
            --job-name="gym-cleanup-$main_job_id" \
            --output="$submit_dir/slurm-logs/%j-gym-cleanup-$main_job_id.log" \
            "$submit_dir/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py" \
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
