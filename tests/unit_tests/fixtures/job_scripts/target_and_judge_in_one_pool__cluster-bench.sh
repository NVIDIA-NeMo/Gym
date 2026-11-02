#!/bin/bash
#SBATCH --job-name=gym-bench
#SBATCH --account=acct
#SBATCH --chdir=/jobs/bench
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8



# Resolve the head node IP for multi-node vLLM services (spanning nodes via Ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP"
gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export GYM_POOL_GPU_NODES="$(IFS=,; echo "${gym_nodes[*]:0:2}")"

# service: otel_collector
env OTEL_TOKEN=${OTEL_TOKEN} SLURM_JOB_ID=${SLURM_JOB_ID} srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --output=logs/otel_collector-$SLURM_JOB_ID.log otelcol-contrib --config /jobs/bench/otel/collector.yaml &
OTEL_COLLECTOR_PID=$!

# service: target
srun --overlap --nodes=2 --ntasks=2 --no-container-mount-home --container-image=vllm:img --kill-on-bad-exit=1 --wait=1 --output=logs/target-$SLURM_JOB_ID.log bash -lc '
    if [ "$SLURM_NODEID" = "0" ]; then
        vllm serve org/model --port 8000 --tensor-parallel-size 4 --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-address "$HEAD_NODE_IP" --data-parallel-rpc-port 13345
    else
        vllm serve org/model --port 8000 --tensor-parallel-size 4 --data-parallel-size 2 --data-parallel-size-local 1 --data-parallel-address "$HEAD_NODE_IP" --data-parallel-rpc-port 13345 --headless --data-parallel-start-rank $(( SLURM_NODEID * 1 ))
    fi
' &
TARGET_PID=$!

# service: judge
srun --overlap --nodes=2 --ntasks=2 --no-container-mount-home --container-image=vllm:img --kill-on-bad-exit=1 --wait=1 --output=logs/judge-$SLURM_JOB_ID.log bash -lc '
    export RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT=600
    export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=1.0
    export RAY_SERVE_PROXY_PREFER_LOCAL_NODE_ROUTING=0
    command -v ray >/dev/null 2>&1 || pip install -q "ray[default]"
    if [ "${SLURM_NODEID:-0}" != "0" ]; then
        echo "Waiting up to ${RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT}s for the Ray head at $RAY_HEAD_NODE_IP..."
        deadline=$((SECONDS + RAY_SYMMETRIC_RUN_CLUSTER_WAIT_TIMEOUT))
        until (: >"/dev/tcp/${RAY_HEAD_NODE_IP%:*}/${RAY_HEAD_NODE_IP##*:}") 2>/dev/null; do
            if [ "$SECONDS" -ge "$deadline" ]; then
                echo "ERROR: timed out waiting for the Ray head at $RAY_HEAD_NODE_IP." >&2
                exit 1
            fi
            sleep 1
        done
    fi
    if ray symmetric-run --help >/dev/null 2>&1; then
        ray symmetric-run \
            --address "$RAY_HEAD_NODE_IP" \
            --min-nodes 2 \
            --num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE} \
            -- vllm serve org/judge --port 8001 --tensor-parallel-size 2 --distributed-executor-backend ray
    elif [ "$SLURM_NODEID" = "0" ]; then
        ray start --head --port=6379 --num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE}
        vllm serve org/judge --port 8001 --tensor-parallel-size 2 --distributed-executor-backend ray
    else
        ray start --address="$RAY_HEAD_NODE_IP" --num-cpus=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE} --num-gpus=${SLURM_GPUS_PER_TASK:-$SLURM_GPUS_ON_NODE} --block
    fi
' &
JUDGE_PID=$!

# Wait for otel_collector (try multiple health endpoints)
echo "Waiting for otel_collector at http://localhost:13133..."
OTEL_COLLECTOR_READY=0
for _i in $(seq 1 60); do
    if curl -sf "http://localhost:13133/" > /dev/null 2>&1; then
        echo "  otel_collector ready."
        OTEL_COLLECTOR_READY=1
        break
    fi
    if [ -n "${OTEL_COLLECTOR_PID:-}" ] && ! kill -0 $OTEL_COLLECTOR_PID 2>/dev/null; then
        echo "  otel_collector died during startup."
        exit 1
    fi
    sleep 5
done
if [ $OTEL_COLLECTOR_READY -eq 0 ]; then
    echo "ERROR: otel_collector did not become healthy after 60 attempts."
    exit 1
fi


# Wait for target (try multiple health endpoints)
echo "Waiting for target at http://localhost:8000..."
TARGET_READY=0
for _i in $(seq 1 12); do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "  target ready."
        TARGET_READY=1
        break
    fi
    if [ -n "${TARGET_PID:-}" ] && ! kill -0 $TARGET_PID 2>/dev/null; then
        echo "  target died during startup."
        exit 1
    fi
    sleep 5
done
if [ $TARGET_READY -eq 0 ]; then
    echo "ERROR: target did not become healthy after 12 attempts."
    exit 1
fi


# Wait for judge (try multiple health endpoints)
echo "Waiting for judge at http://localhost:8001..."
JUDGE_READY=0
for _i in $(seq 1 12); do
    if curl -sf "http://localhost:8001/health" > /dev/null 2>&1; then
        echo "  judge ready."
        JUDGE_READY=1
        break
    fi
    if [ -n "${JUDGE_PID:-}" ] && ! kill -0 $JUDGE_PID 2>/dev/null; then
        echo "  judge died during startup."
        exit 1
    fi
    sleep 5
done
if [ $JUDGE_READY -eq 0 ]; then
    echo "ERROR: judge did not become healthy after 12 attempts."
    exit 1
fi




GYM_CMD=(
    gym eval run
    +output_jsonl_fpath=/jobs/bench/artifacts/rollouts.jsonl
    --model-type openai_model
    +observability_enabled=True
    +policy_base_url=http://localhost:8000/v1
    +policy_model_name=org/model
    +policy_api_key=dummy
    +model_call_capture_dir=/jobs/bench/model-calls
    +require_complete=True
)
if [ "${_gym_resubmit:-${GYM_RESUBMIT:-}}" = 1 ]; then GYM_CMD+=(++resubmit=True ++resume_from_cache=True); fi
env NEMO_GYM_OTEL_ENABLED=1 NEMO_GYM_OTEL_RUN_ID=jobs NEMO_GYM_OTEL_SPAN_GROUPS=default,verify NEMO_GYM_OTEL_LOGS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf srun --overlap --no-container-mount-home --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --container-mounts=/jobs/bench:/jobs/bench --container-image=img --output=logs/driver-$SLURM_JOB_ID.log "${GYM_CMD[@]}" &
_gym_driver_pid=$!
_gym_service_exited() {
    # A successful run tears its services down as the driver ends: if the driver is already gone, its status stands.
    if ! kill -0 "$_gym_driver_pid" 2>/dev/null; then
        wait "$_gym_driver_pid"
        return
    fi
    wait "$2"
    local status=$?
    echo "ERROR: service $1 exited (status $status) while the driver was running; stopping the driver."
    kill -TERM "$_gym_driver_pid" 2>/dev/null
    wait "$_gym_driver_pid"
    return 1
}
_gym_run_driver() {
    while kill -0 "$_gym_driver_pid" 2>/dev/null; do
        kill -0 $TARGET_PID 2>/dev/null || { _gym_service_exited target $TARGET_PID; return; }
        kill -0 $JUDGE_PID 2>/dev/null || { _gym_service_exited judge $JUDGE_PID; return; }
        sleep 5
    done
    wait "$_gym_driver_pid"
}
_gym_run_driver
DRIVER_RC=$?
sleep 20
pkill -TERM -u "$USER" -f -- '^otelcol\-contrib --config /jobs/bench/otel/collector\.yaml' || true
for _i in $(seq 1 30); do kill -0 $OTEL_COLLECTOR_PID 2>/dev/null || break; sleep 1; done
kill -TERM $OTEL_COLLECTOR_PID 2>/dev/null || true
exit $DRIVER_RC
