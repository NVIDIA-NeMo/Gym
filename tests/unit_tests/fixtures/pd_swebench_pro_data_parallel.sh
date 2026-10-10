#!/bin/bash
#SBATCH --job-name=gym-swebench_pro
#SBATCH --account=acct
#SBATCH --chdir=/jobs/swebench_pro
#SBATCH --partition=batch
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4



# Resolve the head node IP for multi-node vLLM services (spanning nodes via Ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP"
gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export GYM_POOL_PREFILL_NODES="$(IFS=,; echo "${gym_nodes[*]:0:4}")"
export GYM_POOL_DECODE_NODES="$(IFS=,; echo "${gym_nodes[*]:4:6}")"

# service: otel_collector
env OTEL_TOKEN=${OTEL_TOKEN} SLURM_JOB_ID=${SLURM_JOB_ID} srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --output=logs/otel_collector-$SLURM_JOB_ID.log otelcol-contrib --config /jobs/swebench_pro/otel/collector.yaml &
OTEL_COLLECTOR_PID=$!

# service: policy-prefill
srun --overlap --nodelist="${GYM_POOL_PREFILL_NODES}" --nodes=4 --ntasks=4 --no-container-mount-home --container-image=vllm:img --output=logs/policy-prefill-$SLURM_JOB_ID.log bash -c 'export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600
exec bash -lc '"'"'
    if [ "$SLURM_NODEID" = "0" ]; then
        vllm serve /ckpt --port 8001 --tensor-parallel-size 4 --served-model-name super35 --max-model-len 131072 --kv-transfer-config '"'"'"'"'"'"'"'"'{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'"'"'"'"'"'"'"'"' --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-address ${gym_nodes[0]} --data-parallel-rpc-port 13345
    else
        vllm serve /ckpt --port 8001 --tensor-parallel-size 4 --served-model-name super35 --max-model-len 131072 --kv-transfer-config '"'"'"'"'"'"'"'"'{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'"'"'"'"'"'"'"'"' --data-parallel-size 4 --data-parallel-size-local 1 --data-parallel-address ${gym_nodes[0]} --data-parallel-rpc-port 13345 --headless --data-parallel-start-rank $(( SLURM_NODEID * 1 ))
    fi
'"'"'' &
POLICY_PREFILL_PID=$!

# service: policy-decode
env UCX_TLS=rc,cuda_copy srun --overlap --nodelist="${GYM_POOL_DECODE_NODES}" --nodes=6 --ntasks=6 --no-container-mount-home --container-image=vllm:img --output=logs/policy-decode-$SLURM_JOB_ID.log bash -c 'export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)
export VLLM_NIXL_SIDE_CHANNEL_PORT=5601
export NCCL_DEBUG=WARN
exec bash -lc '"'"'
    if [ "$SLURM_NODEID" = "0" ]; then
        vllm serve /ckpt --port 8002 --tensor-parallel-size 4 --served-model-name super35 --kv-transfer-config '"'"'"'"'"'"'"'"'{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'"'"'"'"'"'"'"'"' --data-parallel-size 6 --data-parallel-size-local 1 --data-parallel-address ${gym_nodes[4]} --data-parallel-rpc-port 13346
    else
        vllm serve /ckpt --port 8002 --tensor-parallel-size 4 --served-model-name super35 --kv-transfer-config '"'"'"'"'"'"'"'"'{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'"'"'"'"'"'"'"'"' --data-parallel-size 6 --data-parallel-size-local 1 --data-parallel-address ${gym_nodes[4]} --data-parallel-rpc-port 13346 --headless --data-parallel-start-rank $(( SLURM_NODEID * 1 ))
    fi
'"'"'' &
POLICY_DECODE_PID=$!

# service: policy
srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --no-container-mount-home --container-image=vllm:img --output=logs/policy-$SLURM_JOB_ID.log vllm-router --prefill-policy cache_aware --decode-policy cache_aware --vllm-pd-disaggregation --prefill "http://${gym_nodes[0]}:8001" --decode "http://${gym_nodes[4]}:8002" --host 0.0.0.0 --port 8000 --intra-node-data-parallel-size 1 --request-timeout-secs 86400 --log-level error &
POLICY_PID=$!

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


# Wait for policy-prefill (try multiple health endpoints)
echo "Waiting for policy-prefill at http://localhost:8001..."
POLICY_PREFILL_READY=0
for _i in $(seq 1 12); do
    if curl -sf "http://localhost:8001/health" > /dev/null 2>&1; then
        echo "  policy-prefill ready."
        POLICY_PREFILL_READY=1
        break
    fi
    if [ -n "${POLICY_PREFILL_PID:-}" ] && ! kill -0 $POLICY_PREFILL_PID 2>/dev/null; then
        echo "  policy-prefill died during startup."
        exit 1
    fi
    sleep 5
done
if [ $POLICY_PREFILL_READY -eq 0 ]; then
    echo "ERROR: policy-prefill did not become healthy after 12 attempts."
    exit 1
fi


# Wait for policy-decode (try multiple health endpoints)
echo "Waiting for policy-decode at http://${gym_nodes[4]}:8002..."
POLICY_DECODE_READY=0
for _i in $(seq 1 12); do
    if curl -sf "http://${gym_nodes[4]}:8002/health" > /dev/null 2>&1; then
        echo "  policy-decode ready."
        POLICY_DECODE_READY=1
        break
    fi
    if [ -n "${POLICY_DECODE_PID:-}" ] && ! kill -0 $POLICY_DECODE_PID 2>/dev/null; then
        echo "  policy-decode died during startup."
        exit 1
    fi
    sleep 5
done
if [ $POLICY_DECODE_READY -eq 0 ]; then
    echo "ERROR: policy-decode did not become healthy after 12 attempts."
    exit 1
fi


# Wait for policy (try multiple health endpoints)
echo "Waiting for policy at http://localhost:8000..."
POLICY_READY=0
for _i in $(seq 1 12); do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "  policy ready."
        POLICY_READY=1
        break
    fi
    if [ -n "${POLICY_PID:-}" ] && ! kill -0 $POLICY_PID 2>/dev/null; then
        echo "  policy died during startup."
        exit 1
    fi
    sleep 5
done
if [ $POLICY_READY -eq 0 ]; then
    echo "ERROR: policy did not become healthy after 12 attempts."
    exit 1
fi




GYM_CMD=(
    gym eval run
    +output_jsonl_fpath=/jobs/swebench_pro/artifacts/rollouts.jsonl
    --model-type openai_model
    +observability_enabled=True
    +policy_base_url=http://localhost:8000/v1
    +policy_model_name=super35
    +policy_api_key=dummy
    +model_call_capture_dir=/jobs/swebench_pro/model-calls
    +require_complete=True
)
env NEMO_GYM_OTEL_ENABLED=1 NEMO_GYM_OTEL_RUN_ID=jobs NEMO_GYM_OTEL_SPAN_GROUPS=default,verify NEMO_GYM_OTEL_LOGS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf srun --overlap --no-container-mount-home --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --container-mounts=/jobs/swebench_pro:/jobs/swebench_pro --container-image=img --output=logs/driver-$SLURM_JOB_ID.log bash -c '
    set -euo pipefail
    if python3 -c "import opentelemetry.exporter.otlp.proto.grpc" 2>/dev/null; then export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4317; else export OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://localhost:4318; fi
    exec "$@"
' -- "${GYM_CMD[@]}"
DRIVER_RC=$?
sleep 20
pkill -TERM -u "$USER" -f -- '^otelcol\-contrib --config /jobs/swebench_pro/otel/collector\.yaml' || true
for _i in $(seq 1 30); do kill -0 $OTEL_COLLECTOR_PID 2>/dev/null || break; sleep 1; done
kill -TERM $OTEL_COLLECTOR_PID 2>/dev/null || true
exit $DRIVER_RC
