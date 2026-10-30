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
# service declaration: which model service runs on which node and GPUs
_gym_decl_gpus=$(timeout 120 srun --overlap --wait=0 --nodelist="${gym_nodes[0]},${gym_nodes[1]},${gym_nodes[2]},${gym_nodes[3]},${gym_nodes[4]},${gym_nodes[5]},${gym_nodes[6]},${gym_nodes[7]},${gym_nodes[8]},${gym_nodes[9]}" --nodes=10 --ntasks=10 bash -c 'out=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader); s=$?; if [ "$s" -ne 0 ]; then echo "$SLURMD_NODENAME, nvidia-smi-exit, $s"; else printf "%s\n" "$out" | sed "s/^/$SLURMD_NODENAME, /"; fi' 2>>logs/service_declaration-$SLURM_JOB_ID.log)
_gym_decl_status=$?
case "$_gym_decl_status" in
    0) _gym_decl_reason="nvidia-smi listed no GPUs in a job step on this node" ;;
    124) _gym_decl_reason="the GPU listing step timed out after 120 s" ;;
    *) _gym_decl_reason="the GPU listing step failed with exit status $_gym_decl_status" ;;
esac
_gym_decl_nodes=()
_gym_decl_rows=()
_gym_decl_node() {
    # <node index> <hostname> <services on the node> <its GPU-using services>
    local host="$2" rows smi_status reason
    rows=$(printf '%s\n' "$_gym_decl_gpus" | GYM_DECL_SERVICES="$4" awk -F' *, *' -v host="$host" -v node="$1" '
        $1 == host && $2 ~ /^[0-9]+$/ && $3 ~ /^GPU-[0-9A-Za-z-]+$/ {
            printf "%s{\"gpu_uuid\":\"%s\",\"node_index\":%d,\"gpu_index\":%d,\"services\":%s}", sep, $3, node, $2, ENVIRON["GYM_DECL_SERVICES"]
            sep = ","
        }')
    if [ -n "$rows" ]; then
        _gym_decl_rows+=("$rows")
        _gym_decl_nodes+=("{\"index\":$1,\"hostname\":\"$host\",\"services\":$3,\"gpus_observed\":true}")
    else
        smi_status=$(printf '%s\n' "$_gym_decl_gpus" | awk -F' *, *' -v host="$host" '$1 == host && $2 == "nvidia-smi-exit" && $3 ~ /^[0-9]+$/ {print $3; exit}')
        reason="$_gym_decl_reason"
        [ -n "$smi_status" ] && reason="nvidia-smi failed on this node with exit status $smi_status"
        _gym_decl_nodes+=("{\"index\":$1,\"hostname\":\"$host\",\"services\":$3,\"gpus_observed\":false,\"gpus_unavailable_reason\":\"$reason\"}")
    fi
}
_gym_decl_node 0 "${gym_nodes[0]}" '["gym-vllm/policy-prefill","gym-vllm/policy"]' '["gym-vllm/policy-prefill"]'
_gym_decl_node 1 "${gym_nodes[1]}" '["gym-vllm/policy-prefill"]' '["gym-vllm/policy-prefill"]'
_gym_decl_node 2 "${gym_nodes[2]}" '["gym-vllm/policy-prefill"]' '["gym-vllm/policy-prefill"]'
_gym_decl_node 3 "${gym_nodes[3]}" '["gym-vllm/policy-prefill"]' '["gym-vllm/policy-prefill"]'
_gym_decl_node 4 "${gym_nodes[4]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
_gym_decl_node 5 "${gym_nodes[5]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
_gym_decl_node 6 "${gym_nodes[6]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
_gym_decl_node 7 "${gym_nodes[7]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
_gym_decl_node 8 "${gym_nodes[8]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
_gym_decl_node 9 "${gym_nodes[9]}" '["gym-vllm/policy-decode"]' '["gym-vllm/policy-decode"]'
mkdir -p /jobs/swebench_pro/artifacts
printf '{"schema_version":1,"status":"available","run_id":%s,"benchmark":%s,"slurm_job_id":"%s","services":%s,"nodes":[%s],"gpus":[%s]}\n' '"jobs"' '"swebench_pro"' "$SLURM_JOB_ID" '[{"service":"gym-vllm/policy-prefill","gym_service":"policy-prefill","kind":"vllm_pd_tier","pool_role":"prefill","node_pool":"prefill","node_indices":[0,1,2,3],"port":8001,"tensor_parallel_size":4,"pipeline_parallel_size":1,"data_parallel_size":4,"ray_serve":false,"server_per_node":false,"gpus_per_node":4},{"service":"gym-vllm/policy-decode","gym_service":"policy-decode","kind":"vllm_pd_tier","pool_role":"decode","node_pool":"decode","node_indices":[4,5,6,7,8,9],"port":8002,"tensor_parallel_size":4,"pipeline_parallel_size":1,"data_parallel_size":6,"ray_serve":false,"server_per_node":false,"gpus_per_node":4},{"service":"gym-vllm/policy","gym_service":"policy","kind":"vllm_pd_router","pool_role":"router","node_indices":[0],"port":8000,"gpus_per_node":0}]' "$(IFS=,; echo "${_gym_decl_nodes[*]}")" "$(IFS=,; echo "${_gym_decl_rows[*]}")" > /jobs/swebench_pro/artifacts/rollouts_service_declaration.json.tmp && cp /jobs/swebench_pro/artifacts/rollouts_service_declaration.json.tmp logs/service_declaration-$SLURM_JOB_ID.json && mv /jobs/swebench_pro/artifacts/rollouts_service_declaration.json.tmp /jobs/swebench_pro/artifacts/rollouts_service_declaration.json

# service: otel_collector
env OTEL_TOKEN=${OTEL_TOKEN} SLURM_JOB_ID=${SLURM_JOB_ID} srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --output=logs/otel_collector-$SLURM_JOB_ID.log otelcol-contrib --config /jobs/swebench_pro/otel/collector.yaml &
OTEL_COLLECTOR_PID=$!

# service: policy-prefill
srun --overlap --nodelist="${GYM_POOL_PREFILL_NODES}" --nodes=4 --ntasks=4 --no-container-mount-home --container-image=vllm:img --kill-on-bad-exit=1 --wait=1 --output=logs/policy-prefill-$SLURM_JOB_ID.log bash -c 'export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)
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
env UCX_TLS=rc,cuda_copy srun --overlap --nodelist="${GYM_POOL_DECODE_NODES}" --nodes=6 --ntasks=6 --no-container-mount-home --container-image=vllm:img --kill-on-bad-exit=1 --wait=1 --output=logs/policy-decode-$SLURM_JOB_ID.log bash -c 'export VLLM_NIXL_SIDE_CHANNEL_HOST=$(hostname)
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
srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --no-container-mount-home --container-image=vllm:img --kill-on-bad-exit=1 --wait=1 --output=logs/policy-$SLURM_JOB_ID.log vllm-router --prefill-policy cache_aware --decode-policy cache_aware --vllm-pd-disaggregation --prefill "http://${gym_nodes[0]}:8001" --decode "http://${gym_nodes[4]}:8002" --host 0.0.0.0 --port 8000 --intra-node-data-parallel-size 1 --request-timeout-secs 86400 --log-level error &
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
if [ "${_gym_resubmit:-${GYM_RESUBMIT:-}}" = 1 ]; then GYM_CMD+=(++resubmit=True ++resume_from_cache=True); fi
env NEMO_GYM_OTEL_ENABLED=1 NEMO_GYM_OTEL_RUN_ID=jobs NEMO_GYM_OTEL_SPAN_GROUPS=default,verify NEMO_GYM_OTEL_LOGS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf srun --overlap --no-container-mount-home --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --container-mounts=/jobs/swebench_pro:/jobs/swebench_pro --container-image=img --output=logs/driver-$SLURM_JOB_ID.log "${GYM_CMD[@]}" &
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
        kill -0 $POLICY_PREFILL_PID 2>/dev/null || { _gym_service_exited policy-prefill $POLICY_PREFILL_PID; return; }
        kill -0 $POLICY_DECODE_PID 2>/dev/null || { _gym_service_exited policy-decode $POLICY_DECODE_PID; return; }
        kill -0 $POLICY_PID 2>/dev/null || { _gym_service_exited policy $POLICY_PID; return; }
        sleep 5
    done
    wait "$_gym_driver_pid"
}
_gym_run_driver
DRIVER_RC=$?
sleep 20
pkill -TERM -u "$USER" -f -- '^otelcol\-contrib --config /jobs/swebench_pro/otel/collector\.yaml' || true
for _i in $(seq 1 30); do kill -0 $OTEL_COLLECTOR_PID 2>/dev/null || break; sleep 1; done
kill -TERM $OTEL_COLLECTOR_PID 2>/dev/null || true
exit $DRIVER_RC
