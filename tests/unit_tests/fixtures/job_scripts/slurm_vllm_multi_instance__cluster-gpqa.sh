#!/bin/bash
#SBATCH --job-name=gym-gpqa
#SBATCH --account=myaccount
#SBATCH --time=01:00:00
#SBATCH --chdir=/jobs/gpqa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8





# service: otel_collector
env OTEL_TOKEN=${OTEL_TOKEN} SLURM_JOB_ID=${SLURM_JOB_ID} srun --overlap --output=logs/otel_collector-$SLURM_JOB_ID.log otelcol-contrib --config /jobs/gpqa/otel/collector.yaml &
OTEL_COLLECTOR_PID=$!

# service: vllm_model
env HF_TOKEN=hf-golden TRUST_REMOTE_CODE=1 NEL_INVOCATION_ID=${NEL_INVOCATION_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun --overlap --no-container-mount-home --container-mounts=/lustre/datasets:/data --container-image=vllm/vllm-openai:v0.9.0 --kill-on-bad-exit=1 --wait=1 --output=logs/vllm_model-$SLURM_JOB_ID.log vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --tensor-parallel-size 2 --data-parallel-size 4 --trust-remote-code &
VLLM_MODEL_PID=$!

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


# Wait for vllm_model (try multiple health endpoints)
echo "Waiting for vllm_model at http://localhost:8000..."
VLLM_MODEL_READY=0
for _i in $(seq 1 240); do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "  vllm_model ready."
        VLLM_MODEL_READY=1
        break
    fi
    if [ -n "${VLLM_MODEL_PID:-}" ] && ! kill -0 $VLLM_MODEL_PID 2>/dev/null; then
        echo "  vllm_model died during startup."
        exit 1
    fi
    sleep 5
done
if [ $VLLM_MODEL_READY -eq 0 ]; then
    echo "ERROR: vllm_model did not become healthy after 240 attempts."
    exit 1
fi




GYM_CMD=(
    gym eval run
    +output_jsonl_fpath=/jobs/gpqa/artifacts/rollouts.jsonl
    --model-type openai_model
    +observability_enabled=True
    +split=benchmark
    '+config_paths=[benchmarks/gpqa/config.yaml]'
    +policy_base_url=http://localhost:8000/v1
    +policy_model_name=Qwen/Qwen2.5-7B-Instruct
    +policy_api_key=dummy
    +model_call_capture_dir=/jobs/gpqa/model-calls
    +require_complete=True
)
if [ "${_gym_resubmit:-${GYM_RESUBMIT:-}}" = 1 ]; then GYM_CMD+=(++resubmit=True ++resume_from_cache=True); fi
env NEMO_GYM_OTEL_ENABLED=1 NEMO_GYM_OTEL_RUN_ID=jobs NEMO_GYM_OTEL_SPAN_GROUPS=default,verify NEMO_GYM_OTEL_LOGS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf HF_TOKEN=hf-golden srun --overlap --no-container-mount-home --container-mounts=/jobs/gpqa:/jobs/gpqa --container-image=python:3.12 --output=logs/driver-$SLURM_JOB_ID.log bash -c '
    set -euo pipefail
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    GYM_SRC="$(mktemp -d /tmp/gym-install-XXXXXX)"
    (command -v git >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)) && git clone https://github.com/NVIDIA-NeMo/gym "$GYM_SRC/gym" && cd "$GYM_SRC/gym" && git checkout main
    uv venv --seed .venv
    source .venv/bin/activate
    uv pip install -e ".[telemetry,ray]"
    exec "$@"
' -- "${GYM_CMD[@]}" &
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
        kill -0 $VLLM_MODEL_PID 2>/dev/null || { _gym_service_exited vllm_model $VLLM_MODEL_PID; return; }
        sleep 5
    done
    wait "$_gym_driver_pid"
}
_gym_run_driver
DRIVER_RC=$?
sleep 20
pkill -TERM -u "$USER" -f -- '^otelcol\-contrib --config /jobs/gpqa/otel/collector\.yaml' || true
for _i in $(seq 1 30); do kill -0 $OTEL_COLLECTOR_PID 2>/dev/null || break; sleep 1; done
kill -TERM $OTEL_COLLECTOR_PID 2>/dev/null || true
exit $DRIVER_RC
