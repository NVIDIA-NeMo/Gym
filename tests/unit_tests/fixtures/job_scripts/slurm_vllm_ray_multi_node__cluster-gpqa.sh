#!/bin/bash
#SBATCH --job-name=gym-gpqa
#SBATCH --account=myaccount
#SBATCH --time=01:00:00
#SBATCH --chdir=/jobs/gpqa
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

# --- Auto-resume chain ---
OUTPUT_DIR="$PWD"
_this_script="$OUTPUT_DIR/job.sh"
_prev_slurm_job_id="${1:-}"
_walltime_file="$OUTPUT_DIR/.gym_accumulated_walltime"
_retry_file="$OUTPUT_DIR/.gym_infra_retries"
_incomplete_file="$OUTPUT_DIR/.gym_chain_incomplete"

# The chain only stops here after a job that did not finish the run: a finished run's job COMPLETEs,
# and its follow-up never starts. So the stop exits non-zero, and sacct does not read the run as done.
_gym_stop_incomplete() {
    echo "$1" > "$_incomplete_file"
    echo "$1"
    exit 1
}

# A job queued by `gym eval resubmit` starts a fresh retry count and a fresh walltime budget: a chain
# stopped at the walltime cap says "resubmit to continue". The variable is read once and unset,
# so the follow-up queued below (which inherits this environment) is an ordinary chain hop.
_gym_resubmit="${GYM_RESUBMIT:-}"
unset GYM_RESUBMIT
if [[ "$_gym_resubmit" == 1 ]]; then
    rm -f "$_retry_file" "$_walltime_file"
fi

if [[ "$_prev_slurm_job_id" != "" ]]; then
    for _sacct_try in 1 2 3 4 5 6; do
        _prev_state=$(sacct -j $_prev_slurm_job_id -P -n -o State | head -n 1)
        [[ -n "$_prev_state" ]] && break
        sleep 10
    done
    _prev_exit=$(sacct -j $_prev_slurm_job_id -P -n -o ExitCode | head -n 1)
    _prev_elapsed=$(sacct -j $_prev_slurm_job_id -P -n -o ElapsedRaw | head -n 1)
    _prev_elapsed=${_prev_elapsed:-0}
    _gym_accumulated=$(cat "$_walltime_file" 2>/dev/null || echo 0)
    _gym_accumulated=$((_gym_accumulated + _prev_elapsed))
    echo $_gym_accumulated > "$_walltime_file"

    if [[ -z "$_prev_state" ]]; then
        # Accounting lag: the state is unknown, so resume without spending an infra retry.
        echo "Previous job $_prev_slurm_job_id: state unknown (sacct returned nothing). Resuming..."
    elif [[ $_prev_state == 'COMPLETED' ]]; then
        echo "Previous job $_prev_slurm_job_id completed successfully. Exiting."
        exit 0
    elif [[ $_prev_state == CANCELLED* ]]; then
        _gym_stop_incomplete "Previous job $_prev_slurm_job_id was cancelled. Chain stopped with rollouts missing; resubmit to continue."
    elif [[ $_prev_state == 'FAILED' && $_prev_exit == '76:0' ]]; then
        _gym_stop_incomplete "Previous evaluation is incomplete with no eligible retries. Stopping chain; inspect its artifacts."
    elif [[ $_prev_state == 'FAILED' && $_prev_exit == '75:0' ]]; then
        # The collector uses exit 1 if no new outcome was saved, so repeated
        # empty invocations take the bounded infrastructure-retry branch below.
        echo "Previous evaluation made progress and has retryable unfinished work. Resuming..."

    elif [[ $_prev_state == 'TIMEOUT' || $_prev_state == 'PREEMPTED' || $_prev_state == 'NODE_FAIL' ]]; then
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Resuming..."

    else
        _retries=$(cat "$_retry_file" 2>/dev/null || echo 0)
        _retries=$((_retries + 1))
        echo $_retries > "$_retry_file"
        if [[ $_retries -gt 3 ]]; then
            _gym_stop_incomplete "Infra retry limit (3) reached after $_prev_state. Stopping."
        fi
        echo "Previous job $_prev_slurm_job_id: $_prev_state. Infra retry $_retries/3..."
    fi
fi

# This job continues the run, so an earlier stop no longer describes it.
rm -f "$_incomplete_file"
echo "$SLURM_JOB_ID" >> "$OUTPUT_DIR/.gym_job_chain"
# A requeued job keeps its job id and runs this script again. One follow-up per job id: the marker is a
# claim on it, taken with an O_EXCL create (bash noclobber, not mkdir: uutils coreutils 0.10.0's mkdir tells
# concurrent callers that both succeeded). A claim with no follow-up pending on this job was left by a shell
# that died before sbatch answered, or by an sbatch that failed; replace it.
_guard_marker="$OUTPUT_DIR/.gym_followup_queued.$SLURM_JOB_ID"
_gym_claim_followup() { (set -C; : > "$_guard_marker") 2>/dev/null; }
_gym_queue_followup=1
if ! _gym_claim_followup; then
    _gym_queue_followup=0
    # squeue's %E lists the dependencies, e.g. "afternotok:123(unfulfilled)", and %Z the job's working directory.
    # The follow-up runs in this job dir (--chdir), so a dependent job of another run does not count. %Z is the
    # --chdir path as given, while $PWD in the job can be the resolved one, so the two are compared with -ef
    # (same file), never as strings.
    # --all: without it squeue does not list jobs, ours included, in a hidden partition.
    if ! _pending=$(squeue --me --all --noheader --states=PENDING --format='%i|%E|%Z' 2>&1); then
        # Unknown is not none: a second follow-up would run the job dir twice, so keep the claim.
        echo "WARNING: squeue failed ($_pending); cannot tell whether job $SLURM_JOB_ID has a follow-up, so none is queued."
    elif _followup=$(
            while IFS='|' read -r _id _deps _cwd; do
                [[ -n "$_cwd" && "$_cwd" -ef "$OUTPUT_DIR" ]] || continue
                if grep -qE "(^|[ ,])afternotok:$SLURM_JOB_ID([(,]|$)" <<< "$_deps"; then
                    echo "$_id"
                    break
                fi
            done <<< "$_pending"
        ) && [[ -n "$_followup" ]]; then
        echo "Auto-resume follow-up for job $SLURM_JOB_ID is already queued: $_followup (requeued job). Not queueing another."
    else
        echo "Follow-up claim of job $SLURM_JOB_ID has no pending follow-up (submission interrupted or failed). Queueing one."
        rm -rf -- "$_guard_marker"
        _gym_claim_followup && _gym_queue_followup=1
    fi
fi
if [[ $_gym_queue_followup == 1 ]]; then
    # kill-on-invalid-dep: without it, clusters lacking kill_invalid_depend keep the successor pending forever.
    if _next_output=$(sbatch --dependency=afternotok:$SLURM_JOB_ID --kill-on-invalid-dep=yes "$_this_script" $SLURM_JOB_ID 2>&1); then
        _next_id=$(echo "$_next_output" | grep -oE '[0-9]+')
        if [[ -n "$_next_id" ]]; then
            echo "$_next_id" > "$_guard_marker"
            echo "Auto-resume follow-up queued: $_next_id (afternotok:$SLURM_JOB_ID)"
        fi
    else
        # Keep the claim: sbatch can fail after the job was created (a lost reply). A requeue of this job asks
        # squeue, and queues the follow-up only if none is pending.
        echo "WARNING: Failed to submit auto-resume follow-up. The chain may not continue on failure; a requeue of this job queues one if squeue shows none pending."
    fi
fi


# Resolve the head node IP for multi-node vLLM services (spanning nodes via Ray).
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node_hostname=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node_hostname" | awk '{print $1}')
export HEAD_NODE_IP="$head_node_ip"
export RAY_HEAD_NODE_IP="$head_node_ip:6379"
echo "Head node IP address: $HEAD_NODE_IP"
gym_nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export GYM_POOL_COMPUTE_NODES="$(IFS=,; echo "${gym_nodes[*]:0:2}")"

# service: otel_collector
env OTEL_TOKEN=${OTEL_TOKEN} SLURM_JOB_ID=${SLURM_JOB_ID} srun --overlap --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --output=logs/otel_collector-$SLURM_JOB_ID.log otelcol-contrib --config /jobs/gpqa/otel/collector.yaml &
OTEL_COLLECTOR_PID=$!

# service: vllm_model
srun --overlap --nodes=2 --ntasks=2 --no-container-mount-home --container-mounts=/lustre/datasets:/data --container-image=vllm/vllm-openai:v0.9.0 --kill-on-bad-exit=1 --wait=1 --output=logs/vllm_model-$SLURM_JOB_ID.log bash -lc '
    if [ "$SLURM_NODEID" = "0" ]; then
        vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000 --tensor-parallel-size 2 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-address "$HEAD_NODE_IP" --data-parallel-rpc-port 13345 --trust-remote-code
    else
        vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000 --tensor-parallel-size 2 --data-parallel-size 8 --data-parallel-size-local 4 --data-parallel-address "$HEAD_NODE_IP" --data-parallel-rpc-port 13345 --trust-remote-code --headless --data-parallel-start-rank $(( SLURM_NODEID * 4 ))
    fi
' &
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
    +policy_model_name=Qwen/Qwen2.5-72B-Instruct
    +policy_api_key=dummy
    +model_call_capture_dir=/jobs/gpqa/model-calls
    +require_complete=True
    +resume_from_cache=True
)
if [ "${_gym_resubmit:-${GYM_RESUBMIT:-}}" = 1 ]; then GYM_CMD+=(++resubmit=True ++resume_from_cache=True); fi
env NEMO_GYM_OTEL_ENABLED=1 NEMO_GYM_OTEL_RUN_ID=jobs NEMO_GYM_OTEL_SPAN_GROUPS=default,verify NEMO_GYM_OTEL_LOGS_ENABLED=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf srun --overlap --no-container-mount-home --nodelist="${gym_nodes[0]}" --nodes=1 --ntasks=1 --container-mounts=/jobs/gpqa:/jobs/gpqa --container-image=python:3.13 --output=logs/driver-$SLURM_JOB_ID.log bash -c '
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
