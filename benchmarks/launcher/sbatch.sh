#!/usr/bin/env bash
set -euo pipefail

# Called by eval.sh; positional args are Gym eval args.
# Expected exported environment (set by eval.sh and the selected profile):
#   CHECKPOINT, VLLM_CONFIG            checkpoint and profile script
#   GYM_CONTAINER, VLLM_CONTAINER, ROUTER_CONTAINER  image paths or registry refs
#   MOUNTS                             Pyxis container mounts
#   REPLICAS_PER_NODE                   workers per node
#   NUM_NODES (optional, default 1)     requested allocation size
#   EXPERIMENT_NAME, RUNS_DIR, BENCHMARK run naming and output location
#   OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY
#   GYM_RUN_ARGS                       shell-quoted shared Gym run settings
#   GYM_INFERENCE_METRICS_ENABLED      sampler switch supplied by eval.sh
#   SBATCH_ACCOUNT, SBATCH_PARTITION, SBATCH_QOS    Slurm submission defaults
# Optional: ROUTER_BALANCE_ABS_THRESHOLD (default 40), ROUTER_BALANCE_REL_THRESHOLD (default 2),
#   NEMO_GYM_USER (defaults to USER), ROUTER_RUST_LOG,
#   WANDB_PROJ, WANDB_API_KEY, WANDB_ENTITY, WANDB_MODE.
# VLLM_CONFIG is sourced inside workers for serving arguments and model environment settings.
# Slurm supplies SLURM_* variables; the shell supplies USER.
export NEMO_GYM_USER=${NEMO_GYM_USER:-$USER}
export ROUTER_SERVER_PORT=8000
printf -v GYM_ARGS '%q ' "$@"
export GYM_ARGS

# 2) Inference — configured number of TP1 replicas per node.
export serving_command=$(cat <<'SERVING'
#!/usr/bin/env bash
set -euo pipefail
source "$VLLM_CONFIG"
host=$(hostname)
IFS=, read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
(( ${#gpus[@]} >= REPLICAS_PER_NODE )) || { echo "Not enough visible GPUs for replicas" >&2; exit 1; }
logs="$RUN_DIR/inference${RUN_NAME:+-$SLURM_JOB_ID}"
mkdir -p "$logs"
pids=()
cleanup() {
    status=$?
    trap - EXIT
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((replica=0; replica<REPLICAS_PER_NODE; replica++)); do
    CUDA_VISIBLE_DEVICES="${gpus[replica]}" \
    vllm serve "$CHECKPOINT" --served-model-name "$MODEL_NAME" \
        "${VLLM_COMMON_ARGS[@]}" --host "$host" --port "$((8001 + replica))" \
        > "$logs/$host-replica$replica.log" 2>&1 &
    pids+=("$!")
done
# Any server exiting ends the Slurm step, which triggers the normal job cleanup.
status=0
wait -n "${pids[@]}" || status=$?
(( status != 0 )) || status=1
exit "$status"
SERVING
)

# Router — separate container, same node as the first inference worker.
export router_command=$(cat <<'ROUTER'
set -euo pipefail
export RUST_LOG="${ROUTER_RUST_LOG:-vllm_router_rs=info,vllm_router_rs::policies::cache_aware=debug}"
read -r -a nodes <<< "$ALL_NODES"
urls=()
for node in "${nodes[@]}"; do
    for ((replica=0; replica<REPLICAS_PER_NODE; replica++)); do
        urls+=("http://$node:$((8001 + replica))")
    done
done
exec vllm-router --host 0.0.0.0 --port "$ROUTER_SERVER_PORT" \
    --worker-urls "${urls[@]}" --policy cache_aware \
    --balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD:-40}" \
    --balance-rel-threshold "${ROUTER_BALANCE_REL_THRESHOLD:-2}" \
    --intra-node-data-parallel-size 1 --request-timeout-secs 86400 \
    --prometheus-host 0.0.0.0 --prometheus-port 29000 \
    --log-level info
ROUTER
)

# 3) Evaluation — Gym prepares the tasks and waits for inference to be ready.
export eval_command=$(cat <<'EVAL'
eval "set -- $GYM_ARGS"
gym_args=("$@")
eval "set -- $GYM_RUN_ARGS"
gym_run_args=("$@")
set -euo pipefail

# Copy the read-only development source into this container's private filesystem.
if [[ -d /mnt/gym-dev ]]; then
    echo "Copying development Gym checkout into the container"
    rm -rf /opt/nemo-gym
    mkdir -p /opt/nemo-gym
    tar -C /mnt/gym-dev \
        --exclude=.git --exclude=.env --exclude=env.yaml \
        --exclude=.venv --exclude=__pycache__ --exclude='*.egg-info' \
        --exclude=cache --exclude=.cache --exclude=logs --exclude=results \
        --exclude=runs --exclude=wandb --exclude='swe_*_setup' \
        -cf - . | tar -C /opt/nemo-gym -xf -
    # setup.cfg uses egg_base=cache; the exclude above drops that dir and the editable install fails without it
    mkdir -p /opt/nemo-gym/cache
fi

source /opt/nemo_gym_venv/bin/activate
cd /opt/nemo-gym

# Pin Ray in Gym so the main environment and server venvs agree.
uv add --no-sync 'ray[default]==2.56.1'
uv pip install --python /opt/nemo_gym_venv/bin/python -e .

export NEMO_GYM_RUN_ID="$SLURM_JOB_ID"

# Scrape each backend directly: the router does not expose every replica's metrics.
if [[ "$GYM_INFERENCE_METRICS_ENABLED" == true ]]; then
    inference_metrics_config="$RUN_DIR/inference-metrics.yaml"
    read -r -a nodes <<< "$ALL_NODES"
    {
        printf 'inference_metrics:\n  enabled: true\n  endpoints:\n'
        for node_index in "${!nodes[@]}"; do
            for ((replica=0; replica<REPLICAS_PER_NODE; replica++)); do
                printf '    node%s_replica%s: "http://%s:%s/metrics"\n' \
                    "$node_index" "$replica" "${nodes[node_index]}" "$((8001 + replica))"
            done
        done
        printf '  router_endpoints:\n    main: "http://%s:29000/metrics"\n' "$ROUTER_NODE"
    } > "$inference_metrics_config"
    gym_run_args+=(--config "$inference_metrics_config")
fi

gym eval prepare "${gym_args[@]}" +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_$SLURM_JOB_ID/date_$(date +%Y%m%d_%H%M%S)
gym eval run \
    "${gym_args[@]}" "${gym_run_args[@]}" \
    +wandb_name=$experiment_name \
    "+nemo_gym_log_dir=$RUN_DIR/gym" \
    "++model_call_capture_dir=$RUN_DIR/captures" \
    "++output_jsonl_fpath=$RUN_DIR/rollouts.jsonl" \
    ++policy_base_url=http://$(getent hosts "$ROUTER_NODE" | awk 'NR == 1 {print $1}'):$ROUTER_SERVER_PORT/v1
EVAL
)

# 4) Slurm job — run inference, router, and Gym; stop all when any exits.
export batch_command=$(cat <<'BATCH'
set -euo pipefail
export RUN_DIR="${RUN_NAME:+$RUNS_DIR/$RUN_NAME}"
: "${RUN_DIR:=$RUNS_DIR/$SLURM_JOB_ID-$BENCHMARK}"
if [[ -f "$RUN_DIR/rollouts_aggregate_metrics.json" ]]; then
    echo "Run $RUN_DIR already has final metrics; nothing to do."
    exit 0
fi
mkdir -p "$RUN_DIR/inference${RUN_NAME:+-$SLURM_JOB_ID}"
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
export ALL_NODES="${nodes[*]}" ROUTER_NODE="${nodes[0]}"
container_args=(
    --container-mounts="$MOUNTS" --no-container-entrypoint
    --container-workdir=/ --no-container-mount-home
)
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 --container-image="$VLLM_CONTAINER" "${container_args[@]}" \
    bash -c "$serving_command" &
pids+=("$!")

srun --overlap --exact --nodes=1 --ntasks=1 --gpus=0 \
    --cpus-per-task=8 --nodelist="$ROUTER_NODE" \
    --container-image="$ROUTER_CONTAINER" --container-workdir=/ \
    --no-container-mount-home --no-container-entrypoint bash -c "$router_command" \
    > "$RUN_DIR/inference${RUN_NAME:+-$SLURM_JOB_ID}/router.log" 2>&1 &
pids+=("$!")

srun --overlap --exact --nodes=1 --ntasks=1 --gpus=0 \
    --cpus-per-task="$SLURM_CPUS_ON_NODE" --nodelist="${nodes[1]:-${nodes[0]}}" \
    --container-image="$GYM_CONTAINER" "${container_args[@]}" bash -c "$eval_command" &
pids+=("$!")

eval_pid=${pids[2]}
status=0
wait -n -p finished "${pids[@]}" || status=$?
if [[ $finished != "$eval_pid" ]]; then
    echo "Inference or router exited before Gym finished" >&2
    exit 1
fi
exit "$status"
BATCH
)

# Allocate four GPUs per node, independently of replica count.
# Hold briefly so the output directory exists before Slurm opens its log.
NUM_NODES=${NUM_NODES:-1}
if [[ -n ${RUN_NAME:-} ]]; then
    # Named run: one shared directory, one Slurm job name, jobs serialized with singleton.
    run_dir="$RUNS_DIR/$RUN_NAME"
    mkdir -p "$run_dir"
    naming=(--job-name="gym-$RUN_NAME-$USER" --dependency=singleton --output="$run_dir/slurm-%j.log")
else
    naming=(--job-name="gym-$EXPERIMENT_NAME-$USER" --output="$RUNS_DIR/%j-$BENCHMARK/slurm.log")
fi
job=$(sbatch --hold --parsable --nodes="$NUM_NODES" --ntasks-per-node=1 --gpus-per-node=4 \
    --exclusive --segment="$NUM_NODES" --time=04:00:00 \
    "${naming[@]}" \
    --wrap 'exec bash -c "$batch_command"')
job=${job%%;*}
if [[ -z ${RUN_NAME:-} ]]; then
    run_dir="$RUNS_DIR/$job-$BENCHMARK"
    mkdir -p "$run_dir" || { scancel "$job"; exit 1; }
fi
printf 'Submitted eval job %s\nLogs and results: %s\n' "$job" "$run_dir"

# 5) Sandbox cleanup — also runs if the GPU job fails or is cancelled.
export CLEANUP_RUN_ID="$job"
export cleanup_command=$(cat <<'CLEANUP'
set -euo pipefail
# The base image may predate cleanup_sandboxes.py; prefer the mounted dev checkout when present.
script=/opt/nemo-gym/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py
[[ -f /mnt/gym-dev/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py ]] \
    && script=/mnt/gym-dev/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py
exec /opt/nemo_gym_venv/bin/python "$script" \
    --domain "$OPENSANDBOX_DOMAIN" --api-key "$OPENSANDBOX_API_KEY" \
    --run-id "$CLEANUP_RUN_ID" --user "$NEMO_GYM_USER" --reap
CLEANUP
)
unset SBATCH_RESERVATION
if ! sbatch --parsable --dependency="afterany:$job" \
    --partition=cpu --qos=cpu-normal --gres=none --gpus-per-node=0 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G --time=00:30:00 \
    --job-name="gym-cleanup-$job" --output="$run_dir/cleanup.log" \
    --wrap 'exec srun --container-image="$GYM_CONTAINER" ${GYM_DEV_CHECKOUT:+--container-mounts=$GYM_DEV_CHECKOUT:/mnt/gym-dev:ro} --container-workdir=/opt/nemo-gym --no-container-mount-home --no-container-entrypoint bash -c "$cleanup_command"'; then
    scancel "$job"
    echo "Cancelled held eval job $job: sandbox cleanup could not be scheduled." >&2
    exit 1
fi

scontrol release "$job" || { scancel "$job"; exit 1; }
