# Nemo-RL `ray.sub` Integration

Scope: packageable guidance for running Nemo Gym reward profiling inside a Nemo-RL Slurm/Ray allocation. This is not vanilla Gym behavior; it is an execution pattern for large profiling runs that need GPU model serving, optional judge serving, or optional sandbox sidecars.

## Mental Model

Use Nemo-RL `ray.sub` to allocate GPU nodes and start a Ray cluster. Then run one driver command inside the Ray head container. That driver usually:

1. starts the policy model server, often vLLM with Ray distributed execution
2. waits for the policy `/v1/models` endpoint
3. starts Gym servers with `ng_run`
4. collects rollout outputs with `ng_collect_rollouts`
5. profiles completed rollouts with `ng_reward_profile`
6. shuts down child processes

Do not launch separate CPU collection jobs unless there is a specific reason. For reward profiling, the simpler pattern is to run model serving and Gym collection inside the same GPU/Ray allocation so network, mounts, sandbox, and logs are all attached to one job.

## Base `ray.sub` Hooks

When starting from the base NeMo-RL `ray.sub`, use the hooks it already provides before adding new launch machinery:

- `COMMAND`: non-interactive command run on the Ray head container after all Ray workers connect.
- `BASE_LOG_DIR`: shared filesystem base for `<job_id>-logs`.
- `CONTAINER` and `MOUNTS`: the container image and shared mounts used by all Ray nodes.
- `ray-driver.log`: stdout/stderr from `COMMAND`.
- `ray-head.log` and `ray-worker-*.log`: Ray process logs under the same log dir.
- `RAY_LOG_SYNC_FREQUENCY`: optional sync of `/tmp/ray/session_*/logs` into the shared log dir.
- attach script fallback: if `COMMAND` is empty, the job idles and prints an attach script.

In the base flow, Ray head and workers start first. The submit script polls `ray status` until `worker_units == GPUS_PER_NODE * SLURM_JOB_NUM_NODES`, prints `All workers connected!`, then runs:

```bash
srun --overlap --container-name=ray-head ... -o "$LOG_DIR/ray-driver.log" bash -c "$COMMAND"
```

That is enough for a reward profiling driver that starts vLLM, runs `ng_run`, collects rollouts, and runs `ng_reward_profile`.

## Minimal Base Edits

When editing base `ray.sub`, keep changes small and local to launch behavior:

- Prefer passing a short `COMMAND="bash /shared/path/reward_profile_driver.sh /shared/path/reward_profile.env"` rather than embedding a long script in `COMMAND`.
- If nested quoting becomes painful, patch base `ray.sub` to write `COMMAND` into `$LOG_DIR/driver_command.sh` and run that file inside the head container.
- Keep `BASE_LOG_DIR` on a shared filesystem visible from the submit host and containers.
- Keep port ranges non-overlapping: Ray internals, Gym servers, policy serving, judge serving, and sandbox ports should not compete.
- Add sandbox startup only behind `SANDBOX_CONTAINER` and `SANDBOX_COMMAND`; no sandbox variables should be required for sandboxless envs.
- Add optional node/IP map logging only if the driver or debugging workflow needs it; base `ray.sub` does not require one for reward profiling.
- Do not change Gym APIs for this. The driver command should still run the normal `ng_run`, `ng_collect_rollouts`, and `ng_reward_profile` flow.

### Optional command-file patch

Base `ray.sub` can run `COMMAND` directly with `bash -c "$COMMAND"`. For reward profiling, a safer patch is to materialize the command:

```bash
DRIVER_COMMAND_FILE=""
if [[ -n "$COMMAND" ]]; then
  DRIVER_COMMAND_FILE="$LOG_DIR/driver_command.sh"
  printf '%s' "$COMMAND" > "$DRIVER_COMMAND_FILE"
  chmod +x "$DRIVER_COMMAND_FILE"
fi
```

Then replace the driver launch:

```bash
if [[ -n "$DRIVER_COMMAND_FILE" ]]; then
  srun --no-container-mount-home --overlap \
    --container-name=ray-head \
    --container-workdir="$CONTAINER_CWD" \
    --nodes=1 --ntasks=1 -w "$head_node" \
    -o "$LOG_DIR/ray-driver.log" \
    bash "$DRIVER_COMMAND_FILE"
else
  # existing interactive attach behavior
fi
```

Only do this if the base command string is becoming hard to quote. A short `COMMAND="bash /shared/script /shared/config"` is usually enough.

## Optional Sandbox Patch

For sandbox support, patch base `ray.sub` to start sidecars only when requested:

- create a sandbox log/ready directory under `LOG_DIR`
- start one sandbox task per node with `srun --overlap`
- write `SANDBOX_READY_<hostname>` after each sandbox listens on its local port
- make the Ray head wait for all ready markers before running `COMMAND`
- keep sandbox logs under the same `<job_id>-logs` tree as Ray logs
- include any extra sandbox mounts explicitly; the sandbox container does not automatically need the same mounts as the Ray container

The sandbox should run only when the env needs it. Tool envs can also implement tools directly in the resource server.

Patch shape:

```bash
SANDBOX_PORTS_DIR=""
if [[ -n "${SANDBOX_CONTAINER:-}" ]] && [[ -n "${SANDBOX_COMMAND:-}" ]]; then
  SANDBOX_PORTS_DIR="$LOG_DIR/sandbox"
  mkdir -p "$SANDBOX_PORTS_DIR"

  srun --output "$SANDBOX_PORTS_DIR/sandbox-%t.log" \
    --container-image="$SANDBOX_CONTAINER" \
    --container-mounts="$SANDBOX_PORTS_DIR:$SANDBOX_PORTS_DIR${SHARED_TEMP_DIR:+,$SHARED_TEMP_DIR:$SHARED_TEMP_DIR}" \
    --no-container-mount-home \
    --mpi=pmix \
    -A "$SLURM_JOB_ACCOUNT" \
    -p "$SLURM_JOB_PARTITION" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --overlap \
    --nodes="$SLURM_JOB_NUM_NODES" \
    --ntasks="$SLURM_JOB_NUM_NODES" \
    --ntasks-per-node=1 \
    --export=ALL,SANDBOX_PORTS_DIR="$SANDBOX_PORTS_DIR",SANDBOX_WORKER_BASE_PORT="${SANDBOX_BASE_PORT:-6001}",NGINX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}" \
    bash -xc '
      ('"$SANDBOX_COMMAND"') &
      SANDBOX_PID=$!
      deadline=$((SECONDS + 300))
      while ! (echo > /dev/tcp/localhost/'"${NEMO_SKILLS_SANDBOX_PORT:-6000}"') 2>/dev/null; do
        if ! kill -0 "$SANDBOX_PID" 2>/dev/null; then
          echo "sandbox process died before readiness on $(hostname)" >&2
          exit 1
        fi
        if (( SECONDS > deadline )); then
          echo "sandbox readiness timed out on $(hostname)" >&2
          exit 1
        fi
        sleep 2
      done
      touch '"$SANDBOX_PORTS_DIR"'/SANDBOX_READY_$(hostname)
      wait "$SANDBOX_PID"
    ' &
  SRUN_PIDS["sandbox"]=$!
fi
```

Before launching the driver command, gate on sandbox readiness if `SANDBOX_PORTS_DIR` is non-empty:

```bash
if [[ -n "$SANDBOX_PORTS_DIR" ]]; then
  SANDBOX_DEADLINE=$((SECONDS + 600))
  while true; do
    ready_count=$(ls -1 "$SANDBOX_PORTS_DIR"/SANDBOX_READY_* 2>/dev/null | wc -l)
    [[ "$ready_count" -eq "$SLURM_JOB_NUM_NODES" ]] && break
    if (( SECONDS > SANDBOX_DEADLINE )); then
      echo "timed out waiting for sandbox readiness: $ready_count/$SLURM_JOB_NUM_NODES" >&2
      touch "$LOG_DIR/ENDED"
      exit 1
    fi
    sleep 2
  done
fi
```

## Driver Script Template

This is the script passed through `COMMAND`. It is intentionally a template: fill the config values explicitly or source a config file that is visible inside the container. Do not rely on arbitrary submit-host environment variables crossing into the container unless the target `ray.sub` explicitly bakes them in.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Optional: source explicit config written on the shared filesystem.
# Example: bash reward_profile_driver.sh /shared/run/reward_profile.env
if [[ $# -gt 0 ]]; then
    source "$1"
fi

: "${GYM_REPO_DIR:?}"
: "${POLICY_MODEL:?}"
: "${POLICY_MODEL_NAME:?}"
: "${POLICY_BASE_URL:?}"        # Example shape: http://host:port/v1
: "${POLICY_API_KEY:?}"
: "${POLICY_PORT:?}"
: "${POLICY_DP_SIZE:?}"
: "${VLLM_PYTHON:?}"
: "${CONFIG_PATHS:?}"
: "${DATA_JSONL:?}"
: "${ROLLOUTS_JSONL:?}"
: "${NUM_SAMPLES_PARALLEL:?}"
: "${NUM_REPEATS:?}"

AGENT_NAME="${AGENT_NAME:-}"
NG_RUN_EXTRA_ARGS="${NG_RUN_EXTRA_ARGS:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
GYM_PORT_RANGE_LOW="${GYM_PORT_RANGE_LOW:-5000}"
GYM_PORT_RANGE_HIGH="${GYM_PORT_RANGE_HIGH:-5999}"
VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-1800}"
MATERIALIZED_JSONL="${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"

mkdir -p "$(dirname "$ROLLOUTS_JSONL")"

cleanup() {
    local code=$?
    kill "${NG_RUN_PID:-}" "${VLLM_PID:-}" 2>/dev/null || true
    wait "${NG_RUN_PID:-}" "${VLLM_PID:-}" 2>/dev/null || true
    exit "$code"
}
trap cleanup EXIT

# Start policy serving on the Ray cluster created by ray.sub.
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$POLICY_MODEL" \
    --served-model-name "$POLICY_MODEL_NAME" \
    --port "$POLICY_PORT" \
    --distributed-executor-backend ray \
    --data-parallel-backend ray \
    --data-parallel-size "$POLICY_DP_SIZE" \
    $VLLM_EXTRA_ARGS &
VLLM_PID=$!

deadline=$((SECONDS + VLLM_READY_TIMEOUT_SECONDS))
until curl -sf "${POLICY_BASE_URL}/models" >/dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "policy server died before readiness" >&2
        exit 1
    fi
    if (( SECONDS > deadline )); then
        echo "timed out waiting for policy server: ${POLICY_BASE_URL}/models" >&2
        exit 1
    fi
    sleep 10
done

cd "$GYM_REPO_DIR"

# Avoid accidental connection to the Nemo-RL Ray cluster from Gym internals.
unset RAY_ADDRESS 2>/dev/null || true
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/nemo_gym_ray_$$}"
mkdir -p "$RAY_TMPDIR"

# Optional: activate a Gym environment if the container does not already expose ng_run.
if [[ -n "${GYM_VENV_DIR:-}" ]]; then
    source "${GYM_VENV_DIR}/bin/activate"
fi

ng_run "+config_paths=[$CONFIG_PATHS]" \
    ++port_range_low="$GYM_PORT_RANGE_LOW" \
    ++port_range_high="$GYM_PORT_RANGE_HIGH" \
    +policy_model_name="$POLICY_MODEL_NAME" \
    +policy_base_url="$POLICY_BASE_URL" \
    +policy_api_key="$POLICY_API_KEY" \
    $NG_RUN_EXTRA_ARGS &
NG_RUN_PID=$!

# Prefer a target-specific readiness probe when available.
sleep "${GYM_READY_SLEEP_SECONDS:-60}"

agent_args=()
if [[ -n "$AGENT_NAME" ]]; then
    agent_args=(+agent_name="$AGENT_NAME")
fi

ng_collect_rollouts \
    "${agent_args[@]}" \
    +input_jsonl_fpath="$DATA_JSONL" \
    +output_jsonl_fpath="$ROLLOUTS_JSONL" \
    +responses_create_params.temperature="${TEMPERATURE:-1.0}" \
    +responses_create_params.top_p="${TOP_P:-1.0}" \
    +num_samples_in_parallel="$NUM_SAMPLES_PARALLEL" \
    +num_repeats="$NUM_REPEATS" \
    +resume_from_cache="${RESUME_FROM_CACHE:-True}"

# Optional, if the target Gym checkout supports inflight reward profiling:
# add +inflight_reward_profile=True and optionally
# +inflight_reward_profile_fpath="$INFLIGHT_REWARD_PROFILE_FPATH".

ng_reward_profile \
    +materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    +rollouts_jsonl_fpath="$ROLLOUTS_JSONL"

out_count=$(wc -l < "$ROLLOUTS_JSONL")
mat_count=$(wc -l < "$MATERIALIZED_JSONL")
echo "completed rollouts: ${out_count}/${mat_count}"
test "$out_count" -eq "$mat_count"
```

## Example Config File

Write a config file on a shared mount and pass it to the driver. This avoids fragile nested shell quoting and avoids relying on submit-host env propagation.

```bash
GYM_REPO_DIR="/shared/path/to/Gym"
GYM_VENV_DIR="/shared/path/to/Gym/.venv"

POLICY_MODEL="/shared/path/to/model"
POLICY_MODEL_NAME="policy_model"
POLICY_PORT="7000"
POLICY_BASE_URL="http://127.0.0.1:${POLICY_PORT}/v1"
POLICY_API_KEY="dummy-or-real-key"
POLICY_DP_SIZE="2"
VLLM_PYTHON="/path/in/container/to/python"
VLLM_EXTRA_ARGS="--tensor-parallel-size 4 --gpu-memory-utilization 0.85 --trust-remote-code"

CONFIG_PATHS="responses_api_models/vllm_model/configs/vllm_model.yaml,resources_servers/<env>/configs/<env>.yaml"
NG_RUN_EXTRA_ARGS=""
AGENT_NAME="<agent-name-or-empty>"
DATA_JSONL="/shared/path/to/data.jsonl"
ROLLOUTS_JSONL="/shared/path/to/outputs/rollouts.jsonl"
NUM_SAMPLES_PARALLEL="128"
NUM_REPEATS="16"
```

Use `127.0.0.1` only when the driver, vLLM server, and Gym collection all run in the same head container. If the policy server is separate, set `POLICY_BASE_URL` to that reachable endpoint.

## Submit Commands

No sandbox:

```bash
cd "$NEMO_RL_DIR"
mkdir -p "$LOG_BASE" "$SLURM_LOG_DIR"

CONTAINER="$CONTAINER_IMAGE" \
MOUNTS="$MOUNTS" \
COMMAND="bash $DRIVER_SCRIPT $DRIVER_CONFIG" \
BASE_LOG_DIR="$LOG_BASE" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
RAY_LOG_SYNC_FREQUENCY=60 \
sbatch --parsable \
    --nodes="$MODEL_NODES" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --job-name="$JOB_NAME" \
    --output="$SLURM_LOG_DIR/%x-%j.out" \
    --export=ALL \
    ray.sub
```

With sandbox sidecars:

```bash
cd "$NEMO_RL_DIR"
mkdir -p "$LOG_BASE" "$SLURM_LOG_DIR"

CONTAINER="$CONTAINER_IMAGE" \
MOUNTS="$MOUNTS" \
COMMAND="bash $DRIVER_SCRIPT $DRIVER_CONFIG" \
BASE_LOG_DIR="$LOG_BASE" \
GPUS_PER_NODE="$GPUS_PER_NODE" \
RAY_LOG_SYNC_FREQUENCY=60 \
SANDBOX_CONTAINER="$SANDBOX_IMAGE" \
SANDBOX_COMMAND="$SANDBOX_START_COMMAND" \
NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}" \
SANDBOX_BASE_PORT="${SANDBOX_BASE_PORT:-6001}" \
SHARED_TEMP_DIR="${SHARED_TEMP_DIR:-}" \
sbatch --parsable \
    --nodes="$MODEL_NODES" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --job-name="$JOB_NAME" \
    --output="$SLURM_LOG_DIR/%x-%j.out" \
    --export=ALL \
    ray.sub
```

If the env needs a judge model, either point `NG_RUN_EXTRA_ARGS` at an already-running judge endpoint or submit a separate judge `ray.sub` job first and pass its base URL into the collection driver. Scale judge serving separately from policy serving.

## Logs and Readiness

Expected logs:

- Slurm stdout: `--output` path from `sbatch`
- Ray job dir: `$BASE_LOG_DIR/<job_id>-logs`
- driver log: `$BASE_LOG_DIR/<job_id>-logs/ray-driver.log`
- Ray head/worker logs: `$BASE_LOG_DIR/<job_id>-logs/ray-head.log`, `ray-worker-*.log`
- optional sandbox logs: `$BASE_LOG_DIR/<job_id>-logs/sandbox/sandbox-*.log`

Useful checks:

```bash
tail -f "$BASE_LOG_DIR/<job_id>-logs/ray-driver.log"
wc -l "$ROLLOUTS_JSONL" "${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"
```

If `ray-driver.log` never appears, debug `ray.sub` worker readiness first, then sandbox readiness if a sandbox was configured. If `ray-driver.log` appears and vLLM is ready, debug Gym config/data/verifier next.

## Common Failure Modes

- `COMMAND` path not visible inside container: use a shared absolute path or a path relative to the submit workdir.
- arbitrary env vars missing inside container: put values into the driver script/config file or patch `ray.sub` to bake them.
- Gym accidentally uses Nemo-RL's Ray cluster: unset `RAY_ADDRESS` and set a separate `RAY_TMPDIR` before `ng_run`.
- sandbox logs missing for a sandbox env: confirm `SANDBOX_CONTAINER` and `SANDBOX_COMMAND` were exported into the `sbatch ray.sub` call.
- collection completes with fewer lines than materialized inputs: inspect verifier/model errors before profiling.
