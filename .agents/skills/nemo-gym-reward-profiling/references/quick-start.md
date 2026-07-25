# Reward Profiling Quick Start

Substitute environment-specific config paths, input data, model endpoint, and output paths.

## Minimal Flow

```bash
POLICY_MODEL_NAME="your_policy_model_name"
POLICY_BASE_URL="your_policy_base_url"
POLICY_ENDPOINT_KEY="your_policy_endpoint_key"

DATA_JSONL="/path/to/your_input.jsonl"
ROLLOUTS_JSONL="/path/to/your_rollouts.jsonl"
MATERIALIZED_JSONL="${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"

AGENT_NAME="your_agent_name"
NUM_REPEATS=2
NUM_SAMPLES_IN_PARALLEL=8

gym env start \
    --config your_model_config_paths \
    --config your_env_config_paths \
    --model "$POLICY_MODEL_NAME" \
    --model-url "$POLICY_BASE_URL" \
    --model-api-key "$POLICY_ENDPOINT_KEY" &
NG_RUN_PID=$!
trap 'kill "$NG_RUN_PID" 2>/dev/null || true' EXIT
./scripts/wait_for_servers.sh "$NG_RUN_PID"

agent_args=()
if [[ -n "$AGENT_NAME" ]]; then
    agent_args=(--agent "$AGENT_NAME")
fi

gym eval run --no-serve \
    "${agent_args[@]}" \
    --input "$DATA_JSONL" \
    --output "$ROLLOUTS_JSONL" \
    --num-repeats "$NUM_REPEATS" \
    --concurrency "$NUM_SAMPLES_IN_PARALLEL"

gym eval profile \
    --inputs "$MATERIALIZED_JSONL" \
    --rollouts "$ROLLOUTS_JSONL"
```

If rows already contain `agent_ref`, leave `AGENT_NAME` empty. Passing `+agent_name` supplies a default for rows without one.

## Partial Rollouts

By default, `gym eval profile` expects every materialized input row to have a matching rollout row. If collection stopped early, profile the completed rollouts with:

```bash
gym eval profile \
    --inputs "$MATERIALIZED_JSONL" \
    --rollouts "$ROLLOUTS_JSONL" \
    ++allow_partial_rollouts=True
```

Partial profiling writes rows only for original input tasks with at least one completed rollout. The command prints how many input tasks were complete, partial, or dropped because they had no rollout.
