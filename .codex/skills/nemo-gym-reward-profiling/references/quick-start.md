# Reward Profiling Quick Start

Use this as the general shape for a first reward profiling run. Substitute environment-specific config paths, input data, model endpoint, and output paths.

## Minimal Flow

```bash
CONFIG_PATHS="responses_api_models/vllm_model/configs/vllm_model.yaml,__ENV_CONFIG_PATHS__"

POLICY_MODEL_NAME="__POLICY_MODEL_NAME__"
POLICY_BASE_URL="__POLICY_BASE_URL__"
POLICY_API_KEY="__POLICY_API_KEY__"

DATA_JSONL="__DATA_JSONL__"
ROLLOUTS_JSONL="__ROLLOUTS_JSONL__"
MATERIALIZED_JSONL="${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"

AGENT_NAME="__AGENT_NAME__"
NUM_REPEATS="__NUM_REPEATS__"
NUM_SAMPLES_IN_PARALLEL="__NUM_SAMPLES_IN_PARALLEL__"

ng_run "+config_paths=[$CONFIG_PATHS]" \
    +policy_model_name="$POLICY_MODEL_NAME" \
    +policy_base_url="$POLICY_BASE_URL" \
    +policy_api_key="$POLICY_API_KEY" &
NG_RUN_PID=$!
trap 'kill "$NG_RUN_PID" 2>/dev/null || true' EXIT

# Replace this with a real readiness check when possible.
sleep 60

agent_args=()
if [[ -n "$AGENT_NAME" ]]; then
    agent_args=(+agent_name="$AGENT_NAME")
fi

ng_collect_rollouts \
    "${agent_args[@]}" \
    +input_jsonl_fpath="$DATA_JSONL" \
    +output_jsonl_fpath="$ROLLOUTS_JSONL" \
    +num_repeats="$NUM_REPEATS" \
    +num_samples_in_parallel="$NUM_SAMPLES_IN_PARALLEL" \
    +resume_from_cache=False

# ng_collect_rollouts writes ${ROLLOUTS_JSONL%.jsonl}_reward_profiling.jsonl
# by default in current Gym. Run ng_reward_profile when you want to regenerate
# or validate the profile from completed artifacts.
ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    ++rollouts_jsonl_fpath="$ROLLOUTS_JSONL"
```

If rows already contain `agent_ref`, leave `AGENT_NAME` empty. Passing `+agent_name` supplies a default for rows without one.

## First-Run Settings

Start small:

```bash
NUM_REPEATS=2
NUM_SAMPLES_IN_PARALLEL=8
```

Then inspect line counts and sample rows before increasing scale. For real profiling, use enough repeats to make per-task variability visible.

## Inflight Profiling Default

Current Gym enables inflight reward profiling by default during collection. The default profile path is:

```text
${ROLLOUTS_JSONL%.jsonl}_reward_profiling.jsonl
```

This writes a partial profile file while collection is running, then rewrites the final file through the same `RewardProfiler` path as `ng_reward_profile`. Disable it only for rollout-only collection:

```bash
+inflight_reward_profile=False
```
