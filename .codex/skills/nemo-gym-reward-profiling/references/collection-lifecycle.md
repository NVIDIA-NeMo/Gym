# Collection Lifecycle and Command Flow

Scope: packageable Nemo Gym guidance.

## Mental Model

The reusable reward-profiling lifecycle is:

1. Start Gym servers with `ng_run`.
2. Collect rollout outputs with `ng_collect_rollouts`.
3. Profile those rollout outputs with `ng_reward_profile`.

Repeated rollouts are often the point of reward profiling because they expose per-task variability. They are not required by `ng_reward_profile`; with one rollout per task, per-task aggregates naturally collapse to that single rollout.

Launchers and Slurm wrappers can wrap this flow, but they should not change the core data contract.

`ng_collect_rollouts` turns a source JSONL into two important artifacts:

- `*_materialized_inputs.jsonl`: fully expanded inputs after repeats, agent defaults, prompt templating, task index assignment, and rollout index assignment.
- rollout output JSONL: one response/verify result per completed materialized row.
- optionally, `*_reward_profiling.jsonl` during collection when the checkout supports `+inflight_reward_profile=True`.

The materialized file is the source of truth for joining results back to original samples. When cache/resume is enabled, stale materialized inputs can make a run use old data even if the source JSONL has changed.

## Minimal Flow Template

Substitute the placeholders and remove flags not supported by the target checkout. Use `ng_collect_rollouts +h=true` and inspect `nemo_gym/reward_profile.py` when in doubt.

```bash
# Gym config composition. Include the policy model server config plus the
# resource server / agent / judge configs needed by the env.
CONFIG_PATHS="responses_api_models/vllm_model/configs/vllm_model.yaml,__ENV_CONFIG_PATHS__"

# Policy model endpoint that rollout collection should call.
# POLICY_MODEL_NAME must match the served model name expected by the model server.
POLICY_MODEL_NAME="__POLICY_MODEL_NAME__"
POLICY_BASE_URL="__POLICY_BASE_URL__"   # Example shape: http://host:port/v1
POLICY_API_KEY="__POLICY_API_KEY__"     # Whatever the policy endpoint expects.

# Rollout input and output files. ng_collect_rollouts writes the materialized
# inputs next to the rollout output with this conventional suffix.
DATA_JSONL="__DATA_JSONL__"
ROLLOUTS_JSONL="__ROLLOUTS_JSONL__"
MATERIALIZED_JSONL="${ROLLOUTS_JSONL%.jsonl}_materialized_inputs.jsonl"

# Collection knobs.
# Leave AGENT_NAME empty when rows already contain agent_ref.
AGENT_NAME="__AGENT_NAME__"
NUM_SAMPLES_PARALLEL="__NUM_SAMPLES_PARALLEL__"  # In-flight rollout requests; not GPU count.
NUM_REPEATS="__NUM_REPEATS__"                    # Rollouts per source task; 1 is valid but degenerate.

# Extra args are for env-specific overrides such as judge server names,
# verifier defaults, or explicit port ranges when multiple jobs share a host.
ng_run "+config_paths=[$CONFIG_PATHS]" \
    +policy_model_name="$POLICY_MODEL_NAME" \
    +policy_base_url="$POLICY_BASE_URL" \
    +policy_api_key="$POLICY_API_KEY" \
    __NG_RUN_EXTRA_ARGS__ &
NG_RUN_PID=$!
trap 'kill "$NG_RUN_PID" 2>/dev/null || true' EXIT

# A real launcher should use an actual readiness check; sleep keeps the example compact.
sleep 60

agent_name_args=()
if [[ -n "$AGENT_NAME" ]]; then
    agent_name_args=(+agent_name="$AGENT_NAME")
fi

# These response overrides are collection-time defaults. Remove them when the
# data rows or env config should own temperature/top_p.
ng_collect_rollouts \
    "${agent_name_args[@]}" \
    +input_jsonl_fpath="$DATA_JSONL" \
    +output_jsonl_fpath="$ROLLOUTS_JSONL" \
    +responses_create_params.temperature=1.0 \
    +responses_create_params.top_p=1.0 \
    +num_samples_in_parallel="$NUM_SAMPLES_PARALLEL" \
    +num_repeats="$NUM_REPEATS" \
    +resume_from_cache=True

ng_reward_profile \
    +materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    +rollouts_jsonl_fpath="$ROLLOUTS_JSONL"
```

Notes:

- If the target CLI expects `+input_jsonl_fpath` or `+output_jsonl_fpath` for `ng_reward_profile`, use that version's help/code. The common conceptual input is still original/materialized task identity plus rollout results.
- To omit an explicit agent, set `AGENT_NAME=""` and rely on row-level `agent_ref`.
- If rows already carry `agent_ref`, `+agent_name` can be omitted. Passing `+agent_name` supplies a default for rows without an agent.
- `num_samples_in_parallel` is request concurrency. It is not the same thing as GPU count; it should be tuned against model throughput, verifier latency, judge latency, and tool/sandbox latency.
- `ng_reward_profile` works when `num_repeats=1`. In that case, task-level `mean`, `max`, `min`, and `median` equal the single rollout value, and task-level `std` is zero or otherwise degenerate. Agent metrics still aggregate across all rollout rows for that agent.

## Optional Inflight Profiling

Some checkouts support writing reward profiling rows during rollout collection:

```bash
ng_collect_rollouts \
    "${agent_name_args[@]}" \
    +input_jsonl_fpath="$DATA_JSONL" \
    +output_jsonl_fpath="$ROLLOUTS_JSONL" \
    +num_samples_in_parallel="$NUM_SAMPLES_PARALLEL" \
    +num_repeats="$NUM_REPEATS" \
    +resume_from_cache=True \
    +inflight_reward_profile=True
```

The default inflight output path is:

```text
${ROLLOUTS_JSONL%.jsonl}_reward_profiling.jsonl
```

Use `+inflight_reward_profile_fpath=...` when the profile file should live somewhere else.

Behavior to expect:

- while collection is running, the profile file is useful for partial inspection
- when collection completes, the final file should be rewritten through `RewardProfiler.profile_from_data(...)`
- final inflight output should match a post-hoc `ng_reward_profile` run on the same materialized inputs and rollout JSONL
- `VERIFY_OFFLINE_PROFILE` is a wrapper-script convention, not a core Gym CLI flag

When validating inflight profiling, compare structure and values against the canonical offline path. Byte-for-byte comparison is possible when the wrapper snapshots the final inflight file, then reruns `ng_reward_profile` on the same outputs.

## Scaling Reward Profiling Runs

Reward profiling scale is mostly rollout-collection scale. The expected completed rollout count is roughly:

```text
source rows after limit/filtering * num_repeats
```

`num_samples_in_parallel` controls in-flight rollout requests. It does not directly equal GPU count. Tune it against the slowest active layer:

- policy model serving throughput
- resource server and verifier latency
- judge model throughput, if the verifier calls a judge
- tool or sandbox latency, if the env uses tools
- output/logging overhead on the shared filesystem

Main scaling levers:

- Increase `num_samples_in_parallel` until throughput stops improving or timeouts/error rates increase.
- Increase policy serving data parallelism/replicas when the policy model is saturated and the verifier/judge/tool path has headroom.
- Increase judge serving data parallelism/replicas separately when judge calls dominate latency.
- Scale resource-server or tool capacity when verifier/tool execution is the bottleneck; more policy GPUs will not fix that.
- Reduce generation length, tool timeouts, or verbose logging only when they are known contributors.

Scale in stages:

1. Run a tiny clean-cache smoke test and inspect source, materialized, rollout, and profile rows.
2. Increase `num_repeats` to the profiling target and confirm expected materialized/output line counts.
3. Raise `num_samples_in_parallel` until throughput stops improving or verifier/model/tool timeouts appear.
4. Add model or judge capacity only after identifying the bottleneck.
5. If one run becomes too large to manage, split the source JSONL yourself into independent shards and run/profile each shard with its own output paths.

Operational checks:

- output JSONL should grow steadily during collection
- materialized line count should match expected repeated rows
- rollout line count should match materialized line count when complete
- manual shards should never share rollout or materialized output paths
- profile only after collection is complete, unless the target checkout explicitly supports inflight profiling
- after data schema, agent, verifier, or tool config changes, prefer a clean output path over `resume_from_cache`

## Identity

Use the global names from the target repo rather than hard-coding if possible:

- task index: stable original task/sample identifier
- rollout index: repeated rollout number for that task

Every profiling output that maps rewards to usage, lengths, or raw responses must preserve both identifiers.

Important details:

- Task indices are assigned from source row identity before repeat expansion.
- Rollout indices are assigned per task after repeat expansion.
- Repeated identical source rows may collapse to the same task identity in some implementations because task assignment can key off serialized source row content. Inspect the target code if duplicate rows matter.
- Results can finish out of order. Profiling code should sort or join by `(task index, rollout index)`, not rely on file order.

## Cache/Resume Checks

When outputs look wrong:

```bash
wc -l path/to/rollouts.jsonl path/to/rollouts_materialized_inputs.jsonl
stat -c '%y %s %n' path/to/source.jsonl path/to/rollouts_materialized_inputs.jsonl path/to/rollouts.jsonl
```

If materialized inputs predate a source/schema change, rerun with a clean output path or remove stale materialized/output files.

Common stale-cache symptoms:

- source JSONL was fixed, but materialized inputs still contain old fields or old types
- verifier error mentions data shapes that no longer appear in the current source file
- rollout output resumes from an older partial run with a different agent/config
- profiling row counts do not match the expected repeat count

## Safe Inspection Snippet

```bash
python - <<'PY'
import json
from collections import Counter
p = "PATH_TO_JSONL"
keys = Counter()
types = Counter()
with open(p) as f:
    for i, line in enumerate(f, 1):
        obj = json.loads(line)
        keys.update(obj.keys())
        for k, v in obj.items():
            types[(k, type(v).__name__)] += 1
        if i >= 1000:
            break
print("top keys", keys.most_common(30))
print("types", types.most_common(40))
PY
```

## Reward-to-Usage Join Sketch

For raw reward to length/token mappings, build one row per rollout. Keep it intentionally boring:

```python
join_key = (row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME])
usage = (result.get("response") or {}).get("usage") or {}
raw_mapping_row = {
    TASK_INDEX_KEY_NAME: join_key[0],
    ROLLOUT_INDEX_KEY_NAME: join_key[1],
    "reward": result.get("reward"),
    "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
    "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
    "total_tokens": usage.get("total_tokens"),
}
```

Keep missing token fields as `null`/missing rather than fabricating zeros unless downstream explicitly wants zeros.
