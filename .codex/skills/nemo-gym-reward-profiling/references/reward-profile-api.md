# Reward Profile API

Scope: packageable Nemo Gym guidance for `nemo_gym/reward_profile.py`.

## Current Direction

Reward profiling should stay centered on `RewardProfiler`. Check the target checkout before editing because the CLI and output fields can drift across branches.

Key concepts:

- `RewardProfileConfig`: points at materialized inputs and rollout JSONL.
- `RewardProfiler.profile_from_data(rows, results)`: aligns materialized rows with rollout results by task/rollout key and builds task-level and agent-level metrics.
- `RewardProfiler.write_to_disk(...)`: writes `*_reward_profiling.jsonl` and, when requested, `*_agent_metrics.json`.
- `compute_pass_majority_metrics(...)`: shared pass@k, majority@k, no-answer, and variance metric utility for resource servers.
- `compute_aggregate_metrics(...)`: converts raw verify responses into aggregate metrics, with optional env-specific hooks.

Prefer extending this path for generally useful profiling behavior. Inflight collection and post-hoc `ng_reward_profile` should both use this same code path for their final profile output.

## Input Contract

The profiler receives:

- `rows`: materialized rollout input rows, including task index, rollout index, and agent ref.
- `results`: rollout output rows, including task index, rollout index, response payload, reward, and verifier metrics.

Current profiling aligns rows by `(_ng_task_index, _ng_rollout_index)`. It should reject duplicate or mismatched keys rather than silently relying on async output order.

The profiler extracts numeric fields from each result. It also merges numeric `response.usage` fields into the metric surface when usage exists. Booleans become `0`/`1`; numbers stay numeric; strings, lists, dicts, and nulls are ignored for numeric aggregation.

## Output Shape

Common output files:

- `*_reward_profiling.jsonl`: one task-level metric row per original task.
- `*_agent_metrics.json`: agent/global metrics across all rollouts, unless the caller intentionally disables agent metric writing.

Current task-level profile rows include:

- `_ng_task_index`: original task/sample identifier.
- `sample`: materialized input row for the task, with task and rollout indexes removed from the sample copy.
- `num_rollouts`: number of rollout results aggregated for the task.
- `rollout_infos`: compact per-rollout records sorted by `_ng_rollout_index`.
- aggregate metric keys such as `mean/reward`, `max/reward`, `min/reward`, `median/reward`, and `std/reward`.

`rollout_infos` entries are intentionally compact. They include:

- `rollout_id`: string identifier like `task_idx:rollout_idx`.
- `_ng_task_index`
- `_ng_rollout_index`
- `reward`, when present
- numeric `response.usage` fields such as `input_tokens`, `output_tokens`, or `total_tokens`, when present
- numeric verifier/result fields, when present

Full response bodies are not copied into `rollout_infos`. Join back to `rollouts.jsonl` on `(_ng_task_index, _ng_rollout_index)` when the raw model output or full verifier payload is needed.

With one rollout per task, task-level `mean`, `max`, `min`, and `median` for a field all describe that one rollout, and `std` is zero or otherwise degenerate. That is expected output, not a profiling failure. Repeated rollouts are needed when the user wants within-task pass-rate, variance, or reward-to-length distribution analysis.

## Recovering Pass-Rate-Style Fields

Do not require dedicated `pass_rate` columns when `rollout_infos` are present. Recover the original fields from rollout rewards:

```python
infos = row["rollout_infos"]
pass_rate_passed = sum(1 for info in infos if info.get("reward") == 1.0)
pass_rate_total = row["num_rollouts"]
pass_rate = pass_rate_passed / pass_rate_total if pass_rate_total else None
```

For non-binary reward scales, choose the threshold explicitly in downstream code and record that threshold near the derived metric.

## Raw Reward to Length/Token Mappings

For additions such as raw reward-to-length/token mappings:

1. Keep the canonical input contract: materialized inputs plus rollout results.
2. Build or preserve rows keyed by both task index and rollout index.
3. Extract numeric usage fields from `response.usage` defensively.
4. Preserve the raw reward exactly as emitted by the verifier when possible.
5. Do not assume every response has token usage, text length, or a normal response body.
6. Prefer compact rollout info inside task-level rows unless a downstream consumer needs a separate flat file.

Useful compact fields:

- `reward`
- `input_tokens` / `prompt_tokens`
- `output_tokens` / `completion_tokens`
- `total_tokens`
- response text length, if there is a stable response extraction helper in the repo
- verifier-specific numeric fields, when they describe the rollout rather than the task

Avoid:

- deriving task identity from list position after async collection
- collapsing rollout rows before preserving task and rollout identity
- converting missing usage to zero unless the downstream contract explicitly defines zero as "missing"
- copying full response payloads into task profile rows when a join back to `rollouts.jsonl` is enough

## Scaling API Behavior

`ng_reward_profile` reads materialized inputs and rollout results into memory, aligns by task/rollout identity, and builds pandas dataframes for aggregation. For very large runs, plan around memory and output size rather than treating profiling as a streaming reducer.

Practical implications:

- profile matching materialized/rollout pairs from the same run
- if the dataset is too large for one run, split the source JSONL yourself before collection and keep each shard's outputs isolated
- combine shard-level artifacts only with an explicit downstream merge step that preserves task/rollout identity
- watch for missing numeric fields: non-numeric verifier outputs are ignored by aggregation
- token usage only appears in aggregate metrics and `rollout_infos` when responses include numeric `response.usage` fields

## Expected Tests

Add focused tests for:

- multiple rollouts per task
- missing `response.usage`
- non-numeric fields in verify responses
- stable task and rollout identifiers
- out-of-order rollout results
- mismatched or duplicate task/rollout keys
- recoverability of pass-rate-style metrics from `rollout_infos`
- serialization output does not include unserializable histogram objects
