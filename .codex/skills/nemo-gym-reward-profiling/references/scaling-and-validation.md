# Scaling and Validation

Reward profiling scale is mostly rollout collection scale. Expected completed rollout count is:

```text
source rows after limit/filtering * num_repeats
```

## Main Knobs

- `num_repeats`: rollouts per source task. Increase this to measure per-task variability.
- `num_samples_in_parallel`: in-flight rollout request concurrency. This is not GPU count.
- policy model capacity: increase serving parallelism when the policy endpoint is saturated.
- judge/verifier/tool capacity: scale these separately when they dominate latency.

Increase one knob at a time. More model throughput will not fix a slow verifier, judge, or tool server.

## Smoke-Test Checks

Before a large run:

```bash
wc -l "$DATA_JSONL" "$MATERIALIZED_JSONL" "$ROLLOUTS_JSONL"
head -n 1 "$MATERIALIZED_JSONL"
head -n 1 "$ROLLOUTS_JSONL"
```

Expected line counts:

- materialized inputs: source rows after limit/filtering times `num_repeats`
- rollout JSONL: same as materialized inputs when collection completes
- reward profile JSONL: one row per original task after `ng_reward_profile`

After rollout collection completes, run:

```bash
ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    ++rollouts_jsonl_fpath="$ROLLOUTS_JSONL"
```

Then inspect:

```bash
head -n 1 "${ROLLOUTS_JSONL%.jsonl}_reward_profiling.jsonl"
```

If rollout collection stopped early, strict profiling fails. Use partial profiling only when you intentionally want to profile completed rollouts; see `quick-start.md` for the command and `output-format.md` for the output semantics.
