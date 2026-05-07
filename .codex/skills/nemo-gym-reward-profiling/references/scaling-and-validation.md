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

If rollout collection stopped early, strict profiling fails. Use partial profiling only when you intentionally want to profile completed rollouts:

```bash
ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$MATERIALIZED_JSONL" \
    ++rollouts_jsonl_fpath="$ROLLOUTS_JSONL" \
    ++allow_partial_rollouts=True
```

The command prints completion status, including complete input tasks, partial input tasks, and input tasks dropped because they have no rollout.

## Cache and Resume

`resume_from_cache=True` can reuse existing materialized inputs and rollout outputs. That is useful for restarts, but confusing after data or config changes.

Use a clean output path when changing:

- source data schema
- agent routing
- verifier behavior
- prompt/template behavior
- repeat count

Stale-cache symptoms:

- materialized inputs contain old fields or old types
- verifier errors mention data shapes not present in current source rows
- rollout counts do not match the expected repeat count
- profile rows look like they came from an older run

## Safe JSONL Inspection

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

## Offline Validation

To validate reward profiling output, rerun `ng_reward_profile` on the same materialized inputs and rollout results in a scratch directory, then compare outputs:

```bash
tmpdir="$(mktemp -d /tmp/nemo-gym-profile-compare.XXXXXX)"
cp "$ROLLOUTS_JSONL" "$tmpdir/rollouts.jsonl"
cp "$MATERIALIZED_JSONL" "$tmpdir/rollouts_materialized_inputs.jsonl"

ng_reward_profile \
    ++materialized_inputs_jsonl_fpath="$tmpdir/rollouts_materialized_inputs.jsonl" \
    ++rollouts_jsonl_fpath="$tmpdir/rollouts.jsonl"

cmp -s "${ROLLOUTS_JSONL%.jsonl}_reward_profiling.jsonl" "$tmpdir/rollouts_reward_profiling.jsonl"
```
