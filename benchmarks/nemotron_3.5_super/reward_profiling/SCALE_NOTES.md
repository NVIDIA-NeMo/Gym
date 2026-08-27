# Full-scale prepare timings

Whole manifest, no `LIMIT_PER_ENTRY`. Login node, Lustre fs1, 36 workers.
Source: 36.9 GB across 36 entries, `num_repeats: 8`.

| | `--no-expand` (default) | `EXPAND=1` |
|---|---|---|
| rows written | 726,121 | 5,808,968 |
| size | **34 GB** | 291.5 GB |
| wall time | **682 s** | 2,087 s |
| after sharding x16 | ~68 GB | ~583 GB |

Both are one-time per (manifest, checkpoint); later jobs resume from the result.

## Why `--no-expand` is the default

`materialize` writes one row per task and Gym expands `num_repeats` at collection time
(`rollout_collection.py:828`), honouring the `_ng_task_index` stamped here. That expansion is the
step PR #2816 took from 8 min to 1 min, and each shard only expands its own slice.

The reason it matters beyond disk is prefix caching. A task's repeats have byte-identical prompts.
Unexpanded, all 8 land in one shard and repeats 2-8 hit the vLLM prefix cache. Expanded, the 8
repeats are consecutive lines, so round-robin dealing scatters them across 8 jobs and every repeat
is a cold prefill. (In a single unsharded job expanded repeats also cache fine -- this only bites
once sharded.)

## Failure behaviour, measured against 16 shards

Shards resume, so a dead shard is a delay rather than a loss, and `03_run_sharded.sh` resubmits it
automatically up to `MAX_ROUNDS`. What differs is what a *partial* profile looks like if you stop
early:

- **`--no-expand`**: the ~45k tasks in the dead shard have zero rollouts; the other 15/16 have all 8.
  Damage is deep and local.
- **`EXPAND=1`**: repeats scatter, so one dead shard costs a single repeat from half the tasks
  (which half depends on how `num_repeats` divides into the shard count). Damage is shallow and
  broad.

Neither is clearly better for reward profiling, which wants per-task variance: 7 of 8 repeats still
estimates it, and a random 15/16 of tasks with all 8 repeats is a clean sample.

## Sizing consequence

At 291.5 GB the sharded copy doubles peak disk to ~583 GB, which is worth checking against quota
before a run. At 34 GB it is not a consideration.
