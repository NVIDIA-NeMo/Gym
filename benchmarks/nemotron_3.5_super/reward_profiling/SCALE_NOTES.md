# Full-scale prepare timings

Whole manifest, no `LIMIT_PER_ENTRY`. Login node, Lustre fs1, 36 workers.
Source: 36.9 GB across 36 entries, `num_repeats: 8`.

| | `NO_EXPAND=1` | default |
|---|---|---|
| rows written | 726,121 | 5,808,968 |
| size | **34 GB** | 291.5 GB |
| wall time | **682 s** | 2,087 s |
| after sharding x16 | ~68 GB | ~583 GB |

Both are one-time per (manifest, checkpoint); later jobs resume from the result -- which is
exactly what `--no-expand` gives up.

## Why `--no-expand` is NOT the default

It cannot be collected with `--resume`. `_load_from_cache` keys the materialized inputs on
`(_ng_task_index, _ng_rollout_index)` (`rollout_collection.py:740`) and an unexpanded row has only
the task index, so collection dies about 90 seconds in:

```
KeyError: '_ng_rollout_index'          job 6597590
```

`--resume` is what skips preprocessing on restart and what makes a walltime kill recoverable, so
losing it costs more than the disk saves. `03_run.sh` now refuses an unexpanded sweep at submit
rather than letting the job discover this.

The savings below are real and it stays available via `NO_EXPAND=1` for a first run that will not
need resume -- or if Gym ever keys the resume cache on task index alone, at which point it should
become the default.

## What `--no-expand` would buy

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
