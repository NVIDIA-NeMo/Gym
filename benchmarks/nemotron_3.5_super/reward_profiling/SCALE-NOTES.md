# Full-scale prepare timings

Whole manifest, no `LIMIT_PER_ENTRY`. Login node, Lustre fs1, 36 workers.
Source: 36.9 GB across 36 entries, `num_repeats: 8`.

| | |
|---|---|
| rows written | 5,808,968 |
| size | 291.5 GB |
| wall time | 2,087 s |
| after sharding x16 | ~583 GB |

One-time per (manifest, checkpoint); later jobs resume from the result rather than redoing it.

## Sizing consequence

The sharded copy roughly doubles peak disk to ~583 GB. Worth checking against quota before a full
run — it is the largest thing this pipeline writes.

## Why repeats are always expanded here

`01_prepare_sweep.sh` writes `num_repeats` copies of every row rather than one row per task. Gym
can expand repeats itself at collection time (`rollout_collection.py:828`), and PR #2816 took that
from 8 min to 1 min, which would save 8x the rows — 34 GB and 682 s, measured.

It is not an option, because unexpanded inputs cannot be collected with `--resume`.
`_load_from_cache` keys the materialized inputs on `(_ng_task_index, _ng_rollout_index)`
(`rollout_collection.py:740`); a row with only a task index dies about 90 seconds in:

```
KeyError: '_ng_rollout_index'          job 6597590
```

`--resume` is what skips preprocessing on restart and what makes a walltime kill recoverable. Every
launcher depends on it, so the disk saving is not available. If Gym ever keys the resume cache on
task index alone, this is the decision to revisit.

The other thing given up is prefix caching. A task's repeats have byte-identical prompts, and
expanded repeats are consecutive lines, so round-robin dealing scatters them across shards and each
repeat is a cold prefill. (Within a single unsharded job they still cache fine — this only bites
once sharded.)

## Failure behaviour, measured against 16 shards

Shards resume, so a dead shard is a delay rather than a loss, and `03_run_sharded.sh` resubmits it
automatically up to `MAX_ROUNDS`. What it changes is the shape of a *partial* profile if you stop
early: because repeats scatter across shards, one dead shard costs a single repeat from about half
the tasks rather than all 8 repeats from a contiguous block. Which half depends on how `num_repeats`
divides into the shard count.

That is the good direction for reward profiling, which wants per-task variance — 7 of 8 repeats
still estimates it.
