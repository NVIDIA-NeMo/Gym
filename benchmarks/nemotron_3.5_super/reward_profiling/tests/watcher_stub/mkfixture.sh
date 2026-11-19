#!/bin/bash
# Build a tiny sweep directory that the watcher will accept: N shards, M rollouts each.
# Everything is 2-field JSON -- the watcher only ever reads _ng_task_index / _ng_rollout_index.
set -euo pipefail
D=${1:?usage: mkfixture.sh <dir> [num_shards] [rows_per_shard]}
N=${2:-2}; M=${3:-10}
rm -rf "$D"; mkdir -p "$D/shards"
printf '{"nickname":"stub","num_shards":%d,"entries":{}}\n' "$N" > "$D/sweep_report.json"
printf '{"num_shards":%d}\n' "$N" > "$D/shards/shard_report.json"
: > "$D/rollouts.jsonl"
for ((i=0;i<N;i++)); do
    s=$(printf "$D/shards/shard_%03d" "$i"); mkdir -p "$s"
    : > "$s/rollouts.jsonl"; : > "$s/rollouts_failures.jsonl"
    for ((k=0;k<M;k++)); do
        printf '{"_ng_task_index":%d,"_ng_rollout_index":%d}\n' "$((i*M+k))" 0
    done > "$s/rollouts_materialized_inputs.jsonl"
done
