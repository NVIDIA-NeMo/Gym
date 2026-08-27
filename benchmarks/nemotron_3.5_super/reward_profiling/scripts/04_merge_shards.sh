#!/bin/bash
# Unshard: concatenate every shard's rollouts back into the parent sweep.
#
#   SWEEP_DIR=<outputs/sweeps>/<nickname> bash .../scripts/04_merge_shards.sh
#
# Deduplicates on (_ng_task_index, _ng_rollout_index), the same key Gym resumes on, so a shard
# rerun after a walltime kill cannot double-count. Run this before resharding to a different
# NUM_SHARDS: 02_shard.sh carries work from the parent, so anything left only in the old shards
# would be recollected.
#
# 03_run_sharded.sh already does this at the end; use this directly when shards were launched
# individually, or to merge partial results mid-run.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
OUTPUT=${OUTPUT:-$SWEEP_DIR/rollouts.jsonl}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

cd "$REPO_ROOT"
python -m nemo_gym.sweep merge "$SHARDS_DIR" --output "$OUTPUT"
