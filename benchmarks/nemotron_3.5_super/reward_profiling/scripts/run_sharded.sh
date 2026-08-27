#!/bin/bash
# Run one sweep across several jobs, then merge the results back.
#
# One job cannot use 256 nodes: --segment needs a topology-contiguous allocation and an NVL72 rack
# is 18 nodes, and a single driver at 512 x decode_nodes concurrency would exceed the aiohttp
# per-host connector limit long before the GPUs saturated. N identical jobs over disjoint slices is
# the shape that scales. With the P4D12 default of 16 nodes, NUM_SHARDS=16 is a 256-node run.
#
#   MODEL=<checkpoint> CONTAINER=<sqsh> SWEEP_DIR=<out>/<nickname> NUM_SHARDS=16 \
#   SANDBOX_CONTAINER=<sandbox sqsh> \
#   bash benchmarks/nemotron_3.5_super/reward_profiling/scripts/run_sharded.sh
#
# Safe to re-run. Shards carry whatever the parent sweep has already collected, and merge
# deduplicates on the same (task, rollout) key Gym resumes on, so a rerun resumes rather than
# recollecting. To change the shard count mid-run: merge first, then re-run this with a new
# NUM_SHARDS -- shard_sweep carries from the parent, not from the old shards.
set -euo pipefail

for _required in MODEL CONTAINER SWEEP_DIR; do
    if [[ -z "${!_required:-}" ]]; then
        echo "ERROR: $_required is required. See the header of $0." >&2
        exit 2
    fi
done

NUM_SHARDS=${NUM_SHARDS:-16}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RP_DIR/../../.." && pwd)"
# Poll rather than `wait`: sbatch returns immediately, so there is no child to wait on.
POLL_S=${POLL_S:-60}

cd "$REPO_ROOT"

echo ">>> dealing $SWEEP_DIR into $NUM_SHARDS shards"
python -m nemo_gym.sweep shard "$SWEEP_DIR" --num-shards "$NUM_SHARDS" --out-dir "$SHARDS_DIR"

job_ids=()
for shard_dir in "$SHARDS_DIR"/shard_*/; do
    [[ -d "$shard_dir" ]] || continue
    shard_name="$(basename "$shard_dir")"
    remaining=$(python - "$shard_dir" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
key = lambda r: (r.get("_ng_task_index"), r.get("_ng_rollout_index"))
done = {key(json.loads(l)) for l in open(d / "rollouts.jsonl") if l.strip()}
todo = sum(1 for l in open(d / "rollouts_materialized_inputs.jsonl")
           if l.strip() and key(json.loads(l)) not in done)
print(todo)
PY
)
    if [[ "$remaining" -eq 0 ]]; then
        echo "    $shard_name already complete; not submitting"
        continue
    fi

    submit_output=$(
        EXPERIMENT_NAME="${EXPERIMENT_NAME:-rp}-$shard_name" \
        SWEEP_DIR="$shard_dir" \
        bash "$RP_DIR/scripts/sbatch_reward_profiling.sh"
    )
    job_id=$(grep -oE '[0-9]+$' <<<"$submit_output" | tail -1)
    job_ids+=("$job_id")
    echo "    $shard_name -> job $job_id ($remaining rollouts outstanding)"
done

if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo ">>> nothing to run; every shard was already complete"
else
    echo ">>> waiting on ${#job_ids[@]} jobs: ${job_ids[*]}"
    while :; do
        live=$(squeue -h -j "$(IFS=,; echo "${job_ids[*]}")" -o "%i" 2>/dev/null | wc -l)
        [[ "$live" -eq 0 ]] && break
        echo "    $live/${#job_ids[@]} still running"
        sleep "$POLL_S"
    done
fi

echo ">>> merging shard rollouts back into $SWEEP_DIR"
python -m nemo_gym.sweep merge "$SHARDS_DIR" --output "$SWEEP_DIR/rollouts.jsonl"

echo ">>> splitting by manifest entry"
python -m nemo_gym.sweep split "$SWEEP_DIR"

echo
echo "Merged rollouts : $SWEEP_DIR/rollouts.jsonl"
echo "Per-entry output: $SWEEP_DIR/by_label/"
echo "Profile the whole sweep, or any single entry, with:"
echo "  gym eval profile --inputs <dir>/rollouts_materialized_inputs.jsonl \\"
echo "                   --rollouts <dir>/rollouts.jsonl ++allow_partial_rollouts=True"
