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
#   bash benchmarks/nemotron_3.5_super/reward_profiling/scripts/03_run_sharded.sh
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
# Bounds resubmission of a shard that keeps dying for a permanent reason.
MAX_ROUNDS=${MAX_ROUNDS:-4}

cd "$REPO_ROOT"

# nemo_gym needs its dependencies importable. Inside the eval container that is automatic; on a
# login node it is not, and the failure is otherwise a bare ModuleNotFoundError from deep inside
# the CLI. GYM_SITE_PACKAGES points PYTHONPATH at a venv's site-packages if you are not in one.
if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
        echo "ERROR: cannot import nemo_gym and its deps." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi


echo ">>> dealing $SWEEP_DIR into $NUM_SHARDS shards"
python -m nemo_gym.sweep shard "$SWEEP_DIR" --num-shards "$NUM_SHARDS" --out-dir "$SHARDS_DIR"

# Outstanding rollouts for a shard: inputs whose (task, rollout) pair is not yet in its
# rollouts.jsonl.
shard_outstanding() {
    python - "$1" <<'INNER'
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
done_pairs = set()
for line in open(d / "rollouts.jsonl"):
    if line.strip():
        r = json.loads(line)
        done_pairs.add((r.get("_ng_task_index"), r.get("_ng_rollout_index")))

outstanding = 0
for line in open(d / "rollouts_materialized_inputs.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    outstanding += (r.get("_ng_task_index"), r.get("_ng_rollout_index")) not in done_pairs
print(outstanding)
INNER
}

declare -A shard_rounds
round=0
while :; do
    round=$((round + 1))
    unset live_jobs; declare -A live_jobs=()
    submitted=0

    for shard_dir in "$SHARDS_DIR"/shard_*/; do
        [[ -d "$shard_dir" ]] || continue
        shard_name="$(basename "$shard_dir")"
        outstanding=$(shard_outstanding "$shard_dir")

        if [[ "$outstanding" -eq 0 ]]; then
            echo "    $shard_name complete"
            continue
        fi
        attempts=${shard_rounds[$shard_name]:-0}
        if [[ "$attempts" -ge "$MAX_ROUNDS" ]]; then
            echo "    $shard_name still has $outstanding outstanding after $attempts attempts; giving up" >&2
            continue
        fi

        submit_output=$(
            EXPERIMENT_NAME="${EXPERIMENT_NAME:-rp}-$shard_name" \
            SWEEP_DIR="$shard_dir" \
            bash "$RP_DIR/scripts/03_run.sh"
        )
        job_id=$(grep -oE '[0-9]+$' <<<"$submit_output" | tail -1)
        live_jobs[$job_id]=$shard_name
        shard_rounds[$shard_name]=$((attempts + 1))
        submitted=$((submitted + 1))
        echo "    $shard_name -> job $job_id (attempt $((attempts + 1)), $outstanding outstanding)"
    done

    if [[ "$submitted" -eq 0 ]]; then
        echo ">>> round $round: nothing left to submit"
        break
    fi

    echo ">>> round $round: waiting on $submitted job(s)"
    while :; do
        live=$(squeue -h -j "$(IFS=,; echo "${!live_jobs[*]}")" -o "%i" 2>/dev/null | wc -l)
        [[ "$live" -eq 0 ]] && break
        sleep "$POLL_S"
    done
    echo ">>> round $round done; rechecking for shards that died with work outstanding"
done

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
