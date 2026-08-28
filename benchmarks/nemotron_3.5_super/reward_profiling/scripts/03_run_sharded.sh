#!/bin/bash
# 03 - Run one sweep across N jobs, watch them, resubmit failures, then merge and split.
# Each shard's own job profiles itself; run 05_profile.sh afterwards for the whole sweep.
#
# One job cannot use 256 nodes: --segment needs a topology-contiguous allocation and an NVL72 rack
# is 18 nodes, and a single driver at 512 x decode_nodes concurrency would exceed the aiohttp
# per-host connector limit long before the GPUs saturated. N identical jobs over disjoint slices is
# the shape that scales. At the 16-node default, NUM_SHARDS=16 is a 256-node run.
#
# USAGE
#   MODEL=<ckpt> CONTAINER=<sqsh> SANDBOX_CONTAINER=<sandbox sqsh> \
#     SWEEP_DIR=<out>/<nickname> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
#
# REQUIRED + OPTIONAL
#   everything 03_run_single.sh takes -- it submits one 03_run_single.sh per shard -- plus:
#   NUM_SHARDS    how many jobs                            (default: 16)
#                 1 is legitimate: it is how a single-job run gets resubmission
#   SHARDS_DIR    where shard_NNN/ go                      (default: SWEEP_DIR/shards)
#   POLL_S        seconds between squeue checks            (default: 60)
#   EXPERIMENT_NAME  job-name prefix                       (default: rp)
#   GYM_SITE_PACKAGES a venv's site-packages, if nemo_gym is not already importable
#   MAX_ROUNDS    resubmissions per shard before giving up (default: 4)
#
# This runs in the FOREGROUND for hours, so detach it. It locks the shards directory and refuses
# to start twice -- each watcher resubmits independently, so several pile up jobs against the
# node limit:
#
#   setsid nohup bash -lc "... bash $R/scripts/03_run_sharded.sh" > watcher.log 2>&1 &
#
# Safe to re-run. Shards carry whatever the parent has already collected, and merge deduplicates
# on the same (task, rollout) key Gym resumes on, so a rerun resumes rather than recollecting. To
# change the shard count mid-run: merge first, then re-run with a new NUM_SHARDS -- shard_sweep
# carries from the parent, not from the old shards.
set -euo pipefail

# One watcher per sweep. Each instance resubmits dead shards independently, so a second one
# doubles the submissions and they pile up against the account's node limit -- observed with three
# watchers holding six jobs against a four-node cap. The lock is the sweep directory, so different
# sweeps still run concurrently. flock releases it when this shell exits, however it exits.
_lock_dir=${SHARDS_DIR:-${SWEEP_DIR:-/tmp}}
mkdir -p "$_lock_dir" 2>/dev/null || true
exec {_lock_fd}>"$_lock_dir/.watcher.lock"
if ! flock -n "$_lock_fd"; then
    echo "ERROR: another 03_run_sharded.sh is already watching $_lock_dir." >&2
    echo "       Two watchers resubmit the same shards independently. Stop that one first:" >&2
    echo "         ps -eo pid,args | grep '[0]3_run_shard[e]d'" >&2
    exit 2
fi

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
            bash "$RP_DIR/scripts/03_run_single.sh"
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
