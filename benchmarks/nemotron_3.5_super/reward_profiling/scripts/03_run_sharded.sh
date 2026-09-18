#!/bin/bash
# 03 - Run one sweep across N jobs, watch them, resubmit failures, then merge and split.
# Each shard's own job profiles itself; run 05_profile.sh afterwards for the whole sweep.
#
# One job cannot use 256 nodes: --segment needs a topology-contiguous allocation and an NVL72 rack
# is 18 nodes. N identical jobs over disjoint slices is the shape that scales. (This used to also
# cite the aiohttp per-host connector limit; that was wrong -- the limit is per Gym server process
# and there are 63 of them, so it has never bound. See 03_run_single.sh's note on it.)
#
# Sharding also cuts the per-job driver preflight, which is a linear scan of the input before the
# first dispatch: ~20 min over the full 271.5 GiB, about a minute over a 1/16 shard, and it is paid
# again on every resubmission. Nodes = NUM_SHARDS x (prefill + decode), so 16 shards at the 2+8
# default is 160 nodes.
#
# USAGE
#   MODEL=<ckpt> \
#     SWEEP_DIR=<out>/<nickname> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
#
# REQUIRED + OPTIONAL
#   everything 03_run_single.sh takes -- it submits one 03_run_single.sh per shard -- plus:
#   NUM_SHARDS    how many jobs                            (default: 16)
#                 1 is legitimate: it is how a single-job run gets resubmission
#   SHARDS_DIR    where shard_NNN/ go                      (default: SWEEP_DIR/shards)
#   POLL_S        seconds between squeue checks            (default: 60)
#   EXPERIMENT_NAME  job-name prefix                       (default: rp)
#   GYM_SITE_PACKAGES a venv's site-packages, if orjson/yaml/pydantic are not importable
#   MAX_ROUNDS    attempts per shard, first submission included  (default: 100)
#                 Production needs 7-17: a shard is ~26 h of collection at 16 shards and a vLLM
#                 engine assertion ends a job every ~1.5 h.
#
# This runs in the FOREGROUND for hours, so detach it. It locks the sweep directory and refuses
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
# The sweep dir, never SHARDS_DIR: that is not defaulted until further down, so one watcher
# with it set and one without would lock different files and both run -- the pile-up this
# lock exists to prevent.
_lock_dir=${SWEEP_DIR:?}
mkdir -p "$_lock_dir" 2>/dev/null || true
exec {_lock_fd}>"$_lock_dir/.watcher.lock"
if ! flock -n "$_lock_fd"; then
    echo "ERROR: another 03_run_sharded.sh is already watching $_lock_dir." >&2
    echo "       Two watchers resubmit the same shards independently. Stop that one first:" >&2
    echo "         ps -eo pid,args | grep '[0]3_run_shard[e]d'" >&2
    exit 2
fi

# Only SWEEP_DIR. This script uses neither MODEL nor CONTAINER -- it passes the environment
# through to 03_run_single.sh, which resolves both from the shard's sweep_report.json when the
# manifest names them. Requiring them here rejects a manifest that supplies its own.
for _required in SWEEP_DIR; do
    if [[ -z "${!_required:-}" ]]; then
        echo "ERROR: $_required is required. See the header of $0." >&2
        exit 2
    fi
done

# num_shards from the manifest when this environment does not set it. Nodes = NUM_SHARDS x
# (vllm.prefill_nodes + vllm.decode_nodes), so 16 shards at the default 2+8 is 160 nodes.
# 2>/dev/null || true to match 02_shard.sh:28, which guards the identical block. Without it a torn
# sweep_report.json or a missing python aborts the whole run at startup.
_manifest_shards=$(python - "$SWEEP_DIR" 2>/dev/null <<'PY_SHARDS'
import json, sys
from pathlib import Path
try:
    print(json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text()).get("num_shards") or "")
except OSError:
    print("")
PY_SHARDS
) || _manifest_shards=""
NUM_SHARDS=${NUM_SHARDS:-${_manifest_shards:-16}}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$RP_DIR/../../.." && pwd)"
# Poll rather than `wait`: sbatch returns immediately, so there is no child to wait on.
POLL_S=${POLL_S:-60}
# Bounds resubmission of a shard that keeps dying for a permanent reason.
# Attempts per shard. This is the only stop condition: a shard finishes when it runs out of work,
# or when it has been retried this many times. 4 was far too low -- a production shard is ~26 h of
# collection at 16 shards while a vLLM engine assertion ends a job every ~1.5 h, so 7-17 attempts
# is the normal healthy case. Set it well above what you expect to need; a shard that is genuinely
# broken shows up as "no progress last attempt" in this log, repeatedly.
MAX_ROUNDS=${MAX_ROUNDS:-100}

cd "$REPO_ROOT"

# `infra` needs orjson/yaml/pydantic importable; it does not import nemo_gym at all. Inside the
# eval container that is automatic; on a
# login node it is not, and the failure is otherwise a bare ModuleNotFoundError from deep inside
# the CLI. GYM_SITE_PACKAGES points PYTHONPATH at a venv's site-packages if you are not in one.
if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
        echo "ERROR: cannot import infra and its deps (orjson, yaml, pydantic)." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi


# True when the shards still need dealing. Restarting the watcher must NOT re-deal an existing
# layout: _carry_existing_rollouts opens every shard's rollouts.jsonl with "w" (shard.py:239), so
# a re-deal truncates files that live jobs are appending to. The early-return at shard.py:235 only
# protects the very first run -- after the first absorb the parent file is non-empty forever, so
# every later restart truncates. Fails safe toward dealing: a missing or unreadable report means
# we cannot prove the layout is right, and dealing a fresh sweep is cheap.
need_reshard() {
    local shards_dir=$1 want=$2 verdict
    # Prints "deal" or "skip". Anything else -- python missing, crash -- also means deal.
    verdict=$(python - "$shards_dir" "$want" <<'INNER' 2>/dev/null
import json, sys
from pathlib import Path
try:
    have = json.loads((Path(sys.argv[1]) / "shard_report.json").read_text())["num_shards"]
except Exception:
    print("deal"); sys.exit(0)       # cannot prove the layout is right
print("skip" if have == int(sys.argv[2]) else "deal")
INNER
)
    [[ "$verdict" != "skip" ]]
}

# Independent of the count check: if any shard's rollouts.jsonl was written in the last hour, a job
# is almost certainly live and re-dealing would truncate it (shard.py:239 opens them "w"). This
# matters because NUM_SHARDS resolves to the manifest value (16) when the environment does not set
# it -- so restarting a NUM_SHARDS=8 run without that variable would otherwise compare 8 against 16,
# decide to deal, and truncate eight live shards.
# -mmin, not -newermt: this cluster's `find` is bfs, which rejects -newermt '-60 minutes' as an
# invalid timestamp -- and with stderr discarded that silently matched nothing, disabling the guard.
_recent=$(find "$SHARDS_DIR" -maxdepth 2 -name rollouts.jsonl -mmin -60 2>/dev/null | head -1) || _recent=""
if [[ -n "$_recent" ]]; then
    echo ">>> $SHARDS_DIR has rollouts written in the last hour; refusing to reshard (jobs are live)"
elif need_reshard "$SHARDS_DIR" "$NUM_SHARDS"; then
    echo ">>> dealing $SWEEP_DIR into $NUM_SHARDS shards"
    SWEEP_DIR="$SWEEP_DIR" NUM_SHARDS="$NUM_SHARDS" SHARDS_DIR="$SHARDS_DIR" \
        bash "$RP_DIR/scripts/02_shard.sh"
else
    echo ">>> already dealt $NUM_SHARDS ways; not resharding (jobs may be live)"
fi

# Work a shard has left, defined exactly as Gym defines it.
#
# This MUST match rollout_collection.py:1213-1216, which runs
#   inputs - (successes | terminal | maxed_out)
# where `terminal` and `maxed_out` come from rollouts_failures.jsonl and are deliberately never
# retried. Counting only successes -- which this did -- means any rollout that goes terminal (a
# judge 500, a sandbox timeout, three failed attempts) stays "outstanding" forever: the watcher
# resubmits a job that dispatches nothing, and the sweep has no terminating condition at all.
# Across 5.8M rollouts terminal failures are a certainty, so this affected every shard.
#
# Tolerates a torn final line -- a walltime-killed job leaves one, which is the exact event this
# function exists to observe -- and a missing rollouts.jsonl.
shard_outstanding() {
    python - "$1" "${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-3}" <<'INNER'
import json, sys
from collections import Counter
from pathlib import Path

d, max_attempts = Path(sys.argv[1]), int(sys.argv[2])

def rows(name, required=False):
    try:
        fh = open(d / name)
    except FileNotFoundError:
        if required:
            raise          # a missing inputs file must not read as "nothing left to do"
        return
    with fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue          # torn final line from a killed job

def key(r):
    return (r.get("_ng_task_index"), r.get("_ng_rollout_index"))

gated = {key(r) for r in rows("rollouts.jsonl")}

attempts, terminal = Counter(), set()
for r in rows("rollouts_failures.jsonl"):
    k = key(r)
    attempts[k] += 1
    if r.get("_ng_failure_terminal"):
        terminal.add(k)
gated |= terminal
gated |= {k for k, n in attempts.items() if n >= max_attempts}

print(sum(key(r) not in gated for r in rows("rollouts_materialized_inputs.jsonl", required=True)))
INNER
}

declare -a incomplete=()      # shards that ran out of attempts, for the exit status
declare -A shard_rounds
round=0
while :; do
    round=$((round + 1))
    unset live_jobs; declare -A live_jobs=()
    submitted=0

    # An unmatched glob leaves the literal pattern, every [[ -d ]] fails, submitted stays 0 and the
    # loop "finishes" having done nothing. A Lustre blip or a wrong SHARDS_DIR must be an error.
    compgen -G "$SHARDS_DIR/shard_*/" >/dev/null \
        || { echo "ERROR: no shard directories under $SHARDS_DIR" >&2; exit 2; }

    for shard_dir in "$SHARDS_DIR"/shard_*/; do
        [[ -d "$shard_dir" ]] || continue
        shard_name="$(basename "$shard_dir")"

        # Every "$(cmd)" below is guarded. This loop runs for days under set -euo pipefail, where
        # an unguarded assignment turns any transient failure into a silent end to the whole sweep:
        # the watcher exits, the running jobs finish, and nothing ever resubmits them.
        outstanding=$(shard_outstanding "$shard_dir") || outstanding=""
        if [[ ! "$outstanding" =~ ^[0-9]+$ ]]; then
            echo "    $shard_name: could not read outstanding; retrying next round" >&2
            submitted=$((submitted + 1))
            continue
        fi

        if [[ "$outstanding" -eq 0 ]]; then
            echo "    $shard_name complete"
            continue
        fi

        # A restart -- the normal case over a multi-day run -- must not submit a second job for a
        # shard that already has one. Two drivers appending to the same rollouts.jsonl interleave
        # inside multi-KB records, and merge silently drops the torn lines.
        _jobname="gym-${EXPERIMENT_NAME:-rp}-$shard_name-$USER"
        _running=$(squeue -h -u "$USER" -n "$_jobname" -o "%i" 2>/dev/null | head -1) || _running=""
        if [[ -z "${SBATCH_JOB_NAME:-}" && -n "$_running" ]]; then
            echo "    $shard_name already has job $_running; adopting it"
            live_jobs[$_running]=$shard_name
            submitted=$((submitted + 1))
            continue
        fi

        attempts=${shard_rounds[$shard_name]:-0}
        if [[ "$attempts" -ge "$MAX_ROUNDS" ]]; then
            echo "    $shard_name still has $outstanding outstanding after $attempts attempts; giving up" >&2
            [[ " ${incomplete[*]-} " == *" $shard_name:"* ]] || incomplete+=("$shard_name:$outstanding")
            continue
        fi

        if ! submit_output=$(
                EXPERIMENT_NAME="${EXPERIMENT_NAME:-rp}-$shard_name" \
                SWEEP_DIR="$shard_dir" \
                bash "$RP_DIR/scripts/03_run_single.sh" 2>&1); then
            echo "    $shard_name: submit failed, retrying next round: $(tail -1 <<<"$submit_output")" >&2
            submitted=$((submitted + 1))   # not done, so the round must not count as finished
            continue                       # and a failed submit is not an attempt
        fi
        # No $ anchor: a federated sbatch prints "Submitted batch job N on cluster foo".
        job_id=$(grep -oE '[0-9]+' <<<"$submit_output" | tail -1) || job_id=""
        if [[ -z "$job_id" ]]; then
            echo "    $shard_name: no job id in submit output; retrying next round" >&2
            submitted=$((submitted + 1))
            continue
        fi
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
    while (( ${#live_jobs[@]} > 0 )); do
        # Count our ids among the user's live jobs rather than asking squeue about specific ones.
        # `squeue -j <id>` exits 1 once an id is purged (MinJobAge=300 here), and with pipefail
        # that status propagates past wc -l and kills the watcher -- which happens deterministically
        # as soon as one shard finishes while its siblings are still running.
        live=$(squeue -h -u "$USER" -o "%i" 2>/dev/null \
               | grep -cxF -f <(printf '%s\n' "${!live_jobs[@]}")) || live=0
        [[ "$live" -eq 0 ]] && break
        sleep "$POLL_S"
    done
    echo ">>> round $round done; rechecking for shards that died with work outstanding"
done

echo ">>> merging shard rollouts back into $SWEEP_DIR"
SWEEP_DIR="$SWEEP_DIR" SHARDS_DIR="$SHARDS_DIR" OUTPUT="$SWEEP_DIR/rollouts.jsonl" \
    bash "$RP_DIR/scripts/04_merge_shards.sh" || echo "WARNING: merge failed" >&2

echo ">>> splitting by manifest entry"
PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m infra split "$SWEEP_DIR" \
    || echo "WARNING: split failed" >&2

echo
echo "Merged rollouts : $SWEEP_DIR/rollouts.jsonl"
echo "Per-entry output: $SWEEP_DIR/by_label/"
echo "Profile the whole sweep, or any single entry, with:"
echo "  gym eval profile --inputs <dir>/rollouts_materialized_inputs.jsonl \\"
echo "                   --rollouts <dir>/rollouts.jsonl ++allow_partial_rollouts=True"

# Merging is unconditional because partial data is still worth having, but the exit status must
# not be. Previously this printed the same banner and exited 0 whether every shard finished or
# every shard exhausted its attempts -- and since ALLOW_PARTIAL_ROLLOUTS defaults to True, a
# 60%-complete sweep then profiles cleanly and reports a reward number. A partial sweep that is
# indistinguishable from a finished one is the worst outcome this pipeline can produce.
if (( ${#incomplete[@]} > 0 )); then
    echo
    echo "INCOMPLETE: ${#incomplete[@]} shard(s) ran out of attempts with work outstanding." >&2
    for _entry in "${incomplete[@]}"; do
        echo "  ${_entry%%:*}: ${_entry##*:} rollouts never collected" >&2
    done
    echo "The merge above contains only what was collected. Do not treat it as a full sweep." >&2
    exit 1
fi
