#!/bin/bash
# Assert the shard / reshard / merge / split / profile invariants against a real sweep.
#
# Run after changing anything in nemo_gym/sweep/. Read-only with respect to SWEEP_DIR: everything
# happens on a scratch copy.
#
# USAGE
#   SWEEP_DIR=<a sweep with rollouts> bash $R/scripts/debug_selftest.sh
#
# REQUIRED
#   SWEEP_DIR     a sweep directory with a non-empty rollouts.jsonl
#
# OPTIONAL
#   SHAPES        shard counts to cycle through           (default: "4 7 2 9 3 5")
#   PROFILE_JOBS  passed to 05_profile.sh                 (default: 12)
#   KEEP=1        leave the scratch copy behind for inspection
#
# Asserts after every reshape that inputs partition exactly, every rollout sits in the shard
# owning its input, collected work survives, and merge round-trips; then that the split attributes
# every row to an entry inside its task_index_range, and that profiling yields one row per source
# task. Uses real collected rollouts rather than synthetic rows, because the failures worth
# catching only appear once inputs and rollouts disagree the way collection makes them disagree.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to a sweep directory that has rollouts.jsonl}
SHAPES=${SHAPES:-"4 7 2 9 3 5"}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -s "$SWEEP_DIR/rollouts.jsonl" ]]; then
    echo "ERROR: $SWEEP_DIR/rollouts.jsonl is empty. Merge shards first, or point at a finished sweep." >&2
    exit 2
fi

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/sweep_selftest_XXXXXX")
cleanup() { [[ "${KEEP:-0}" == "1" ]] || rm -rf "$SCRATCH"; }
trap cleanup EXIT

echo ">>> copying $SWEEP_DIR to scratch"
mkdir -p "$SCRATCH/sweep"
for f in rollouts.jsonl rollouts_materialized_inputs.jsonl sweep_config.yaml sweep_report.json; do
    cp "$SWEEP_DIR/$f" "$SCRATCH/sweep/$f"
done
D="$SCRATCH/sweep"

python - "$D" "$SHAPES" <<'PY'
import json, subprocess, sys
from pathlib import Path

d, shapes = Path(sys.argv[1]), [int(n) for n in sys.argv[2].split()]
TASK, ROLLOUT = "_ng_task_index", "_ng_rollout_index"
failures = []


def keys(fpath):
    """The (task, rollout) key multiset of a jsonl file."""
    out = []
    if not Path(fpath).is_file():
        return out
    with open(fpath) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out.append((row.get(TASK), row.get(ROLLOUT)))
    return out


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail and not ok else ''}")
    if not ok:
        failures.append(name)


baseline_inputs = sorted(keys(d / "rollouts_materialized_inputs.jsonl"))
baseline_rollouts = sorted(keys(d / "rollouts.jsonl"))
print(f"\nbaseline: {len(baseline_inputs):,} inputs, {len(baseline_rollouts):,} rollouts\n")
check("baseline inputs are unique", len(set(baseline_inputs)) == len(baseline_inputs))
check("baseline rollouts are unique", len(set(baseline_rollouts)) == len(baseline_rollouts))
check("every rollout has an input", set(baseline_rollouts) <= set(baseline_inputs))

# Rollouts live in the parent to start with. Each reshard must fold whatever the previous layout
# collected back into the parent before rewriting shard directories, so nothing is stranded.
for n in shapes:
    print(f"\n--- reshard to {n} ---")
    subprocess.run([sys.executable, "-m", "nemo_gym.sweep", "shard", str(d), "--num-shards", str(n)],
                   check=True, capture_output=True)
    shard_dirs = sorted(p for p in (d / "shards").glob("shard_*") if p.is_dir())

    check(f"{n} shard dirs exist, no stale ones", len(shard_dirs) == n, f"found {len(shard_dirs)}")

    all_in, all_out = [], []
    misfiled = 0
    for s in shard_dirs:
        si, so = keys(s / "rollouts_materialized_inputs.jsonl"), keys(s / "rollouts.jsonl")
        all_in += si
        all_out += so
        owned = set(si)
        misfiled += sum(1 for k in so if k not in owned)
    check("inputs partition exactly, no loss or duplication", sorted(all_in) == baseline_inputs,
          f"{len(all_in)} vs {len(baseline_inputs)}")
    check("every rollout sits in the shard that owns its input", misfiled == 0, f"{misfiled} misfiled")
    check("collected work survives the reshape", sorted(all_out) == baseline_rollouts,
          f"{len(all_out)} vs {len(baseline_rollouts)}")

    # Balance: round-robin should not leave a shard starved, which would waste a whole allocation.
    sizes = [len(keys(s / "rollouts_materialized_inputs.jsonl")) for s in shard_dirs]
    check("shards are balanced to within one row", max(sizes) - min(sizes) <= 1, f"sizes {sizes}")

    merged = subprocess.run([sys.executable, "-m", "nemo_gym.sweep", "merge", str(d / "shards"),
                             "--output", str(d / "rollouts.jsonl")],
                            check=True, capture_output=True, text=True)
    check("merge round-trips to the original rollout set",
          sorted(keys(d / "rollouts.jsonl")) == baseline_rollouts, merged.stdout.strip()[-80:])

print("\n--- split ---")
subprocess.run([sys.executable, "-m", "nemo_gym.sweep", "split", str(d)], check=True, capture_output=True)
report = json.loads((d / "by_label" / "split_report.json").read_text())
labels = report["labels"]
check("no row is unattributable to an entry",
      report["unmapped_inputs"] == 0 and report["unmapped_rollouts"] == 0,
      f"{report['unmapped_inputs']} inputs, {report['unmapped_rollouts']} rollouts")
check("per-label rollouts sum to the whole",
      sum(v["rollouts"] for v in labels.values()) == len(baseline_rollouts))
check("per-label inputs sum to the whole",
      sum(v["inputs"] for v in labels.values()) == len(baseline_inputs))
check("no label silently collected nothing", not report["labels_without_rollouts"],
      str(report["labels_without_rollouts"]))

# A label's rows must all fall inside the task_index_range the report claims for it, or reward
# profiling attributes an environment's scores to a different environment.
ranges = {k: v["task_index_range"] for k, v in json.loads((d / "sweep_report.json").read_text())["entries"].items()}
out_of_range = 0
for label, meta in labels.items():
    lo, hi = ranges[label]
    for t, _ in keys(Path(meta["dir"]) / "rollouts.jsonl"):
        if not (lo <= t <= hi):
            out_of_range += 1
check("every split row falls in its entry's task_index_range", out_of_range == 0, f"{out_of_range} stray")

print("\n" + ("FAILED: " + ", ".join(failures) if failures else "ALL INVARIANTS HELD"))
sys.exit(1 if failures else 0)
PY

echo
echo ">>> profiling the resharded copy (final shape check)"
SWEEP_DIR="$D" PROFILE_JOBS="${PROFILE_JOBS:-12}" bash "$(dirname "${BASH_SOURCE[0]}")/05_profile.sh" >"$SCRATCH/profile.log" 2>&1 || true

python - "$D" "$SWEEP_DIR" <<'PY'
import json, sys
from pathlib import Path
d, orig = Path(sys.argv[1]), Path(sys.argv[2])
src_tasks = len(json.loads((d / "sweep_report.json").read_text())["entries"]) and len(
    {json.loads(l)["_ng_task_index"] for l in open(d / "rollouts_materialized_inputs.jsonl") if l.strip()})
rows = [json.loads(l) for l in open(d / "rollouts_reward_profiling.jsonl") if l.strip()]
ok = len(rows) == src_tasks
print(f"  {'PASS' if ok else 'FAIL'}  profiled output is one row per source task ({len(rows)} vs {src_tasks})")
need = {"mean/reward", "std/reward", "min/reward", "max/reward", "num_rollouts", "sample"}
missing = need - set(rows[0]) if rows else need
print(f"  {'PASS' if not missing else 'FAIL'}  each row carries the reward distribution"
      + (f"  -- missing {sorted(missing)}" if missing else ""))
sys.exit(0 if ok and not missing else 1)
PY

echo
echo "SELFTEST PASSED"
