#!/bin/bash
# Exercises shard_outstanding() as it actually exists in 03_run_sharded.sh.
# Run: bash test_probe.sh    (exit 0 = all pass)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../../scripts/03_run_sharded.sh"
pass=0; fail=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1 -- $2"; fail=$((fail+1)); }

# Pull the real function out of the launcher, so this tests shipped code.
eval "$(sed -n '/^shard_outstanding() {/,/^}/p' "$SCRIPT")"
type shard_outstanding >/dev/null 2>&1 || { echo "could not extract shard_outstanding"; exit 2; }

mk() { bash "$HERE/mkfixture.sh" "$1" 1 "${2:-10}"; }

# --- T4: rows Gym has permanently gated must not count as outstanding -------------
d=/tmp/probe_t4; mk $d 10; s=$d/shards/shard_000
for k in 0 1 2 3 4 5 6; do printf '{"_ng_task_index":%d,"_ng_rollout_index":0}\n' $k; done > $s/rollouts.jsonl
# tasks 7,8,9 each failed 3 times -> Gym gates them, will never dispatch again
for k in 7 8 9; do for _ in 1 2 3; do printf '{"_ng_task_index":%d,"_ng_rollout_index":0}\n' $k; done; done > $s/rollouts_failures.jsonl
out=$(shard_outstanding "$s" 2>/dev/null)
[[ "$out" == "0" ]] && ok "T4 gated rows count as done" || bad "T4 gated rows count as done" "got '$out', want 0 (shard loops forever)"

# --- T1: a torn final line must not kill the probe --------------------------------
d=/tmp/probe_t1; mk $d 10; s=$d/shards/shard_000
printf '{"_ng_task_index":0,"_ng_rollout_index":0}\n' > $s/rollouts.jsonl
printf '{"_ng_task_ind' >> $s/rollouts.jsonl        # walltime-killed job leaves this
out=$(shard_outstanding "$s" 2>/dev/null); rc=$?
[[ $rc -eq 0 && -n "$out" ]] && ok "T1 survives a torn final line" || bad "T1 survives a torn final line" "rc=$rc out='$out'"

# --- T10: a missing rollouts.jsonl must not kill the probe ------------------------
d=/tmp/probe_t10; mk $d 10; s=$d/shards/shard_000; rm -f $s/rollouts.jsonl
out=$(shard_outstanding "$s" 2>/dev/null); rc=$?
[[ $rc -eq 0 && "$out" == "10" ]] && ok "T10 survives a missing rollouts.jsonl" || bad "T10 survives a missing rollouts.jsonl" "rc=$rc out='$out'"

# --- T11: the happy path still works ----------------------------------------------
d=/tmp/probe_t11; mk $d 10; s=$d/shards/shard_000
for k in 0 1 2; do printf '{"_ng_task_index":%d,"_ng_rollout_index":0}\n' $k; done > $s/rollouts.jsonl
out=$(shard_outstanding "$s" 2>/dev/null)
[[ "$out" == "7" ]] && ok "T11 counts plain outstanding correctly" || bad "T11 counts plain outstanding correctly" "got '$out', want 7"

echo "  --- $pass passed, $fail failed"
exit $(( fail > 0 ))
