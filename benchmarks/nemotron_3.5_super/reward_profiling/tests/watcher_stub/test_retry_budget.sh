#!/bin/bash
# Rollouts must never be silently abandoned, and the two scripts must agree on the budget.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE="$HERE/../../scripts/03_run_single.sh"; SHARDED="$HERE/../../scripts/03_run_sharded.sh"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass+1)); }
bad() { echo "  FAIL  $1 -- $2"; fail=$((fail+1)); }

a=$(grep -oE 'MAX_ROLLOUT_ATTEMPTS:-[0-9]+' "$SINGLE"  | head -1 | cut -d- -f2)
b=$(grep -oE 'MAX_ROLLOUT_ATTEMPTS:-[0-9]+' "$SHARDED" | head -1 | cut -d- -f2)
[[ -n "$a" && "$a" == "$b" ]] && ok "T12a both scripts default to the same budget ($a)" \
                              || bad "T12a both scripts default to the same budget" "single=$a sharded=$b"
[[ -n "$a" && "$a" -ge 100 ]] && ok "T12b budget is high enough not to drop hard rollouts" \
                              || bad "T12b budget is high enough not to drop hard rollouts" "got $a; Gym's default 3 drops them"
grep -q 'export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS' "$SINGLE" \
  && ok "T12c budget is exported into the eval container" || bad "T12c budget is exported into the eval container" "Gym reads os.environ"

# the probe must honour the env var rather than hardcoding 3
eval "$(sed -n '/^shard_outstanding() {/,/^}/p' "$SHARDED")"
d=/tmp/budget_t; bash "$HERE/mkfixture.sh" $d 1 5; s=$d/shards/shard_000
: > $s/rollouts.jsonl
for k in 0 1 2 3 4; do for _ in 1 2 3; do printf '{"_ng_task_index":%d,"_ng_rollout_index":0}\n' $k; done; done > $s/rollouts_failures.jsonl
got_hi=$(NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=1000 shard_outstanding "$s")
got_lo=$(NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3    shard_outstanding "$s")
[[ "$got_hi" == "5" ]] && ok "T12d with a high budget, 3x-failed rows are still outstanding" \
                       || bad "T12d with a high budget, 3x-failed rows are still outstanding" "got $got_hi, want 5"
[[ "$got_lo" == "0" ]] && ok "T12e with budget 3, Gym has gated them (watcher agrees)" \
                       || bad "T12e with budget 3, Gym has gated them (watcher agrees)" "got $got_lo, want 0"

echo "  --- $pass passed, $fail failed"
exit $(( fail > 0 ))
