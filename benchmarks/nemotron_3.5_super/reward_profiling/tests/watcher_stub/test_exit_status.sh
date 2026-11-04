#!/bin/bash
# P0.3: a sweep that gave up must not exit 0 with a success banner.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../../scripts/03_run_sharded.sh"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass+1)); }
bad() { echo "  FAIL  $1 -- $2"; fail=$((fail+1)); }

grep -q 'incomplete+=(' "$SCRIPT" \
  && ok "T6a give-up is recorded" || bad "T6a give-up is recorded" "shards vanish silently"

# the assertion must sit AFTER the merge (partial data is still written) and exit non-zero
mline=$(grep -n 'merging shard rollouts' "$SCRIPT" | cut -d: -f1)
aline=$(grep -n 'INCOMPLETE:' "$SCRIPT" | cut -d: -f1)
[[ -n "$mline" && -n "$aline" && "$aline" -gt "$mline" ]] \
  && ok "T6b merge still runs before the assertion" || bad "T6b merge still runs before the assertion" "merge=$mline assert=$aline"

awk -v a="$aline" 'NR>a && /^ *exit 1/ {found=1} END {exit !found}' "$SCRIPT" \
  && ok "T6c exits non-zero when incomplete" || bad "T6c exits non-zero when incomplete" "no exit 1 after the banner"

# and the happy path must still be exit 0
awk '/^if \(\( \$\{#incomplete\[@\]\} > 0 \)\); then/,/^fi$/' "$SCRIPT" | grep -q 'exit 1' \
  && ok "T6d exit 1 is inside the incomplete branch only" || bad "T6d exit 1 is inside the incomplete branch only" "unconditional exit"

echo "  --- $pass passed, $fail failed"
exit $(( fail > 0 ))
