#!/bin/bash
# P0.2: restarting the watcher must not re-deal shards that are already dealt, because
# _carry_existing_rollouts opens every shard's rollouts.jsonl with "w" (shard.py:239) --
# truncating files that live jobs are appending to.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../../scripts/03_run_sharded.sh"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass+1)); }
bad() { echo "  FAIL  $1 -- $2"; fail=$((fail+1)); }

eval "$(sed -n '/^need_reshard() {/,/^}/p' "$SCRIPT")" 2>/dev/null
if ! type need_reshard >/dev/null 2>&1; then
    echo "  FAIL  need_reshard() does not exist -- watcher reshards unconditionally"
    echo "  --- 0 passed, 1 failed"; exit 1
fi

# already dealt at the requested count -> must NOT reshard
d=/tmp/rs_match; bash "$HERE/mkfixture.sh" $d 2 5
need_reshard "$d/shards" 2 && bad "T3a skips reshard when layout matches" "wants to reshard" \
                           || ok "T3a skips reshard when layout matches"

# dealt at a different count -> MUST reshard
d=/tmp/rs_diff; bash "$HERE/mkfixture.sh" $d 2 5
printf '{"num_shards":4}\n' > "$d/shards/shard_report.json"
need_reshard "$d/shards" 2 && ok "T3b reshards when the count changed" \
                           || bad "T3b reshards when the count changed" "skipped it"

# no report at all -> MUST reshard
d=/tmp/rs_none; bash "$HERE/mkfixture.sh" $d 2 5; rm -f "$d/shards/shard_report.json"
need_reshard "$d/shards" 2 && ok "T3c reshards when never dealt" \
                           || bad "T3c reshards when never dealt" "skipped it"

# corrupt report -> MUST reshard (fail safe toward doing the work)
d=/tmp/rs_bad; bash "$HERE/mkfixture.sh" $d 2 5
printf 'not json' > "$d/shards/shard_report.json"
need_reshard "$d/shards" 2 && ok "T3d reshards when the report is corrupt" \
                           || bad "T3d reshards when the report is corrupt" "skipped it"

echo "  --- $pass passed, $fail failed"
exit $(( fail > 0 ))
