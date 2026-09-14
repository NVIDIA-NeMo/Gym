#!/bin/bash
# The watcher runs set -euo pipefail for days. Every construct below, as written in the script,
# must survive a transient failure instead of killing the run. Each case is the real line.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../../scripts/03_run_sharded.sh"
pass=0; fail=0
ok()  { echo "  PASS  $1"; pass=$((pass+1)); }
bad() { echo "  FAIL  $1 -- $2"; fail=$((fail+1)); }

# D1: a purged job id makes squeue exit 1; pipefail carries it past wc -l
grep -q 'grep -cxF -f' "$SCRIPT" \
  && ok "D1 squeue poll tolerates purged/unknown job ids" \
  || bad "D1 squeue poll tolerates purged/unknown job ids" "still 'squeue | wc -l', dies on rc=1"

# D2: a transient sbatch rejection must not end the sweep
grep -q 'if ! submit_output=' "$SCRIPT" \
  && ok "D2 failed submit is caught" || bad "D2 failed submit is caught" "bare \$( ) assignment is fatal"

# D3: grep with no match exits 1 under pipefail
grep -qE 'job_id=\$\(.*\) \|\| job_id=' "$SCRIPT" \
  && ok "D3 job-id capture tolerates no match" || bad "D3 job-id capture tolerates no match" "unguarded grep"

# D4: the probe must not kill the watcher, and must not read 'complete' from a missing inputs file
grep -q 'outstanding=$(shard_outstanding "$shard_dir") || outstanding=' "$SCRIPT" \
  && ok "D4a probe failure is caught" || bad "D4a probe failure is caught" "unguarded assignment"
grep -q 'required=True\|, True)' "$SCRIPT" \
  && ok "D4b missing inputs file is an error, not 'complete'" \
  || bad "D4b missing inputs file is an error, not 'complete'" "silently reports 0 outstanding"

# D5: after a restart, a shard whose job is still running must not get a second one
grep -q 'already has job' "$SCRIPT" \
  && ok "D5 adopts a running job instead of double-submitting" \
  || bad "D5 adopts a running job instead of double-submitting" "two writers on one rollouts.jsonl"

# D6: an unmatched glob must not read as 'nothing left to do'
grep -q 'compgen -G' "$SCRIPT" \
  && ok "D6 missing shard dirs is an error, not success" \
  || bad "D6 missing shard dirs is an error, not success" "merges an empty sweep and exits 0"

echo "  --- $pass passed, $fail failed"
exit $(( fail > 0 ))
