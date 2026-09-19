#!/bin/bash
# All watcher-logic tests. Seconds, no cluster, no GPUs.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
for t in "$HERE"/test_*.sh; do
    echo "== $(basename "$t")"
    bash "$t" || rc=1
done
echo
[[ $rc -eq 0 ]] && echo "ALL WATCHER TESTS PASS" || echo "WATCHER TESTS FAILED"
exit $rc
