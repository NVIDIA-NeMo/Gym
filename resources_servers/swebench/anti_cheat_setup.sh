_T0=$(date +%s 2>/dev/null || echo 0)
if ! cd $WORKING_DIRECTORY 2>/dev/null; then echo "GIT_CLEANUP skipped reason=no_workdir wd=$WORKING_DIRECTORY"; exit 0; fi
git config --global --add safe.directory $WORKING_DIRECTORY 2>/dev/null
if ! git rev-parse --git-dir >/dev/null 2>&1; then echo "GIT_CLEANUP skipped reason=not_a_git_repo wd=$WORKING_DIRECTORY"; exit 0; fi
_BASE=$(git rev-parse HEAD 2>/dev/null)
if [ -z "$_BASE" ]; then echo "GIT_CLEANUP skipped reason=no_head wd=$WORKING_DIRECTORY"; exit 0; fi
_BEFORE=$(git rev-list --all --count 2>/dev/null || echo '?')
_LOOSE_B=$(git count-objects -v 2>/dev/null | sed -n 's/^count: //p')
_PACK_B=$(git count-objects -v 2>/dev/null | sed -n 's/^in-pack: //p')
mkdir -p .git/refs/heads
echo "$_BASE" > .git/refs/heads/_nel_work
git symbolic-ref HEAD refs/heads/_nel_work 2>/dev/null
rm -f .git/packed-refs .git/ORIG_HEAD .git/FETCH_HEAD .git/MERGE_HEAD
find .git/refs -type f ! -path '*/heads/_nel_work' -delete 2>/dev/null
rm -rf .git/refs/tags .git/refs/remotes .git/logs
for _r in $(git remote 2>/dev/null); do git remote remove "$_r" 2>/dev/null; done
git reflog expire --expire=now --expire-unreachable=now --all 2>/dev/null
git gc --prune=now --quiet 2>/dev/null
_GC_RC=$?
_AFTER=$(git rev-list --all --count 2>/dev/null || echo '?')
_LEFT=$(git rev-list --all --not "$_BASE" --count 2>/dev/null || echo '?')
_LOOSE_A=$(git count-objects -v 2>/dev/null | sed -n 's/^count: //p')
_PACK_A=$(git count-objects -v 2>/dev/null | sed -n 's/^in-pack: //p')
_ELAPSED=$(( $(date +%s 2>/dev/null || echo 0) - _T0 ))
echo "GIT_CLEANUP base=$_BASE commits_before=$_BEFORE commits_after=$_AFTER non_ancestor_left=$_LEFT gc_rc=$_GC_RC objects_before=$_LOOSE_B+$_PACK_B objects_after=$_LOOSE_A+$_PACK_A elapsed_s=$_ELAPSED"
