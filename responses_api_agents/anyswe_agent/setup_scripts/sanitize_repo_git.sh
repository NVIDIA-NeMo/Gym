#!/bin/bash
# Leave the task repo with only HEAD and its ancestry. Task images often keep the upstream clone's
# other branches, tags and remotes, so `git log --all` would show the future fix commit.
# Repo paths come from the arguments. Without any, the usual task repo locations are tried.
[ "$#" -gt 0 ] || set -- /testbed /workspace/repo /app /root/repo
for repo in "$@"; do
  [ -d "$repo/.git" ] || continue
  cd "$repo" || continue
  git config --global --add safe.directory "$repo"
  base=$(git rev-parse --verify -q HEAD) || continue
  branch=$(git symbolic-ref -q --short HEAD || true)
  git for-each-ref --format='%(refname)' | while read -r ref; do git update-ref -d "$ref"; done
  for remote in $(git remote); do git remote remove "$remote"; done
  if [ -n "$branch" ]; then
    git update-ref "refs/heads/$branch" "$base"
  else
    git update-ref --no-deref HEAD "$base"
  fi
  rm -rf .git/logs .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD .git/refs/original
  rm -f .git/objects/info/alternates
  git reflog expire --expire=now --all
  # -a -d drops every object not reachable from HEAD or the index; prune clears loose ones.
  git -c gc.auto=0 -c repack.writeBitmaps=false repack -a -d -q
  git prune --expire=now
  echo "sanitized $repo at $base"
done
