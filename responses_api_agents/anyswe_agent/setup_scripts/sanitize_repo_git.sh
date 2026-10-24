#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Leave the task repo with only HEAD and its ancestry. Task images often keep the upstream clone's
# other branches, tags and remotes, so `git log --all` would show the future fix commit.
# Repo paths come from the arguments. Without any, the runner's repo locations are tried.
# Any failed step fails the script, so the agent never starts on a half-sanitized repo.
set -eu
[ "$#" -gt 0 ] || set -- /testbed /workspace/repo /app /root/repo
for repo in "$@"; do
  if [ -f "$repo/.git" ]; then
    echo "cannot sanitize gitfile repository: $repo" >&2
    exit 1
  fi
  [ -d "$repo/.git" ] || continue
  cd "$repo"
  git config --global --add safe.directory "$repo"
  base=$(git rev-parse --verify -q HEAD)
  branch=$(git symbolic-ref -q HEAD || true)
  refs=$(git for-each-ref --format='%(refname)')
  for ref in $refs; do git update-ref -d "$ref"; done
  remotes=$(git remote)
  for remote in $remotes; do git remote remove "$remote"; done
  if [ -n "$branch" ]; then
    git update-ref "$branch" "$base"
  else
    git update-ref --no-deref HEAD "$base"
  fi
  rm -rf .git/logs .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD .git/refs/original
  git reflog expire --expire=now --all
  # -a -d drops every object not reachable from HEAD or the index. It runs before the alternates
  # are dropped, so objects a `clone --shared` repo borrows are copied in first.
  git -c gc.auto=0 -c repack.writeBitmaps=false repack -a -d -q
  rm -f .git/objects/info/alternates
  git prune --expire=now
  echo "sanitized $repo at $base"
done
