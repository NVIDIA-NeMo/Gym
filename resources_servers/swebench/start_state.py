# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Seed a sandbox at a curriculum start state without revealing where it differs.

A start-state task begins with part of the reference fix already applied (``setup_patch``);
the agent must finish it. The applied part must not point at where to work, so seeding:

1. applies the patch and commits the result with the base commit's author, committer, dates
   and message, either onto the base commit's parents or, with ``squash_history``, as a single
   root commit (``git show HEAD`` / ``git log -p`` then reveal nothing);
2. leaves the anti-cheat scrub (run next by the caller) to expire the reflog and prune the
   original commit and every now-unreachable object;
3. gives every tracked file one modification time and deletes Python and pytest caches, so
   ``stat``, ``ls -t`` and ``find -newer`` reveal nothing either.

Rows that only set ``squash_history`` get the same treatment without a patch, so parents and
start-state tasks in one curriculum look alike. Rows with neither field are untouched.

Grading applies ``setup_patch`` before the agent's patch, which the server diffs against the
seeded HEAD (see ``seeded_head``).
"""

import shlex
from typing import Any

from resources_servers.swebench.anti_cheat import apply_anti_cheat_setup


SETUP_PATCH_PATH = "/tmp/nemo_gym_setup.diff"

_COMMIT_SCRIPT = r"""set -e
git config --global --add safe.directory "$PWD" 2>/dev/null || true
{prepare}
base=$(git rev-parse HEAD)
if [ -s {setup} ]; then git apply --index {setup}; fi
export GIT_AUTHOR_NAME="$(git log -1 --format=%an "$base")"
export GIT_AUTHOR_EMAIL="$(git log -1 --format=%ae "$base")"
export GIT_AUTHOR_DATE="$(git log -1 --format=%aD "$base")"
export GIT_COMMITTER_NAME="$(git log -1 --format=%cn "$base")"
export GIT_COMMITTER_EMAIL="$(git log -1 --format=%ce "$base")"
export GIT_COMMITTER_DATE="$(git log -1 --format=%cD "$base")"
tree=$(git write-tree)
parents=""
if [ "{squash}" != 1 ]; then
  for p in $(git log -1 --format=%P "$base"); do parents="$parents -p $p"; done
fi
# The raw message bytes: --format=%B appends a newline the original does not have.
commit=$(git cat-file commit "$base" | sed '1,/^$/d' | git commit-tree "$tree" $parents)
git update-ref HEAD "$commit"
git reset -q --hard "$commit"
rm -f {setup}
"""

_NORMALIZE_SCRIPT = r"""set -e
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache
# Snapshot repos (git init over the image) can track caches; only untracked ones may go.
git ls-files -z --deleted | xargs -0 -r git checkout --
stamp=$(date +%s)
git ls-files -z | xargs -0 -r touch -h -d "@$stamp"
git update-index -q --refresh || true
"""


def wants_start_state(setup_patch: str, squash_history: bool) -> bool:
    return bool(setup_patch.strip()) or squash_history


def setup_files(setup_patch: str) -> dict[str, str]:
    """Create-time sandbox files for ``commit_start_state``."""
    return {SETUP_PATCH_PATH: setup_patch} if setup_patch.strip() else {}


async def _run(sandbox: Any, workdir: str, script: str, what: str) -> None:
    result = await sandbox.exec(f"bash -c {shlex.quote(script)}", cwd=workdir, timeout_s=600)
    if result.return_code != 0:
        raise RuntimeError(f"{what} failed ({result.return_code}): {result.stderr or result.stdout}")


async def commit_start_state(sandbox: Any, workdir: str, *, squash_history: bool, keep_worktree: bool = False) -> None:
    """Step 1. Raises: an agent must never start from a wrong or half-made state.

    ``keep_worktree`` folds the image's own edits to tracked files into the start-state commit
    instead of discarding them, for servers whose grading runs on the image's tree as shipped.
    """
    script = _COMMIT_SCRIPT.format(
        setup=SETUP_PATCH_PATH,
        squash="1" if squash_history else "0",
        prepare="git add -u" if keep_worktree else "git reset -q --hard",
    )
    await _run(sandbox, workdir, script, "start-state commit")


async def normalize_tree(sandbox: Any, workdir: str) -> None:
    """Step 3, after the anti-cheat scrub."""
    await _run(sandbox, workdir, _NORMALIZE_SCRIPT, "start-state normalization")


async def assert_unreachable(sandbox: Any, workdir: str, original: str) -> None:
    """The anti-cheat scrub is best-effort; a surviving original commit would hold the fix.

    Squashing a parentless base without a setup patch rebuilds the very same commit, which
    reveals nothing and so may remain.
    """
    if await seeded_head(sandbox, workdir) == original:
        return
    result = await sandbox.exec(f"git -C {shlex.quote(workdir)} cat-file -e {shlex.quote(original)}^{{commit}}")
    if result.return_code == 0:
        raise RuntimeError("The original base commit survived the anti-cheat scrub")


async def seeded_head(sandbox: Any, workdir: str) -> str:
    result = await sandbox.exec(f"git -C {shlex.quote(workdir)} rev-parse HEAD")
    if result.return_code != 0 or not (result.stdout or "").strip():
        raise RuntimeError(f"Could not read the seeded HEAD: {result.stderr}")
    return result.stdout.strip()


async def seed_start_state(
    sandbox: Any,
    workdir: str,
    instance_id: str,
    log_prefix: str,
    *,
    setup_patch: str,
    squash_history: bool,
    apply_anti_cheating: bool,
) -> None:
    """Steps 1-3 around the anti-cheat scrub, for a sandbox whose ``workdir`` is a git repo
    at the base commit and whose grading applies patches to the image's tree as shipped.
    Without a start state this is only the configured scrub; with one the scrub is required,
    since the rewritten HEAD must leave the original unreachable.
    """
    start_state = wants_start_state(setup_patch, squash_history)
    if start_state:
        original = await seeded_head(sandbox, workdir)
        await commit_start_state(sandbox, workdir, squash_history=squash_history, keep_worktree=True)
    if apply_anti_cheating or start_state:
        await apply_anti_cheat_setup(sandbox, workdir, instance_id, log_prefix)
    if start_state:
        await assert_unreachable(sandbox, workdir, original)
        await normalize_tree(sandbox, workdir)


SETUP_PATCH_FAILED_EXIT = 96


def apply_setup_patch_command(setup_patch: str) -> str:
    """The grading script's first change to the checkout. Unlike the agent's patch, the start
    state is the task itself: if it does not apply exactly, the run says nothing about the
    agent and must not be graded.
    """
    if not setup_patch.strip():
        return ""
    return f"git apply {SETUP_PATCH_PATH} || exit {SETUP_PATCH_FAILED_EXIT}"
