# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage literal initial actor state through the shared task identity boundary."""

import json
import re
from collections.abc import Sequence
from shlex import quote

from resources_servers.agentic_if.episode.runtime import BoundSandbox, private_staging, upload_text


# The base every workspace diff of an episode is taken against: the tree of the commit HEAD names when seed first
# observes the repository, or the empty tree on a branch with no commit yet. Staging and commits do not move it, so a
# diff describes the files, whatever the actor staged or committed. HEAD naming a missing commit is Git's own fatal
# repository state (78).
_WORKSPACE_BASE = """
if_base="$(git rev-parse -q --verify 'HEAD^{tree}')" || {
    if_head="$(git symbolic-ref -q HEAD)" && ! git rev-parse -q --verify "$if_head" >/dev/null || exit 78
    if_base="$(git hash-object -t tree /dev/null)" || exit 1
}
printf '%s\\n' "$if_base"
"""


# Match the profile plugin's per-tool snapshots: the files against the base tree (`if_base`), untracked files included
# through a temporary copy of the actor's Git index, which itself stays unchanged. Keep the index timestamp so Git still
# checks same-size, same-second edits. The positional parameters are the diff's pathspecs.
WORKSPACE_DIFF = """
if_git_dir="$(git rev-parse --absolute-git-dir)" || {
    if_status=$?
    [ "$if_status" -eq 128 ] && exit 78
    exit "$if_status"
}
# Validate the original index read-only before attempting any temporary writes.
git ls-files --stage >/dev/null || {
    if_status=$?
    [ "$if_status" -eq 128 ] && exit 78
    exit "$if_status"
}
if_index_dir="$(mktemp -d /tmp/ng-if-index.XXXXXXXX)" || exit 1
trap 'rm -f "$if_index_dir/index"; rmdir "$if_index_dir"' EXIT
if [ -f "$if_git_dir/index" ]; then
    cp "$if_git_dir/index" "$if_index_dir/index" && touch -r "$if_git_dir/index" "$if_index_dir/index" || exit 1
fi
export GIT_INDEX_FILE="$if_index_dir/index"
# Git also uses 128 for index.lock/disk failures: writes must remain failures,
# not be mistaken for deliberately broken repository state in the task image.
git add -N --ignore-removal . || exit "$?"
git diff --binary --full-index "$if_base" -- "$@"
if_status=$?
[ "$if_status" -eq 128 ] && exit 78
exit "$if_status"
"""


def workspace_diff_command(base: str, *, exclude: Sequence[str] = ()) -> str:
    """``WORKSPACE_DIFF`` against the base tree ``base``, leaving out the repository-relative paths in ``exclude``."""
    pathspecs = (".", *(":(top,literal,exclude)" + path for path in exclude)) if exclude else ()
    return f"if_base={quote(base)}\nset -- {' '.join(quote(spec) for spec in pathspecs)}\n" + WORKSPACE_DIFF


_DISCOVER_WORKSPACE = """
command -v git >/dev/null 2>&1 || exit 77
if_probe="$(LC_ALL=C git rev-parse --is-inside-work-tree 2>&1)"
if_status=$?
if [ "$if_status" -ne 0 ]; then
    printf '%s\\n' "$if_probe" >&2
    [ "$if_status" -eq 128 ] && exit 78
    exit "$if_status"
fi
case "$if_probe" in
    true) ;;
    false) exit 77 ;;
    *) exit 1 ;;
esac
"""


async def workspace_base(sandbox: BoundSandbox) -> str | None:
    """The episode's base tree (``_WORKSPACE_BASE``), observed as the actor; None without a usable Git worktree."""
    result = await sandbox.exec(command=_DISCOVER_WORKSPACE + _WORKSPACE_BASE, timeout_s=60)
    if result.return_code in {77, 78} and not result.error_type:
        return None
    base = (result.stdout or "").strip()
    if result.return_code or result.error_type or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", base):
        raise RuntimeError("could not resolve the workspace's base tree")
    return base


async def snapshot_workspace(sandbox: BoundSandbox, base: str | None) -> str | None:
    """Observe initial state as the actor, against the episode's base tree; unavailable Git is not a clean tree.

    A task may deliberately contain broken Git metadata. That state is unknown,
    not clean, and is recorded explicitly for the scorer's argument fallback.
    Transport, timeout and temporary-index I/O failures still abort preparation.
    """
    if base is None:
        return None
    # 77 means no Git/worktree; 78 means Git's own fatal repository error (128).
    # Neither implies an empty diff. Never repair task state to obtain a baseline.
    command = _DISCOVER_WORKSPACE + workspace_diff_command(base)
    # Initial trees may contain substantial pre-existing untracked state. Use
    # the same bounded budget as replay's identical diff operation, not the
    # short metadata-command budget. BoundSandbox still caps the task lifetime.
    result = await sandbox.exec(command=command, timeout_s=600)
    if result.return_code in {77, 78} and not result.error_type:
        return None
    if result.return_code or result.error_type:
        raise RuntimeError("could not snapshot initial workspace")
    return result.stdout or ""


# Directory descriptors prevent both existing symlinks and symlink replacement
# races from redirecting writes away from declared sandbox paths. Never elevate
# beyond the task identity, including for explicitly declared absolute files.
_STAGE = """
import json, os, stat, sys
with open(sys.argv[1], encoding='utf-8') as manifest:
    files = json.load(manifest)
resolved = [(os.path.abspath(path), text) for path, text in files.items()]
paths = {path for path, _ in resolved}
if len(paths) != len(resolved) or any(
    '/'.join(path.split('/')[:i]) in paths
    for path in paths for i in range(1, len(path.split('/')))
):
    raise ValueError('workspace paths collide after resolving the working directory')
root = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
try:
    for path, text in resolved:
        directory = os.dup(root)
        try:
            parts = path.removeprefix('/').split('/')
            for part in parts[:-1]:
                try:
                    os.mkdir(part, 0o755, dir_fd=directory)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o644, dir_fd=directory)
            with os.fdopen(fd, 'wb') as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError('workspace target must be a regular file')
                stream.truncate(0)
                stream.write(text.encode('utf-8'))
        finally:
            os.close(directory)
finally:
    os.close(root)
"""


async def verify_materialized_cwd(sandbox: BoundSandbox, expected: object) -> None:
    """Check an upstream prompt fact against the actual task, without rewriting it.

    Resolve both paths as the actor, allowing symlink aliases. An omitted fact
    changes nothing and requires no extra provider call.
    """
    if expected is None:
        return
    if not isinstance(expected, str) or not expected.startswith("/") or any(c in expected for c in "\0\n\r"):
        raise ValueError("materialized_cwd must be an absolute single-line path")
    result = await sandbox.exec(command=f"pwd -P && (cd -- {quote(expected)} && pwd -P)", timeout_s=30)
    paths = (result.stdout or "").splitlines()
    if result.return_code or result.error_type or len(paths) != 2 or not all(path.startswith("/") for path in paths):
        raise RuntimeError("could not verify materialized_cwd in the task sandbox")
    if paths[0] != paths[1]:
        raise ValueError(f"materialized_cwd {expected!r} differs from the task working directory {paths[0]!r}")


async def stage_workspace_files(sandbox: BoundSandbox, files: dict[str, str]) -> None:
    """Write exact UTF-8 bytes once before discovery."""
    if not files:
        return
    if sandbox.workdir is None:
        result = await sandbox.exec(command="pwd -P", timeout_s=30)
        workdir = (result.stdout or "").removesuffix("\n")
        if result.return_code or result.error_type or not workdir.startswith("/"):
            raise RuntimeError("could not resolve initial workspace working directory")
        sandbox.workdir = workdir
    # The files go by upload, not on the command line, which caps a single argument at 128 KiB.
    async with private_staging(sandbox, "workspace") as staging:
        manifest = f"{staging}/files.json"
        await upload_text(sandbox, manifest, json.dumps(files))
        result = await sandbox.exec(command=f"python3 -c {quote(_STAGE)} {quote(manifest)}", timeout_s=30)
    if result.return_code or result.error_type:
        raise RuntimeError("could not stage initial workspace files")
