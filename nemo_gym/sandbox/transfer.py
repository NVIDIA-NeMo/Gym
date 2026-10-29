# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandbox transfer helpers.

Files move between this host and a sandbox as tar archives: :func:`upload_dir` and
:func:`download_dir` pack on one side and unpack on the other, and run the sandbox side through
:func:`exec_as_root`, so the task's file modes travel unchanged and a target the image's default
user cannot write (``/tests``, ``/logs``) is still reachable. :func:`upload_path` and
:func:`download_path` move a single file or a directory the same way, and :func:`remote_kind` tells
which a path is. :func:`upload_archive` replays the archive a download kept (``archive_copy``) into
another sandbox, so what the verifier receives is byte for byte what the agent's sandbox packed.
:func:`tar_extract_command` is the shell line that unpacks an uploaded archive with GNU or BusyBox tar. :class:`SandboxTransferError` reports a transfer command that failed inside the
sandbox, :class:`SandboxProbeError` a probe the sandbox could not answer, and
:class:`SandboxUnavailableError` a command the provider could not run at all, which is Gym's outage
rather than the task's. Environment servers and the agents that seed a sandbox share this module.
"""

import asyncio
import contextlib
import logging
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult


LOGGER = logging.getLogger(__name__)


class SandboxTransferError(RuntimeError):
    """A transfer command failed inside the sandbox."""


class SandboxUnavailableError(RuntimeError):
    """The provider could not run a transfer command at all (its result carries an ``error_type``).

    A timeout, a lost container or an exec the provider refused is Gym's outage, not a property of
    the task or the sandbox's files, so it is not a :class:`SandboxTransferError`: a caller that
    scores a deterministic transfer failure (permission denied, cannot mkdir) lets this one propagate
    and the episode is retried.
    """


class SandboxProbeError(RuntimeError):
    """The sandbox could not answer a probe or run a command (lost container, provider timeout).

    Not a :class:`SandboxTransferError`: a caller that treats a failed transfer as "this artifact is
    not there" must not treat an outage the same way.
    """


def _pack(source: Path, archive: Path, *, ignore: Callable[[str], bool] | None = None) -> None:
    if Path(source).is_symlink():
        # The root is never followed: a link the agent left where a folder was expected would otherwise
        # archive whatever it points at on this host.
        raise SandboxTransferError(f"Refusing to archive {source}: it is a symbolic link")

    def member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        # ``info.name`` is the entry's path relative to ``source``; a folder left out takes its subtree with it.
        return None if ignore is not None and ignore(info.name) else info

    with tarfile.open(archive, "w:gz") as tar:
        # Add the children, not the folder itself: an archive entry for "." makes tar restore the
        # source folder's mode onto the target, which fails on a root-owned world-writable target.
        for child in sorted(Path(source).iterdir()):
            tar.add(child, arcname=child.name, filter=member)


def _unpack(
    archive: Path,
    target: Path,
    *,
    names: tuple[str, ...] | None = None,
    skip_unsafe: bool = False,
    links: str = "data",
    protect: tuple[str, ...] = (),
) -> None:
    """Extract ``archive`` into ``target``.

    ``protect`` names top-level entries of ``target`` that belong to this host (a transcript copy, a
    staging folder with its manifest): a member at or under one of them is left out of the extraction,
    with a warning, so what the sandbox holds under the same name (an agent-planted file in a folder the
    agent could write) never replaces the host's record. ``links="data"`` applies :func:`tarfile.data_filter`, which refuses links that point outside the
    extraction root: right for verifier output that is read on this host. ``links="keep"`` applies
    :func:`tarfile.tar_filter` instead, which keeps symbolic links as links wherever they point (an
    agent's virtualenv links to ``/usr/bin/python3``): right for artifacts that only pass through
    this host on their way into another sandbox, where they are re-archived and never followed.
    Hard links are created on this host whatever the mode, so their target must stay inside
    ``target`` in both modes, as must every member's own path, and a device node or FIFO is refused
    in both modes (``tar_filter`` alone would create it on this host). In ``"keep"`` mode files and
    folders are made readable by their owner (as ``data_filter`` does), so a tree the agent left
    unreadable can still be re-archived; the host copy's modes are a staging convenience, the modes
    the agent left are in the archive itself (see :func:`upload_archive`).
    """

    def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
        if names is not None and member.name.removeprefix("./") not in names:
            return None
        if protect and member.name.removeprefix("./").split("/", 1)[0] in protect:
            LOGGER.warning("Skipping archive member %r: the host owns that name under %s", member.name, target)
            return None
        strict = links == "data" or member.islnk()
        try:
            if member.isdev():
                raise tarfile.SpecialFileError(member)
            member = (tarfile.data_filter if strict else tarfile.tar_filter)(member, destination)
            if member is not None and not strict and (member.isfile() or member.isdir()):
                member = member.replace(mode=member.mode | (0o700 if member.isdir() else 0o600))
            return member
        except tarfile.FilterError:
            if not skip_unsafe:
                raise
            LOGGER.warning("Skipping unsafe diagnostic archive member %r", member.name)
            return None

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter=safe_member)


# Transfers run as root: the image's default user may not be allowed to create `/tests`,
# or `/logs`. Preserve the task's file modes. Some images run without the capability to switch
# users at all; those fall back to the default user.
TRANSFER_USER = "root"
_NO_ROOT_MARKERS = ("CAP_SETUID", "identity switch", "switching to uid=0", "operation not permitted")


async def exec_as_root(sandbox: AsyncSandbox, command: str, *, timeout_s: float) -> SandboxExecResult:
    """Run ``command`` as root, or as the default user when the sandbox cannot switch users."""
    try:
        result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s, user=TRANSFER_USER)
    except Exception as exc:  # the SDK surfaces the refusal as a runtime error on some servers
        if not any(marker in str(exc) for marker in _NO_ROOT_MARKERS):
            raise
        # The fallback goes through the same provider-error check below as the result-shaped refusal does.
        result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
    else:
        text = f"{result.stderr or ''} {result.stdout or ''}"
        if result.return_code and any(marker in text for marker in _NO_ROOT_MARKERS):
            result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
    if result.error_type:
        raise SandboxUnavailableError(
            f"The sandbox could not run {command[:80]!r}: {result.error_type}: {result.stderr or result.stdout or ''}"
        )
    return result


def tar_extract_command(remote: str, target: str, *, replace_links: bool = False) -> str:
    """The shell line that unpacks the uploaded archive ``remote`` into ``target`` with GNU or BusyBox tar.

    With ``replace_links``, a symbolic link found at the path of any member of the archive is removed first, in the
    same shell line, before the extraction: the tars differ on what a link in the way means. GNU tar replaces a
    link at a directory member's path with the directory; BusyBox tar (Alpine and other minimal images) keeps the
    link and writes the member's children through it, wherever it points, so an archive merged as root into a tree
    another user has written to (a step's ``workdir/`` into the agent's working directory) would land where that
    user's link says. Harbor's ``docker cp`` removes the entry in the way (moby's ``archive.Unpack``), as GNU tar
    does; the pre-clearing gives both tars that outcome. The members are read as ``tar -t`` prints them, so a
    member whose name holds a newline (GNU tar escapes it in the listing) is not matched; a link at the target
    itself is followed by either tar, as ``docker cp`` resolves its destination.

    The clearing and the extraction are two commands, not one operation: a process still running in the target's
    tree (one a previous turn's agent left behind in the working directory it owns) can plant a link again at a
    member's path after the loop removed it and before ``tar -x`` reaches it, and BusyBox tar then writes that
    member through it as root. The pre-clearing gives BusyBox tar GNU tar's outcome against a link already in place,
    not against one re-planted during the extraction; GNU tar and ``docker cp`` remove a link in the way right before
    they create each member and carry a window of the same shape, so Harbor's merge shares it. Unpacking into a
    root-owned staging folder and moving the entries over would not close it: a directory member merges into a
    directory the previous agents may already have at that name, so the move is a recursive copy through the same
    user-owned tree, with the same window at every level, and a link at the destination that points at a directory
    sends ``mv`` into that directory. What such a process gains is where the archive's own members land, not their
    content. The window is noted here rather than closed.
    """
    quoted = shlex.quote(target)
    probe = "opts=''; tar --no-overwrite-dir --help >/dev/null 2>&1 && opts=--no-overwrite-dir; "
    extract = f"tar -xzf {remote} --no-same-owner $opts -C {quoted}"
    if not replace_links:
        return f"{probe}mkdir -p {quoted} && {extract}"
    # Directory members are listed before their children and with a trailing slash; a link removed at a parent
    # leaves nothing at the children's paths to test. The group's status is the extraction's.
    clear = (
        f"tar -tzf {remote} | while IFS= read -r m; do m=${{m%/}}; "
        f'if [ -L {quoted}/"$m" ]; then rm -- {quoted}/"$m"; fi; done; '
    )
    return f"{probe}mkdir -p {quoted} && {{ {clear}{extract}; }}"


async def upload_dir(
    sandbox: AsyncSandbox,
    source: Path,
    target: str,
    *,
    timeout_s: float = 600,
    make_readable: bool = False,
    ignore: Callable[[str], bool] | None = None,
    replace_links: bool = False,
) -> None:
    """Copy the contents of ``source`` into ``target`` inside the sandbox.

    File modes travel with the archive. ``make_readable`` additionally opens the tree to every user, for a
    solution the task user must read; a verifier tree keeps its modes because tests may check them.
    ``ignore`` is called with each entry's path relative to ``source`` (posix form); an entry it names is left
    out, its subtree with it. The environment server passes the task digest's rule
    (:func:`nemo_gym.data.tasks.harbor.digest.is_ignored`) for a task's own folders, so a ``tests/helper.pyc`` or an
    ``environment/__pycache__`` that the digest skips never reaches the sandbox; artifacts restored by
    :func:`upload_path` travel whole. ``replace_links`` is for a merge into a tree another user has written to:
    a symbolic link at any member's path is removed, as root, before the extraction, so the member lands where its
    name says on BusyBox tar as it does on GNU tar (:func:`tar_extract_command`).
    """
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        # Archiving is blocking file work; keep it off the event loop.
        await asyncio.to_thread(_pack, source, archive, ignore=ignore)
        # The readability fix-up is best effort, as the extraction's handling of the target folder is.
        fixup = f"chmod -R a+rX {shlex.quote(target)} 2>/dev/null || true; " if make_readable else ""
        await _upload_and_extract(
            sandbox, archive, target, timeout_s=timeout_s, fixup=fixup, what=source, replace_links=replace_links
        )


async def _upload_and_extract(
    sandbox: AsyncSandbox,
    archive: Path,
    target: str,
    *,
    timeout_s: float,
    fixup: str = "",
    what: object = None,
    replace_links: bool = False,
) -> None:
    """Upload ``archive`` and extract it as root into ``target`` inside the sandbox, then remove the upload."""
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    await sandbox.upload(archive, remote)
    # --no-overwrite-dir keeps the target folder's own ownership and mode: it may be a root-owned
    # world-writable folder the extracting user cannot chmod. GNU tar has the option; BusyBox tar
    # (Alpine images) rejects options it does not know, so it is passed only where `tar --help`
    # accepts it. `_pack` leaves the `.` entry out of the archive, so without the option only an
    # explicit entry could touch the target's own metadata.
    extract = tar_extract_command(remote, target, replace_links=replace_links)
    result = await exec_as_root(
        sandbox,
        f"{extract}; status=$?; {fixup}rm -f {remote}; exit $status",
        timeout_s=timeout_s,
    )
    if result.return_code:
        raise SandboxTransferError(
            f"Failed to unpack {what or archive} into {target}: {result.stderr or result.stdout}"
        )


async def download_dir(
    sandbox: AsyncSandbox,
    source: str,
    target: Path,
    *,
    timeout_s: float = 600,
    names: tuple[str, ...] | None = None,
    skip_unsafe: bool = False,
    links: str = "data",
    exclude: tuple[str, ...] = (),
    archive_copy: Path | None = None,
    protect: tuple[str, ...] = (),
) -> None:
    """Copy the contents of ``source`` inside the sandbox into the local ``target`` folder.

    ``links`` is passed to :func:`_unpack`: ``"keep"`` for artifacts that are restored into another
    sandbox, ``"data"`` (the default) for files read on this host, as is ``protect``: top-level names
    under ``target`` this host owns, which no member of the archive may write (see there). ``exclude`` holds an artifact's
    ``exclude`` rules, passed to ``tar --exclude`` exactly as Harbor passes them, unanchored: so
    ``node_modules`` leaves out every directory of that name at any depth, while ``./cache`` matches
    only the top-level one because member names start with ``./``. ``archive_copy`` keeps the archive
    the sandbox's root ``tar`` produced at that path, with the modes the sandbox held, for
    :func:`upload_archive`; it stays only when the host copy unpacked, so a refused member (one that
    escapes, a device node) leaves no archive behind either.
    """
    target.mkdir(parents=True, exist_ok=True)
    remote = f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    options = "".join(f" --exclude={shlex.quote(pattern)}" for pattern in exclude)
    command = f"tar -czf {remote}{options} -C {shlex.quote(source)} ."
    if names is not None:
        # An empty archive is valid: a verifier that writes no reward scores a measured zero. The names that exist
        # are passed as arguments; when none does, the end-of-archive marker (two zero blocks) gzipped is the
        # empty archive, without asking tar for one. ``-T`` and ``--null`` are not in every tar (BusyBox builds,
        # the tar of Alpine task images, lack one or both), which lost the reward of every task on such an image.
        # The names that exist become the shell's positional parameters, so a name with whitespace (a task's single
        # file artifact through :func:`download_path`) stays one argument, and ``--`` keeps one that starts with ``-``
        # from being read as an option. The group after ``&&`` runs only once ``cd`` succeeded: a source that is gone
        # fails the collection (as the single ``tar -C`` form does) instead of archiving the names from the exec's
        # own working directory or producing an empty archive.
        entries = " ".join(shlex.quote(name) for name in names)
        command = (
            f"cd {shlex.quote(source)} && {{ set --; "
            f'for name in {entries}; do if test -e "$name" || test -L "$name"; then set -- "$@" "$name"; fi; done; '
            f'if [ "$#" -gt 0 ]; then tar -czf {remote}{options} -- "$@"; '
            f"else head -c 1024 /dev/zero | gzip -c > {remote}; fi; }}"
        )
    result = await exec_as_root(sandbox, command, timeout_s=timeout_s)
    if result.return_code:
        raise SandboxTransferError(f"Failed to archive {source}: {result.stderr or result.stdout}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download.tar.gz"
            if archive_copy is not None:
                archive_copy.parent.mkdir(parents=True, exist_ok=True)
                archive = archive_copy
            try:
                await sandbox.download(remote, archive)
                await asyncio.to_thread(
                    _unpack, archive, target, names=names, skip_unsafe=skip_unsafe, links=links, protect=protect
                )
            except BaseException:
                if archive_copy is not None:
                    archive_copy.unlink(missing_ok=True)
                raise
    finally:
        # Best effort: a sandbox that is gone cannot clean up, and must not hide why the download failed.
        with contextlib.suppress(SandboxUnavailableError):
            await exec_as_root(sandbox, f"rm -f {remote}", timeout_s=60)


async def remote_kind(sandbox: AsyncSandbox, path: str) -> str | None:
    """``"dir"``, ``"file"`` or ``None`` for a path inside the sandbox.

    Only an answer counts as absence: a probe the sandbox could not run (a provider error, a lost
    container, a non-zero exit of the test itself) raises :class:`SandboxProbeError`, so an outage is
    never read as "the agent wrote nothing".
    """
    quoted = shlex.quote(path)
    result = await exec_as_root(
        sandbox,
        f"if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; else echo none; fi",
        timeout_s=60,
    )
    if result.error_type or result.return_code:
        raise SandboxProbeError(
            f"Could not probe {path}: {result.error_type or result.return_code}: {result.stderr or result.stdout}"
        )
    kind = (result.stdout or "").strip().splitlines()[-1:] or ["none"]
    return None if kind[0] == "none" else kind[0]


async def download_path(
    sandbox: AsyncSandbox,
    source: str,
    target: Path,
    *,
    timeout_s: float = 600,
    links: str = "data",
    exclude: tuple[str, ...] = (),
    archive_copy: Path | None = None,
) -> str | None:
    """Copy a file or directory out of the sandbox; returns what it was, or ``None`` when absent.

    A single file travels through the same archive as a directory, so its mode bits (an executable
    the agent built) arrive with it; ``exclude`` applies to directories only. ``archive_copy`` is
    passed to :func:`download_dir`: a file's archive holds that one member, named as the file is.
    """
    kind = await remote_kind(sandbox, source)
    if kind == "dir":
        await download_dir(
            sandbox, source, target, timeout_s=timeout_s, links=links, exclude=exclude, archive_copy=archive_copy
        )
    elif kind == "file":
        remote = PurePosixPath(source)
        await download_dir(
            sandbox,
            str(remote.parent),
            target.parent,
            timeout_s=timeout_s,
            names=(remote.name,),
            links=links,
            archive_copy=archive_copy,
        )
        if remote.name != target.name:
            (target.parent / remote.name).rename(target)
    return kind


async def upload_path(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Restore a collected file or directory into the sandbox at ``target``.

    A directory replaces whatever the image ships at ``target`` (a file the agent deleted stays
    deleted) and is made world-writable, as Harbor does when it restores artifacts: verifiers drop
    privileges before running agent output and need to write next to it. A file lands in its
    parent, which is created world-writable when missing; an existing parent is made world-writable
    when the sandbox can (``a+rwx`` keeps a sticky bit) and otherwise only has to be writable, so a
    root-owned ``/tmp`` keeps its mode and an image that cannot switch to root can still receive the
    file. Modes travel in both cases.
    """
    if source.is_symlink():
        raise SandboxTransferError(f"Refusing to restore {source} to {target}: it is a symbolic link")
    directory = await _prepare_restore_target(sandbox, source.is_dir(), target)
    if source.is_dir():
        await upload_dir(sandbox, source, target, timeout_s=timeout_s)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # One-file staging folder: the archive carries the file's mode, and `--no-overwrite-dir` keeps the parent's.
        staged = Path(tmp) / "file"
        staged.mkdir()
        shutil.copy2(source, staged / PurePosixPath(target).name, follow_symlinks=False)
        await upload_dir(sandbox, staged, directory, timeout_s=timeout_s)


async def _prepare_restore_target(sandbox: AsyncSandbox, is_dir: bool, target: str) -> str:
    """Clear and open ``target`` (a directory) or open its parent (a file) as :func:`upload_path` describes.

    Returns the directory an archive is then extracted into. The named path is what is cleared and opened:
    a link already there is left as a link, which the extraction then follows (see ``keep_out`` on
    :func:`upload_archive`).
    """
    if is_dir:
        directory = target
        quoted = shlex.quote(target)
        prepare = f"mkdir -p {quoted} && find {quoted} -mindepth 1 -delete && chmod 777 {quoted}"
    else:
        directory = str(PurePosixPath(target).parent)
        quoted = shlex.quote(directory)
        prepare = (
            f"if [ -d {quoted} ]; then chmod a+rwx {quoted} 2>/dev/null || test -w {quoted}; "
            f"else mkdir -p {quoted} && chmod 777 {quoted}; fi"
        )
    result = await exec_as_root(sandbox, prepare, timeout_s=60)
    if result.return_code:
        raise SandboxTransferError(f"Could not prepare {quoted} for {target}: {result.stderr or result.stdout}")
    return directory


async def upload_archive(
    sandbox: AsyncSandbox,
    archive: Path,
    kind: str,
    target: str,
    *,
    timeout_s: float = 600,
    keep_out: tuple[str, ...] = (),
) -> bool:
    """Restore at ``target`` the archive another sandbox's root ``tar`` produced (a download's ``archive_copy``).

    ``kind`` is what :func:`download_path` reported, ``"dir"`` or ``"file"``. The target is prepared as
    :func:`upload_path` prepares it; then root ``tar`` extracts the archive there, so the files arrive
    with the modes the agent's sandbox held (read-only, world-writable and set-id bits included), as
    Harbor's ``docker cp`` round trip keeps them. None of those modes is applied on this host: the
    staging copy :func:`download_dir` unpacked beside the archive went through :func:`_unpack`'s
    filter, and the archive exists only because that copy was accepted. A file's archive holds the one
    member named as the file is, so it is extracted into ``target``'s parent, which ``target`` must
    name the same way.

    ``keep_out`` names directories the restore must land clear of when a link diverts it: before anything
    touches the target, the directory the archive would be extracted into is resolved in the sandbox to
    its physical place (:func:`_lands_clear`), and when that place is not the one named (a link an
    earlier restore left in the way, say) and is one of ``keep_out``, inside one or above one, nothing is
    prepared or extracted, a warning says so and ``False`` is returned. The probe comes first because
    preparing the target is destructive: it empties the directory and creates what is missing, through
    whatever link sits in the path, with or without a trailing slash. A restore that lands where its
    name says is the caller's to judge by that name. Returns ``True`` once the archive is extracted.
    """
    directory = target if kind == "dir" else str(PurePosixPath(target).parent)
    if keep_out and not await _lands_clear(sandbox, directory, keep_out):
        return False
    await _prepare_restore_target(sandbox, kind == "dir", target)
    await _upload_and_extract(sandbox, archive, directory, timeout_s=timeout_s)
    return True


# Where a directory physically is, read without touching anything: `cd -P` into it, or into its nearest
# existing ancestor with the missing tail kept aside, then `pwd -P`. Two lines out: `tail=` (empty when the
# directory exists) and the physical place. `cd -P` resolves a `..` after the links before it, as the kernel
# would resolve the path a `mkdir -p` then creates.
_PHYSICAL_PROBE = (
    'p={quoted}; t=\'\'; while ! cd -P "$p" 2>/dev/null; do [ "$p" = / ] && exit 3; '
    't="$(basename "$p")/$t"; p=$(dirname "$p"); done; printf \'tail=%s\\n\' "$t"; pwd -P'
)


async def _lands_clear(sandbox: AsyncSandbox, directory: str, keep_out: tuple[str, ...]) -> bool:
    """Whether extracting into ``directory`` stays clear of ``keep_out`` as :func:`upload_archive` describes.

    The probe only reads: ``directory`` need not exist yet, and nothing is created, emptied or opened for it.
    A sandbox that answers the probe with nothing (a fake) leaves only the name to judge; one that cannot
    run it is an outage and raises.
    """
    result = await exec_as_root(sandbox, _PHYSICAL_PROBE.format(quoted=shlex.quote(directory)), timeout_s=60)
    if result.error_type or result.return_code:
        raise SandboxTransferError(
            f"Could not resolve {directory}: {result.error_type or result.return_code}: {result.stderr or result.stdout}"
        )
    lines = [line.strip() for line in (result.stdout or "").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return True
    # One leading slash everywhere: POSIX lets "//x" name something other than "/x" and the shell keeps that
    # spelling for such a cwd, while PurePosixPath keeps the two apart.
    physical = PurePosixPath("/" + lines[-1].lstrip("/"))
    if len(lines) > 1 and lines[-2].startswith("tail=") and (tail := lines[-2][len("tail=") :].strip("/")):
        physical = physical / tail
    if physical == PurePosixPath("/" + directory.lstrip("/")):
        return True
    for kept in keep_out:
        kept_path = PurePosixPath("/" + kept.lstrip("/"))
        if physical.is_relative_to(kept_path) or kept_path.is_relative_to(physical):
            LOGGER.warning("Not extracting into %s: it physically is %s, which reaches %s", directory, physical, kept)
            return False
    return True
