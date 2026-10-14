# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory transfers through sandbox exec, upload and download."""

import asyncio
import logging
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult


LOGGER = logging.getLogger(__name__)


class SandboxTransferError(RuntimeError):
    """A transfer command failed inside the sandbox."""


class SandboxProbeError(RuntimeError):
    """The sandbox could not answer a probe or run a command (lost container, provider timeout).

    Not a :class:`SandboxTransferError`: a caller that treats a failed transfer as "this artifact is
    not there" must not treat an outage the same way.
    """


def _pack(source: Path, archive: Path) -> None:
    if Path(source).is_symlink():
        # The root is never followed: a link the agent left where a folder was expected would otherwise
        # archive whatever it points at on this host.
        raise SandboxTransferError(f"Refusing to archive {source}: it is a symbolic link")
    with tarfile.open(archive, "w:gz") as tar:
        # Add the children, not the folder itself: an archive entry for "." makes tar restore the
        # source folder's mode onto the target, which fails on a root-owned world-writable target.
        for child in sorted(Path(source).iterdir()):
            tar.add(child, arcname=child.name)


def _unpack(
    archive: Path,
    target: Path,
    *,
    names: tuple[str, ...] | None = None,
    skip_unsafe: bool = False,
    links: str = "data",
) -> None:
    """Extract ``archive`` into ``target``.

    ``links="data"`` applies :func:`tarfile.data_filter`, which refuses links that point outside the
    extraction root: right for verifier output that is read on this host. ``links="keep"`` applies
    :func:`tarfile.tar_filter` instead, which keeps symbolic links as links wherever they point (an
    agent's virtualenv links to ``/usr/bin/python3``): right for artifacts that only pass through
    this host on their way into another sandbox, where they are re-archived and never followed.
    Hard links are created on this host whatever the mode, so their target must stay inside
    ``target`` in both modes, as must every member's own path. In ``"keep"`` mode files and folders
    are made readable by their owner (as ``data_filter`` does), so a tree the agent left unreadable
    can still be re-archived.
    """

    def safe_member(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo | None:
        if names is not None and member.name.removeprefix("./") not in names:
            return None
        strict = links == "data" or member.islnk()
        try:
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


async def _exec_as_root(sandbox: AsyncSandbox, command: str, *, timeout_s: float) -> SandboxExecResult:
    """Run ``command`` as root, or as the default user when the sandbox cannot switch users."""
    try:
        result = await sandbox.exec(command, cwd="/", timeout_s=timeout_s, user=TRANSFER_USER)
    except Exception as exc:  # the SDK surfaces the refusal as a runtime error on some servers
        if not any(marker in str(exc) for marker in _NO_ROOT_MARKERS):
            raise
        return await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
    text = f"{result.stderr or ''} {result.stdout or ''}"
    if result.return_code and any(marker in text for marker in _NO_ROOT_MARKERS):
        return await sandbox.exec(command, cwd="/", timeout_s=timeout_s)
    return result


async def upload_dir(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Copy the contents of ``source`` into ``target`` inside the sandbox.

    File modes travel with the archive; the target directory keeps its own mode and ownership.
    """
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        # Archiving is blocking file work; keep it off the event loop.
        await asyncio.to_thread(_pack, source, archive)
        await sandbox.upload(archive, remote)
    # --no-overwrite-dir keeps the target folder's own ownership and mode: it may be a root-owned
    # world-writable folder the extracting user cannot chmod.
    result = await _exec_as_root(
        sandbox,
        f"mkdir -p {shlex.quote(target)} && tar -xzf {remote} --no-same-owner --no-overwrite-dir -C {shlex.quote(target)}; "
        f"status=$?; rm -f {remote}; exit $status",
        timeout_s=timeout_s,
    )
    if result.return_code:
        raise SandboxTransferError(f"Failed to unpack {source} into {target}: {result.stderr or result.stdout}")


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
) -> None:
    """Copy the contents of ``source`` inside the sandbox into the local ``target`` folder.

    ``links`` is passed to :func:`_unpack`: ``"keep"`` for artifacts that are restored into another
    sandbox, ``"data"`` (the default) for files read on this host. ``exclude`` holds patterns
    relative to ``source`` (an artifact's ``exclude`` rules); each is anchored at the source, so
    ``node_modules`` leaves out ``source/node_modules`` and nothing deeper with that name.
    """
    target.mkdir(parents=True, exist_ok=True)
    remote = f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    options = "".join(
        f" --exclude={shlex.quote('./' + pattern.removeprefix('./').lstrip('/'))}" for pattern in exclude
    )
    if options:
        options = " --anchored" + options
    command = f"tar -czf {remote}{options} -C {shlex.quote(source)} ."
    if names is not None:
        # An empty archive is valid: a verifier that writes no reward scores a measured zero.
        entries = " ".join(shlex.quote(name) for name in names)
        command = (
            f"cd {shlex.quote(source)} && "
            f'for name in {entries}; do if test -e "$name" || test -L "$name"; '
            f"then printf '%s\\0' \"$name\"; fi; done | tar -czf {remote}{options} --null -T -"
        )
    result = await _exec_as_root(sandbox, command, timeout_s=timeout_s)
    if result.return_code:
        raise SandboxTransferError(f"Failed to archive {source}: {result.stderr or result.stdout}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download.tar.gz"
            await sandbox.download(remote, archive)
            await asyncio.to_thread(_unpack, archive, target, names=names, skip_unsafe=skip_unsafe, links=links)
    finally:
        await _exec_as_root(sandbox, f"rm -f {remote}", timeout_s=60)


async def remote_kind(sandbox: AsyncSandbox, path: str) -> str | None:
    """``"dir"``, ``"file"`` or ``None`` for a path inside the sandbox.

    Only an answer counts as absence: a probe the sandbox could not run (a provider error, a lost
    container, a non-zero exit of the test itself) raises :class:`SandboxProbeError`, so an outage is
    never read as "the agent wrote nothing".
    """
    quoted = shlex.quote(path)
    result = await _exec_as_root(
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
) -> str | None:
    """Copy a file or directory out of the sandbox; returns what it was, or ``None`` when absent.

    A single file travels through the same archive as a directory, so its mode bits (an executable
    the agent built) arrive with it; ``exclude`` applies to directories only.
    """
    kind = await remote_kind(sandbox, source)
    if kind == "dir":
        await download_dir(sandbox, source, target, timeout_s=timeout_s, links=links, exclude=exclude)
    elif kind == "file":
        remote = PurePosixPath(source)
        await download_dir(
            sandbox, str(remote.parent), target.parent, timeout_s=timeout_s, names=(remote.name,), links=links
        )
        if remote.name != target.name:
            (target.parent / remote.name).rename(target)
    return kind


async def upload_path(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Restore a collected file or directory into the sandbox at ``target``.

    A directory replaces whatever the image ships at ``target`` (a file the agent deleted stays
    deleted) and is made world-writable, as Harbor does when it restores artifacts: verifiers drop
    privileges before running agent output and need to write next to it. A file lands in its
    parent, created world-writable the same way. Modes travel in both cases.
    """
    if source.is_symlink():
        raise SandboxTransferError(f"Refusing to restore {source} to {target}: it is a symbolic link")
    if source.is_dir():
        directory = shlex.quote(target)
        prepare = f"mkdir -p {directory} && find {directory} -mindepth 1 -delete && chmod 777 {directory}"
    else:
        directory = shlex.quote(str(PurePosixPath(target).parent))
        prepare = f"mkdir -p {directory} && chmod 777 {directory}"
    result = await _exec_as_root(sandbox, prepare, timeout_s=60)
    if result.return_code:
        raise SandboxTransferError(f"Could not prepare {directory} for {target}: {result.stderr or result.stdout}")
    if source.is_dir():
        await upload_dir(sandbox, source, target, timeout_s=timeout_s)
        return
    with tempfile.TemporaryDirectory() as tmp:
        # One-file staging folder: the archive carries the file's mode, and `--no-overwrite-dir` keeps the parent's.
        staged = Path(tmp) / "file"
        staged.mkdir()
        shutil.copy2(source, staged / PurePosixPath(target).name, follow_symlinks=False)
        await upload_dir(sandbox, staged, str(PurePosixPath(target).parent), timeout_s=timeout_s)
