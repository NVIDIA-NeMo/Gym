# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory transfers through sandbox exec, upload and download."""

import asyncio
import logging
import shlex
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult


LOGGER = logging.getLogger(__name__)


class SandboxTransferError(RuntimeError):
    """A transfer command failed inside the sandbox."""


def _pack(source: Path, archive: Path) -> None:
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
) -> None:
    """Copy the contents of ``source`` inside the sandbox into the local ``target`` folder.

    ``links`` is passed to :func:`_unpack`: ``"keep"`` for artifacts that are restored into another
    sandbox, ``"data"`` (the default) for files read on this host.
    """
    target.mkdir(parents=True, exist_ok=True)
    remote = f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    command = f"tar -czf {remote} -C {shlex.quote(source)} ."
    if names is not None:
        # An empty archive is valid: a verifier that writes no reward scores a measured zero.
        entries = " ".join(shlex.quote(name) for name in names)
        command = (
            f"cd {shlex.quote(source)} && "
            f'for name in {entries}; do if test -e "$name" || test -L "$name"; '
            f"then printf '%s\\0' \"$name\"; fi; done | tar -czf {remote} --null -T -"
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
    """``"dir"``, ``"file"`` or ``None`` for a path inside the sandbox."""
    quoted = shlex.quote(path)
    result = await sandbox.exec(
        f"if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; else echo none; fi", timeout_s=60
    )
    kind = (result.stdout or "").strip().splitlines()[-1:] or ["none"]
    return None if result.return_code or kind[0] == "none" else kind[0]


async def download_path(
    sandbox: AsyncSandbox, source: str, target: Path, *, timeout_s: float = 600, links: str = "data"
) -> str | None:
    """Copy a file or directory out of the sandbox; returns what it was, or ``None`` when absent."""
    kind = await remote_kind(sandbox, source)
    if kind == "dir":
        await download_dir(sandbox, source, target, timeout_s=timeout_s, links=links)
    elif kind == "file":
        target.parent.mkdir(parents=True, exist_ok=True)
        await sandbox.download(source, target)
    return kind


async def upload_path(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Copy a local file or directory into the sandbox at ``target``, creating parents."""
    if source.is_dir():
        await upload_dir(sandbox, source, target, timeout_s=timeout_s)
        return
    parent = str(PurePosixPath(target).parent)
    result = await sandbox.exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=60)
    if result.return_code:
        raise SandboxTransferError(f"Could not create {parent}: {result.stderr or result.stdout}")
    await sandbox.upload(source, target)
