# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory transfers through sandbox exec, upload and download."""

import asyncio
import shlex
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox


class SandboxTransferError(RuntimeError):
    """A transfer command failed inside the sandbox."""


def _pack(source: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tar:
        # Add the children, not the folder itself: an archive entry for "." makes tar restore the
        # source folder's mode onto the target, which fails on a root-owned world-writable target.
        for child in sorted(Path(source).iterdir()):
            tar.add(child, arcname=child.name)


def _unpack(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter="data")


# Transfers run as root: the image's default user may not be allowed to create `/tests`,
# `/solution` or `/logs`, and the files they unpack must be readable by every user. Some
# images run without the capability to switch users at all; those fall back to the default user.
TRANSFER_USER = "root"
_NO_ROOT_MARKERS = ("CAP_SETUID", "identity switch", "switching to uid=0", "operation not permitted")


async def _exec_as_root(sandbox: AsyncSandbox, command: str, *, timeout_s: float):
    """Run ``command`` as root, or as the default user when the sandbox cannot switch users."""
    try:
        result = await sandbox.exec(command, timeout_s=timeout_s, user=TRANSFER_USER)
    except Exception as exc:  # the SDK surfaces the refusal as a runtime error on some servers
        if not any(marker in str(exc) for marker in _NO_ROOT_MARKERS):
            raise
        return await sandbox.exec(command, timeout_s=timeout_s)
    text = f"{result.stderr or ''} {result.stdout or ''}"
    if result.return_code and any(marker in text for marker in _NO_ROOT_MARKERS):
        return await sandbox.exec(command, timeout_s=timeout_s)
    return result


async def upload_dir(
    sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600, make_readable: bool = False
) -> None:
    """Copy the contents of ``source`` into ``target`` inside the sandbox.

    File modes travel with the archive. ``make_readable`` additionally opens the tree to every user, for a
    solution the task user must read; a verifier tree keeps its modes because tests may check them.
    """
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        # Archiving is blocking file work; keep it off the event loop.
        await asyncio.to_thread(_pack, source, archive)
        await sandbox.upload(archive, remote)
    # --no-overwrite-dir keeps the target folder's own ownership and mode: it may be a root-owned
    # world-writable folder the extracting user cannot chmod. The readability fix-up is best effort
    # for the same reason.
    fixup = f"chmod -R a+rX {shlex.quote(target)} 2>/dev/null || true; " if make_readable else ""
    result = await _exec_as_root(
        sandbox,
        f"mkdir -p {shlex.quote(target)} && tar -xzf {remote} --no-same-owner --no-overwrite-dir -C {shlex.quote(target)}; "
        f"status=$?; {fixup}rm -f {remote}; exit $status",
        timeout_s=timeout_s,
    )
    if result.return_code:
        raise SandboxTransferError(f"Failed to unpack {source} into {target}: {result.stderr or result.stdout}")


async def download_dir(sandbox: AsyncSandbox, source: str, target: Path, *, timeout_s: float = 600) -> None:
    """Copy the contents of ``source`` inside the sandbox into the local ``target`` folder."""
    target.mkdir(parents=True, exist_ok=True)
    remote = f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    result = await _exec_as_root(sandbox, f"tar -czf {remote} -C {shlex.quote(source)} .", timeout_s=timeout_s)
    if result.return_code:
        raise SandboxTransferError(f"Failed to archive {source}: {result.stderr or result.stdout}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download.tar.gz"
            await sandbox.download(remote, archive)
            await asyncio.to_thread(_unpack, archive, target)
    finally:
        await _exec_as_root(sandbox, f"rm -f {remote}", timeout_s=60)


async def remote_kind(sandbox: AsyncSandbox, path: str) -> str | None:
    """``"dir"``, ``"file"`` or ``None`` for a path inside the sandbox."""
    quoted = shlex.quote(path)
    result = await _exec_as_root(
        sandbox,
        f"if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; else echo none; fi",
        timeout_s=60,
    )
    kind = (result.stdout or "").strip().splitlines()[-1:] or ["none"]
    return None if result.return_code or kind[0] == "none" else kind[0]


async def download_path(sandbox: AsyncSandbox, source: str, target: Path, *, timeout_s: float = 600) -> str | None:
    """Copy a file or directory out of the sandbox; returns what it was, or ``None`` when absent."""
    kind = await remote_kind(sandbox, source)
    if kind == "dir":
        await download_dir(sandbox, source, target, timeout_s=timeout_s)
    elif kind == "file":
        target.parent.mkdir(parents=True, exist_ok=True)
        await sandbox.download(source, target)
    return kind


async def upload_path(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Restore a collected file or directory into the sandbox at ``target``.

    The directory that receives it is made world-writable, as Harbor does when it restores
    artifacts: verifiers drop privileges before running agent output and need to write next to it.
    """
    directory = target if source.is_dir() else str(PurePosixPath(target).parent)
    result = await _exec_as_root(
        sandbox, f"mkdir -p {shlex.quote(directory)} && chmod 777 {shlex.quote(directory)}", timeout_s=60
    )
    if result.return_code:
        raise SandboxTransferError(f"Could not create {directory}: {result.stderr or result.stdout}")
    if source.is_dir():
        await upload_dir(sandbox, source, target, timeout_s=timeout_s)
    else:
        await sandbox.upload(source, target)
