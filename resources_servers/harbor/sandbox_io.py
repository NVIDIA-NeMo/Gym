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
        tar.add(source, arcname=".")


def _unpack(archive: Path, target: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target, filter="data")


async def upload_dir(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Copy the contents of ``source`` into ``target`` inside the sandbox."""
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        # Archiving is blocking file work; keep it off the event loop.
        await asyncio.to_thread(_pack, source, archive)
        await sandbox.upload(archive, remote)
    result = await sandbox.exec(
        f"mkdir -p {shlex.quote(target)} && tar -xzf {remote} -C {shlex.quote(target)}; "
        f"status=$?; rm -f {remote}; exit $status",
        timeout_s=timeout_s,
    )
    if result.return_code:
        raise SandboxTransferError(f"Failed to unpack {source} into {target}: {result.stderr or result.stdout}")


async def download_dir(sandbox: AsyncSandbox, source: str, target: Path, *, timeout_s: float = 600) -> None:
    """Copy the contents of ``source`` inside the sandbox into the local ``target`` folder."""
    target.mkdir(parents=True, exist_ok=True)
    remote = f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    result = await sandbox.exec(f"tar -czf {remote} -C {shlex.quote(source)} .", timeout_s=timeout_s)
    if result.return_code:
        raise SandboxTransferError(f"Failed to archive {source}: {result.stderr or result.stdout}")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download.tar.gz"
            await sandbox.download(remote, archive)
            await asyncio.to_thread(_unpack, archive, target)
    finally:
        await sandbox.exec(f"rm -f {remote}", timeout_s=60)


async def remote_kind(sandbox: AsyncSandbox, path: str) -> str | None:
    """``"dir"``, ``"file"`` or ``None`` for a path inside the sandbox."""
    quoted = shlex.quote(path)
    result = await sandbox.exec(
        f"if [ -d {quoted} ]; then echo dir; elif [ -e {quoted} ]; then echo file; else echo none; fi", timeout_s=60
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
    """Copy a local file or directory into the sandbox at ``target``, creating parents."""
    if source.is_dir():
        await upload_dir(sandbox, source, target, timeout_s=timeout_s)
        return
    parent = str(PurePosixPath(target).parent)
    result = await sandbox.exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=60)
    if result.return_code:
        raise SandboxTransferError(f"Could not create {parent}: {result.stderr or result.stdout}")
    await sandbox.upload(source, target)
