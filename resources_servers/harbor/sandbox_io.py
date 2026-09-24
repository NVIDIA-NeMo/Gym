# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory transfers through sandbox exec, upload and download."""

import shlex
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox


class SandboxTransferError(RuntimeError):
    """A transfer command failed inside the sandbox."""


async def upload_dir(sandbox: AsyncSandbox, source: Path, target: str, *, timeout_s: float = 600) -> None:
    """Copy the contents of ``source`` into ``target`` inside the sandbox."""
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".")
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
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target, filter="data")
    finally:
        await sandbox.exec(f"rm -f {remote}", timeout_s=60)
