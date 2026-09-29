# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File and directory transfer through Gym sandbox operations."""

import hashlib
import shlex
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4


async def upload_file(sandbox, source, target):
    await sandbox.exec(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}", timeout_s=60)
    await sandbox.upload(Path(source), target)


async def upload_dir(sandbox, source, target):
    source = Path(source)
    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".")
        await sandbox.upload(archive, remote)
    result = await sandbox.exec(
        f"mkdir -p {shlex.quote(target)} && tar -xzf {remote} -C {shlex.quote(target)}; "
        f"status=$?; rm -f {remote}; exit $status",
        timeout_s=600,
    )
    if result.return_code:
        for path in sorted(source.rglob("*")):
            destination = str(PurePosixPath(target) / path.relative_to(source).as_posix())
            if path.is_dir():
                await sandbox.exec(f"mkdir -p {shlex.quote(destination)}", timeout_s=60)
            elif path.is_file():
                await upload_file(sandbox, path, destination)


async def download_file(sandbox, source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    await sandbox.download(source, target)


async def download_dir(sandbox, source, target, *, exclude=None, exec_command=None, shared_archive=None):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    remote = shared_archive or f"/tmp/.nemo-gym-download-{uuid4().hex}.tar.gz"
    retained = False
    flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in (exclude or []))
    command = f"tar -czf {remote} {flags} -C {shlex.quote(source)} ."
    # The reference uses role shell/root for exclusions; plain directory
    # transfers use the provider's default execution user and shell.
    if exclude and exec_command:
        result = await exec_command(command, timeout_sec=600, user="root")
    else:
        result = await sandbox.exec(command, timeout_s=600)
    try:
        if result.return_code:
            if exclude:
                raise RuntimeError(f"Failed to archive {source}: {result.stderr}")
            listing = await sandbox.exec(f"find {shlex.quote(source)} -type f", timeout_s=120)
            if listing.return_code:
                raise RuntimeError(f"Failed to list {source}: {listing.stderr}")
            for line in (listing.stdout or "").splitlines():
                if line.strip():
                    relative = PurePosixPath(line.strip()).relative_to(PurePosixPath(source))
                    await download_file(sandbox, line.strip(), target / relative)
            return
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "download.tar.gz"
            await sandbox.download(remote, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target, filter="data")
            if shared_archive:
                with archive.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                retained = True
                return digest
    finally:
        if not retained:
            await sandbox.exec(f"rm -f {remote}", timeout_s=60)


async def prepare_directory(environment, path, *, empty=False):
    quoted = shlex.quote(path)
    commands = []
    if empty:
        commands.append(f"if [ -L {quoted} ] || {{ [ -e {quoted} ] && [ ! -d {quoted} ]; }}; then rm -rf {quoted}; fi")
    commands.append(f"mkdir -p {quoted}")
    if empty:
        commands.append(f"find {quoted} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +")
    commands.append(f"chmod 777 {quoted}")
    # Reference preparation is best effort; the following transfer/test detects
    # an unusable directory and retains the provider's concrete error.
    await environment.exec(" && ".join(commands), user="root")
