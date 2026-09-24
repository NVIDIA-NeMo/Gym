# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File and directory transfer through Gym sandbox operations."""

import hashlib
import json
import shlex
import tarfile
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult


async def stage_trusted_directory(sandbox: AsyncSandbox, source: Path, target: str) -> None:
    """Stage host-owned solution/tests as root, without changing task workspace permissions."""
    source = source.resolve()
    if target not in ("/solution", "/tests") or not source.is_dir():
        raise ValueError("Trusted staging requires a solution/tests directory and a fixed destination")
    for path in source.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(source):
            raise ValueError(f"Staged link escapes its directory: {path.relative_to(source)}")

    def metadata(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = 0
        member.uname = member.gname = "root"
        if target == "/solution":
            # The declared agent must be able to read reference assets staged by
            # the harness. This grants no new access to the image's workspace.
            member.mode |= 0o555 if member.isdir() else 0o444
        return member

    remote = f"/tmp/.nemo-gym-trusted-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "trusted.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".", filter=metadata)
        await sandbox.upload(archive, remote)
    try:
        result = await sandbox.exec(
            f"test ! -L {target} && mkdir -p {target} && "
            f"find {target} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + && "
            f"tar --no-same-owner --same-permissions -xzf {remote} -C {target}",
            user="root",
            timeout_s=600,
        )
        if result.return_code:
            raise RuntimeError(f"Trusted staging into {target} failed: {result.stderr}")
    finally:
        await sandbox.exec(f"rm -f {remote}", user="root", timeout_s=60)


def artifact_metadata_path(directory: Path, host_path: Path) -> Path:
    """Keep trusted transfer metadata outside the agent-controlled artifact tree."""
    key = hashlib.sha256(host_path.as_posix().encode()).hexdigest()
    return directory.parent / "artifact-metadata" / f"{key}.json"


def _write_metadata(path: Path, metadata: dict[str, list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, sort_keys=True))


def _read_metadata(path: Path) -> dict[str, list[int]]:
    if not path.is_file():
        raise RuntimeError(f"Artifact ownership/permissions metadata is missing: {path}")
    return json.loads(path.read_text())


async def upload_file(sandbox: AsyncSandbox, source: Path, target: str, *, metadata_path: Path | None = None) -> None:
    metadata = _read_metadata(metadata_path)["."] if metadata_path else None
    await sandbox.exec(f"mkdir -p {shlex.quote(str(PurePosixPath(target).parent))}", timeout_s=60)
    await sandbox.upload(Path(source), target)
    if metadata:
        uid, gid, mode = metadata
        quoted = shlex.quote(target)
        result = await sandbox.exec(
            f"test ! -L {quoted} && chown {uid}:{gid} -- {quoted} && chmod {mode:o} -- {quoted}",
            user="root",
            timeout_s=60,
        )
        if result.return_code:
            raise RuntimeError(f"Failed to restore artifact metadata for {target}: {result.stderr}")


async def upload_dir(sandbox: AsyncSandbox, source: Path, target: str, *, metadata_path: Path | None = None) -> None:
    source = Path(source)
    metadata = _read_metadata(metadata_path) if metadata_path else None

    def headers(member: tarfile.TarInfo) -> tarfile.TarInfo:
        if metadata is not None:
            key = PurePosixPath(member.name).as_posix()
            if key not in metadata:
                raise RuntimeError(f"Artifact metadata is missing for {key}")
            member.uid, member.gid, member.mode = metadata[key]
            # Names in two images may resolve to different numeric identities.
            member.uname = member.gname = ""
        return member

    remote = f"/tmp/.nemo-gym-upload-{uuid4().hex}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "upload.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname=".", filter=headers)
        await sandbox.upload(archive, remote)
    flags = "--numeric-owner --same-owner --same-permissions " if metadata is not None else ""
    result = await sandbox.exec(
        f"mkdir -p {shlex.quote(target)} && tar {flags}-xzf {remote} -C {shlex.quote(target)}; "
        f"status=$?; rm -f {remote}; exit $status",
        timeout_s=600,
        **({"user": "root"} if metadata is not None else {}),
    )
    if result.return_code:
        if metadata is not None:
            raise RuntimeError(f"Failed to restore artifact {target} with its metadata: {result.stderr}")
        for path in sorted(source.rglob("*")):
            destination = str(PurePosixPath(target) / path.relative_to(source).as_posix())
            if path.is_dir():
                await sandbox.exec(f"mkdir -p {shlex.quote(destination)}", timeout_s=60)
            elif path.is_file():
                await upload_file(sandbox, path, destination)


async def download_file(
    sandbox: AsyncSandbox, source: str, target: Path, *, metadata_path: Path | None = None
) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path:
        result = await sandbox.exec(f"stat -Lc '%u %g %a' -- {shlex.quote(source)}", timeout_s=60)
        if result.return_code:
            raise RuntimeError(f"Failed to read artifact metadata for {source}: {result.stderr}")
        uid, gid, mode = result.stdout.split()
        metadata = {".": [int(uid), int(gid), int(mode, 8) & 0o777]}
    await sandbox.download(source, target)
    if metadata_path:
        _write_metadata(metadata_path, metadata)


async def download_dir(
    sandbox: AsyncSandbox,
    source: str,
    target: Path,
    *,
    exclude: list[str] | None = None,
    exec_command: Callable[..., Awaitable[SandboxExecResult]] | None = None,
    shared_archive: str | None = None,
    metadata_path: Path | None = None,
) -> str | None:
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
            if exclude or metadata_path:
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
            metadata = {}

            def data_filter(member: tarfile.TarInfo, destination: str) -> tarfile.TarInfo:
                filtered = tarfile.data_filter(member, destination)
                # Keep the safe host extraction, but never mistake host ownership
                # or umask-derived directory modes for the sandbox's metadata.
                # Set-ID/sticky bits are not propagated from untrusted artifacts.
                metadata[PurePosixPath(filtered.name).as_posix()] = [member.uid, member.gid, member.mode & 0o777]
                return filtered

            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target, filter=data_filter)
            if metadata_path:
                _write_metadata(metadata_path, metadata)
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
