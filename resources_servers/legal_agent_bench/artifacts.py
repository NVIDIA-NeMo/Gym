# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB-only resource limits and publication for sandbox artifacts."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import psutil

from nemo_gym.sandbox import AsyncSandbox
from resources_servers.legal_agent_bench.archive_worker import ArchiveLimits


_WORKER = Path(__file__).with_name("archive_worker.py")


async def _run_worker(request: dict[str, object], *, limits: ArchiveLimits) -> None:
    # Provider credentials travel over stdin, never command arguments or artifact files.
    payload = json.dumps({**request, "limits": asdict(limits)}).encode()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        str(_WORKER),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    observed: dict[int, psutil.Process] = {}
    error_tail = bytearray()

    async def drain_errors() -> None:
        while chunk := await process.stderr.read(8192):
            error_tail.extend(chunk)
            del error_tail[:-8192]

    errors = asyncio.create_task(drain_errors())
    try:
        async with asyncio.timeout(limits.timeout_seconds):
            process.stdin.write(payload)
            await process.stdin.drain()
            process.stdin.close()
            parent = psutil.Process(process.pid)
            while process.returncode is None:
                try:
                    children = parent.children(recursive=True)
                    observed.update((child.pid, child) for child in children)
                    memory = parent.memory_info().rss
                    for child in children:
                        try:
                            memory += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    if memory > limits.worker_memory_bytes:
                        raise RuntimeError("LAB archive exceeds archive_limits.worker_memory_bytes")
                except psutil.NoSuchProcess:
                    break
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.1)
                except TimeoutError:
                    pass
            await process.wait()
            await errors
        if process.returncode:
            detail = error_tail.decode(errors="replace").strip() if request["operation"] == "extract" else ""
            raise ValueError(
                f"LAB archive {request['operation']} worker exited {process.returncode}; "
                f"check archive byte, memory and CPU limits. {detail}"
            )
    except TimeoutError as error:
        raise TimeoutError("LAB archive exceeds archive_limits.timeout_seconds") from error
    finally:
        # Reap the worker and any provider CLI children on errors, timeouts, and cancellation.
        try:
            if process.returncode is None:
                try:
                    observed.update((p.pid, p) for p in psutil.Process(process.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for child in observed.values():
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        await asyncio.shield(process.wait())
        if not errors.done():
            errors.cancel()
        await asyncio.gather(errors, return_exceptions=True)


async def download_archive(
    sandbox: AsyncSandbox,
    remote_path: str,
    destination: Path,
    *,
    provider_config: Mapping[str, object],
    limits: ArchiveLimits,
) -> None:
    """Download with resource limits using a borrowed connection; the resources server owns the sandbox."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lab-download-", dir=destination.parent) as temporary:
        target = Path(temporary) / "archive.tar.gz"
        target.touch(mode=0o600)
        await _run_worker(
            {
                "operation": "download",
                "provider": dict(provider_config),
                "descriptor": await sandbox.serialize(),
                "remote_path": remote_path,
                "target": str(target.resolve()),
            },
            limits=limits,
        )
        info = target.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > limits.max_archive_bytes:
            raise ValueError("LAB archive download is not a regular file within archive_limits.max_archive_bytes")
        os.replace(target, destination)


async def extract_outputs(source: Path, destination: Path, *, prefix: str, limits: ArchiveLimits) -> None:
    """Publish validated output only after bounded extraction completes successfully."""
    if prefix not in {"output", "verification"}:
        raise ValueError("Unexpected LAB artifact prefix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lab-extract-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        await _run_worker(
            {
                "operation": "extract",
                "source": str(source.absolute()),
                "staging": str(staging),
                "prefix": prefix,
            },
            limits=limits,
        )
        destination.mkdir(exist_ok=True)
        target = destination / prefix
        previous = staging / "previous"
        # A retry may replace a prior successful extraction, but never merge into its tree.
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise ValueError("LAB artifact destination is not a regular directory")
            target.rename(previous)
        try:
            (staging / prefix).rename(target)
        except BaseException:
            if previous.exists():
                previous.rename(target)
            raise
        if previous.exists():
            shutil.rmtree(previous)
