# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded LAB artifact transfer/extraction worker; launched by artifacts.py."""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import resource
import stat
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO


@dataclass(frozen=True)
class ArchiveLimits:
    """Per-archive limits, deliberately larger than ordinary LAB deliverables."""

    max_archive_bytes: int = 4 * 1024**3
    max_expanded_bytes: int = 16 * 1024**3
    max_file_bytes: int = 8 * 1024**3
    max_entries: int = 100_000
    max_path_bytes: int = 4096
    max_path_depth: int = 64
    worker_memory_bytes: int = 8 * 1024**3
    timeout_seconds: float = 900

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"archive_limits.{name} must be positive and finite")
            if name != "timeout_seconds" and not isinstance(value, int):
                raise ValueError(f"archive_limits.{name} must be an integer")


def _check_limit(value: int, maximum: int, name: str) -> None:
    if value > maximum:
        raise ValueError(f"LAB archive exceeds archive_limits.{name} ({maximum})")


class _LimitedReader:
    """Count actual decoded bytes, including tar headers, padding, and PAX metadata."""

    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source = source
        self.limit = limit
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit - self.count
        data = self.source.read(min(size if size >= 0 else remaining + 1, remaining + 1))
        self.count += len(data)
        _check_limit(self.count, self.limit, "max_expanded_bytes")
        return data


def _extract_archive(source: Path, staging: Path, *, prefix: str, limits: ArchiveLimits) -> None:
    """Stream into a private directory. The parent publishes it only after worker success."""
    if prefix not in {"output", "verification"}:
        raise ValueError("Unexpected LAB artifact prefix")
    # Do not follow an unexpected link returned by a provider.
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as raw:
        info = os.fstat(raw.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("LAB archive must be a regular file")
        _check_limit(info.st_size, limits.max_archive_bytes, "max_archive_bytes")
        with gzip.GzipFile(fileobj=raw) as decoded:
            reader = _LimitedReader(decoded, limits.max_expanded_bytes)
            paths: dict[str, tuple[str, bool]] = {}
            entries: set[str] = set()
            total = 0
            # stream=True prevents TarFile from retaining every TarInfo object.
            with tarfile.open(fileobj=reader, mode="r|", stream=True) as archive:
                for index, member in enumerate(archive, start=1):
                    _check_limit(index, limits.max_entries, "max_entries")
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or not path.parts
                        or path.parts[0] != prefix
                        or "\\" in member.name
                        or any(ord(c) < 32 for c in member.name)
                        or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
                        or member.sparse is not None
                        or member.size < 0
                        or (member.isdir() and member.size != 0)
                        or (len(path.parts) == 1 and not member.isdir())
                    ):
                        raise ValueError(f"Unsafe LAB output archive entry: {member.name[:200]!r}")
                    _check_limit(len(member.name.encode("utf-8")), limits.max_path_bytes, "max_path_bytes")
                    _check_limit(len(path.parts), limits.max_path_depth, "max_path_depth")
                    key = str(path).casefold()
                    if key in entries:
                        raise ValueError("Duplicate LAB archive path")
                    entries.add(key)
                    # Include implicit directories in the filesystem-entry budget and reject aliases/conflicts.
                    for depth in range(1, len(path.parts) + 1):
                        part = PurePosixPath(*path.parts[:depth])
                        directory = depth < len(path.parts) or member.isdir()
                        old = paths.setdefault(str(part).casefold(), (str(part), directory))
                        if old != (str(part), directory):
                            raise ValueError("Conflicting LAB archive paths")
                    _check_limit(len(paths), limits.max_entries, "max_entries")
                    target = staging.joinpath(*path.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    _check_limit(member.size, limits.max_file_bytes, "max_file_bytes")
                    total += member.size
                    _check_limit(total, limits.max_expanded_bytes, "max_expanded_bytes")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = archive.extractfile(member)
                    if content is None:
                        raise ValueError("Missing LAB archive file content")
                    with content, target.open("xb") as output:
                        while chunk := content.read(1024 * 1024):
                            output.write(chunk)
            # Consume remaining compressed data too: padding/trailing gzip members must fit the budget and CRC.
            while reader.read(1024 * 1024):
                pass
    (staging / prefix).mkdir(exist_ok=True)


async def _download_archive(request: dict[str, object]) -> None:
    # The worker starts by filename, so add only this trusted checkout for Gym imports.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nemo_gym.sandbox import AsyncSandbox, create_provider

    provider = create_provider(request["provider"])
    box = None
    try:
        box = await AsyncSandbox.connect(request["descriptor"], provider=provider)
        await box.download(request["remote_path"], Path(request["target"]))
    finally:
        if box is not None:
            await box.disconnect()  # The Resources server still owns the sandbox.
        else:
            await provider.aclose()


def _set_resource_limits(limits: ArchiveLimits, *, download: bool) -> None:
    def cap(kind: int, value: int) -> None:
        _, hard = resource.getrlimit(kind)
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (value, value))

    cap(resource.RLIMIT_CORE, 0)
    cap(resource.RLIMIT_FSIZE, limits.max_archive_bytes if download else limits.max_file_bytes)
    cap(resource.RLIMIT_CPU, max(1, math.ceil(limits.timeout_seconds)))
    # Darwin does not enforce RLIMIT_AS; the parent also watches resident memory on all platforms.
    if sys.platform == "linux":
        cap(resource.RLIMIT_AS, limits.worker_memory_bytes)


def _main() -> None:
    request = json.load(sys.stdin)
    limits = ArchiveLimits(**request["limits"])
    download = request["operation"] == "download"
    _set_resource_limits(limits, download=download)
    try:
        if download:
            asyncio.run(_download_archive(request))
        else:
            _extract_archive(
                Path(request["source"]), Path(request["staging"]), prefix=request["prefix"], limits=limits
            )
    except Exception as error:
        # Provider exceptions can contain credentials; do not return their text to the caller.
        detail = type(error).__name__ if download else str(error)[:500]
        print(f"LAB archive {request['operation']} failed: {detail}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    _main()
