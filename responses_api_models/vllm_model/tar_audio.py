# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read audio members from uncompressed tar archives without extracting files."""

import os
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import BinaryIO


def validate_relative_path(value: str, *, field: str) -> None:
    """Require a normalized relative POSIX path for a portable audio reference."""
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a nonempty relative POSIX path.")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field}={value!r} must be normalized and cannot contain absolute paths or traversal.")


def resolve_archive_path(archive: str, *, audio_root: str | None) -> Path:
    """Resolve a relative tar path inside audio_root, including symlink checks."""
    validate_relative_path(archive, field="metadata.audio_ref.archive")
    if Path(archive).suffix.lower() != ".tar":
        raise ValueError("metadata.audio_ref.archive must name an uncompressed .tar file.")
    if not audio_root:
        raise ValueError(
            "metadata.audio_ref requires VLLMModelConfig.audio_root to resolve its relative archive path."
        )
    root = Path(audio_root).resolve()
    resolved = (root / archive).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"metadata.audio_ref.archive={archive!r} resolves outside audio_root={audio_root!r}.")
    return resolved


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@dataclass(frozen=True)
class _ArchiveIndex:
    identity: tuple[int, int, int, int, int]
    # None records non-regular entries so attempts to read a link fail explicitly.
    members: dict[str, tuple[int, int] | None]


class TarAudioReader:
    """Cache validated member offsets and give every read its own file handle.

    Cache entries are invalidated when the archive's file identity changes.
    The lock protects initial indexing; subsequent reads have independent seek
    positions and do not retain open descriptors or decoded audio in the cache.
    """

    def __init__(self) -> None:
        self._indexes: dict[Path, _ArchiveIndex] = {}
        self._lock = Lock()

    @staticmethod
    def _build_index(handle: BinaryIO, *, identity: tuple[int, int, int, int, int]) -> _ArchiveIndex:
        members: dict[str, tuple[int, int] | None] = {}
        try:
            with tarfile.open(fileobj=handle, mode="r:") as archive:
                for member in archive:
                    name = member.name.rstrip("/") if member.isdir() else member.name
                    validate_relative_path(name, field="Tar member name")
                    if name in members:
                        raise ValueError(f"Audio archive contains duplicate member {name!r}.")
                    if member.isreg() and not member.issparse():
                        if member.offset_data < 0 or member.size < 0 or member.offset_data + member.size > identity[2]:
                            raise ValueError(f"Audio archive member {name!r} extends beyond the archive file.")
                        members[name] = member.offset_data, member.size
                    else:
                        members[name] = None
        except tarfile.TarError as error:
            raise ValueError("Audio archive must be a valid uncompressed tar file.") from error
        return _ArchiveIndex(identity=identity, members=members)

    def read_member(self, archive_path: Path, *, member: str) -> bytes:
        """Read one regular, contiguous member and reject stale or truncated data."""
        validate_relative_path(member, field="metadata.audio_ref.member")
        try:
            handle = archive_path.open("rb")
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Audio archive {str(archive_path)!r} does not exist.") from error
        with handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"Audio archive {str(archive_path)!r} must be a regular file.")
            identity = _file_identity(info)
            with self._lock:
                index = self._indexes.get(archive_path)
                if index is None or index.identity != identity:
                    index = self._build_index(handle, identity=identity)
                    self._indexes[archive_path] = index
            if member not in index.members:
                raise FileNotFoundError(f"Audio archive {str(archive_path)!r} has no member {member!r}.")
            location = index.members[member]
            if location is None:
                raise ValueError(f"Audio archive member {member!r} must be a regular, non-sparse file.")
            offset, size = location
            handle.seek(offset)
            audio = handle.read(size)
            if len(audio) != size or _file_identity(os.fstat(handle.fileno())) != identity:
                raise ValueError(
                    f"Audio archive {str(archive_path)!r} was truncated or changed while reading {member!r}."
                )
            return audio
