# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming plain/Zstandard JSONL IO; JSON encoding belongs to callers."""

import hashlib
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, TextIO


if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd


class _FrameFlushingZstdFile(zstd.ZstdFile):
    """Make ordinary flush + fsync publish a complete, independently readable frame."""

    def flush(self, mode=zstd.ZstdFile.FLUSH_FRAME):
        super().flush(mode)


def open_jsonl(path: str | Path, mode: str = "rt") -> BinaryIO | TextIO:
    """Open plain or .zst JSONL without materializing the decompressed file.

    Compressed reads accept concatenated frames and raise on truncated frames. Writers
    finish a frame on flush(), preserving append/resume after each flush + fsync.
    No JSON parsing or serialization is performed. Text streams use UTF-8.
    """
    if mode not in {"r", "rt", "rb", "w", "wt", "wb", "a", "at", "ab"}:
        raise ValueError(f"Unsupported JSONL mode: {mode!r}")
    path = Path(path)
    binary = "b" in mode
    if path.suffix != ".zst":
        return path.open(mode, **({} if binary else {"encoding": "utf-8"}))
    options = (
        None
        if mode.startswith("r")
        else {
            zstd.CompressionParameter.compression_level: 3,
            zstd.CompressionParameter.checksum_flag: 1,
        }
    )
    stream = _FrameFlushingZstdFile(path, mode[0] + "b", options=options)
    if not mode.startswith("r"):
        # Initialize an empty frame too: upstream ZstdFile otherwise leaves an
        # unwritten output empty, which a strict reader correctly rejects.
        stream.write(b"")
    return stream if binary else io.TextIOWrapper(stream, encoding="utf-8")


def _digest(stream: BinaryIO) -> bytes:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.digest()


def same_jsonl_bytes(plain: Path, compressed: Path) -> bool:
    """Compare without loading either full file; also validates compressed framing."""
    with plain.open("rb") as source, open_jsonl(compressed, "rb") as restored:
        return _digest(source) == _digest(restored)


def compress_jsonl(path: str | Path, *, remove_source: bool = True) -> Path:
    """Atomically publish a byte-verified .zst copy of a quiescent plain file.

    The caller must exclude concurrent writers, including throughout source removal.
    An existing destination is never overwritten. If an earlier publish succeeded but
    source removal was interrupted, identical copies are safely reconciled.
    """
    path = Path(path)
    if path.suffix == ".zst":
        raise ValueError("Expected an uncompressed JSONL path")
    destination = Path(str(path) + ".zst")
    if destination.exists():
        if not same_jsonl_bytes(path, destination):
            raise FileExistsError(f"Conflicting plain and compressed JSONL files: {path}")
    else:
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".zst", dir=path.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with path.open("rb") as source, open_jsonl(temporary, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                # mkstemp defaults to 0600. Preserve ordinary source access permissions
                # for shared archives, without copying special setuid/setgid/sticky bits.
                os.fchmod(target.fileno(), os.fstat(source.fileno()).st_mode & 0o777)
                os.fsync(target.fileno())
            if not same_jsonl_bytes(path, temporary):
                raise OSError(f"Compressed JSONL verification failed: {path}")
            # link publishes without replacing another publisher's destination.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    # Persist publication before removing the only original copy.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
        if remove_source:
            path.unlink()
            os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination
