# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse content verification only while the same file identity remains unchanged."""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


class VerificationCache:
    """Keep attributable hash receipts outside immutable sources and run inputs.

    A cache hit reuses an earlier digest after checking path, device, inode, size,
    mode and nanosecond modification/change times. It is recorded as reused
    evidence, not as newly hashed bytes. Changed or unreadable receipts cause a
    fresh hash. These receipts are local filesystem evidence, not remote signatures.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.events: list[dict[str, object]] = []

    @staticmethod
    def identity(path: Path) -> dict[str, object]:
        """Identify the actual file behind a model snapshot's possible symlink."""
        resolved = path.resolve(strict=True)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Verification requires a regular file: {path}")
        return {
            "path": str(path.absolute()),
            "resolved": str(resolved),
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mode": info.st_mode,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }

    def verify(self, path: Path, *, algorithm: str, expected: str) -> None:
        """Verify one expected digest or reuse its unchanged file-identity receipt."""
        if algorithm not in {"sha256", "git_blob_sha1"}:
            raise ValueError(f"Unsupported content hash: {algorithm}")
        before = self.identity(path)
        key = hashlib.sha256(json.dumps([str(path.absolute()), algorithm, expected]).encode()).hexdigest()
        receipt_path = self.directory / f"{key}.json"
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            receipt = None
        wanted = {"version": 1, "identity": before, "algorithm": algorithm, "digest": expected}
        reused = receipt == wanted
        if not reused:
            hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git_blob_sha1":
                hasher.update(f"blob {before['size']}\0".encode())
            with path.open("rb") as stream:
                while block := stream.read(4 * 1024 * 1024):
                    hasher.update(block)
            if hasher.hexdigest() != expected:
                raise ValueError(f"Content differs from its expected {algorithm}: {path}")
        if self.identity(path) != before:
            raise ValueError(f"File changed during verification: {path}")
        if not reused:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=self.directory)
            try:
                with os.fdopen(fd, "w") as stream:
                    json.dump(wanted, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, receipt_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self.events.append(
            {
                "path": str(path),
                "algorithm": algorithm,
                "digest": expected,
                "evidence": "reused_unchanged_file_identity" if reused else "content_hashed",
                "receipt": str(receipt_path),
            }
        )
