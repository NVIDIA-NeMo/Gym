# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked append and incremental reads shared by capture streams."""

import fcntl
import os
from collections.abc import Iterator
from pathlib import Path


def append_jsonl(path: Path, line: bytes, *, durable: bool = True) -> None:
    """Flush one encoded record before returning, serialized across writers."""
    with path.open("ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def jsonl_lines(path: Path) -> Iterator[bytes]:
    """Read nonempty lines under a shared lock, including an unfinished tail."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            for line in handle:
                if stripped := line.strip():
                    yield stripped
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
