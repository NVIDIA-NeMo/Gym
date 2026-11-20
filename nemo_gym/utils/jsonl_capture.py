# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked append and incremental reads shared by capture streams."""

import fcntl
import os
from collections.abc import Iterator
from pathlib import Path


def append_jsonl(path: Path, line: bytes, *, durable: bool = True, recover_tail: bool = False) -> None:
    """Flush one encoded record before returning, serialized across writers.

    With ``recover_tail``, an unterminated last line (a writer that died mid-append) is ended first, so it stays a
    damaged record of its own instead of swallowing this one. That needs read access to the file.
    """
    with path.open("a+b" if recover_tail else "ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if recover_tail:
                handle.seek(0, os.SEEK_END)
                if handle.tell():
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
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
