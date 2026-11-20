# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounds for the data-only tool workspace and model-visible results."""

from collections.abc import Mapping
from pathlib import PurePosixPath


MAX_FILE_BYTES = 4_000_000
MAX_OUTPUT_BYTES = 4_000_000
MAX_VIRTUAL_BYTES = 64_000_000


class CoverageGap(ValueError):
    """The frozen inputs do not determine this operation's result."""


def checked_path(path: str) -> str:
    """Reject relative paths and parent traversal before any virtual file access."""
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path or len(path.encode()) > 4096:
        raise CoverageGap("Virtual file paths must be absolute")
    if ".." in PurePosixPath(path).parts:
        raise CoverageGap("Virtual file path traverses a parent directory")
    return path


def bounded_output(text: str) -> str:
    """Refuse oversized results without silently truncating evidence."""
    if not isinstance(text, str) or len(text.encode()) > MAX_OUTPUT_BYTES:
        raise CoverageGap("Tool output exceeds the recorded-context ceiling")
    return text


class VirtualFiles(dict[str, str]):
    """A bounded in-memory filesystem shared by all modeled write paths."""

    def __init__(self, values: Mapping[str, str]) -> None:
        super().__init__()
        self._bytes = 0
        self.update(values)

    @staticmethod
    def _size(path: str, content: str) -> int:
        checked_path(path)
        if not isinstance(content, str) or len(content.encode()) > MAX_FILE_BYTES:
            raise CoverageGap("Virtual file exceeds the recorded-context ceiling")
        return len(content.encode())

    def __setitem__(self, path: str, content: str) -> None:
        self.update({path: content})

    def update(self, values: Mapping[str, str]) -> None:
        sizes = {path: self._size(path, content) for path, content in values.items()}
        total = self._bytes + sum(size - len(self.get(path, "").encode()) for path, size in sizes.items())
        if total > MAX_VIRTUAL_BYTES:
            raise CoverageGap("Virtual workspace exceeds the recorded-context ceiling")
        # Search materializes keys below a temporary root, ignoring repeated slashes and ".".
        keys = self.keys() | values.keys()
        paths = {PurePosixPath("/", key.lstrip("/")) for key in keys}
        if len(paths) != len(keys) or PurePosixPath("/") in paths:
            raise CoverageGap("Virtual files must have distinct paths below the workspace root")
        if any(parent in paths for path in paths for parent in path.parents):
            raise CoverageGap("Virtual files cannot also act as directories")
        super().update(values)
        self._bytes = total
