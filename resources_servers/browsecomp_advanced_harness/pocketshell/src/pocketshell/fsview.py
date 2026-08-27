# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Workspace-confined filesystem access.

This is the module that replaces the sandbox. The previous design ran
``bash -c`` as a subprocess and relied on a command-NAME deny/allow list, which
never inspected path arguments -- so ``cat /home/<user>/.bashrc``,
``cat /etc/passwd`` and ``grep -r API_KEY /home/<user>`` were all accepted by
the guard. Here every path is resolved and checked against the workspace root
before any byte is read, so escape is structurally impossible rather than
enumerated.
"""

from __future__ import annotations

import os
from pathlib import Path


__all__ = ["Workspace", "PathEscape"]

# Read cap per file, so one pathological page cannot exhaust memory.
MAX_FILE_BYTES = 64 * 1024 * 1024


class PathEscape(PermissionError):
    """Raised when a path argument resolves outside the workspace."""


class Workspace:
    """A directory tree that commands may read, and nothing else."""

    def __init__(self, root: str | os.PathLike, cwd: str | os.PathLike | None = None):
        self.root = Path(root).resolve()
        self.cwd = Path(cwd).resolve() if cwd else self.root
        if not self._inside(self.cwd):
            raise PathEscape(f"cwd {self.cwd} is outside workspace {self.root}")

    def _inside(self, p: Path) -> bool:
        try:
            return p == self.root or self.root in p.parents
        except Exception:
            return False

    def resolve(self, arg: str) -> Path:
        """Resolve a user-supplied path argument, confined to the workspace.

        Symlinks are resolved before the check, so a symlink inside the
        workspace cannot be used to read outside it.
        """
        p = Path(arg)
        if not p.is_absolute():
            p = self.cwd / p
        try:
            rp = p.resolve()
        except (OSError, RuntimeError) as exc:
            raise PathEscape(f"{arg}: cannot resolve path") from exc
        if not self._inside(rp):
            raise PathEscape(f"{arg}: Permission denied (outside workspace)")
        return rp

    def display(self, p: Path) -> str:
        """Path as the agent should see it: relative to cwd where possible."""
        try:
            return str(p.relative_to(self.cwd))
        except ValueError:
            try:
                return str(p.relative_to(self.root))
            except ValueError:
                return str(p)

    # -- reads ------------------------------------------------------------
    def read_text(self, arg: str) -> str:
        p = self.resolve(arg)
        if p.is_dir():
            raise IsADirectoryError(f"{arg}: Is a directory")
        with open(p, "r", encoding="utf-8", errors="surrogateescape") as fh:
            return fh.read(MAX_FILE_BYTES)

    def read_bytes(self, arg: str) -> bytes:
        p = self.resolve(arg)
        if p.is_dir():
            raise IsADirectoryError(f"{arg}: Is a directory")
        with open(p, "rb") as fh:
            return fh.read(MAX_FILE_BYTES)

    def chdir(self, arg: str) -> None:
        p = self.resolve(arg)
        if not p.is_dir():
            raise NotADirectoryError(f"{arg}: Not a directory")
        self.cwd = p
