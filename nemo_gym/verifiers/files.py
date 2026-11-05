# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File checks for Python verifiers: the agent's workspace is a directory, these read it.

Standard library only (see :mod:`nemo_gym.verifiers`). Paths are relative to the workspace; a path that
escapes it (``..`` or an absolute path outside it) reads as absent, so a task cannot be graded on files the
agent never owned.
"""

from __future__ import annotations

from pathlib import Path


__all__ = ["file_exists", "read_text", "text_file_equals"]


def _without_trailing_newline(text: str) -> str:
    # Not the 3.9+ string method for it: this module runs inside the task image under whatever ``python3`` it has,
    # and the vendored package keeps to Python 3.8 (the floor the shim's test.sh checks).
    return text[:-1] if text.endswith("\n") else text


def _resolve(workspace: Path | str, path: str | Path) -> Path | None:
    root = Path(workspace).resolve()
    candidate = (root / path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def file_exists(workspace: Path | str, path: str | Path) -> bool:
    """Whether ``path`` under the workspace is a regular file."""
    resolved = _resolve(workspace, path)
    return resolved is not None and resolved.is_file()


def read_text(
    workspace: Path | str, path: str | Path, *, encoding: str = "utf-8", strip_trailing_newline: bool = True
) -> str | None:
    """The text of ``path`` under the workspace, or ``None`` when it is absent or not a file.

    One trailing newline is dropped by default (``echo`` and most editors end a file with one, and a verifier
    comparing ``read_text(...) == "answer"`` should not fail on it); pass ``strip_trailing_newline=False`` for the
    bytes as written. Undecodable bytes are replaced rather than raised, so a verifier comparing text never
    crashes on binary output.
    """
    resolved = _resolve(workspace, path)
    if resolved is None or not resolved.is_file():
        return None
    try:
        text = resolved.read_text(encoding=encoding, errors="replace")
    except OSError:
        return None
    return _without_trailing_newline(text) if strip_trailing_newline else text


def text_file_equals(
    workspace: Path | str,
    *,
    path: str | Path,
    content: str,
    strip_trailing_newline: bool = True,
) -> float:
    """``1.0`` when ``path`` under the workspace holds exactly ``content`` (one trailing newline tolerated).

    A complete verifier: ``verify = functools.partial(text_file_equals, path="hello.txt", content="hello")``.
    """
    actual = read_text(workspace, path, strip_trailing_newline=False)
    if actual is None:
        return 0.0
    if strip_trailing_newline:
        actual = _without_trailing_newline(actual)
        content = _without_trailing_newline(content)
    return float(actual == content)
