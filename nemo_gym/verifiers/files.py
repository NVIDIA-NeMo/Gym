# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers and verifiers for workspace files."""

from typing import Protocol


class Workspace(Protocol):
    async def read_text(self, path: str) -> str: ...


class Attempt(Protocol):
    workspace: Workspace


class TextFileVerifierInput(Protocol):
    path: str
    content: str


async def read_text(attempt: Attempt, path: str) -> str | None:
    """Read UTF-8 workspace text, returning ``None`` when it is unavailable."""
    try:
        return await attempt.workspace.read_text(path)
    except (OSError, UnicodeError):
        return None


async def text_file_equals(attempt: Attempt, verifier_input: TextFileVerifierInput) -> float:
    """Score a text file, accepting the conventional single final newline."""
    actual = await read_text(attempt, verifier_input.path)
    if actual is None:
        return 0.0
    matches = actual.removesuffix("\n") == verifier_input.content.removesuffix("\n")
    return float(matches)
