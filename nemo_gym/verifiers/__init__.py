# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for a Harbor task's Python verifier (``tests/verifier.py``, or ``verifier.py`` at the dataset root).

A Python verifier is a module with a ``verify`` function. It runs inside the task's verifier sandbox, never
on the host: the ``harbor`` environment server (and ``gym dataset publish --harbor``) wraps it in a
``tests/test.sh`` shim that carries this package along, so ``from nemo_gym.verifiers import ...`` works in
the sandbox with nothing installed. Everything here is therefore standard library only, and this package
imports nothing else from ``nemo_gym``.

``verify`` declares the arguments it wants and the runner passes those alone (by name, or by position for a
positional-only parameter such as ``def verify(workspace, /)``):

- ``workspace`` (``pathlib.Path``): the task's working directory, where the agent worked;
- ``task_data`` (``dict``): the task's own fields (``task_data.json``, a parameterized row);
- ``response`` (``dict`` or ``None``): the agent's final Responses API response (not available to a
  published Harbor task, whose ``test.sh`` sees only the container);
- ``state`` (``dict``): ``{"workspace": Path}``, the RFC's spelling of the workspace.

It returns a number (the reward) or a dict with a ``reward`` key and other numeric keys as metrics; an
``async def verify`` is awaited. A ``verify`` that requires ``response`` without a default is a contract error (no
reward) wherever no response exists, which is every published Harbor task.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


__all__ = ["exact_match", "final_text", "numeric_match"]


def final_text(response: Mapping[str, Any] | None) -> str:
    """The text of the agent's final assistant message in a Responses API response (``""`` when there is none).

    Walks ``response["output"]`` from the end and returns the concatenated ``output_text`` parts of the last
    ``message`` item; a response that is ``None`` or carries no message yields ``""``.
    """
    if not isinstance(response, Mapping):
        return ""
    for item in reversed(response.get("output") or []):
        if not isinstance(item, Mapping) or item.get("type", "message") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        parts = []
        for part in content or []:
            if isinstance(part, Mapping) and part.get("type", "output_text") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def exact_match(actual: str, expected: str, *, strip: bool = True, case_sensitive: bool = True) -> float:
    """``1.0`` when the two strings agree, after stripping surrounding whitespace by default."""
    left, right = (actual.strip(), expected.strip()) if strip else (actual, expected)
    if not case_sensitive:
        left, right = left.casefold(), right.casefold()
    return float(left == right)


_NUMBER = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def _parse_number(text: str) -> float | None:
    match = None
    for match in _NUMBER.finditer(text):
        pass
    if match is None:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def numeric_match(actual: str | float, expected: str | float, *, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> float:
    """``1.0`` when the last number in ``actual`` equals ``expected`` within the tolerances.

    Strings are searched for their last number (``"The answer is 72."`` reads as ``72``; thousands separators
    are dropped); a string without one scores ``0.0``.
    """
    left = _parse_number(actual) if isinstance(actual, str) else float(actual)
    right = _parse_number(expected) if isinstance(expected, str) else float(expected)
    if left is None or right is None:
        return 0.0
    return float(math.isclose(left, right, rel_tol=rel_tol, abs_tol=abs_tol))
