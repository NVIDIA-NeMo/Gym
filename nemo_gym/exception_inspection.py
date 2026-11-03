# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read facts from any Python exception: its explicit causes, its type name and its HTTP status.

These functions know nothing about Gym's failure types; ``nemo_gym.typed_exceptions`` builds on them.
"""

from collections.abc import Iterator


TRUNCATION_MARKER = " [truncated]"


def explicit_causes(error: BaseException) -> Iterator[BaseException]:
    """``error`` and then each ``__cause__`` (``raise ... from``), outermost first.

    ``__context__`` (an exception that was only being handled, such as a retry's earlier attempt) is not a
    cause and is not followed. A cycle stops the walk.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def qualified_type_name(error: BaseException) -> str:
    """``module.QualName`` of ``error``'s type; a builtin type is its bare name."""
    kind = type(error)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def error_to_http_status(error: BaseException) -> int | None:
    """The HTTP status ``error`` carries as an int: ``status`` (aiohttp), ``status_code`` (FastAPI, SDKs) or
    ``http_status`` (Gym); ``None`` when it carries none."""
    for name in ("status", "status_code", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def bounded_text(text: str, max_chars: int) -> str:
    """``text``, or its start with ``TRUNCATION_MARKER`` when it is longer than ``max_chars``."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
