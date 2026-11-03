# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read facts from any Python exception: its explicit causes, its type name and its HTTP status.

These functions know nothing about Gym's failure types; ``nemo_gym.core.typed_exceptions`` builds on them.
"""

import re
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


def provider_error_reason(error: BaseException) -> str:
    """The provider's own code for an SDK error (OpenSandbox's ``error.code``, e.g. ``SANDBOX::ENDED``), else the
    error's type name."""
    code = getattr(getattr(error, "error", None), "code", None)
    return code if isinstance(code, str) and code else type(error).__name__


def sandbox_failure_prefix(
    *,
    phase: str,
    provider: str | None = None,
    sandbox_id: str | None = None,
    reason: str | None = None,
    http_status: int | None = None,
    exit_code: int | None = None,
    signal: str | None = None,
    command: str | None = None,
) -> str:
    """``[sandbox phase=... provider=... sandbox=... reason=... http=... exit=... signal=... command=...] ``: what a
    producer knows about a sandbox failure, as a fixed-order prefix of its ``failure_reason``, so a saved record says
    in which phase, on which sandbox and with which provider status it failed. Unknown facts are left out; each value
    is one bounded token. It decides nothing: no reader classifies a failure from it."""
    facts = {
        "phase": phase,
        "provider": provider,
        "sandbox": sandbox_id,
        "reason": reason,
        "http": http_status,
        "exit": exit_code,
        "signal": signal,
        "command": command,
    }
    tokens = [
        f"{name}={re.sub(r'[\s\[\]=]+', '_', str(value))[:128]}"
        for name, value in facts.items()
        if value not in (None, "")
    ]
    return "[sandbox " + " ".join(tokens) + "] "


def bounded_text(text: str, max_chars: int) -> str:
    """``text``, or its start with ``TRUNCATION_MARKER`` when it is longer than ``max_chars``."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
