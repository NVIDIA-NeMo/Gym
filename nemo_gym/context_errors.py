# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared context-window error classification for model clients and agents."""

from __future__ import annotations

import json
import re
from typing import Any


_CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length|"
    r"context[_ ]length[_ ]exceeded|"
    r"context length is (?:only )?\d+ tokens|"
    r"maximum input length|"
    r"please reduce the length of the input|"
    r"exceed.*context (?:limit|window|length)|"
    r"context window exceeds|"
    r"exceeds maximum length|"
    r"too long.*tokens.*maximum|"
    r"too large for model with \d+ maximum context length|"
    r"longer than the model's context length|"
    r"too many tokens|"
    r"prompt is too long|"
    r"maximum prompt length|"
    r"input length should be|"
    r"sent message larger than max|"
    r"input tokens exceeded|"
    r"(?:messages?|total length).*too long|"
    r"payload.*too large|"
    r"string too long|"
    r"input exceeded the context window|"
    r"cannot be greater than max_model_len|"
    r"`inputs` tokens \+ `max_new_tokens`",
    re.IGNORECASE,
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def error_text(error: Any) -> str:
    """Return searchable text, including HTTP bodies omitted by exception ``str()``."""

    parts = [_stringify(error)]
    for attribute in ("response_content", "body"):
        text = _stringify(getattr(error, attribute, None))
        if text and text not in parts:
            parts.append(text)
    return " ".join(part for part in parts if part)


def is_context_overflow_error(error: Any) -> bool:
    """Whether text, a response body, or an exception describes context overflow."""

    return _CONTEXT_OVERFLOW_RE.search(error_text(error)) is not None
