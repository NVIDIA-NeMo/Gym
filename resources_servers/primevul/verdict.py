# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse PrimeVul's published YES/NO classification response."""

import re


# Thinking models emit their chain of thought in these blocks. A draft verdict inside one is not
# the answer, so remove it before looking for an explicit option.
_REASONING_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_BINARY_VERDICT = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
_NUMBERED_VERDICT = re.compile(r"^\s*\(?([12])\)?\s*$")


def strip_reasoning(text: str) -> str:
    """Drop `<think>`/`<thinking>` blocks from a model message."""
    return _REASONING_BLOCK.sub("", text or "")


def parse_verdict(text: str) -> dict:
    """Return the final YES/NO token, tolerating common option formatting."""
    cleaned = strip_reasoning(text)
    matches = list(_BINARY_VERDICT.finditer(cleaned))
    if matches:
        return {"is_vulnerable": matches[-1].group(1).upper() == "YES", "parse_error": False}
    numbered = _NUMBERED_VERDICT.fullmatch(cleaned)
    if numbered:
        return {"is_vulnerable": numbered.group(1) == "1", "parse_error": False}
    return {
        "is_vulnerable": None,
        "parse_error": True,
        "raw": (text or "")[:500],
    }
