# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse PrimeVul's published YES/NO classification response."""

import re
from typing import Optional


# Thinking models emit their chain of thought in these blocks. A draft verdict inside one is not
# the answer, so remove it before looking for an explicit option. (Models that return reasoning as
# a separate Responses-API output item are already handled upstream, by reading only the final
# assistant message.)
_REASONING_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)

# Each tier is (pattern, take_last, numbered), strongest evidence first. The first tier with any
# match decides the verdict.
_VERDICT_TIERS = (
    # 1. The parenthesized option number, e.g. "(1)" or "my answer is (2)". Unambiguous, and
    #    last-wins so a reply that quotes both options before answering is still read correctly.
    (re.compile(r"\(([12])\)"), True, True),
    # 2. The labelled option, e.g. "YES: A security vulnerability detected." The colon marks it as
    #    the option being chosen rather than a word in a sentence.
    (re.compile(r"\b(YES|NO)\s*:", re.IGNORECASE), True, False),
    # 3. A line- or sentence-initial verdict, e.g. "YES. The function ...". FIRST match wins: the
    #    reply leads with its answer, so a trailing "No bounds check is performed." is explanation.
    (re.compile(r"(?:^|[.!?]\s+)\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE), False, False),
    # 4. A standalone token in upper case, e.g. "... so the answer is YES." Case-sensitive on
    #    purpose: this is the tier where a lower-case English "no" would otherwise be read as a
    #    verdict, and the prompt presents both options in upper case.
    (re.compile(r"\b(YES|NO)\b"), True, False),
    # 5. A bare option number as the entire reply, e.g. "1". Deliberately not matched anywhere
    #    else, where a digit could just as easily be a line number or part of a CWE id.
    (re.compile(r"\A\W*([12])\W*\Z"), True, True),
)


def strip_reasoning(text: str) -> str:
    """Drop `<think>`/`<thinking>` blocks from a model message."""
    return _REASONING_BLOCK.sub("", text or "")


def parse_verdict(text: str) -> dict:
    """Return the chosen option, tolerating common option formatting."""
    cleaned = strip_reasoning(text)
    for pattern, take_last, numbered in _VERDICT_TIERS:
        tokens = pattern.findall(cleaned)
        if tokens:
            token = tokens[-1] if take_last else tokens[0]
            return {"is_vulnerable": _is_vulnerable(token, numbered), "parse_error": False}
    return {
        "is_vulnerable": None,
        "parse_error": True,
        "raw": (text or "")[:500],
    }


def _is_vulnerable(token: str, numbered: bool) -> Optional[bool]:
    """Option (1)/YES is the vulnerable label; option (2)/NO is the benign one."""
    return token == "1" if numbered else token.upper() == "YES"
