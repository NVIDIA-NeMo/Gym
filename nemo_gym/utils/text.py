# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Text helpers for user-facing messages."""

import difflib
import json
import re
from typing import Any, Iterable


def did_you_mean(value: str, candidates: Iterable[str]) -> str:
    """A ` Did you mean \\`X\\`?` fragment for the closest candidate to `value`, or `""` if none is close enough."""
    # Exclude the exact input so it is not suggested as its own correction.
    matches = difflib.get_close_matches(
        value,
        [candidate for candidate in candidates if candidate != value],
        n=1,
    )
    return f" Did you mean `{matches[0]}`?" if matches else ""


def hydra_quote(value: str) -> str:
    """Single-quote `value` as a literal for Hydra's override grammar, so it parses back to the same string.

    `json.dumps` is not that encoder: Hydra keeps `\\uXXXX`, `\\t` and a doubled backslash literally, so a Unicode
    or backslash-containing filename would reach the command under another name. Hydra un-escapes a backslash
    only when its run ends at a quote character, so only those runs (and the quote itself) are escaped here.
    Everything else, Unicode included, is passed through unchanged.
    """
    escaped = re.sub(r"(\\*)(')", lambda m: m.group(1) * 2 + "\\'", value)
    escaped = re.sub(r"(\\+)$", lambda m: m.group(1) * 2, escaped)
    return f"'{escaped}'"


def _hydra_parses_to(text: str, expected: Any) -> bool:
    """Whether Hydra's override grammar reads `text` back as exactly `expected`, type included."""
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.errors import HydraException

    try:
        parsed = OverridesParser.create().parse_overrides([f"+setting={text}"])[0].value()
    except HydraException:
        return False
    return json.dumps(parsed, sort_keys=True, default=str) == json.dumps(expected, sort_keys=True, default=str)


def hydra_literal(value: Any) -> str:
    """`value` as an override value that Hydra parses back to the same value and type.

    A string is written as it is when Hydra reads it back unchanged, and as a quoted literal otherwise, so
    `"1"` stays a string and `nightly(run)` survives the grammar. The check is Hydra's own parser, not a guess
    at its rules. Numbers, booleans, null and lists of those are written in their own syntax; anything else
    raises `ValueError`.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value if _hydra_parses_to(value, value) else hydra_quote(value)
    if isinstance(value, (list, tuple)):
        for quote_strings in (False, True):
            items = (hydra_quote(v) if quote_strings and isinstance(v, str) else hydra_literal(v) for v in value)
            text = "[" + ",".join(items) + "]"
            if _hydra_parses_to(text, list(value)):
                return text
    raise ValueError(f"{value!r} cannot be written as a Hydra override value.")


def listed_names(names: list[str], *, limit: int = 10) -> str:
    """Quote and join names for an error message, showing at most ``limit`` of them."""
    shown = ", ".join(repr(name) for name in names[:limit])
    return f"{shown} and {len(names) - limit} more" if len(names) > limit else shown
