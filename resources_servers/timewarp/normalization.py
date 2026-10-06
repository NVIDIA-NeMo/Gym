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
#
# Copied from TimeWarp, src/browsergym/timewarp/normalization.py at commit 4978e69
# (https://github.com/sparklabutah/timewarp), which declares the MIT License (reproduced
# below). The copy is unmodified apart from formatting; it is vendored because it defines
# how TimeWarp's deterministic verifiers read a free-text answer. Modifications Copyright (c)
# 2026 NVIDIA CORPORATION & AFFILIATES and contributors, licensed under the Apache License 2.0
# (SPDX header above).
#
# MIT License
#
# Copyright (c) 2026 The TimeWarp Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Deterministic answer-normalization utilities for TimeWarp verifiers.

Standard library only -- no new dependencies.

TimeWarp agents answer in free text via ``send_msg_to_user``, so a verifier
cannot assume the answer is a bare value: it is usually a sentence (often with
markdown) that *contains* the value. These helpers canonicalize such text and
provide word-boundary containment so that reference tokens match real answers
without matching lookalikes (``"10"`` must not match ``"100"``, ``"no"`` must
not match ``"north"``).

The design mirrors webarena_verified's normalization layer (type-aware
canonicalization, any-of alternatives, regex leaves) scoped down to the value
types that actually occur in TimeWarp golds: strings, numbers and lists.
"""

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional, Sequence


# Separator for "any of these is acceptable" alternatives inside a single
# reference entry, e.g. "kangaroos |OR| kangaroo". Same spelling as WebArena.
# Surrounding whitespace is optional when splitting: an entry written
# "kangaroos|OR|kangaroo" would otherwise become a literal that can never match,
# and that failure is silent at scoring time.
ALTERNATIVE_SEPARATOR = " |OR| "
_ALTERNATIVE_RE = re.compile(r"\s*\|OR\|\s*")


def split_alternatives(entry: str) -> list:
    """Split a reference entry into its interchangeable alternatives."""
    return [part for part in _ALTERNATIVE_RE.split(entry) if part.strip()]


# Typographic characters that carry no semantic weight but break naive matching.
_CHAR_MAP = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    " ": " ",
    "​": "",
    "﻿": "",
    "…": "...",
}

# Stripped from both ends of a value. Includes markdown emphasis and code
# fences because agents routinely answer with "**Biology**" or "`Biology`".
# Deliberately excludes "$", "%" and "#", which can be semantically load-bearing.
_EDGE_CHARS = " \t\r\n\"'`*_.,;:!?()[]{}<>«»"


def fold_unicode(text: str) -> str:
    """Map typographic characters to ASCII and drop accents/symbols.

    Accents are removed via canonical decomposition (``Beyoncé`` ->
    ``Beyonce``). Remaining non-ASCII punctuation and symbols (``™``, ``®``,
    ``°``) become spaces; non-ASCII *letters* (e.g. CJK, Cyrillic) are kept so
    they can still be matched literally.
    """
    for src, dst in _CHAR_MAP.items():
        text = text.replace(src, dst)

    decomposed = unicodedata.normalize("NFD", text)
    kept = []
    for char in decomposed:
        if unicodedata.combining(char):
            continue
        if ord(char) < 128:
            kept.append(char)
            continue
        if unicodedata.category(char)[0] in ("P", "S"):
            kept.append(" ")
            continue
        kept.append(char)
    return "".join(kept)


def normalize_text(text: Optional[str]) -> str:
    """Canonicalize a value or an answer for matching.

    Folds unicode, collapses all whitespace runs to a single space, strips
    surrounding quotes/markdown/punctuation, and lowercases. Internal
    punctuation is preserved so that ``9.99`` and ``u.s.`` survive intact.
    """
    if text is None:
        return ""
    folded = fold_unicode(str(text))
    collapsed = re.sub(r"\s+", " ", folded)
    return collapsed.strip(_EDGE_CHARS).lower()


# A sentence boundary is a terminator that is not part of a decimal number and
# is followed by whitespace or end-of-string. Closing markdown/quote characters
# may sit between the two, because "**Yes.** ..." is a very common answer shape
# and treating it as one long sentence would defeat the purpose. Splitting
# slightly early (e.g. at "U.S.") is harmless for the intended use -- isolating
# a leading yes/no verdict -- whereas splitting late would not be.
_SENTENCE_END = re.compile(r"""(?<!\d)[.!?](?=[)"'*_`\]]*(?:\s|$))|[\n\r]""")


def first_sentence(text: Optional[str]) -> str:
    """Return the leading sentence of ``text`` (the whole string if unsplittable).

    Used by the ``"scope": "first_sentence"`` option: agents state their verdict
    first and then justify it, and the justification frequently contains tokens
    that contradict a naive whole-answer match ("Yes. It is not listed under
    ...").
    """
    if not text:
        return ""
    match = _SENTENCE_END.search(text)
    if match is None:
        return text
    head = text[: match.start()].strip()
    return head if head else text


def is_regex_entry(entry: str) -> bool:
    """True if ``entry`` is a regex leaf, i.e. explicitly anchored as ``^...$``."""
    entry = entry.strip()
    return len(entry) > 2 and entry.startswith("^") and entry.endswith("$")


def _containment_pattern(reference: str) -> str:
    """Build a word-boundary containment pattern for a normalized reference.

    Runs of hyphens/whitespace in the reference match any run of hyphens or
    whitespace in the answer, so "e-mail" also matches "e mail". The custom
    lookaround guards (rather than ``\\b``) keep "10" from matching inside "100"
    even when the reference ends in punctuation such as "u.s.".
    """
    parts = [re.escape(part) for part in re.split(r"[-\s]+", reference) if part]
    if not parts:
        return ""
    body = r"[-\s]+".join(parts)
    return r"(?<![a-z0-9])" + body + r"(?![a-z0-9])"


def find_reference(text: str, reference: str) -> Optional[int]:
    """Return the earliest match offset of ``reference`` in ``text``, else ``None``.

    ``text`` is raw answer text; both sides are normalized here. ``reference``
    may be a ``^regex$`` leaf, which is matched against the whole normalized
    answer.
    """
    normalized_text = normalize_text(text)
    reference = reference.strip()
    if not reference:
        return None

    if is_regex_entry(reference):
        try:
            match = re.search(reference, normalized_text, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid regex reference {reference!r}: {exc}") from exc
        return match.start() if match else None

    pattern = _containment_pattern(normalize_text(reference))
    if not pattern:
        return None
    match = re.search(pattern, normalized_text)
    return match.start() if match else None


def find_entry(text: str, entry: str) -> Optional[int]:
    """Like :func:`find_reference` but ``entry`` may hold ``|OR|`` alternatives.

    Returns the earliest offset among the alternatives that match.
    """
    offsets = [
        offset
        for alternative in split_alternatives(entry)
        for offset in (find_reference(text, alternative),)
        if offset is not None
    ]
    return min(offsets) if offsets else None


def contains_entry(text: str, entry: str) -> bool:
    """True if any ``|OR|`` alternative of ``entry`` occurs in ``text``."""
    return find_entry(text, entry) is not None


def equals_entry(text: str, entry: str) -> bool:
    """True if ``text`` equals any ``|OR|`` alternative of ``entry`` after normalization."""
    normalized_text = normalize_text(text)
    for alternative in split_alternatives(entry):
        alternative = alternative.strip()
        if is_regex_entry(alternative):
            if re.fullmatch(alternative, normalized_text, re.IGNORECASE):
                return True
        elif normalized_text == normalize_text(alternative):
            return True
    return False


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #

_UNIT_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_BIG_SCALE_WORDS = {
    "thousand": Decimal(1000),
    "million": Decimal(10**6),
    "billion": Decimal(10**9),
    "trillion": Decimal(10**12),
}

# Single-letter magnitude suffixes are ambiguous in prose ("5 m" is five
# million or five metres), so both readings are emitted -- see extract_numbers.
_SUFFIX_SCALES = {
    "bn": Decimal(10**9),
    "k": Decimal(1000),
    "m": Decimal(10**6),
    "b": Decimal(10**9),
    "t": Decimal(10**12),
}

_DIGIT_NUMBER = re.compile(r"(?<![\w.])(-?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
_SCALE_WORD_TAIL = re.compile(
    r"(?:st|nd|rd|th)?\s*(hundred|thousand|million|billion|trillion)(?![a-z])",
    re.IGNORECASE,
)
_SCALE_SUFFIX_TAIL = re.compile(r"(?:st|nd|rd|th)?\s?(bn|k|m|b|t)(?![a-z0-9])", re.IGNORECASE)


def _extract_digit_numbers(text: str) -> list:
    values = []
    for match in _DIGIT_NUMBER.finditer(text):
        sign, integer, fraction = match.groups()
        try:
            base = Decimal(f"{sign}{integer.replace(',', '')}{fraction or ''}")
        except InvalidOperation:
            continue

        tail = text[match.end() : match.end() + 24]
        scale_match = _SCALE_WORD_TAIL.match(tail)
        if scale_match:
            word = scale_match.group(1).lower()
            if word == "hundred":
                values.append(base * 100)
            else:
                values.append(base * _BIG_SCALE_WORDS[word])
            continue

        suffix_match = _SCALE_SUFFIX_TAIL.match(tail)
        if suffix_match:
            # Ambiguous: emit both the plain and the scaled reading.
            values.append(base)
            values.append(base * _SUFFIX_SCALES[suffix_match.group(1).lower()])
            continue

        values.append(base)
    return values


def _extract_word_numbers(text: str) -> list:
    """Parse spelled-out cardinals ("thirteen", "seven million", "one hundred two")."""
    values = []
    total = Decimal(0)
    group = Decimal(0)
    active = False

    def flush():
        nonlocal total, group, active
        if active:
            values.append(total + group)
        total = Decimal(0)
        group = Decimal(0)
        active = False

    for token in re.findall(r"[a-z]+", text.lower()):
        if token in _UNIT_WORDS:
            group += Decimal(_UNIT_WORDS[token])
            active = True
        elif token == "hundred" and active:
            group = (group if group else Decimal(1)) * 100
        elif token in _BIG_SCALE_WORDS and active:
            total += (group if group else Decimal(1)) * _BIG_SCALE_WORDS[token]
            group = Decimal(0)
        elif token == "and" and active:
            continue
        else:
            flush()
    flush()
    return values


def extract_numbers(text: Optional[str]) -> list:
    """Extract every numeric reading from free text as :class:`~decimal.Decimal`.

    Handles thousands separators, decimals, currency symbols, percentages,
    ordinals, magnitude words ("57.7 million"), ambiguous magnitude suffixes
    ("7m" -> both 7 and 7000000) and spelled-out cardinals ("thirteen").
    """
    if not text:
        return []
    folded = fold_unicode(str(text))
    values = _extract_digit_numbers(folded) + _extract_word_numbers(folded)

    unique = []
    seen = set()
    for value in values:
        key = value.normalize()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def to_decimal(value) -> Decimal:
    """Coerce a JSON-sourced reference value to :class:`~decimal.Decimal`."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError(f"Expected a number, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        candidates = extract_numbers(value)
        if len(candidates) != 1:
            raise ValueError(
                f"Reference number {value!r} is not an unambiguous number (parsed {len(candidates)} candidates)"
            )
        return candidates[0]
    raise ValueError(f"Cannot interpret {value!r} as a number")


def numbers_match(
    candidate: Decimal,
    expected: Decimal,
    rel_tolerance: Optional[float] = None,
    abs_tolerance: Optional[float] = None,
) -> bool:
    """Compare two numbers, exactly by default.

    If either tolerance is given, the candidate passes when it falls inside
    *any* supplied tolerance.
    """
    if rel_tolerance is None and abs_tolerance is None:
        return candidate == expected

    difference = abs(candidate - expected)
    if abs_tolerance is not None and difference <= Decimal(str(abs_tolerance)):
        return True
    if rel_tolerance is not None:
        allowed = abs(expected) * Decimal(str(rel_tolerance))
        if difference <= allowed:
            return True
    return False


def scope_text(text: Optional[str], scope: str) -> str:
    """Apply a matching scope to the raw answer text."""
    if scope == "full":
        return text or ""
    if scope == "first_sentence":
        return first_sentence(text)
    raise ValueError(f"Unknown scope {scope!r}; expected 'full' or 'first_sentence'")


def as_entry_list(value, field: str) -> Sequence[str]:
    """Normalize a reference field that may be a single string or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"{field} entries must be strings, got {item!r}")
        return list(value)
    raise ValueError(f"{field} must be a string or list of strings, got {value!r}")
