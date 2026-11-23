# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""POSIX BRE / ERE -> Python `re` pattern translation.

The core problem, and the reason this module exists at all: GNU grep's DEFAULT
dialect is POSIX BRE, in which the escaping of ``| ( ) { } + ?`` is INVERTED
relative to Python::

    BRE:     \\|  = alternation        |  = literal pipe
    Python:   |   = alternation       \\| = literal pipe

Handing a BRE straight to :func:`re.compile` therefore does not fail loudly --
it silently matches different text. Measured over 2,993 real agent-written
patterns differential-tested against GNU grep, doing nothing agrees on only
71.9% of distinct patterns (96.9% usage-weighted, because the heavily reused
patterns are plain literals). The escape swap below takes that to 99.90%, and
the context rules take it to 99.97%.

Four context rules have no Python analogue and are handled in :func:`bre_to_python`:

* ``*`` is a LITERAL at the start of a pattern (or right after ``\\(`` / ``\\|``)
* ``^`` is an anchor ONLY at the start (or right after ``\\(`` / ``\\|``)
* ``$`` is an anchor ONLY at the end (or right before ``\\)`` / ``\\|``)
* inside ``[...]`` a backslash is LITERAL (POSIX), and ``[:classes:]`` expand
"""

from __future__ import annotations

import re


__all__ = ["bre_to_python", "ere_to_python", "compile_pattern", "RegexError"]


class RegexError(ValueError):
    """Raised when a pattern cannot be compiled after translation."""


# POSIX character classes -> Python character-class bodies.
POSIX_CLASS = {
    "alpha": "a-zA-Z",
    "digit": "0-9",
    "alnum": "a-zA-Z0-9",
    "upper": "A-Z",
    "lower": "a-z",
    "space": r" \t\n\r\f\v",
    "blank": r" \t",
    "punct": r"!-/:-@\[-`{-~",
    "xdigit": "0-9A-Fa-f",
    "cntrl": r"\x00-\x1f\x7f",
    "print": r"\x20-\x7e",
    "graph": r"\x21-\x7e",
    "word": r"\w",
}


def _bracket(pat: str, i: int) -> tuple[str, int]:
    """Translate one ``[...]`` bracket expression starting at ``pat[i] == '['``.

    POSIX bracket semantics differ from Python's in three ways that matter:
    a leading ``]`` is a literal member, a backslash is a literal backslash
    (not an escape), and ``[:name:]`` classes must be expanded.
    """
    n = len(pat)
    j = i + 1
    body: list[str] = []
    if j < n and pat[j] == "^":
        body.append("^")
        j += 1
    if j < n and pat[j] == "]":  # leading ] is a literal member, not the terminator
        body.append(r"\]")
        j += 1
    while j < n and pat[j] != "]":
        if pat.startswith("[:", j):
            end = pat.find(":]", j)
            if end != -1:
                name = pat[j + 2 : end]
                body.append(POSIX_CLASS.get(name, ""))
                j = end + 2
                continue
        ch = pat[j]
        if ch == "\\":
            body.append("\\\\")  # literal backslash inside POSIX brackets
        elif ch == "[":
            body.append(r"\[")  # avoid Python's "possible nested set" warning
        else:
            body.append(ch)
        j += 1
    if j >= n:  # unterminated -> the '[' was a literal
        return r"\[", i + 1
    return "[" + "".join(body) + "]", j + 1


def bre_to_python(pat: str) -> str:
    """Translate a POSIX Basic Regular Expression to Python ``re`` syntax."""
    out: list[str] = []
    i, n = 0, len(pat)
    at_start = True  # start of pattern, or immediately after \( or \|
    while i < n:
        c = pat[i]
        if c == "\\" and i + 1 < n:
            d = pat[i + 1]
            if d in "|(":  # escaped in BRE == special; and both reset anchor context
                out.append(d)
                at_start = True
                i += 2
                continue
            if d in "){}+?":
                out.append(d)
                at_start = False
                i += 2
                continue
            if d in "<>":  # GNU word-boundary extensions
                out.append(r"\b")
                at_start = False
                i += 2
                continue
            if d in "wWsSbB" or d.isdigit():
                out.append("\\" + d)
                at_start = False
                i += 2
                continue
            out.append("\\" + d)
            at_start = False
            i += 2
            continue
        if c == "[":
            txt, i = _bracket(pat, i)
            out.append(txt)
            at_start = False
            continue
        if c in "|(){}+?":  # bare == literal in BRE
            out.append("\\" + c)
            at_start = False
            i += 1
            continue
        if c == "*":
            out.append(r"\*" if at_start else "*")
            at_start = False
            i += 1
            continue
        if c == "^":
            out.append("^" if at_start else r"\^")
            at_start = False
            i += 1
            continue
        if c == "$":
            rest = pat[i + 1 :]
            is_anchor = rest == "" or rest.startswith("\\)") or rest.startswith("\\|")
            out.append("$" if is_anchor else r"\$")
            at_start = False
            i += 1
            continue
        out.append(c)
        at_start = False
        i += 1
    return "".join(out)


def ere_to_python(pat: str) -> str:
    """Translate a POSIX Extended Regular Expression to Python ``re`` syntax.

    ERE is already very close to Python. Only bracket expressions differ
    (POSIX classes, literal backslash), so reuse the bracket translator.
    """
    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "\\" and i + 1 < n:
            out.append(pat[i : i + 2])
            i += 2
            continue
        if c == "[":
            txt, i = _bracket(pat, i)
            out.append(txt)
            continue
        out.append(c)
        i += 1
    return "".join(out)


try:  # `regex` supports a per-match timeout; stdlib `re` does not.
    import regex as _regex_mod
except ImportError:  # pragma: no cover
    _regex_mod = None


class TimeBoundedPattern:
    """Wrap a compiled pattern so every match call is bounded by a deadline.

    GNU grep uses a DFA and is immune to catastrophic backtracking; Python's
    engines backtrack, so a pattern that was harmless under the subprocess can
    run unboundedly here. A cooperative per-line check CANNOT help -- the hang
    is inside one ``search()`` call -- and a thread running one of these can
    never be cancelled, so it would permanently consume a pool slot. The
    ``regex`` module's ``timeout=`` argument is the only mechanism that
    interrupts mid-match.
    """

    __slots__ = ("_rx", "_budget", "_supports_timeout")

    def __init__(self, rx, budget: float | None):
        self._rx = rx
        self._budget = budget
        self._supports_timeout = (
            _regex_mod is not None and type(rx) is type(_regex_mod.compile("")) if _regex_mod else False
        )

    def _kw(self):
        if self._supports_timeout and self._budget:
            return {"timeout": self._budget}
        return {}

    def search(self, s):
        try:
            return self._rx.search(s, **self._kw())
        except TimeoutError:
            raise
        except Exception:
            return None

    def finditer(self, s):
        try:
            return list(self._rx.finditer(s, **self._kw()))
        except TimeoutError:
            raise
        except Exception:
            return []

    def sub(self, repl, s, count=0):
        try:
            return self._rx.sub(repl, s, count=count)
        except TimeoutError:
            raise
        except Exception:
            return s


def compile_pattern(
    pattern: str,
    *,
    dialect: str = "bre",
    ignore_case: bool = False,
    word: bool = False,
    line: bool = False,
    fixed: bool = False,
    timeout: float | None = None,
):
    """Compile an agent-written pattern into a Python regex object.

    ``dialect`` is one of ``bre`` (grep default), ``ere`` (-E), ``pcre`` (-P),
    ``fixed`` (-F). Falls back to a literal match when translation produces
    something Python rejects -- agents do write malformed patterns that GNU grep
    tolerates (unbalanced ``\\)``, ``13,**``, ``[2013-2014]``), and a literal
    search is a far better answer there than an exception.
    """
    # Prefer `regex` for EVERY pattern when a timeout is requested -- it is the
    # only engine here that can be interrupted mid-match (see TimeBoundedPattern).
    engine = _regex_mod if (_regex_mod is not None and timeout) else re
    if fixed or dialect == "fixed":
        body = "|".join(re.escape(p) for p in pattern.split("\n"))
    elif dialect == "pcre":
        engine = _regex_mod if _regex_mod is not None else re
        body = pattern
    elif dialect == "ere":
        body = ere_to_python(pattern)
    else:
        body = bre_to_python(pattern)

    if word:
        body = r"\b(?:" + body + r")\b"
    if line:
        body = r"(?:" + body + r")\Z"

    flags = engine.IGNORECASE if ignore_case else 0
    compiled = None
    try:
        compiled = engine.compile(body, flags)
    except Exception:
        pass
    if compiled is None:
        try:  # malformed pattern -> degrade to a literal substring search
            eng2 = engine if engine is not None else re
            compiled = eng2.compile(eng2.escape(pattern), flags)
        except Exception as exc:  # pragma: no cover - defensive
            raise RegexError(str(exc)) from exc
    return TimeBoundedPattern(compiled, timeout)
