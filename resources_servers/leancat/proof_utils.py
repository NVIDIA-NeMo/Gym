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

"""Utilities for LeanCat proof processing and evaluation.

The counterpart of ``math_formal_lean/proof_utils.py``, and deliberately laid out the same
way: code extraction, then the checks ``app.py`` runs before it compiles anything.

LeanCat calls a submission valid only when all five of these hold (``EVALUATION.md``
upstream): it compiles under the pinned toolchain, it preserves the statement,
definitions and assumptions, it contains no ``sorry``/``admit``/``axiom``/``unsafe``,
it pulls in no unauthorised dependency, and it keeps the mathematical intent.

Compilation is the sandbox's job (``sandbox_client.py``). This module covers the two
criteria that are mechanically checkable from text alone -- shortcut declarations and
statement preservation -- because a proof that compiles only because the model deleted
a hypothesis is worse than no proof at all: it scores as a success.

``extract_lean_code`` is this server's ``clean_formal_generation``. It keeps the upstream
name and the upstream regex rather than math_formal_lean's, because agreeing with
LeanCat's own ``scripts/eval_common.py`` on which block of a generation gets scored
matters more here than matching a sibling server's spelling.
"""

import re
from typing import List, Optional, Tuple


# Upstream takes the *last* fenced block, on the theory that a reasoning model's final
# block is its answer. An unfenced response falls back to the whole text.
_CODE_BLOCK_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", re.DOTALL)

# Upstream's shortcut list. `axiom` is here because *declaring* an axiom discharges any
# goal; Mathlib's own axioms (`Classical.choice`, ...) are reached by name, not by the
# keyword, so banning the keyword does not ban ordinary classical reasoning.
_BANNED_TOKEN_RE = re.compile(r"\b(sorry|admit|axiom|unsafe)\b")

# The holes in a reference file. Only `sorry` is used as a placeholder upstream, so
# statement preservation splits on it alone rather than on the whole banned list.
_SORRY_RE = re.compile(r"\bsorry\b")

# A line that opens a top-level declaration. Everything above the first such line is
# preamble (imports, `open`, `variable`, ...) and is checked line-by-line rather than as
# one contiguous block, so a model may legitimately insert auxiliary code after the
# imports -- which the LeanCat prompt explicitly permits.
_DECL_START_RE = re.compile(
    r"^\s*(@\[|attribute\b|theorem\b|lemma\b|example\b|def\b|abbrev\b|instance\b|structure\b"
    r"|class\b|inductive\b|noncomputable\b|private\b|protected\b|opaque\b|axiom\b|macro\b|notation\b)"
)


def extract_lean_code(text: str) -> str:
    """Pull the Lean file out of a model response (last fenced block, else raw text)."""
    matches = _CODE_BLOCK_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


def strip_lean_comments_and_strings(code: str) -> str:
    """Blank out comment and string-literal *contents*, preserving line structure.

    Both checks below must not fire on the word ``sorry`` inside a comment or a string,
    and statement preservation must not fail merely because the model annotated the
    statement it copied. Replacing with spaces rather than deleting keeps offsets and
    line counts intact, so anything reported against this text still lines up with the
    original.

    Block comments nest in Lean (``/- /- -/ -/``), so the scanner tracks depth. Doc
    comments (``/-- ... -/``) are just block comments as far as this is concerned.
    """
    out: List[str] = []
    i = 0
    n = len(code)
    depth = 0  # block-comment nesting depth
    in_line_comment = False
    in_string = False

    while i < n:
        ch = code[i]
        two = code[i : i + 2]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            else:
                out.append(" ")
            i += 1
        elif depth > 0:
            if two == "/-":
                depth += 1
                out.append("  ")
                i += 2
            elif two == "-/":
                depth -= 1
                out.append("  ")
                i += 2
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
        elif in_string:
            if ch == "\\" and i + 1 < n:
                # Consume the escape as a unit so a `\"` does not close the string.
                out.append("  ")
                i += 2
            elif ch == '"':
                in_string = False
                out.append(" ")
                i += 1
            else:
                out.append("\n" if ch == "\n" else " ")
                i += 1
        else:
            if two == "/-":
                depth = 1
                out.append("  ")
                i += 2
            elif two == "--":
                in_line_comment = True
                out.append("  ")
                i += 2
            elif ch == '"':
                in_string = True
                out.append(" ")
                i += 1
            else:
                out.append(ch)
                i += 1

    return "".join(out)


def find_banned_tokens(code: str) -> List[str]:
    """Return the shortcut keywords present in ``code``, ignoring comments and strings."""
    stripped = strip_lean_comments_and_strings(code)
    return sorted({match.group(1) for match in _BANNED_TOKEN_RE.finditer(stripped)})


def _normalize(text: str) -> str:
    """Collapse every whitespace run to a single space.

    Lean is whitespace-sensitive for tactic blocks but not for the declaration
    signatures being compared here, and models routinely re-wrap a long signature they
    copied faithfully. Normalising means such a re-wrap is not scored as tampering.
    """
    return " ".join(text.split())


def split_preamble_and_body(formal_statement: str) -> Tuple[List[str], str]:
    """Split a reference file into its preamble lines and its declaration body.

    The preamble is everything above the first top-level declaration: ``import``,
    ``open``, ``variable``, ``universe``, ``namespace``, ``section``. It is returned as
    individual lines because each is checked independently -- the model is allowed to
    add imports and to slot auxiliary declarations in between.
    """
    lines = strip_lean_comments_and_strings(formal_statement).splitlines()
    for idx, line in enumerate(lines):
        if _DECL_START_RE.match(line):
            preamble = [ln for ln in lines[:idx] if ln.strip()]
            return preamble, "\n".join(lines[idx:])
    return [ln for ln in lines if ln.strip()], ""


def check_statement_preserved(formal_statement: str, submission: str) -> Tuple[bool, Optional[str]]:
    """Check the submission kept the reference statement, assumptions and definitions.

    The reference file is the target with holes: split it on ``sorry`` and every
    remaining segment is text the model was told to copy verbatim. Requiring those
    segments to appear *in order and without overlap* in the submission is exactly the
    condition "you filled the holes and changed nothing else", and it generalises for
    free to the nine LeanCat problems that carry more than one ``sorry``.

    Insertions are allowed everywhere the prompt allows them: before the declarations
    (preamble lines are matched individually) and between them (the in-order scan does
    not require segments to be adjacent).

    Returns ``(True, None)`` or ``(False, reason)``.
    """
    submission_norm = _normalize(strip_lean_comments_and_strings(submission))
    preamble, body = split_preamble_and_body(formal_statement)

    for line in preamble:
        needle = _normalize(line)
        if needle and needle not in submission_norm:
            return False, f"Preamble line missing from submission: {needle!r}"

    cursor = 0
    for segment in _SORRY_RE.split(body):
        if not segment.strip():
            continue
        needle = _normalize(segment)
        found = submission_norm.find(needle, cursor)
        if found < 0:
            return False, f"Statement fragment missing or altered: {needle!r}"
        cursor = found + len(needle)

    return True, None
