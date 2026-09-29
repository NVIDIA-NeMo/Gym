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

"""Text-level checks shared by the Lean benchmarks, run before the sandbox call.

These are free; a Mathlib compile is not. Anything that has already lost -- no code block, a
fabricated `axiom`, a weakened statement -- is rejected here rather than after paying for one.

Shared rather than copied per server because these are exactly the functions that accumulate
subtle bugs: comment-aware scanning, fence extraction, `<think>` stripping. Each was wrong at
least once during development, and a fix should land in one place.

Both statement checks are for whole-file tasks, where the model returns the entire file and
could weaken the theorem it was asked to prove. Servers that reassemble the file around a
model-written proof body do not need either.

They differ in what they hold the submission to, because the two task shapes differ:

* ``check_statement_preserved`` splits the *whole reference file* on ``sorry`` and requires
  every remaining fragment back, in order. Right when the reference has exactly one hole, as
  in LeanCat.
* ``check_target_statement_preserved`` requires only the *target declaration's signature*
  back. Right when the file legitimately keeps other ``sorry``s -- a Formal Conjectures file
  pairs a provable lemma with the open conjecture it sanity-checks, so the hole is not
  textually unique and splitting on ``sorry`` throws out honest answers.
"""

import re
from functools import lru_cache
from typing import List, Optional, Tuple


def strip_comments_and_strings(code: str) -> str:
    """Blank out comment and string-literal contents with spaces, preserving offsets and line structure.

    Block comments nest in Lean, so the scanner tracks depth. Doc comments are block comments.
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


def has_unterminated_block_comment(code: str) -> bool:
    """Does a ``/-`` block comment run to end of file without closing?

    Worth asking separately from the checks below. When a model mangles a closing delimiter
    (writing ``- /`` for ``-/``, which happened 48 times in a 12k-rollout run) the rest of the
    file is swallowed by the comment, so ``strip_comments_and_strings`` blanks it and every
    later check sees an empty file. Without this the server reports "statement modified",
    blaming the model for weakening a theorem when it actually just produced malformed output.
    """
    i, n, depth = 0, len(code), 0
    while i < n:
        two = code[i : i + 2]
        if two == "/-":
            depth += 1
            i += 2
        elif two == "-/" and depth:
            depth -= 1
            i += 2
        elif two == "--" and depth == 0:
            j = code.find("\n", i)
            i = n if j < 0 else j + 1
        else:
            i += 1
    return depth > 0


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Drop a reasoning model's thinking so only its answer remains.

    Handles the three shapes seen in practice: closed ``<think>...</think>`` blocks; a bare
    ``</think>`` when the chat template put the opener in the prompt (everything before the
    last close is thinking); and an unclosed ``<think>`` when the model ran out of budget
    (everything after it is thinking).
    """
    text = _THINK_BLOCK_RE.sub("", text)
    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        text = text[closes[-1].end() :]
    opener = _THINK_OPEN_RE.search(text)
    if opener:
        text = text[: opener.start()]
    return text


# Upstream takes the *last* fenced block; an unfenced response falls back to the whole text.
# Upstream's regex, verbatim (`scripts/eval_common.py`): the newline after the tag is
# required. Loosening it makes every fenced block in any language a candidate for "the last
# fenced block" and captures the language tag as code -- a reply that appends a ```bash hint
# after its Lean file would be scored on the bash. Single-line fences are upstream's loss too.
_CODE_BLOCK_RE = re.compile(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", re.DOTALL)

# An unfenced answer counts as a Lean file if some line opens with a file-level keyword.
# `open` and `def` start English sentences as often as Lean files, so they only count when the
# line looks like Lean: a declaration keyword followed by an identifier, or `open` followed by
# one or more capitalised namespaces.
_LEAN_FILE_START_RE = re.compile(
    r"^\s*(?:import\s+[A-Z]|universe\s+\w|variable\s*[\(\{\[]"
    r"|open\s+[A-Z][\w.]*(?:\s+[A-Z][\w.]*)*\s*$"
    # A declaration is a keyword, an optional name, then binders or a type ascription --
    # `def foo :`, `theorem t (x : T)`, `instance : Category C`. Requiring that is what keeps
    # prose like "def the target theorem is about limits" from reading as Lean.
    r"|(?:theorem|lemma|example|def|abbrev|instance)\s+(?:\w[\w.'’]*\s*)?[({\[:])",
    re.M,
)

# Upstream's shortcut list. Banning the `axiom` keyword does not ban classical reasoning:
# Mathlib's axioms are used by name, not declared.
BANNED_TOKENS = ("sorry", "admit", "axiom", "unsafe")

# For tasks whose file is *allowed* to keep holes elsewhere: `sorry` and `admit` are dropped,
# and what remains is only banned as a declaration the model added, not as a bare word.
DECLARED_SHORTCUT_TOKENS = ("axiom", "unsafe")

# The placeholder upstream uses for the holes in a reference file.
_SORRY_RE = re.compile(r"\bsorry\b")

# A line that opens a top-level declaration. Everything above the first such line is
# preamble (imports, `open`, `variable`, ...) and is checked line-by-line, so the model may
# insert auxiliary code after the imports as the prompt permits.
_DECL_START_RE = re.compile(
    r"^\s*(@\[|attribute\b|theorem\b|lemma\b|example\b|def\b|abbrev\b|instance\b|structure\b"
    r"|class\b|inductive\b|noncomputable\b|private\b|protected\b|opaque\b|axiom\b|macro\b|notation\b)"
)


def extract_lean_code(text: str) -> str:
    """Pull the Lean file out of a model response.

    Thinking is dropped first. Then, in order: the last fenced block in the answer; the
    unfenced answer itself if it reads as a Lean file (some provers, e.g. StepFun-Prover,
    emit their final file bare); the last fenced block anywhere, which is upstream's rule;
    else the raw answer.
    """
    text = text or ""
    answer = strip_thinking(text)
    matches = _CODE_BLOCK_RE.findall(answer)
    if matches:
        return matches[-1].strip()
    if _LEAN_FILE_START_RE.search(answer):
        return answer.strip()
    matches = _CODE_BLOCK_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return answer.strip()


@lru_cache(maxsize=None)
def _banned_token_re(tokens: Tuple[str, ...], declarations_only: bool) -> re.Pattern:
    alternation = "|".join(tokens)
    if declarations_only:
        # A keyword opening a declaration, not the same word used anywhere. `axiom` appears in
        # ordinary Mathlib names (`Classical.axiom_of_choice`), and a file that is allowed to
        # keep holes must not be rejected for mentioning one.
        return re.compile(rf"^\s*({alternation})\s", re.MULTILINE)
    return re.compile(rf"\b({alternation})\b")


def find_banned_declarations(
    code: str,
    tokens: Tuple[str, ...] = BANNED_TOKENS,
    *,
    declarations_only: bool = False,
) -> List[str]:
    """Return the shortcut keywords present in ``code``, ignoring comments and strings.

    ``tokens`` and ``declarations_only`` exist because the benchmarks disagree on what counts
    as cheating: a single-hole task bans ``sorry`` outright, while a task whose file keeps
    other holes on purpose can only ban a *declaration* the model added. See
    ``DECLARED_SHORTCUT_TOKENS``.
    """
    stripped = strip_comments_and_strings(code)
    pattern = _banned_token_re(tuple(tokens), declarations_only)
    return sorted({match.group(1) for match in pattern.finditer(stripped)})


def _normalize(text: str) -> str:
    """Collapse every whitespace run to a single space, so a re-wrapped signature still matches."""
    return " ".join(text.split())


def split_preamble_and_body(formal_statement: str) -> Tuple[List[str], str]:
    """Split a reference file into its preamble lines (checked individually) and its declaration body."""
    lines = strip_comments_and_strings(formal_statement).splitlines()
    for idx, line in enumerate(lines):
        if _DECL_START_RE.match(line):
            preamble = [ln for ln in lines[:idx] if ln.strip()]
            return preamble, "\n".join(lines[idx:])
    return [ln for ln in lines if ln.strip()], ""


def check_statement_preserved(formal_statement: str, submission: str) -> Tuple[bool, Optional[str]]:
    """Check the submission kept the reference statement, assumptions and definitions.

    The reference is split on ``sorry``; every remaining segment must appear in the
    submission in order and without overlap, whitespace-normalised. Preamble lines are
    matched individually so the model may add imports and auxiliary declarations.

    Returns ``(True, None)`` or ``(False, reason)``.
    """
    submission_norm = _normalize(strip_comments_and_strings(submission))
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


# `lemma` is notation for `theorem` in Lean 4 -- interchangeable, and neither weakens anything.
# Comparing them verbatim rejected 53 correct submissions in a 12k-rollout Formal Conjectures
# run, purely because the model wrote `theorem` where upstream wrote `lemma`.
_DECL_KEYWORD_RE = re.compile(r"\blemma\b")


def _normalize_signature(text: str) -> str:
    """Whitespace- and keyword-normalise a declaration so cosmetic edits do not read as edits."""
    stripped = _DECL_KEYWORD_RE.sub("theorem", strip_comments_and_strings(text))
    return _normalize(stripped)


def check_target_statement_preserved(target_statement: str, submission: str) -> Tuple[bool, Optional[str]]:
    """Check the submission still contains the target declaration's signature, unaltered.

    For files that legitimately keep other ``sorry``s, where
    :func:`check_statement_preserved`'s split on ``sorry`` does not apply: it would demand the
    surrounding open conjectures back fragment-by-fragment and reject honest answers (measured
    on Formal Conjectures: 23 of 100).

    Only the target's recorded signature has to survive. Comments, indentation and
    ``lemma``/``theorem`` are normalised away; a changed hypothesis, binder, conclusion or name
    is not. This exists because the whole-file format lets a model weaken the theorem and hand
    back something that compiles, which the compiler by definition cannot catch.

    Returns ``(True, None)`` or ``(False, reason)``.
    """
    needle = _normalize_signature(target_statement)
    if not needle:
        return False, "No target statement recorded for this task."
    if needle not in _normalize_signature(submission):
        return False, f"The target statement was altered or removed: expected {needle[:100]!r}"
    return True, None
