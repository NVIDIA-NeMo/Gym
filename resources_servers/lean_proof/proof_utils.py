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

``check_statement_preserved`` is for whole-file tasks, where the model returns the entire file
and could weaken the theorem it was asked to prove. Servers that reassemble the file around a
model-written proof body do not need it.
"""

import re
from typing import List, Optional, Tuple


# A Lean char literal: `'a'`, `'\n'`, `'\x41'`, `'\u03b1'`, and `'"'` / `'\"'`. Without this the scanner
# reads the quote inside `'"'` as a string opener and blanks everything up to the next `"`.
_CHAR_LITERAL_RE = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'", re.DOTALL)

# `r"..."` and `r#"..."#`: no escapes, so `r"\"` is a complete string, not an open one.
_RAW_STRING_OPEN_RE = re.compile(r'r(#*)"')


def _is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_'"


def _blank(text: str) -> str:
    return "".join("\n" if c == "\n" else " " for c in text)


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
            elif ch == "'" and not (i and _is_ident_char(code[i - 1])) and (m := _CHAR_LITERAL_RE.match(code, i)):
                out.append(_blank(m.group(0)))
                i = m.end()
            elif ch == "r" and not (i and _is_ident_char(code[i - 1])) and (m := _RAW_STRING_OPEN_RE.match(code, i)):
                close = code.find('"' + m.group(1), m.end())
                end = n if close < 0 else close + 1 + len(m.group(1))
                out.append(_blank(code[i:end]))
                i = end
            elif ch == '"':
                in_string = True
                out.append(" ")
                i += 1
            else:
                out.append(ch)
                i += 1

    return "".join(out)


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
#
# Also the ways a file can run code while Lean compiles it (`#eval`, `run_cmd`, `elab`, ...) or
# switch off the kernel (`skipKernelTC`): a proof has no use for them, and code that executes
# during elaboration can tamper with the sandbox and with the checks run on the file. No stored
# LeanCat rollout passed while using any of them.
_BANNED_TOKEN_RE = re.compile(
    r"(?<!\w)(#eval|sorry|admit|axiom|unsafe|run_cmd|run_tac|run_elab|elab|elab_rules"
    r"|initialize|builtin_initialize|command_elab|term_elab|skipKernelTC)(?!\w)"
)

# Appended to the submission and run by the same compile: asks Lean which axioms every declaration
# in the file depends on, so it does not matter how an `axiom` or `sorry` was written or hidden.
# `Lean.ofReduceBool` is what `native_decide` adds; upstream does not ban it, so neither do we.
# The token is random per compile so a submission cannot print the success line itself; one that
# stops the file early (`#exit`, which Lean accepts with exit 0) never prints it.
_AXIOM_CHECK = """
open Lean Elab Command in
#eval show CommandElabM Unit from do
  let ok := [``propext, ``Quot.sound, ``Classical.choice, ``Lean.ofReduceBool]
  let mut bad : Array String := #[]
  for (n, _) in (← getEnv).constants.map₂.toList do
    for a in (← collectAxioms n) do
      unless ok.contains a do bad := bad.push s!"{n} uses {a}"
  IO.println (if bad.isEmpty then "%TOKEN%_OK" else s!"%TOKEN%_BAD: {bad}")
"""


def axiom_check_suffix(token: str) -> str:
    return _AXIOM_CHECK.replace("%TOKEN%", token)


def axiom_check_passed(stdout: str, token: str) -> bool:
    return f"{token}_OK" in stdout and f"{token}_BAD" not in stdout


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


def find_banned_declarations(code: str) -> List[str]:
    """Return the shortcut keywords present in ``code``, ignoring comments and strings."""
    stripped = strip_comments_and_strings(code)
    return sorted({match.group(1) for match in _BANNED_TOKEN_RE.finditer(stripped)})


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
